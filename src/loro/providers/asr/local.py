"""Local Nemotron ASR provider (overlapping windows + worker subprocess).

Long audio is processed in overlapping windows (R10): each window is cut to its
own wav, transcribed by one worker invocation (model loaded once), and persisted
as `asr/win_NNNN.json` the moment its NDJSON line arrives — a crash mid-stage
loses at most one window. Windows are merged at a segment boundary near the
overlap midpoint.

The windowing toolkit (`window_bounds`/`merge_windows`/`_win_artifact`/
`MERGE_EPS`) stays in the asr node and is shared, not owned (KTD3, R7); this
provider drives it (imported lazily inside `transcribe` to avoid the node<->
provider import cycle) and owns the Nemotron worker subprocess and its `WORKER`
path. The asr node owns the cross-engine tail (EN SRT + return).
"""

import json
import logging
import subprocess
import threading
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.harness.retry import StageError, with_retry
from loro.providers.base import AsrResult
from loro.state import Segment
from loro.utils import ffmpeg
from loro.workers.nemotron_worker import MODEL_ID

log = logging.getLogger("loro.asr")

WORKER = Path(__file__).resolve().parents[2] / "workers" / "nemotron_worker.py"


def _run_worker(cfg: Config, asr_dir: Path, jobs: list[dict]) -> None:
    """One worker invocation over every missing window; each NDJSON line is
    persisted as its window artifact immediately."""
    cmd = [str(cfg.nemotron_python), str(WORKER)] + [job["wav"] for job in jobs]
    budget = cfg.asr_timeout_base + cfg.asr_timeout_per_sec * sum(
        job["end"] - job["start"] for job in jobs
    )
    by_path = {job["wav"]: job for job in jobs}
    stderr_log = asr_dir / "worker.log"

    timed_out = threading.Event()
    with open(stderr_log, "a", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err, text=True)
        killer = threading.Timer(budget, lambda: (timed_out.set(), proc.kill()))
        killer.start()
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    job = by_path[payload["path"]]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Library noise on stdout, or a line truncated by the
                    # kill timer — never fatal; missing windows surface below
                    log.warning("skipping non-NDJSON stdout line from worker: %.120s", line)
                    continue
                offset = job["start"]
                data = {
                    "start": job["start"],
                    "end": job["end"],
                    "segments": [
                        {"start": s["start"] + offset, "end": s["end"] + offset, "text": s["text"]}
                        for s in payload["segments"]
                    ],
                    "words": [
                        {"start": w["start"] + offset, "end": w["end"] + offset, "word": w["word"]}
                        for w in payload["words"]
                    ] if payload.get("words") else None,
                }
                artifacts.produce(
                    job["artifact"], job["inputs"], "asr",
                    lambda tmp, data=data: tmp.write_text(
                        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
                    ),
                )
                log.info("window %s transcribed (%.0fs-%.0fs)",
                         job["artifact"].name, job["start"], job["end"])
            returncode = proc.wait()
        finally:
            killer.cancel()
            if proc.poll() is None:
                proc.kill()

    if timed_out.is_set():
        raise TimeoutError(f"nemotron worker exceeded {budget:.0f}s budget")
    if returncode != 0:
        raise StageError("asr", "content", f"worker_exit_{returncode}",
                         f"see {stderr_log}")
    missing = [job["artifact"].name for job in jobs
               if not artifacts.is_valid(job["artifact"], job["inputs"])]
    if missing:
        raise StageError("asr", "content", "missing_windows",
                         f"worker exited 0 but produced no output for {missing}")


class LocalAsrProvider:
    name = "local"
    # The local engine wants the Granite ensemble crosscheck (KTD8); the graph
    # reads this flag instead of `asr_engine == "local"`.
    wants_crosscheck = True
    # The Nemotron path has no language identification, so it requires an explicit
    # --source-lang; preflight rejects source_lang="auto" via this flag (U7, R12).
    detects_language = False

    def transcribe(self, state: dict, cfg: Config, asr_dir) -> AsrResult:
        # The windowing toolkit stays node-side and shared (KTD3); import it lazily
        # to avoid the nodes.asr <-> providers import cycle.
        from loro.nodes.asr import MERGE_EPS, _win_artifact, merge_windows, window_bounds

        python = Path(cfg.nemotron_python)
        audio = state["audio_16k"]
        audio_sha = artifacts.cached_file_sha256(audio)
        duration = ffmpeg.probe_duration(audio)
        bounds = window_bounds(duration, cfg.asr_window, cfg.asr_overlap)
        worker_sha = artifacts.file_sha256(WORKER)
        model_id = MODEL_ID

        jobs = []
        for i, (start, end) in enumerate(bounds):
            inputs = {
                "audio_sha": audio_sha,
                "start": round(start, 3),
                "length": round(end - start, 3),
                "overlap": cfg.asr_overlap,
                "worker_sha": worker_sha,
                "model_id": model_id,
            }
            art = _win_artifact(asr_dir, i)
            if not artifacts.is_valid(art, inputs):
                jobs.append({"index": i, "start": start, "end": end,
                             "artifact": art, "inputs": inputs})

        if jobs:
            if not python.exists():
                raise RuntimeError(
                    f"Nemotron interpreter not found: {python}\n"
                    "Create it with: pyenv virtualenv 3.11.15 nemo && "
                    "~/.pyenv/versions/nemo/bin/pip install 'nemo_toolkit[asr]'\n"
                    "or point NEMOTRON_PYTHON at an env that has nemo_toolkit[asr]."
                )
            for job in jobs:
                if len(bounds) == 1:
                    job["wav"] = str(Path(audio).resolve())
                else:
                    wav = asr_dir / f"win_{job['index']:04d}.wav"
                    ffmpeg.cut_audio(audio, str(wav), job["start"], job["end"])
                    job["wav"] = str(wav)
            log.info("transcribing %d/%d window(s) with Nemotron", len(jobs), len(bounds))

            def attempt() -> None:
                remaining = [j for j in jobs if not artifacts.is_valid(j["artifact"], j["inputs"])]
                if remaining:
                    _run_worker(cfg, asr_dir, remaining)

            with_retry("asr", attempt, attempts=cfg.retry_attempts, base_delay=cfg.retry_base_delay)

        window_payloads = [
            json.loads(_win_artifact(asr_dir, i).read_text(encoding="utf-8"))
            for i in range(len(bounds))
        ]
        # Gather word timestamps (absolute); dedup the overlap regions where a word
        # appears in two adjacent windows. This stream is the input to sentence_seg
        # (which builds the dub backbone) and to the sub-style SRT writers.
        seen: set[float] = set()
        all_words: list[dict] = []
        for payload in window_payloads:
            for w in (payload.get("words") or []):
                key = round(w["start"], 2)
                if key not in seen:
                    seen.add(key)
                    all_words.append(w)
        all_words.sort(key=lambda w: w["start"])

        merge_inputs = {
            "window_hashes": [artifacts.cached_file_sha256(_win_artifact(asr_dir, i))
                              for i in range(len(bounds))],
            "eps": MERGE_EPS,
        }
        manifest = artifacts.produce_json(
            asr_dir / "segments.json", merge_inputs, "asr",
            lambda: {"segments": merge_windows(window_payloads)},
        )

        # Raw acoustic units: sentence_seg consumes these (+ all_words) to produce
        # the sentence backbone, and falls back to them when there is no word timing.
        segments = [
            Segment(index=i, start=s["start"], end=s["end"], text_src=s["text"].strip())
            for i, s in enumerate(seg for seg in manifest["segments"] if seg["text"].strip())
        ]
        if not segments:
            raise RuntimeError("ASR produced no segments — is there speech in the video?")

        log.info("%d raw segment(s) over %d window(s), %d words",
                 len(segments), len(bounds), len(all_words))
        return AsrResult(segments=segments, words=all_words)

    def preflight(self, cfg: Config) -> list[str]:
        """The local engine needs the NeMo interpreter, and Granite (the primary
        verify engine) only when cross-check runs (R9)."""
        problems: list[str] = []
        if not Path(cfg.nemotron_python).exists():
            problems.append(
                f"NeMo interpreter not found: {cfg.nemotron_python} "
                "(create with: pyenv virtualenv 3.11.15 nemo && "
                "~/.pyenv/versions/nemo/bin/pip install 'nemo_toolkit[asr]')"
            )
        # Granite is the primary verify engine; only needed when cross-check runs
        if cfg.enable_cross_check:
            if not Path(cfg.granite_python).exists():
                problems.append(
                    f"Granite interpreter not found: {cfg.granite_python} "
                    "(create with: pyenv virtualenv 3.14.5 granite && "
                    "~/.pyenv/versions/granite/bin/pip install torch torchaudio "
                    "'transformers>=4.52' soundfile accelerate peft librosa; "
                    "or disable cross-check with --no-cross-check)"
                )
            else:
                log.info("note: on the first run the Granite worker downloads the ~5GB model %s "
                         "into the HF cache (preflight cannot gate this step)",
                         cfg.granite_model_id)
        return problems
