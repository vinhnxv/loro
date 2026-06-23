"""Cross-check stage: a weighted three-model ensemble verifies Nemotron's
text per segment (R27-R30). Nemotron keeps timing authority and is the pivot
reading; Granite (the primary verify engine, run via a subprocess worker in
its own venv) re-transcribes every clip; Gemma is a lazy arbiter, consulted
only for segments where Nemotron and Granite diverge on a content word (R29).

Two phases keep peak RAM down — the Granite worker exits before Gemma runs:

  Phase 0  cut one clip per segment (>30s split at silence, R8a), shared by
           both engines.
  Phase 1  one Granite worker invocation over every segment missing its
           reading; each reading is persisted the moment it arrives.
  Phase 2  per segment, vote: skip Gemma when N and G agree, otherwise have
           Gemma re-listen and run the weighted vote (diff.vote3).

Both verify engines transcribe independently — their prompts never contain
Nemotron's text, to avoid anchoring.

Degradation never skips a segment (R8/R32): Granite unavailable falls back to
the two-way Nemotron x Gemma comparison (diff.compare); Gemma failing when its
vote was needed falls back to the Nemotron x Granite vote; both failing keeps
Nemotron flagged low-confidence. Failures write a *retryable* marker (R20) and
count strikes toward the abort window (R5a).
"""

import json
import logging
import os
import re
import subprocess
import threading
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts, diff
from loro.harness.ledger import AbortRun, SkipLedger
from loro.harness.retry import StageError, with_retry
from loro.services import llm
from loro.state import DubState, Segment, segment_id
from loro.utils import ffmpeg, srt

log = logging.getLogger("loro.crosscheck")

PROMPT = (
    "Transcribe this English audio verbatim. Reply with the transcript text "
    "only — no introduction, no quotes, no commentary."
)

GRANITE_WORKER = Path(__file__).resolve().parents[1] / "workers" / "granite_worker.py"

GRANITE_BASE_PROMPT = "transcribe the speech to text."


def granite_prompt(keywords: list[str], cap: int) -> str:
    """Granite's keyword-biased ASR prompt (R31). The model-card syntax is
    `transcribe the speech to text. Keywords: <kw1>, <kw2>, ...`; an empty
    list falls back to plain transcription. Capped to avoid over-biasing."""
    kws = [k.strip() for k in keywords if k.strip()][:cap]
    if not kws:
        return GRANITE_BASE_PROMPT
    return f"{GRANITE_BASE_PROMPT} Keywords: {', '.join(kws)}"


# --- Embedded/sidecar subtitle support (R34/R35/R36) ---

_SENT_SPLIT_RE = re.compile(r"[.!?]+")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'/+.\-]*")
_CAP_TOKEN_RE = re.compile(r"\b[A-Z][A-Za-z0-9'/+.\-]{2,}\b")


def subtitle_keywords(full_text: str) -> list[str]:
    """Heuristic terminology mined from subtitle text (R36): acronyms,
    mid-sentence proper nouns, and capitalized terms that repeat. Pure text
    work — no model call. Ranked by frequency, then alphabetically."""
    candidates: dict[str, int] = {}
    for sentence in _SENT_SPLIT_RE.split(full_text):
        toks = _TOKEN_RE.findall(sentence)
        for j, tok in enumerate(toks):
            acronym = len(tok) >= 2 and tok.isupper()
            proper = j > 0 and tok[0].isupper() and len(tok) >= 3
            if (acronym or proper) and tok.lower() not in diff.STOPWORDS:
                candidates[tok] = candidates.get(tok, 0) + 1
    # Repeated capitalized terms count even when they open a sentence
    freq: dict[str, int] = {}
    for tok in _CAP_TOKEN_RE.findall(full_text):
        if tok.lower() not in diff.STOPWORDS:
            freq[tok] = freq.get(tok, 0) + 1
    for tok, count in freq.items():
        if count >= 2:
            candidates[tok] = max(candidates.get(tok, 0), count)
    return sorted(candidates, key=lambda t: (-candidates[t], t))


def _cue_text_for_span(cue, start: float, end: float) -> tuple[str, float]:
    """The portion of a cue's text inside [start, end], split proportionally
    by time so a cue straddling two segments is not duplicated whole in both
    (R35). Returns (text, overlap_seconds)."""
    o0, o1 = max(cue.start, start), min(cue.end, end)
    if o1 <= o0 or cue.duration <= 0:
        return "", 0.0
    words = cue.text.split()
    if not words:
        return "", o1 - o0
    f0 = (o0 - cue.start) / cue.duration
    f1 = (o1 - cue.start) / cue.duration
    return " ".join(words[round(f0 * len(words)):round(f1 * len(words))]), o1 - o0


def _evaluate_subtitle(seg: Segment, cues: list, cfg: Config) -> dict:
    """Coverage and Nemotron-alignment of the subtitle over one segment (R35).

    qualified = the cue covers enough of the segment's time AND aligns with
    Nemotron's reading on the floor — the guard that rejects bad auto-subs,
    wrong-language subs and desync."""
    pieces, covered = [], 0.0
    for cue in cues:
        text, overlap = _cue_text_for_span(cue, seg.start, seg.end)
        if overlap > 0:
            covered += overlap
            if text:
                pieces.append((cue.start, text))
    sub_text = " ".join(t for _, t in sorted(pieces))
    coverage = min(covered / seg.duration, 1.0) if seg.duration > 0 else 0.0
    align = diff.align_ratio(sub_text, seg.text_src) if sub_text else 0.0
    qualified = coverage >= cfg.sub_coverage_floor and align >= cfg.sub_align_floor
    reason = ""
    if covered > 0 and not qualified:
        reason = "low_coverage" if coverage < cfg.sub_coverage_floor else "below_align"
    return {"sub_text": sub_text, "coverage": round(coverage, 4),
            "align": round(align, 4), "qualified": qualified,
            "covered": covered > 0, "reason": reason}


def _run_granite_worker(cfg: Config, xdir: Path, jobs: list[dict],
                        prompt: str) -> None:
    """One worker invocation over every segment missing its granite artifact;
    each segment's reading is persisted the moment its last clip part arrives
    (a crash loses at most one segment, not the whole batch).

    `jobs`: [{"wavs": [str, ...], "duration": float, "artifact": Path,
              "inputs": dict}]  — a segment >30s carries multiple clip parts
    whose texts are rejoined in order.
    """
    python = Path(cfg.granite_python)
    if not python.exists():
        raise RuntimeError(
            f"Granite interpreter not found: {python}\n"
            "Create it with: pyenv virtualenv 3.14.5 granite && "
            "~/.pyenv/versions/granite/bin/pip install torch torchaudio "
            "'transformers>=4.52' soundfile accelerate peft librosa\n"
            "or point GRANITE_PYTHON at an env that has them."
        )
    # path -> (job index, part index, part count); preserves clip order so a
    # split segment's parts rejoin correctly
    by_path: dict[str, tuple[int, int, int]] = {}
    wav_args: list[str] = []
    for ji, job in enumerate(jobs):
        for pi, wav in enumerate(job["wavs"]):
            by_path[wav] = (ji, pi, len(job["wavs"]))
            wav_args.append(wav)
    parts: dict[int, dict[int, str]] = {}

    cmd = [str(python), str(GRANITE_WORKER)] + wav_args
    budget = cfg.granite_timeout_base + cfg.granite_timeout_per_sec * sum(
        job["duration"] for job in jobs
    )
    env = {**os.environ, "GRANITE_PROMPT": prompt,
           "GRANITE_MODEL_ID": cfg.granite_model_id}
    stderr_log = xdir / "granite_worker.log"

    def persist(ji: int) -> None:
        job = jobs[ji]
        text = " ".join(parts[ji][pi].strip() for pi in sorted(parts[ji]))
        artifacts.produce(
            job["artifact"], job["inputs"], "crosscheck",
            lambda tmp: tmp.write_text(
                json.dumps({"text": text}, ensure_ascii=False, indent=1),
                encoding="utf-8"),
        )
        log.info("granite reading %s persisted", job["artifact"].name)

    timed_out = threading.Event()
    with open(stderr_log, "a", encoding="utf-8") as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err,
                                text=True, env=env)
        killer = threading.Timer(budget, lambda: (timed_out.set(), proc.kill()))
        killer.start()
        try:
            for line in proc.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    ji, pi, count = by_path[payload["path"]]
                    text = payload["text"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    # Library noise on stdout, or a line truncated by the
                    # kill timer — never fatal; missing segments surface below
                    log.warning("skipping non-NDJSON stdout line from granite worker: %.120s",
                                line)
                    continue
                parts.setdefault(ji, {})[pi] = text
                if len(parts[ji]) == count:
                    persist(ji)
            returncode = proc.wait()
        finally:
            killer.cancel()
            if proc.poll() is None:
                proc.kill()

    if timed_out.is_set():
        raise StageError("crosscheck", "infra", "timeout",
                         f"granite worker exceeded {budget:.0f}s budget")
    if returncode != 0:
        raise StageError("crosscheck", "infra", f"worker_exit_{returncode}",
                         f"see {stderr_log}")
    missing = [job["artifact"].name for job in jobs
               if not artifacts.is_valid(job["artifact"], job["inputs"])]
    if missing:
        raise StageError("crosscheck", "content", "missing_clips",
                         f"granite worker exited 0 but produced no output for {missing}")


def split_bounds(duration: float, silences: list[tuple[float, float]],
                 max_len: float = 30.0) -> list[tuple[float, float]]:
    """Split [0, duration] into chunks <= max_len, cutting at the latest
    silence midpoint inside each chunk; hard cut when no silence exists."""
    bounds = []
    pos = 0.0
    while duration - pos > max_len:
        limit = pos + max_len
        candidates = [
            (start + end) / 2
            for start, end in silences
            if pos + 1.0 < (start + end) / 2 <= limit
        ]
        cut = max(candidates) if candidates else limit
        bounds.append((pos, cut))
        pos = cut
    bounds.append((pos, duration))
    return bounds


def _gemma_transcribe_clip(cfg: Config, clip: Path) -> str:
    # Re-listen runs on the audio endpoint (llm_model_audio @ llm_host_audio),
    # not the default: the llama.cpp vision role (26B) has no audio, and putting
    # audio on its own always-hot host keeps the vision/translate router on one
    # resident model. llm_*_audio default to the base, so single-host oMLX is
    # unchanged.
    content = [{"type": "text", "text": PROMPT}, llm.audio_part(clip)]
    return llm.chat(cfg, [{"role": "user", "content": content}],
                     temperature=0.0, max_tokens=1024, stage="crosscheck",
                     role=cfg.llm_role("audio"), enable_thinking=False)


def _cut_clips(cfg: Config, state: DubState, seg: Segment, xdir: Path,
               cache: dict[int, list[Path]]) -> list[Path]:
    """Cut the segment's audio into <=max_clip parts (split at silence, R8a),
    once per segment. A small lead-in pad (crosscheck_clip_pad) is prepended so
    the verify engines hear the segment's onset, which they otherwise drop;
    segment timing is untouched, the pad only widens the verify audio. The same
    clips feed Granite (phase 1) and Gemma (phase 2); the cache holds them until
    the node deletes them at the end."""
    if seg.index in cache:
        return cache[seg.index]
    start = max(0.0, seg.start - cfg.crosscheck_clip_pad)
    span = seg.end - start
    base = xdir / f".clip_{seg.index:04d}.wav"
    ffmpeg.ffmpeg("-i", state["audio_16k"], "-ss", f"{start:.3f}",
                  "-to", f"{seg.end:.3f}", str(base))
    if span <= cfg.crosscheck_max_clip:
        parts = [base]
    else:
        silences = ffmpeg.detect_silences(base, cfg.silence_threshold_db,
                                          cfg.silence_min_duration)
        parts = []
        for i, (s, e) in enumerate(
            split_bounds(span, silences, cfg.crosscheck_max_clip)
        ):
            part = xdir / f".clip_{seg.index:04d}_p{i}.wav"
            ffmpeg.ffmpeg("-i", str(base), "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                          str(part))
            parts.append(part)
        base.unlink(missing_ok=True)  # only the parts are used
    cache[seg.index] = parts
    return parts


def _gemma_reading(cfg: Config, state: DubState, seg: Segment, xdir: Path,
                   cache: dict[int, list[Path]]) -> str:
    """Gemma transcribes each clip part; parts are rejoined with spaces."""
    parts = _cut_clips(cfg, state, seg, xdir, cache)
    return " ".join(_gemma_transcribe_clip(cfg, p).strip() for p in parts)


def _winner_summary(decision: str, regions: list[dict]) -> str:
    """One-line attribution for the report: which engine(s) decided the text."""
    if decision == "keep_low_confidence":
        return "nemotron (granite suspect)"
    if decision != "replace":
        return "nemotron"
    engines: list[str] = []
    for region in regions:
        if region.get("changed"):
            for eng in region["winner_engines"]:
                if eng not in engines:
                    engines.append(eng)
    return "+".join(engines) if engines else "nemotron"


def _granite_inputs(cfg: Config, seg: Segment, audio_sha: str,
                    prompt_granite: str, keywords: list[str], worker_sha: str) -> dict:
    return {
        "audio_sha": audio_sha, "start": seg.start, "end": seg.end,
        "prompt": prompt_granite, "keywords": keywords,
        "model_id": cfg.granite_model_id, "worker_sha": worker_sha,
        "max_clip": cfg.crosscheck_max_clip, "silence_db": cfg.silence_threshold_db,
        "clip_pad": cfg.crosscheck_clip_pad,
    }


def _load_cues(state: DubState) -> list:
    subs_path = state.get("subs_path", "")
    if not subs_path:
        return []
    path = Path(subs_path)
    if not path.exists():
        return []
    return srt.parse_cues(path.read_text(encoding="utf-8"))


def _subtitle_verdict(seg: Segment, ev: dict) -> dict:
    """R35: the subtitle covers and aligns — its text is the effective text,
    no verify engine is consulted."""
    return {
        "decision": "subtitle", "source": "subtitle",
        "text_nemotron": seg.text_src, "text_granite": "", "text_gemma": "",
        "text_effective": ev["sub_text"], "winner": "subtitle",
        "low_confidence": False, "contested": False,
        "sub": {"coverage": ev["coverage"], "align": ev["align"]},
        "metrics": {}, "regions": [], "retryable": False, "attempts": 0,
    }


def _verdict_inputs(cfg: Config, seg: Segment, audio_sha: str,
                    granite_sha: str | None, thresholds: dict,
                    sub_text: str | None) -> dict:
    # The granite reading's content hash is the cascade link (keyword/vision
    # change -> granite re-reads -> new sha -> verdict recomputes). Gemma's
    # text is an *output* of recompute, so it is not a fingerprint input.
    # sub_text + floors fold the subtitle into the same cascade: a sidecar
    # edit changes the covering cue's text -> the right segments recompute.
    return {
        "audio_sha": audio_sha, "start": seg.start, "end": seg.end,
        "text_nemotron": seg.text_src, "granite_sha": granite_sha,
        # The re-listen model is the audio endpoint's model; a verdict recomputes
        # when it changes (and not when the vision model changes), which is what
        # we want — the audio reading is what the verdict depends on (KTD4).
        "gemma_prompt": PROMPT, "audio_model": cfg.llm_model_audio,
        "weights": cfg.crosscheck_weights, "thresholds": thresholds,
        "max_clip": cfg.crosscheck_max_clip, "silence_db": cfg.silence_threshold_db,
        "sub_text": sub_text,
        "sub_floors": {"coverage": cfg.sub_coverage_floor, "align": cfg.sub_align_floor},
    }


def _compute_verdict(cfg: Config, state: DubState, seg: Segment, xdir: Path,
                     clip_cache: dict, granite_art: Path, granite_inputs: dict,
                     granite_error: StageError | None, keywords: list[str],
                     prior_attempts: int) -> dict:
    """Decide the effective text for one segment via the ensemble, handling
    every degradation branch of R32. Returns the verdict dict plus internal
    `_strike`/`_strike_abortproof` keys the caller pops for the ledger.

    Never raises StageError: Gemma failures are translated into degraded
    (retryable) verdicts so the loop's ledger handling stays uniform."""
    n_text = seg.text_src
    w = cfg.crosscheck_weights
    af, mlr = cfg.crosscheck_align_floor, cfg.crosscheck_min_length_ratio
    data = {
        "text_nemotron": n_text, "text_granite": "", "text_gemma": "",
        "text_effective": n_text, "decision": "keep", "source": "vote",
        "low_confidence": False, "contested": False, "winner": "nemotron",
        "metrics": {}, "regions": [], "keywords": keywords,
        "retryable": False, "attempts": prior_attempts,
        "_strike": None, "_strike_abortproof": False,
    }

    if artifacts.is_valid(granite_art, granite_inputs):
        g_text = json.loads(granite_art.read_text(encoding="utf-8"))["text"]
        data["text_granite"] = g_text
        gemma_text = None
        if diff.needs_arbiter(n_text, g_text, align_floor=af, min_length_ratio=mlr):
            try:
                gemma_text = _gemma_reading(cfg, state, seg, xdir, clip_cache)
                data["text_gemma"] = gemma_text
            except StageError as exc:
                # R32: Gemma failed when its vote was needed -> vote {N, G}
                # (Granite wins), mark retryable, count the strike.
                data["error"] = f"{exc.error_class}/{exc.code}"
                data["attempts"] = prior_attempts + 1
                data["retryable"] = prior_attempts + 1 < 2
                data["_strike"] = exc.signature
                data["_strike_abortproof"] = True
        verdict = diff.vote3(n_text, g_text, gemma_text, weights=w,
                             align_floor=af, min_length_ratio=mlr)
        data["decision"] = verdict["decision"]
        data["text_effective"] = verdict["text_effective"]
        data["contested"] = verdict["contested"]
        data["metrics"] = verdict["metrics"]
        data["regions"] = verdict["regions"]
        data["low_confidence"] = verdict["decision"] == "keep_low_confidence"
        data["winner"] = _winner_summary(verdict["decision"], verdict["regions"])
        if verdict["decision"] == "keep_low_confidence" and data["_strike"] is None:
            # R30: the Granite reading itself is suspect; mass occurrence
            # must abort like the old Gemma-gibberish guard.
            data["_strike"] = ("crosscheck", "content", "extreme_mismatch")
        return data

    # R32: Granite reading unavailable -> two-way Nemotron x Gemma fallback.
    data["source"] = "granite_fallback"
    try:
        gemma_text = _gemma_reading(cfg, state, seg, xdir, clip_cache)
        data["text_gemma"] = gemma_text
        verdict = diff.compare(n_text, gemma_text,
                               wer_threshold=cfg.crosscheck_wer_threshold,
                               align_floor=af, min_length_ratio=mlr)
        data["decision"] = verdict["decision"]
        data["text_effective"] = gemma_text if verdict["decision"] == "replace" else n_text
        data["low_confidence"] = verdict["decision"] == "keep_low_confidence"
        data["metrics"] = {k: verdict[k] for k in ("wer", "align_ratio", "content_substitution")}
        data["winner"] = "gemma" if verdict["decision"] == "replace" else "nemotron"
        # Granite is down; retry it on the next run (R20), and count the
        # outage toward the abort window.
        data["attempts"] = prior_attempts + 1
        data["retryable"] = prior_attempts + 1 < 2
        data["_strike"] = granite_error.signature if granite_error else \
            ("crosscheck", "infra", "granite_unavailable")
        data["_strike_abortproof"] = True
    except StageError as exc:
        # Both verify engines down -> keep Nemotron, flag low-confidence.
        data["decision"] = "keep"
        data["text_effective"] = n_text
        data["low_confidence"] = True
        data["winner"] = "nemotron (both engines failed)"
        data["error"] = f"{exc.error_class}/{exc.code}"
        data["attempts"] = prior_attempts + 1
        data["retryable"] = prior_attempts + 1 < 2
        data["_strike"] = exc.signature
        data["_strike_abortproof"] = True
    return data


def crosscheck(state: DubState, cfg: Config) -> DubState:
    workdir = Path(state["workdir"])
    xdir = workdir / "crosscheck"
    segments = state["segments"]

    if not cfg.enable_cross_check:
        inputs = {"passthrough": True,
                  "segments": [(s.index, s.start, s.end, s.text_src) for s in segments]}
        artifacts.produce_json(
            xdir / "segments.json", inputs, "crosscheck",
            lambda: {"passthrough": True, "segments": [s.to_dict() for s in segments]},
        )
        # Still render a sub-style source SRT from the sentence segments: asr's
        # provisional SRT was built from the raw acoustic units, not sentences.
        src_tag = state.get("source_lang", cfg.source_lang).lower()
        srt_src = workdir / f"transcript.{src_tag}.srt"
        srt_src.write_text(
            srt.to_srt_wrapped(segments, state.get("words", []), side="source",
                               max_chars=cfg.srt_max_cue_chars, max_dur=cfg.srt_max_cue_dur),
            encoding="utf-8")
        return {"segments": segments, "srt_src": str(srt_src)}

    keywords = list(state.get("video_keywords", []))
    # Existing subtitle (R34/R35): cues short-circuit covered segments and
    # feed extra terminology into Granite for the rest (R36).
    cues = _load_cues(state)
    sub_eval: dict[int, dict] = {}
    if cues:
        for seg in segments:
            sub_eval[seg.index] = _evaluate_subtitle(seg, cues, cfg)
        for kw in subtitle_keywords(" ".join(c.text for c in cues)):
            if kw not in keywords:
                keywords.append(kw)
    keywords = keywords[:cfg.crosscheck_keyword_cap]
    prompt_granite = granite_prompt(keywords, cfg.crosscheck_keyword_cap)

    audio_sha = artifacts.cached_file_sha256(state["audio_16k"])
    worker_sha = artifacts.file_sha256(GRANITE_WORKER)
    thresholds = {"wer": cfg.crosscheck_wer_threshold, "align": cfg.crosscheck_align_floor,
                  "min_length": cfg.crosscheck_min_length_ratio}
    ledger = SkipLedger.from_cfg(workdir, cfg)
    xdir.mkdir(parents=True, exist_ok=True)
    clip_cache: dict[int, list[Path]] = {}

    def granite_art(seg: Segment) -> Path:
        return xdir / f"seg_{seg.index:04d}.granite.json"

    def granite_inputs(seg: Segment) -> dict:
        return _granite_inputs(cfg, seg, audio_sha, prompt_granite, keywords, worker_sha)

    def is_subtitle(seg: Segment) -> bool:
        ev = sub_eval.get(seg.index)
        return bool(ev and ev["qualified"])

    try:
        # PHASE 1: one Granite invocation over every segment missing its reading.
        # Subtitle-covered segments are skipped — no verify engine touches them.
        granite_jobs = []
        for seg in segments:
            if is_subtitle(seg):
                continue
            if not artifacts.is_valid(granite_art(seg), granite_inputs(seg)):
                wavs = _cut_clips(cfg, state, seg, xdir, clip_cache)
                granite_jobs.append({
                    "wavs": [str(w) for w in wavs], "duration": seg.duration,
                    "artifact": granite_art(seg), "inputs": granite_inputs(seg),
                })
        granite_error: StageError | None = None
        if granite_jobs:
            log.info("granite verify: %d/%d segment(s)", len(granite_jobs), len(segments))

            def attempt() -> None:
                remaining = [j for j in granite_jobs
                             if not artifacts.is_valid(j["artifact"], j["inputs"])]
                if remaining:
                    _run_granite_worker(cfg, xdir, remaining, prompt_granite)

            try:
                with_retry("crosscheck", attempt, attempts=cfg.retry_attempts,
                           base_delay=cfg.retry_base_delay)
            except StageError as exc:
                granite_error = exc
                log.warning("granite verify failed (%s) — missing segments fall back to "
                            "Nemotron x Gemma", exc.code)

        # PHASE 2: per-segment verdict — subtitle wins outright, else lazy Gemma.
        seg_artifacts = []
        for seg in segments:
            art = xdir / f"seg_{seg.index:04d}.json"
            ev = sub_eval.get(seg.index)

            if ev and ev["qualified"]:
                inputs = _verdict_inputs(cfg, seg, audio_sha, None, thresholds,
                                         ev["sub_text"])
                if not artifacts.is_valid(art, inputs):
                    data = _subtitle_verdict(seg, ev)
                    _write(art, inputs, data)
                    ledger.record_ok(segment_id(seg), stage="crosscheck")
                else:
                    data = json.loads(art.read_text(encoding="utf-8"))
                seg.text_src = data["text_effective"]
                seg_artifacts.append(art)
                continue

            g_valid = artifacts.is_valid(granite_art(seg), granite_inputs(seg))
            granite_sha = artifacts.cached_file_sha256(granite_art(seg)) if g_valid else None
            sub_text = ev["sub_text"] if ev else None
            inputs = _verdict_inputs(cfg, seg, audio_sha, granite_sha, thresholds, sub_text)
            data = json.loads(art.read_text(encoding="utf-8")) \
                if artifacts.is_valid(art, inputs) else None

            if data is None or data.get("retryable"):
                prior_attempts = data["attempts"] if data else 0
                data = _compute_verdict(cfg, state, seg, xdir, clip_cache,
                                        granite_art(seg), granite_inputs(seg),
                                        granite_error, keywords, prior_attempts)
                if ev and ev["covered"]:
                    # Sub covered this segment but failed the guard (R35): keep
                    # the ensemble verdict, but record why the sub was rejected
                    data["sub_rejected"] = {"coverage": ev["coverage"],
                                            "align": ev["align"], "reason": ev["reason"]}
                strike = data.pop("_strike")
                abortproof = data.pop("_strike_abortproof")
                _write(art, inputs, data)
                if strike is None:
                    if not data["low_confidence"]:
                        ledger.record_ok(segment_id(seg), stage="crosscheck")
                else:
                    if data.get("error"):
                        log.warning("cross-check segment %d degraded (%s) — %s",
                                    seg.index, data["error"],
                                    "will retry next run" if data["retryable"] else "accepted")
                    try:
                        ledger.record_strike(strike)
                    except AbortRun:
                        # Systemic outage: don't let this attempt consume the
                        # marker's R20 retry budget — the next run retries clean
                        # (R21 spirit). Content strikes (granite-suspect) are
                        # already final, so they are not budget-protected.
                        if abortproof:
                            data["retryable"] = True
                            data["attempts"] = prior_attempts
                            _write(art, inputs, data)
                        raise

            if data["decision"] == "replace":
                log.info("segment %d: thay text [%s] (%r -> %r)", seg.index,
                         data.get("winner"), data["text_nemotron"][:50],
                         data["text_effective"][:50])
            seg.text_src = data["text_effective"]
            seg_artifacts.append(art)
    finally:
        for parts in clip_cache.values():
            for part in parts:
                Path(part).unlink(missing_ok=True)

    merge_inputs = {"seg_hashes": [artifacts.cached_file_sha256(a) for a in seg_artifacts]}
    artifacts.produce_json(
        xdir / "segments.json", merge_inputs, "crosscheck",
        lambda: {"passthrough": False, "segments": [s.to_dict() for s in segments]},
    )

    # The source SRT is derived output; refresh it as sub-style word-timed cues
    # (the dub backbone is whole sentences, the SRT stays readable, U2). Filename
    # derives from the source locale (U10); EN default stays transcript.en.srt.
    src_tag = state.get("source_lang", cfg.source_lang).lower()
    srt_src = workdir / f"transcript.{src_tag}.srt"
    srt_src.write_text(
        srt.to_srt_wrapped(segments, state.get("words", []), side="source",
                           max_chars=cfg.srt_max_cue_chars, max_dur=cfg.srt_max_cue_dur),
        encoding="utf-8")
    return {"segments": segments, "srt_src": str(srt_src)}


def _write(art: Path, inputs: dict, data: dict) -> None:
    # Always rewrite: retryable markers mutate on each attempt
    art.unlink(missing_ok=True)
    artifacts.produce(
        art, inputs, "crosscheck",
        lambda tmp: tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                   encoding="utf-8"),
    )
