"""On-device VieNeu-TTS client: a warm subprocess worker with the same
context-manager surface as HiggsClient.

The worker (workers/vieneu_worker.py) runs in the isolated `vieneu` venv, loads
the model once, and answers one synthesis request per NDJSON line. This client
owns infra resilience: it spawns the worker, blocks until it is ready, and on a
crash / EOF / timeout it respawns and retries the failed request within the
retry budget — so a crash costs one model reload, not the whole run. A worker
that reports a synth *error* (infer raised) is surfaced as a qa-class failure so
the tts node's existing qa-retry-then-skip path handles it without respawning.

Because the worker is local it reads the reference clip path directly — no
temporary HTTP server (KTD6).
"""

import json
import logging
import os
import subprocess
import threading
from pathlib import Path

from loro.config import Config
from loro.harness.retry import INFRA, QA, StageError, with_retry

log = logging.getLogger("loro.vieneu")

WORKER = Path(__file__).resolve().parents[1] / "workers" / "vieneu_worker.py"


class VieNeuClient:
    """Synthesizes speech cloned from one reference voice. Use as a context manager."""

    def __init__(self, cfg: Config, ref_audio: Path, ref_text: str):
        self.cfg = cfg
        self.ref_audio = Path(ref_audio)
        self.ref_text = ref_text
        self._proc: subprocess.Popen | None = None
        self._errlog = None  # worker stderr sink (file handle), opened lazily

    def __enter__(self):
        python = Path(self.cfg.vieneu_python)
        if not python.exists():
            raise RuntimeError(
                f"VieNeu interpreter not found: {python}\n"
                "Create it with: pyenv virtualenv 3.14.5 vieneu && "
                "~/.pyenv/versions/vieneu/bin/pip install vieneu\n"
                "or point VIENEU_PYTHON at an env that has vieneu installed."
            )
        self._spawn()
        return self

    def __exit__(self, *exc):
        self._reap()
        if self._errlog is not None:
            self._errlog.close()
            self._errlog = None

    # --- worker lifecycle -------------------------------------------------

    def _spawn(self) -> None:
        """Launch the worker and block until its `ready` line. Reused for the
        initial spawn and for respawning a dead worker mid-batch."""
        if self._errlog is None:
            try:
                self._errlog = open(self.ref_audio.parent / "vieneu_worker.log",
                                    "a", encoding="utf-8")
            except OSError:
                self._errlog = None
        env = {**os.environ,
               "VIENEU_MODEL": self.cfg.vieneu_model,
               "VIENEU_EMOTION": self.cfg.vieneu_emotion}
        self._proc = subprocess.Popen(
            [str(self.cfg.vieneu_python), str(WORKER)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._errlog or subprocess.DEVNULL, text=True, env=env,
        )
        self._read_until(lambda o: o.get("status") == "ready",
                         self.cfg.vieneu_timeout, "ready")

    def _reap(self) -> None:
        """Close stdin, kill if still alive, and clear the handle so the next
        request respawns."""
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            if proc.stdin and not proc.stdin.closed:
                proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.kill()
        proc.wait()

    # --- protocol ---------------------------------------------------------

    def _send(self, req: dict) -> None:
        try:
            self._proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self._reap()
            raise StageError("tts", INFRA, "worker_pipe", str(exc)[:200]) from exc

    def _read_until(self, match, budget: float, what: str) -> dict:
        """Read worker stdout lines (skipping any non-JSON library noise) until
        one satisfies `match`. A timeout or EOF kills/reaps the worker and raises
        an infra StageError so the caller's with_retry respawns and retries."""
        proc = self._proc
        timed_out = threading.Event()
        killer = threading.Timer(budget, lambda: (timed_out.set(), proc.kill()))
        killer.start()
        try:
            while True:
                line = proc.stdout.readline()
                if line == "":  # EOF: worker exited or was killed
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue  # native library chatter leaked onto stdout
                if match(obj):
                    return obj
        finally:
            killer.cancel()

        self._reap()
        if timed_out.is_set():
            raise StageError("tts", INFRA, "timeout",
                             f"vieneu worker exceeded {budget:.0f}s ({what})")
        raise StageError("tts", INFRA, "worker_eof",
                         f"vieneu worker exited before {what}")

    def synthesize(self, text: str, output: Path, voice: str | None = None) -> None:
        # `voice` is part of the shared synthesize surface for the preset engine;
        # VieNeu clones a reference voice and ignores it (KTD1).
        output = Path(output)

        def call() -> None:
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()  # lazily (re)spawn a dead worker before sending
            self._send({
                "text": text,
                "out": str(output),
                "ref_audio": str(self.ref_audio) if self.ref_audio else None,
                # Audio-only by default (R5); ref_text is plumbed but sent only
                # when cfg.vieneu_ref_text is enabled.
                "ref_text": (self.ref_text or None) if self.cfg.vieneu_ref_text else None,
                "temperature": self.cfg.vieneu_temperature,
                "emotion": self.cfg.vieneu_emotion,
            })
            resp = self._read_until(
                lambda o: o.get("status") in ("ok", "error") and o.get("out") == str(output),
                self.cfg.vieneu_timeout, "synthesis response",
            )
            if resp["status"] == "error":
                # A bad synth, not a dead process: qa-class so the tts node's
                # qa-retry-then-skip path handles it without respawning.
                raise StageError("tts", QA, "vieneu_synth_error",
                                 str(resp.get("error", ""))[:200])
            if not output.exists():
                raise StageError("tts", INFRA, "missing_output",
                                 "worker reported ok but wrote no file")

        # Infra failures (crash/EOF/timeout/pipe) retry here and respawn; qa
        # failures propagate to the caller's own qa-retry layer (like Higgs).
        with_retry("tts", call, attempts=self.cfg.retry_attempts,
                   base_delay=self.cfg.retry_base_delay)
