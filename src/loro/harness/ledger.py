"""Per-segment failure state machine shared by cross-check/translate/TTS.

`skips.json` is rewritten atomically on every update, so a hard kill at any
moment leaves a parseable ledger. It holds two things:

- segments: {segment_id: {status, reason, signature, strikes, input_hash,
  demoted}} — skip bookkeeping per R5/R5b. A skipped segment is retried by
  the next run; a second content/qa failure under the same input hash
  escalates to accepted_skip (not retried until the input changes). Infra
  failures never escalate (R21).

- window: the last W attempt outcomes across all segments. N failures with
  the same (stage, class, code) signature inside the window means systemic
  degradation -> AbortRun (R5a). The strikes that triggered an abort are
  demoted so the next run retries clean instead of instantly re-aborting
  (R21).
"""

import json
from pathlib import Path

from loro.harness.artifacts import atomic_write_bytes

ACCEPTABLE_CLASSES = ("content", "qa")  # only these may escalate to accepted_skip


class AbortRun(RuntimeError):
    def __init__(self, signature: tuple[str, str, str], count: int, window: int):
        self.signature = signature
        self.count = count
        super().__init__(
            f"{count} errors with the same signature {signature} in the last {window} attempts — "
            "the server looks degraded; aborting instead of mass-skipping. Rerun the same command "
            "after addressing the root cause"
        )


class SkipLedger:
    @classmethod
    def from_cfg(cls, workdir: str | Path, cfg) -> "SkipLedger":
        return cls(workdir, cfg.abort_window, cfg.abort_threshold)

    def __init__(self, workdir: str | Path, window: int = 20, abort_threshold: int = 5):
        self.path = Path(workdir) / "skips.json"
        self.window = window
        self.abort_threshold = abort_threshold
        self._segments: dict[str, dict] = {}
        self._window: list[dict] = []
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._segments = data.get("segments", {})
            self._window = data.get("window", [])
        except (OSError, json.JSONDecodeError):
            pass

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"segments": self._segments, "window": self._window}
        atomic_write_bytes(self.path, json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"))

    def entries(self) -> dict[str, dict]:
        return dict(self._segments)

    def should_attempt(self, segment_id: str, input_hash: str) -> bool:
        entry = self._segments.get(segment_id)
        if entry is None:
            return True
        return not (entry["status"] == "accepted_skip" and entry["input_hash"] == input_hash)

    def record_ok(self, segment_id: str, stage: str | None = None) -> None:
        """Record a success. When `stage` is given, only clear the segment's
        skip entry if that entry was created by the same stage — a cross-check
        success must not erase a TTS skip for the same segment id."""
        entry = self._segments.get(segment_id)
        if entry is not None and (stage is None or entry["signature"][0] == stage):
            self._segments.pop(segment_id, None)
        self._push_window({"segment_id": segment_id, "signature": None, "demoted": False})
        self._save()

    def record_failure(
        self,
        segment_id: str,
        input_hash: str,
        signature: tuple[str, str, str],
        reason: str,
    ) -> str:
        """Record a segment failure after retries were exhausted. Returns the
        new status ('skipped' or 'accepted_skip'); raises AbortRun when this
        failure tips the window over the same-signature threshold."""
        error_class = signature[1]
        prev = self._segments.get(segment_id)
        escalate = (
            prev is not None
            and prev["status"] == "skipped"
            and prev["input_hash"] == input_hash
            and not prev.get("demoted")
            and error_class in ACCEPTABLE_CLASSES
            # The prior failure must also be content/qa: an infra skip never
            # contributes to giving up on a segment (R21)
            and prev["signature"][1] in ACCEPTABLE_CLASSES
        )
        status = "accepted_skip" if escalate else "skipped"
        strikes = (prev["strikes"] + 1) if prev and prev["input_hash"] == input_hash else 1
        self._segments[segment_id] = {
            "status": status,
            "reason": reason,
            "signature": list(signature),
            "strikes": strikes,
            "input_hash": input_hash,
            "demoted": False,
        }
        self._push_window({"segment_id": segment_id, "signature": list(signature), "demoted": False})
        self._save()
        self._check_abort(signature)
        return status

    LENGTH_OVERFLOW_SIGNATURE = ("tts", "length", "length_overflow")

    def record_length_overflow(self, segment_id: str) -> None:
        """Record a best-effort length overflow (U6/R7): a non-VI clip that could
        not fit its slot even after the convergence cap. The clip is KEPT (the
        segment is NOT skipped), so this is a report annotation, not a failure —
        it stores a `length_overflow` entry and pushes a PRE-DEMOTED window entry,
        so it can never count toward the abort threshold (the loop's iteration is
        expected, not infra degradation)."""
        sig = list(self.LENGTH_OVERFLOW_SIGNATURE)
        self._segments[segment_id] = {
            "status": "length_overflow",
            "reason": "length_overflow",
            "signature": sig,
            "strikes": 0,
            "input_hash": "",
            "demoted": True,
        }
        self._push_window({"segment_id": segment_id, "signature": sig, "demoted": True})
        self._save()

    def record_strike(self, signature: tuple[str, str, str]) -> None:
        """Window-only strike for failures that never skip (cross-check, R8/R20)."""
        self._push_window({"segment_id": None, "signature": list(signature), "demoted": False})
        self._save()
        self._check_abort(signature)

    def _push_window(self, entry: dict) -> None:
        self._window.append(entry)
        self._window = self._window[-self.window:]

    def _check_abort(self, signature: tuple[str, str, str]) -> None:
        matching = [
            e for e in self._window
            if e["signature"] == list(signature) and not e.get("demoted")
        ]
        if len(matching) < self.abort_threshold:
            return
        # Demote the strikes that fired this abort so the next run starts clean
        for entry in matching:
            entry["demoted"] = True
            seg = self._segments.get(entry["segment_id"] or "")
            if seg is not None:
                seg["demoted"] = True
        self._save()
        raise AbortRun(signature, len(matching), self.window)
