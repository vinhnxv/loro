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

    def _set_overflow_entry(self, segment_id: str, status: str,
                            signature: tuple[str, str, str], *,
                            push_window: bool) -> None:
        """Store a PRE-DEMOTED overflow entry (length_overflow / fit_overflow): the
        clip is KEPT (the segment is NOT skipped), so this is a report annotation,
        not a failure. Does NOT save — callers batch the write.

        A pre-demoted WINDOW entry is pushed only when `push_window`. A
        length_overflow is recorded inside the TTS convergence loop, so it pushes a
        demoted entry to keep that expected iteration off the abort count. A
        fit_overflow is post-hoc and already pre-demoted (it can never count toward
        an abort), so a window push would only evict genuine strikes from the
        bounded window — fit overruns on speed-constrained content pass
        push_window=False (U4/R3)."""
        sig = list(signature)
        self._segments[segment_id] = {
            "status": status,
            "reason": status,
            "signature": sig,
            "strikes": 0,
            "input_hash": "",
            "demoted": True,
        }
        if push_window:
            self._push_window({"segment_id": segment_id, "signature": sig, "demoted": True})

    def _record_non_fatal_overflow(self, segment_id: str, status: str,
                                   signature: tuple[str, str, str], *,
                                   push_window: bool) -> None:
        """Single-record + save for a non-fatal overflow (used by the direct
        record_* entry points; `fit` batches via reconcile_fit_overflows)."""
        self._set_overflow_entry(segment_id, status, signature, push_window=push_window)
        self._save()

    def record_length_overflow(self, segment_id: str) -> None:
        """Record a best-effort length overflow (U6/R7): a non-VI clip that could
        not fit its slot even after the convergence cap. The clip is KEPT (the
        segment is NOT skipped), so this is a report annotation, not a failure —
        it stores a `length_overflow` entry and pushes a PRE-DEMOTED window entry,
        so it can never count toward the abort threshold (the loop's iteration is
        expected, not infra degradation)."""
        self._record_non_fatal_overflow(segment_id, "length_overflow",
                                        self.LENGTH_OVERFLOW_SIGNATURE,
                                        push_window=True)

    FIT_OVERFLOW_SIGNATURE = ("fit", "length", "fit_overflow")

    def record_fit_overflow(self, segment_id: str) -> None:
        """Record a PLACEMENT-layer length overrun (U4/R3): a clip that still
        exceeded its slot after the max_tempo cap and had its spilled tail trimmed
        at the next segment's onset (fit, U2), so dub audio was actually dropped.

        Unlike `length_overflow` — which is CPS-only (measured_duration_active is
        False for VI), deliberately exit-0, and means "best-effort CPS clip kept
        after the re-translation cap" — `fit_overflow` runs for all languages and
        DOES drive exit code 2, so the operator/orchestrating agent sees that a
        dub clip was materially cut. The clip is still KEPT (the segment is NOT
        skipped). The entry is pre-demoted, so recurring overruns on
        speed-constrained content can never accumulate toward the abort threshold;
        unlike length_overflow it pushes NO window entry (push_window=False), so a
        long run of overruns can't evict genuine strikes from the bounded window.
        `fit` calls reconcile_fit_overflows (one save per run); this direct entry
        point exists for callers/tests that record a single id."""
        self._record_non_fatal_overflow(segment_id, "fit_overflow",
                                        self.FIT_OVERFLOW_SIGNATURE,
                                        push_window=False)

    def clear_fit_overflow(self, segment_id: str) -> None:
        """Drop a stale fit_overflow entry when a recompute no longer overruns, so
        a fixed segment doesn't keep the run at exit 2 (U4). Only clears a
        fit_overflow entry — it never erases a real skip or a CPS length_overflow
        for the same id — and only writes when it actually removed one."""
        entry = self._segments.get(segment_id)
        if entry is not None and entry["status"] == "fit_overflow":
            self._segments.pop(segment_id, None)
            self._save()

    def reconcile_fit_overflows(self, overflow_ids) -> None:
        """Batch-reconcile placement-layer fit_overflows for a whole `fit` run in
        ONE write (U4/R3). `overflow_ids` is the set of segment ids whose clip
        materially overran its slot.

        - Records a pre-demoted fit_overflow for each id NOT already carrying an
          entry. A segment with ANY existing entry is left untouched: a real
          skip/accepted_skip is never clobbered, and a CPS length_overflow is
          never promoted (KTD2). An id already at fit_overflow is a no-op, so a
          resumed/cache-hit rerun neither rewrites skips.json nor re-pushes window
          state (the previous per-segment record/clear did both every call).
        - Drops any stale fit_overflow no longer in the set (a recompute that now
          fits clears the exit-2 signal).
        - Writes skips.json at most once, and only when something changed."""
        overflow_ids = set(overflow_ids)
        changed = False
        for sid in overflow_ids:
            if (self._segments.get(sid) or {}).get("status") is None:
                self._set_overflow_entry(sid, "fit_overflow",
                                         self.FIT_OVERFLOW_SIGNATURE, push_window=False)
                changed = True
        stale = [sid for sid, entry in self._segments.items()
                 if entry["status"] == "fit_overflow" and sid not in overflow_ids]
        for sid in stale:
            self._segments.pop(sid, None)
            changed = True
        if changed:
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
