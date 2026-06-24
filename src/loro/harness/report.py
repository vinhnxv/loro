"""Run report built declaratively from durable on-disk markers (R26).

The report never depends on in-run memory: a video completed across three
invocations still reports every skip, replacement and low-confidence segment
from earlier runs, because those live in the ledger and the stage artifacts.
Only stage timings are of-this-invocation. `report.json` is overwritten at
the end of every run — including aborted ones (R22)."""

import json
import re
import time
from pathlib import Path

from loro.harness.artifacts import atomic_write_bytes
from loro.harness.ledger import SkipLedger
from loro.profiles import is_profiled

# A verdict is exactly seg_NNNN.json — never the .granite.json tier-1 reading
# or any .json.meta.json sidecar that the glob also matches.
_VERDICT_RE = re.compile(r"seg_\d+\.json$")


def _current_segment_ids(workdir: Path) -> set[str] | None:
    """Segment ids of the current manifest, so markers left by segments that
    no longer exist (ASR re-segmented) don't poison the report or exit code.
    Returns None when no manifest exists yet (report everything)."""
    for manifest in ("translate/segments.json", "crosscheck/segments.json",
                     "asr/segments.json"):
        try:
            data = json.loads((workdir / manifest).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        segments = data.get("segments", [])
        if segments and "index" in segments[0]:
            return {f"seg_{s['index']:04d}" for s in segments}
    return None


def _crosscheck_report(workdir: Path, valid_ids: set[str] | None) -> dict:
    """Ensemble attribution from durable verdicts (R33): every replacement
    carries all three readings, the winning engine, and the contested flag;
    plus per-engine tallies and the run's effective keyword list."""
    replacements, low_confidence, sub_rejected = [], [], []
    tally = {"replace": 0, "keep": 0, "subtitle": 0, "contested": 0,
             "low_confidence": 0, "granite_fallback": 0}
    by_engine: dict[str, int] = {}
    keywords: list[str] = []

    for path in sorted((workdir / "crosscheck").glob("seg_*.json")):
        if not _VERDICT_RE.fullmatch(path.name):
            continue  # tier-1 .granite.json readings and .meta.json sidecars
        if valid_ids is not None and path.stem not in valid_ids:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        seg = path.stem
        decision = data.get("decision")
        if data.get("keywords") and not keywords:
            keywords = data["keywords"]

        if decision == "replace":
            tally["replace"] += 1
            winner = data.get("winner", "")
            by_engine[winner] = by_engine.get(winner, 0) + 1
            if data.get("source") == "granite_fallback":
                tally["granite_fallback"] += 1
            replacements.append({
                "segment": seg,
                "winner": winner,
                "text_nemotron": data.get("text_nemotron", ""),
                "text_granite": data.get("text_granite", ""),
                "text_gemma": data.get("text_gemma", ""),
                # Effective text; legacy markers only carried text_gemma
                "text_effective": data.get("text_effective") or data.get("text_gemma", ""),
                "contested": data.get("contested", False),
                "source": data.get("source", ""),
            })
        elif decision == "subtitle":
            tally["subtitle"] += 1
        else:
            tally["keep"] += 1

        if data.get("contested"):
            tally["contested"] += 1
        if data.get("low_confidence"):
            tally["low_confidence"] += 1
            low_confidence.append({
                "segment": seg,
                "error": data.get("error"),
                "retryable": data.get("retryable", False),
            })
        if data.get("sub_rejected"):
            sub_rejected.append({"segment": seg, **data["sub_rejected"]})

    return {
        "replacements": replacements,
        "low_confidence": low_confidence,
        "sub_rejected": sub_rejected,
        "summary": {"tally": tally, "by_engine": by_engine, "keywords": keywords},
    }


def _vision_degraded(workdir: Path) -> dict | None:
    try:
        data = json.loads((workdir / "vision" / "context.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("degraded"):
        return {"reason": data.get("reason", "unknown"),
                "hint": "delete vision/context.json to retry",
                "retry_action": {"type": "delete_artifact", "path": "vision/context.json"}}
    return None


def _asr_lid_degraded(workdir: Path) -> dict | None:
    """Surface a mixed / low-confidence Soniox `--source-lang auto` detection
    (B7/R9) from the durable asr/lid.json marker the soniox provider writes —
    auditable without scanning logs. Parallel to _vision_degraded."""
    try:
        data = json.loads((workdir / "asr" / "lid.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("degraded"):
        return {"detected": data.get("detected", "unknown"),
                "hint": "--source-lang auto detection was mixed or low-confidence; "
                        "verify the source language and re-run if wrong"}
    return None


def _overrides_status(workdir: Path, valid_ids: set[str] | None) -> dict | None:
    """Surface which override keys actually match current segments, so a
    re-segmented ASR run makes misapplied overrides visible."""
    try:
        data = json.loads((workdir / "overrides.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    keys = [str(k) for k in data]
    if valid_ids is None:
        return {"applied": keys, "unmatched": []}
    return {"applied": [k for k in keys if k in valid_ids],
            "unmatched": [k for k in keys if k not in valid_ids]}


def build_report(
    workdir: str | Path,
    stage_timings: dict[str, float] | None = None,
    status: str = "completed",
    abort_info: dict | None = None,
    cfg=None,
) -> dict:
    workdir = Path(workdir)
    valid_ids = _current_segment_ids(workdir)
    entries = SkipLedger(workdir).entries()
    if valid_ids is not None:
        entries = {k: v for k, v in entries.items() if k in valid_ids}
    skipped = {k: v for k, v in entries.items() if v["status"] == "skipped"}
    accepted = {k: v for k, v in entries.items() if v["status"] == "accepted_skip"}
    # Best-effort length overflows (U6/R7): the clip was KEPT but could not fit its
    # slot — surfaced for visibility, not counted as a skip (it does not change the
    # exit code).
    length_overflows = {k: v for k, v in entries.items()
                        if v["status"] == "length_overflow"}
    # Placement-layer overruns (U4/R3): a clip that materially overran its slot
    # after the tempo cap and had its spilled tail trimmed (dub audio dropped).
    # Distinct from length_overflow — this DOES drive exit code 2.
    fit_overflows = {k: v for k, v in entries.items()
                     if v["status"] == "fit_overflow"}
    crosscheck = _crosscheck_report(workdir, valid_ids)

    # Language-run config (R22/agent-legibility, #14): which language pair ran and
    # whether it ran on the generic fallback profile, so an agent orchestrating
    # per-language runs can audit the outcome from report.json without replaying
    # the command line. `is_profiled_target=False` + `allow_fallback=True` is the
    # machine-readable signal that the run used the un-calibrated generic profile.
    config = None
    if cfg is not None:
        config = {
            "target_lang": cfg.target_lang,
            "source_lang": cfg.source_lang,
            "allow_fallback": cfg.allow_fallback,
            "is_profiled_target": is_profiled(cfg.target_lang),
            "tts_engine": cfg.tts_engine,
            "asr_engine": cfg.asr_engine,
        }

    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": status,
        "config": config,
        "skipped": skipped,
        "accepted_skips": accepted,
        "length_overflows": length_overflows,
        "fit_overflows": fit_overflows,
        "crosscheck_replacements": crosscheck["replacements"],
        "crosscheck_low_confidence": crosscheck["low_confidence"],
        "crosscheck_sub_rejected": crosscheck["sub_rejected"],
        "crosscheck_summary": crosscheck["summary"],
        "vision_degraded": _vision_degraded(workdir),
        "asr_lid_degraded": _asr_lid_degraded(workdir),
        "overrides": _overrides_status(workdir, valid_ids),
        "stage_timings_sec": {k: round(v, 2) for k, v in (stage_timings or {}).items()},
        "abort": abort_info,
    }


def write_report(workdir: str | Path, report: dict) -> Path:
    path = Path(workdir) / "report.json"
    atomic_write_bytes(path, json.dumps(report, ensure_ascii=False, indent=1).encode("utf-8"))
    return path


def exit_code(report: dict) -> int:
    """0 clean; 2 completed with skips/accepted-skips; 3 aborted; 1 fatal (R25)."""
    if report["status"] == "aborted":
        return 3
    if report["status"] == "failed":
        return 1
    if report["skipped"] or report["accepted_skips"] or report.get("fit_overflows"):
        return 2
    return 0


def console_summary(report: dict) -> str:
    lines = []
    status_label = {"completed": "completed", "aborted": "ABORTED", "failed": "FATAL ERROR"}
    lines.append(f"== Run report: {status_label.get(report['status'], report['status'])} ==")

    if report["abort"]:
        sig = report["abort"].get("signature")
        if isinstance(sig, dict):
            sig = f"{sig.get('stage')}/{sig.get('class')}/{sig.get('code')}"
        lines.append(f"Aborted after {report['abort'].get('count')} errors with the same signature {sig} "
                     "— address the root cause, then rerun the same command.")

    for label, key in (("Segment skip", "skipped"), ("Accepted-skip", "accepted_skips")):
        entries = report[key]
        if entries:
            lines.append(f"{label} ({len(entries)}):")
            for seg, entry in sorted(entries.items()):
                lines.append(f"  - {seg}: {entry['reason']}")

    overflows = report.get("length_overflows") or {}
    if overflows:
        lines.append(f"Length overflow ({len(overflows)}) — clips kept best-effort, "
                     "could not fit slot:")
        for seg in sorted(overflows):
            lines.append(f"  - {seg}")

    fit_overflows = report.get("fit_overflows") or {}
    if fit_overflows:
        lines.append(f"Fit overflow ({len(fit_overflows)}) — clip materially overran "
                     "its slot after the tempo cap; spilled dub audio was trimmed:")
        for seg in sorted(fit_overflows):
            lines.append(f"  - {seg}")

    summary = report.get("crosscheck_summary")
    if summary and any(summary["tally"].values()):
        t = summary["tally"]
        lines.append(
            f"Cross-check: {t['replace']} replaced / {t['keep']} kept / {t['subtitle']} sub"
            f" / {t['contested']} tied / {t['low_confidence']} low-confidence"
            + (f" / {t['granite_fallback']} fallback" if t["granite_fallback"] else ""))
        if summary["by_engine"]:
            engines = ", ".join(f"{eng or '?'}: {n}"
                                for eng, n in sorted(summary["by_engine"].items()))
            lines.append(f"  winning engine (replaced): {engines}")
        if summary["keywords"]:
            shown = ", ".join(summary["keywords"][:15])
            more = "..." if len(summary["keywords"]) > 15 else ""
            lines.append(f"  keyword list (run): {shown}{more}")

    if report["crosscheck_replacements"]:
        lines.append(f"Cross-check text replacements ({len(report['crosscheck_replacements'])}):")
        for r in report["crosscheck_replacements"]:
            winner = f"[{r['winner']}] " if r.get("winner") else ""
            contested = " (has tied region)" if r.get("contested") else ""
            lines.append(f"  - {r['segment']} {winner}{r['text_nemotron'][:50]!r} -> "
                         f"{r['text_effective'][:50]!r}{contested}")

    if report.get("crosscheck_sub_rejected"):
        lines.append(f"Subtitles rejected ({len(report['crosscheck_sub_rejected'])}):")
        for s in report["crosscheck_sub_rejected"]:
            lines.append(f"  - {s['segment']}: {s['reason']} "
                         f"(coverage {s.get('coverage')}, align {s.get('align')})")

    if report["crosscheck_low_confidence"]:
        lines.append(f"Low-confidence ({len(report['crosscheck_low_confidence'])}):")
        for r in report["crosscheck_low_confidence"]:
            note = f" (error {r['error']}, will retry next run)" if r.get("retryable") else ""
            lines.append(f"  - {r['segment']}{note}")

    if report["vision_degraded"]:
        lines.append(f"Vision degraded: {report['vision_degraded']['reason']} "
                     f"({report['vision_degraded']['hint']})")

    if report.get("asr_lid_degraded"):
        lines.append(f"ASR language detection uncertain (detected "
                     f"{report['asr_lid_degraded']['detected']}): "
                     f"{report['asr_lid_degraded']['hint']}")

    if report.get("overrides") and report["overrides"]["unmatched"]:
        lines.append("WARNING: override matches no existing segment: "
                     + ", ".join(report["overrides"]["unmatched"])
                     + " (ASR may have renumbered segments)")

    if report["stage_timings_sec"]:
        timing = ", ".join(f"{k} {v:.0f}s" for k, v in report["stage_timings_sec"].items())
        lines.append(f"Stage durations (this run): {timing}")

    if report["skipped"]:
        lines.append("Rerun the same command to retry skipped segments.")
    if not (report["skipped"] or report["accepted_skips"] or report["abort"]):
        lines.append("No segments were skipped.")
    return "\n".join(lines)
