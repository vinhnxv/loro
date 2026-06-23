"""English transcription with segment timestamps, dispatched to an ASR provider.

The asr node resolves the configured engine's provider, calls its `transcribe`
contract, and owns the cross-engine tail every engine shares: the EN SRT write
(`srt.to_srt_wrapped`, identical across all engines) and the
`{segments, words, srt_src}` return (KTD3, R7).

The overlapping-window toolkit below — `window_bounds`, `merge_windows`,
`_win_artifact`, `MERGE_EPS` — stays here as a SHARED node-side toolkit (KTD3,
R7): the local provider imports and drives it (windows are cut, transcribed by
the Nemotron worker, merged at a segment boundary near the overlap midpoint),
while the cloud providers never reference it. Keeping it node-side lets the local
engine implement the same `transcribe` contract as the cloud engines without
forcing them to carry windowing concerns (R8).
"""

import logging
from pathlib import Path

from loro import providers
from loro.config import Config
from loro.state import DubState
from loro.utils import srt

log = logging.getLogger("loro.asr")

MERGE_EPS = 0.3  # seconds of slack when comparing boundaries


def window_bounds(duration: float, window: float, overlap: float) -> list[tuple[float, float]]:
    """Deterministic window layout for (duration, window, overlap)."""
    if duration <= window:
        return [(0.0, duration)]
    bounds = []
    start = 0.0
    step = window - overlap
    while True:
        end = min(start + window, duration)
        bounds.append((start, end))
        if end >= duration:
            return bounds
        start += step


def merge_windows(windows: list[dict], eps: float = MERGE_EPS) -> list[dict]:
    """Merge per-window transcriptions (absolute times) into one segment list.

    `windows`: [{"start": float, "end": float, "segments": [{start, end, text}]}]
    sorted by window start. In each overlap region the cut point is the
    segment boundary nearest the overlap midpoint; a segment straddling the
    cut is taken from the window where it lies deeper (more interior context).
    """
    merged = list(windows[0]["segments"])
    prev_end = windows[0]["end"]
    for win in windows[1:]:
        b_start = win["start"]
        incoming = list(win["segments"])
        overlap_end = min(prev_end, win["end"])
        mid = (b_start + overlap_end) / 2

        in_overlap = lambda t: b_start - eps <= t <= overlap_end + eps
        candidates = [s["end"] for s in merged if in_overlap(s["end"])]
        candidates += [s["start"] for s in incoming if in_overlap(s["start"])]
        cut = min(candidates, key=lambda c: abs(c - mid)) if candidates else mid

        keep_a = [s for s in merged if s["end"] <= cut + eps]
        keep_b = [s for s in incoming if s["start"] >= cut - eps]

        straddlers_a = [s for s in merged if s["start"] < cut - eps < cut + eps < s["end"]]
        straddlers_b = [s for s in incoming if s["start"] < cut - eps < cut + eps < s["end"]]
        if straddlers_a or straddlers_b:
            # Depth = margin between the segment and its window's risky edge
            depth_a = (prev_end - straddlers_a[-1]["end"]) if straddlers_a else float("-inf")
            depth_b = (straddlers_b[0]["start"] - b_start) if straddlers_b else float("-inf")
            chosen = straddlers_a[-1] if depth_a >= depth_b else straddlers_b[0]
            keep_a = [s for s in keep_a if s["end"] <= chosen["start"] + eps]
            keep_b = [s for s in keep_b if s["start"] >= chosen["end"] - eps]
            merged = keep_a + [chosen] + keep_b
        else:
            merged = keep_a + keep_b
        prev_end = win["end"]
    return sorted(merged, key=lambda s: s["start"])


def _win_artifact(asr_dir: Path, index: int) -> Path:
    return asr_dir / f"win_{index:04d}.json"


def asr(state: DubState, cfg: Config) -> DubState:
    """Transcribe English audio, emitting the {segments, words, srt_src} contract
    every engine shares. The configured engine's provider does the transcription
    (KTD1); this node owns only the shared SRT write + return tail (KTD3)."""
    workdir = Path(state["workdir"])
    asr_dir = workdir / "asr"
    asr_dir.mkdir(parents=True, exist_ok=True)

    result = providers.asr(cfg.asr_engine).transcribe(state, cfg, asr_dir)

    # The resolved input language reaches the translation prompt (U7). A provider
    # that ran language identification (source_lang="auto") returns the detected
    # language; otherwise fall back to the configured source (the local engine
    # never detects, and preflight already rejected `auto` for it).
    source_lang = result.source_lang or cfg.source_lang
    # The source SRT filename derives from the source locale (U10): the EN default
    # keeps transcript.en.srt byte-identical.
    srt_src = workdir / f"transcript.{source_lang.lower()}.srt"
    srt_src.write_text(
        srt.to_srt_wrapped(result.segments, result.words, side="source",
                           max_chars=cfg.srt_max_cue_chars, max_dur=cfg.srt_max_cue_dur),
        encoding="utf-8")
    log.info("%d raw segment(s), %d words, source=%s, source SRT -> %s",
             len(result.segments), len(result.words), source_lang, srt_src)
    return {"segments": result.segments, "words": result.words,
            "source_lang": source_lang, "srt_src": str(srt_src)}
