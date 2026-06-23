"""Minimal SRT writer plus a tolerant SRT/VTT cue reader.

SRT is derived *output* only for the dub — segment state lives in the
per-stage JSON manifests, which carry text_target and indexes the SRT format
cannot round-trip. The cue reader is the reverse: it parses an *input*
subtitle file (embedded track or sidecar) into timed cues (R34/R35)."""

import re
from dataclasses import dataclass

from loro.state import Segment

_TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")
# Cue timestamps tolerate optional hours (VTT short form MM:SS.mmm)
_CUE_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[,.](\d{1,3})")
# Inline VTT markup: <c>, <00:00:01.000>, <v Speaker>, </c>
_TAG_RE = re.compile(r"<[^>]+>")


def fmt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_time(text: str) -> float:
    m = _TIME_RE.fullmatch(text.strip())
    if not m:
        raise ValueError(f"bad SRT timestamp: {text!r}")
    h, mi, s, ms = (int(g) for g in m.groups())
    return h * 3600 + mi * 60 + s + ms / 1000


def to_srt(segments: list[Segment], side: str = "source") -> str:
    """Render plain SRT for one side of the dub. `side="target"` emits the
    translated text; "source" emits the source transcript. The branch keys on
    side, not on a language literal, so any non-VI target still renders its target
    text rather than falling through to the source (U10)."""
    blocks = []
    for i, seg in enumerate(segments, 1):
        text = seg.text_target if side == "target" else seg.text_src
        blocks.append(f"{i}\n{fmt_time(seg.start)} --> {fmt_time(seg.end)}\n{text}\n")
    return "\n".join(blocks)


def _render_cues(cues: list["Cue"]) -> str:
    blocks = []
    for i, cue in enumerate(cues, 1):
        blocks.append(f"{i}\n{fmt_time(cue.start)} --> {fmt_time(cue.end)}\n{cue.text}\n")
    return "\n".join(blocks)


def _words_in(words: list[dict], seg: Segment) -> list[dict]:
    """The word-timestamps that fall inside one segment's span, as a half-open
    interval [seg.start, seg.end). A sentence segment's end is its last word's
    end; the next segment's first word starts at exactly that time when the two
    abut (no inter-word pause), so an inclusive upper bound would duplicate that
    word into both cues and overlap their spans. The exclusive upper bound keeps
    each word in exactly one segment; the segment's own last word (start strictly
    below its end) is still captured."""
    return [w for w in words if seg.start - 1e-6 <= w["start"] < seg.end]


def _source_cues(seg: Segment, words: list[dict], max_chars: int, max_dur: float) -> list["Cue"]:
    """Word-accurate source cues: pack a segment's words into a cue until the next
    word would overflow max_chars or max_dur, then break at that real word
    timestamp. With no word timing, the segment is one full-span cue."""
    if not words:
        return [Cue(seg.start, seg.end, seg.text_src)] if seg.text_src.strip() else []
    cues: list[Cue] = []
    cur: list[dict] = []
    for w in words:
        text = " ".join(x["word"] for x in cur + [w])
        dur = w["end"] - cur[0]["start"] if cur else 0.0
        if cur and (len(text) > max_chars or dur > max_dur):
            cues.append(Cue(cur[0]["start"], cur[-1]["end"],
                            " ".join(x["word"] for x in cur)))
            cur = [w]
        else:
            cur.append(w)
    if cur:
        cues.append(Cue(cur[0]["start"], cur[-1]["end"],
                        " ".join(x["word"] for x in cur)))
    return cues


_CLAUSE_END = (",", ";", ":", ".", "!", "?", "…")
_CLAUSE_CLOSERS = "\"')]}»”’"


def _target_break(toks: list[str], start: int, end: int) -> int:
    """Where to end a target cue that spans toks[start:end). Prefer the last
    clause-punctuation boundary inside the cue so it doesn't end mid-phrase
    (e.g. splitting a multi-syllable word like Vietnamese 'thiết kế'); fall back
    to the char/duration limit when the clause has no internal punctuation."""
    for k in range(end - 1, start, -1):
        if toks[k].rstrip(_CLAUSE_CLOSERS)[-1:] in _CLAUSE_END:
            return k + 1
    return end


def _target_cues(seg: Segment, words: list[dict], max_chars: int, max_dur: float) -> list["Cue"]:
    """The target side has no word timing, so tile text_target by word count across
    [seg.start, seg.end]: grow each cue under the char limit (and the
    proportional-duration limit), prefer to end it at a clause boundary, then
    assign each group a span by its cumulative word offset so cues tile with no
    gap or overlap (KTD4). Cue *times* anchor to the real EN word-timestamp
    curve via `time_at` (KTD1) so a boundary lands on a spoken-time anchor (e.g.
    just before a mid-sentence pause) instead of a uniform-rate midpoint; with
    no covered EN words `time_at` collapses to today's proportional tiling (R2).
    Grouping (char/clause/max_dur) is unchanged — only time assignment moves."""
    toks = seg.text_target.split()
    if not toks:
        return []
    total = len(toks)
    span = max(seg.end - seg.start, 1e-6)

    # Monotone non-decreasing boundary times anchored on the covered EN words:
    # bounds[0] = seg.start, bounds[k] = end of the k-th covered word (clamped
    # into [previous bound, seg.end] so reordering/overlap can't go backwards),
    # the last pinned to seg.end so time_at(1) lands exactly on the span end.
    bounds = [seg.start]
    for w in words:
        bounds.append(min(max(w["end"], bounds[-1]), seg.end))
    if words:
        bounds[-1] = seg.end

    def time_at(frac: float) -> float:
        # frac in [0,1] -> a real time. Empty words: uniform tiling (R2). Else
        # interpolate along the word-boundary curve, so equal token fractions on
        # either side of a pause map to the real pre/post-pause times, and
        # adjacent cues still abut because they share the same frac at the seam.
        if not words:
            return seg.start + frac * span
        p = frac * (len(bounds) - 1)          # position along the boundaries
        i = int(p)
        if i >= len(bounds) - 1:
            return bounds[-1]                  # frac == 1 (and clamp overshoot)
        return bounds[i] + (p - i) * (bounds[i + 1] - bounds[i])

    groups: list[tuple[int, int]] = []
    start = 0
    while start < total:
        end = start + 1
        while end < total:
            chars = len(" ".join(toks[start:end + 1]))
            prop_dur = (end + 1 - start) / total * span
            if chars > max_chars or prop_dur > max_dur:
                break
            end += 1
        if end < total:
            end = _target_break(toks, start, end)  # prefer a clause boundary
        groups.append((start, end))
        start = end
    return [Cue(time_at(a / total), time_at(b / total), " ".join(toks[a:b]))
            for a, b in groups]


def to_srt_wrapped(segments: list[Segment], words: list[dict] | None = None,
                   side: str = "source", *, max_chars: int = 84,
                   max_dur: float = 6.0) -> str:
    """Render sub-style cues from whole-sentence segments (KTD1/KTD4). The SOURCE
    side breaks at real word timestamps from `words` (word-accurate); the TARGET
    side tiles the translated sentence span proportionally (the dub has no
    target-language word timing). The branch keys on side, not on a "vi" literal,
    so any non-VI target tiles its target text instead of word-timing the source
    (U10). Keeps subtitles short and readable even though the dub backbone is
    whole sentences."""
    words = words or []
    cues: list[Cue] = []
    for seg in segments:
        if side == "target":
            cues.extend(_target_cues(seg, _words_in(words, seg), max_chars, max_dur))
        else:
            cues.extend(_source_cues(seg, _words_in(words, seg), max_chars, max_dur))
    return _render_cues(cues)


@dataclass
class Cue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def _cue_ts(field: str) -> float:
    m = _CUE_TS_RE.search(field)
    if not m:
        raise ValueError(f"bad cue timestamp: {field!r}")
    hours, minutes, seconds, ms = m.groups()
    ms = (ms + "000")[:3]  # pad/truncate to milliseconds
    return (int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds)
            + int(ms) / 1000)


def parse_cues(text: str) -> list[Cue]:
    """Parse SRT or WebVTT into timed cues, tolerant of VTT headers, cue
    identifiers, inline tags and cue-setting suffixes. Cues are returned in
    file order; overlapping/out-of-order cues are left as-is for the caller."""
    cues: list[Cue] = []
    lines = text.replace("﻿", "").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if "-->" in line:
            left, _, right = line.partition("-->")
            try:
                start = _cue_ts(left)
                end = _cue_ts(right)  # search() ignores any trailing settings
            except ValueError:
                i += 1
                continue
            j = i + 1
            body = []
            while j < len(lines) and lines[j].strip():
                body.append(lines[j])
                j += 1
            cue_text = _TAG_RE.sub("", " ".join(body)).strip()
            cue_text = re.sub(r"\s+", " ", cue_text)
            if cue_text and end > start:
                cues.append(Cue(start, end, cue_text))
            i = j
        else:
            i += 1
    return cues
