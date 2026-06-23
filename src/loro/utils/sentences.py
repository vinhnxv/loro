"""Turn a word-timestamp stream into complete-sentence spans.

Nemotron under-punctuates monologue (a real fixture: 1266 word tokens, 5
sentence-ending marks, a 405s span with zero internal sentence punctuation), so
sentence boundaries cannot come from word-token punctuation alone — grouping on
`. ! ? …` would yield one 405-second "sentence" and degrade straight back to the
mid-sentence `hard_max` cut this module exists to remove (KTD2).

The strategy, in order:

1. **Pre-split on existing sentence punctuation** (cheap, exact) — short spans
   that Nemotron already punctuated become sentences directly.
2. **LLM segmentation** for any span that is long *and* under-punctuated: an
   injected `llm_fn` re-segments the joined word text into sentences, which are
   then aligned back to the word stream (case/punctuation-insensitive, in order)
   so each sentence inherits exact `start`/`end` from its first/last word. The
   LLM decides boundaries only; the unit text is the original word tokens.
3. **Pause-based fallback** when the LLM is unavailable or its sentences don't
   align to the words: split at inter-word silences so each unit stays within
   the duration budget, never mid-word — strictly better than a `hard_max` cut.

This module is a leaf: no harness import, no model client. `llm_fn` is injected
by the `sentence_seg` node, which wires it to Gemma via `llm.chat`.
"""

import re

# Sentence-ending punctuation. Trailing closing quotes/brackets are stripped
# before the check so `word."` and `idea?)` still register as enders.
_SENT_END = ("." , "!", "?", "…")
_CLOSERS = "\"')]}»”’"
_TOKEN_RE = re.compile(r"[\w']+", flags=re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens, punctuation-insensitive (apostrophes kept). Mirrors
    the cross-check normalize so alignment matches the same token model."""
    return _TOKEN_RE.findall(text.lower())


def _join_words(words: list[dict]) -> str:
    return " ".join(w["word"] for w in words).strip()


def _seg_dict(words: list[dict]) -> dict:
    return {"start": words[0]["start"], "end": words[-1]["end"],
            "text": _join_words(words)}


def _ends_sentence(word: str) -> bool:
    token = word.rstrip(_CLOSERS)
    return bool(token) and token[-1] in _SENT_END


def punct_presplit(words: list[dict]) -> list[list[dict]]:
    """Split the word stream into spans at every word that ends a sentence."""
    spans: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        cur.append(w)
        if _ends_sentence(w["word"]):
            spans.append(cur)
            cur = []
    if cur:
        spans.append(cur)
    return spans


def punct_density(words: list[dict]) -> float:
    """Fraction of words that end a sentence. ~0 for the under-punctuated
    monologue spans this module targets; high for already-punctuated text."""
    if not words:
        return 0.0
    enders = sum(1 for w in words if _ends_sentence(w["word"]))
    return enders / len(words)


def _best_split(words: list[dict], min_pause: float) -> int:
    """Index `i` to break between words[i] and words[i+1]. Prefer the silence
    >= min_pause closest to the span's time-midpoint (most balanced cut); when
    none qualifies, take the largest silence so the duration is still capped.
    The break always lands between whole words, never mid-word."""
    mid = (words[0]["start"] + words[-1]["end"]) / 2
    gaps = []
    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        split_t = (words[i]["end"] + words[i + 1]["start"]) / 2
        gaps.append((gap, i, abs(split_t - mid)))
    qualifying = [g for g in gaps if g[0] >= min_pause]
    if qualifying:
        return min(qualifying, key=lambda g: g[2])[1]  # closest to midpoint
    return max(gaps, key=lambda g: (g[0], -g[2]))[1]    # largest silence


def pause_split(words: list[dict], *, max_dur: float, min_pause: float) -> list[list[dict]]:
    """Recursively split `words` so every group spans <= max_dur, breaking at
    the best available inter-word silence (KTD2/KTD5). Never breaks mid-word; a
    single word (or a sub-`max_dur` group) is returned as-is."""
    words = list(words)
    if len(words) < 2 or words[-1]["end"] - words[0]["start"] <= max_dur:
        return [words]
    i = _best_split(words, min_pause)
    return (pause_split(words[: i + 1], max_dur=max_dur, min_pause=min_pause)
            + pause_split(words[i + 1:], max_dur=max_dur, min_pause=min_pause))


def align_sentences_to_words(sentences: list[str], words: list[dict]) -> list[list[dict]] | None:
    """Map LLM-returned sentences back to word groups by consuming the word
    stream's tokens in order (case/punctuation-insensitive). Returns the word
    groups, or None when the sentences don't reproduce the word stream exactly
    (a dropped/added/paraphrased word) — the caller then falls back to a pause
    split. A sentence boundary that would fall inside one source word is also
    rejected, so spans never overlap."""
    # flat[(token, word_index)] over the word stream; a numeric/abbrev word may
    # expand to several tokens, all mapped to the same word index.
    flat: list[tuple[str, int]] = []
    for wi, w in enumerate(words):
        for tok in _tokens(w["word"]):
            flat.append((tok, wi))
    if not flat:
        return None

    pos = 0
    cursor = 0  # next unassigned word index; keeps grouping lossless
    groups: list[list[dict]] = []
    for sentence in sentences:
        sent_toks = _tokens(sentence)
        if not sent_toks:
            continue
        start = pos
        for st in sent_toks:
            if pos >= len(flat) or flat[pos][0] != st:
                return None  # drift: sentences don't match the word stream
            pos += 1
        w_start, w_end = flat[start][1], flat[pos - 1][1]
        if w_start < cursor:
            return None  # boundary split a source word across two sentences
        # Span from the cursor (not w_start) so a token-less word — a standalone
        # punctuation token Nemotron occasionally emits, which contributes
        # nothing to `flat` — attaches to the adjacent sentence instead of being
        # silently dropped. This preserves the lossless guarantee.
        groups.append(words[cursor: w_end + 1])
        cursor = w_end + 1
    if pos != len(flat):
        return None  # the LLM dropped trailing words
    if cursor < len(words):
        # Trailing token-less words ride with the final sentence.
        if not groups:
            return None
        groups[-1] = groups[-1] + words[cursor:]
    return groups


def window_words(words: list[dict], max_words: int, min_pause: float) -> list[list[dict]]:
    """Chunk a long span into <= max_words windows for the LLM call, backing off
    each cut to the largest silence near the boundary so a window edge doesn't
    land mid-sentence (the LLM re-segments each window independently)."""
    if max_words <= 0 or len(words) <= max_words:
        return [list(words)]
    windows: list[list[dict]] = []
    i, n = 0, len(words)
    while i < n:
        end = min(i + max_words, n)
        if end < n:
            lo = max(i + 1, i + int(max_words * 0.8))
            best, best_gap = end, -1.0
            for j in range(lo, end):
                gap = words[j + 1]["start"] - words[j]["end"]
                if gap > best_gap:
                    best_gap, best = gap, j + 1
            end = best
        windows.append(words[i:end])
        i = end
    return windows


def _llm_segment_span(span: list[dict], *, llm_fn, max_dur: float, min_pause: float,
                      word_window: int) -> tuple[list[list[dict]], bool]:
    """Re-segment one under-punctuated span via the LLM, windowed for bounded
    latency. Any window whose sentences fail to align falls back to a pause
    split and flags the span degraded."""
    out: list[list[dict]] = []
    degraded = False
    for window in window_words(span, word_window, min_pause):
        try:
            sentences = llm_fn(_join_words(window))
            aligned = align_sentences_to_words(sentences, window)
            if not aligned:
                raise ValueError("sentence alignment failed")
            out.extend(aligned)
        except Exception:
            degraded = True
            out.extend(pause_split(window, max_dur=max_dur, min_pause=min_pause))
    return out, degraded


def segment_into_sentences(
    words: list[dict],
    *,
    raw_segments: list[dict] | None = None,
    llm_fn=None,
    max_dur: float,
    min_pause: float,
    max_unpunct_dur: float,
    min_punct_density: float,
    word_window: int = 1000,
) -> tuple[list[dict], bool]:
    """Turn the word stream into complete-sentence segment dicts (KTD2/KTD5).

    Returns `(segments, degraded)` where each segment is
    `{"start", "end", "text"}` and `degraded` is True when an LLM segmentation
    failed and the span fell back to a pause split (the node leaves a degraded
    artifact unfinalized so a later run retries — mirrors translate R5b/R32).

    With no word timing (`words` empty), falls back to the raw acoustic segments
    untouched — this is the legitimate no-words path, not a degradation.
    """
    if not words:
        return ([dict(start=r["start"], end=r["end"], text=str(r.get("text", "")).strip())
                 for r in (raw_segments or []) if str(r.get("text", "")).strip()], False)

    degraded = False
    groups: list[list[dict]] = []
    for span in punct_presplit(words):
        span_dur = span[-1]["end"] - span[0]["start"]
        if (llm_fn is not None and span_dur > max_unpunct_dur
                and punct_density(span) < min_punct_density):
            seg_groups, span_degraded = _llm_segment_span(
                span, llm_fn=llm_fn, max_dur=max_dur, min_pause=min_pause,
                word_window=word_window)
            groups.extend(seg_groups)
            degraded = degraded or span_degraded
        else:
            groups.append(span)

    # KTD5: even a clean sentence may exceed the duration budget — pause-split it.
    capped: list[list[dict]] = []
    for g in groups:
        if g and g[-1]["end"] - g[0]["start"] > max_dur:
            capped.extend(pause_split(g, max_dur=max_dur, min_pause=min_pause))
        else:
            capped.append(g)

    return ([_seg_dict(g) for g in capped if g and _join_words(g)], degraded)
