"""Sentence segmentation from a word-timestamp stream (loro.utils.sentences).

The core regression these pin: Nemotron under-punctuates monologue, so the dub
backbone must come from an LLM segmentation pass (aligned back to word
timestamps) with a pause-based fallback — never a `hard_max` mid-sentence cut.
The under-punctuated path is exercised against the *real* fixture span (1072
words, 405.6s, zero internal sentence punctuation) extracted from the
AI-agent-design-patterns run, with a stub `llm_fn` standing in for Gemma.
"""

import json
from pathlib import Path

import pytest

from loro.utils import sentences as S

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "asr_under_punctuated_span.json"


def _words(spec):
    """spec: list of (start, end, text)."""
    return [{"start": s, "end": e, "word": w} for s, e, w in spec]


def _evenly(n, dur=0.5, gap=0.0, start=0.0, gaps=None, ender_at=None):
    """n words of length `dur`; `gaps[i]` adds extra silence before word i;
    `ender_at` appends a period to that word so it ends a sentence."""
    gaps = gaps or {}
    out = []
    t = start
    for i in range(n):
        t += gaps.get(i, 0.0)
        word = f"w{i}" + ("." if i == ender_at else "")
        out.append({"start": round(t, 3), "end": round(t + dur, 3), "word": word})
        t += dur + gap
    return out


def _regroup_llm(every=8):
    """Stub Gemma: re-segments the joined text into sentences of `every`
    whitespace tokens, preserving every word in order (the contract the real
    prompt enforces). Records its calls so tests can assert it ran or not."""
    calls = []

    def fn(text):
        calls.append(text)
        toks = text.split()
        return [" ".join(toks[i:i + every]) for i in range(0, len(toks), every)]

    fn.calls = calls
    return fn


# --- pause_split (the no-LLM degradation) ---

def test_pause_split_caps_duration_and_keeps_words_in_order():
    words = _evenly(40, dur=0.4, gaps={i: 0.5 for i in range(5, 40, 5)})
    groups = S.pause_split(words, max_dur=4.0, min_pause=0.3)
    flat = [w["word"] for g in groups for w in g]
    assert flat == [w["word"] for w in words]               # lossless, ordered
    for g in groups:
        assert g[-1]["end"] - g[0]["start"] <= 4.0 + 1e-9   # every unit within budget


def test_pause_split_never_breaks_mid_word_without_a_pause():
    # No gap >= min_pause anywhere, but the span is way over budget: it must
    # still cap duration by cutting at the largest available silence.
    words = _evenly(30, dur=0.5, gap=0.1)  # uniform 0.1s gaps, none >= 0.3
    groups = S.pause_split(words, max_dur=3.0, min_pause=0.3)
    assert len(groups) > 1
    for g in groups:
        assert g[-1]["end"] - g[0]["start"] <= 3.0 + 1e-9
        assert len(g) >= 1                                  # never an empty (mid-word) group


def test_pause_split_short_span_is_one_group():
    words = _evenly(4, dur=0.5)
    assert S.pause_split(words, max_dur=18.0, min_pause=0.3) == [words]


# --- align_sentences_to_words ---

def test_align_maps_sentences_to_word_spans():
    words = _words([(0.0, 0.5, "Hello"), (0.6, 1.0, "there."),
                    (1.5, 2.0, "How"), (2.1, 2.5, "are"), (2.6, 3.0, "you?")])
    groups = S.align_sentences_to_words(["Hello there.", "How are you?"], words)
    assert [[w["word"] for w in g] for g in groups] == [["Hello", "there."],
                                                        ["How", "are", "you?"]]
    assert groups[0][0]["start"] == 0.0 and groups[0][-1]["end"] == 1.0
    assert groups[1][0]["start"] == 1.5 and groups[1][-1]["end"] == 3.0


def test_align_returns_none_when_a_word_is_dropped():
    words = _words([(0.0, 0.5, "alpha"), (0.6, 1.0, "beta"), (1.1, 1.5, "gamma")])
    assert S.align_sentences_to_words(["alpha gamma"], words) is None  # 'beta' missing


def test_align_returns_none_when_a_word_is_added():
    words = _words([(0.0, 0.5, "alpha"), (0.6, 1.0, "beta")])
    assert S.align_sentences_to_words(["alpha beta gamma"], words) is None  # extra word


def test_align_keeps_token_less_punctuation_word_lossless():
    # A standalone punctuation token (no alphanumerics) contributes nothing to
    # the token stream; it must ride with a neighbour, never be dropped.
    words = _words([(0.0, 0.5, "alpha"), (0.6, 1.0, "beta"),
                    (1.0, 1.05, "--"), (1.5, 2.0, "gamma"), (2.1, 2.5, "delta")])
    groups = S.align_sentences_to_words(["alpha beta", "gamma delta"], words)
    flat = [w["word"] for g in groups for w in g]
    assert flat == ["alpha", "beta", "--", "gamma", "delta"]  # nothing dropped


# --- segment_into_sentences orchestration ---

def _seg_kwargs(**over):
    base = dict(max_dur=18.0, min_pause=0.4, max_unpunct_dur=30.0,
                min_punct_density=0.04, word_window=1000)
    base.update(over)
    return base


def test_punctuated_happy_path_does_not_call_the_llm():
    words = _words([(0.0, 0.5, "Hello"), (0.6, 1.0, "world."),
                    (1.5, 2.0, "Second"), (2.1, 2.5, "one.")])
    llm = _regroup_llm()
    segs, degraded = S.segment_into_sentences(words, llm_fn=llm, **_seg_kwargs())
    assert llm.calls == []                       # punctuation pre-split was enough
    assert not degraded
    assert [s["text"] for s in segs] == ["Hello world.", "Second one."]


def test_empty_words_falls_back_to_raw_segments():
    raw = [{"start": 1.0, "end": 5.0, "text": "ngắn gọn thôi."},
           {"start": 5.0, "end": 9.0, "text": "câu hai."}]
    segs, degraded = S.segment_into_sentences([], raw_segments=raw, llm_fn=_regroup_llm(),
                                              **_seg_kwargs())
    assert not degraded
    assert segs == raw


def test_under_punctuated_real_span_is_segmented_via_llm():
    words = json.loads(FIXTURE.read_text(encoding="utf-8"))["words"]
    span_dur = words[-1]["end"] - words[0]["start"]
    assert span_dur > 400 and S.punct_density(words) < 0.01  # the real monologue

    llm = _regroup_llm(every=8)
    segs, degraded = S.segment_into_sentences(words, llm_fn=llm, **_seg_kwargs())

    assert llm.calls, "the long under-punctuated span must reach the LLM"
    assert not degraded
    assert len(segs) > 1
    # Every unit is within the duration budget — no 405s blob, no hard_max cut.
    for s in segs:
        assert s["end"] - s["start"] <= 18.0 + 1e-9
    # No word is dropped or reordered: the concatenated unit text equals the
    # original word stream verbatim.
    joined = " ".join(s["text"] for s in segs).split()
    assert joined == [w["word"] for w in words]


def test_llm_unavailable_falls_back_to_pause_split():
    words = _evenly(120, dur=0.3, gap=0.1, gaps={i: 0.6 for i in range(6, 120, 6)})

    def boom(text):
        raise RuntimeError("oMLX down")

    segs, degraded = S.segment_into_sentences(words, llm_fn=boom, **_seg_kwargs(max_dur=6.0))
    assert degraded
    assert len(segs) > 1
    for s in segs:
        assert s["end"] - s["start"] <= 6.0 + 1e-9
    assert " ".join(s["text"] for s in segs).split() == [w["word"] for w in words]


def test_misaligned_llm_output_falls_back_to_pause_split():
    words = _evenly(120, dur=0.3, gap=0.1, gaps={i: 0.6 for i in range(6, 120, 6)})

    def drops_words(text):
        toks = text.split()
        return [" ".join(toks[i:i + 8]) for i in range(0, len(toks), 16)]  # skips half

    segs, degraded = S.segment_into_sentences(words, llm_fn=drops_words,
                                              **_seg_kwargs(max_dur=6.0))
    assert degraded
    for s in segs:
        assert s["end"] - s["start"] <= 6.0 + 1e-9
    assert " ".join(s["text"] for s in segs).split() == [w["word"] for w in words]


def test_long_clean_sentence_over_budget_is_pause_split():
    # One sentence (single trailing period) that runs long: KTD5 splits it at
    # silences down to the budget rather than leaving an over-long unit.
    words = _evenly(60, dur=0.3, gap=0.1, gaps={i: 0.6 for i in range(5, 60, 5)},
                    ender_at=59)
    segs, degraded = S.segment_into_sentences(words, llm_fn=_regroup_llm(),
                                              **_seg_kwargs(max_dur=5.0))
    assert not degraded                          # punctuated -> no LLM, no degradation
    assert len(segs) > 1
    for s in segs:
        assert s["end"] - s["start"] <= 5.0 + 1e-9
