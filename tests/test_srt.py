from pathlib import Path

from loro.config import Config
from loro.state import Segment
from loro.utils import srt


def _en_segment(start=0.0, count=20, dur=0.4, gap=0.1):
    """A sentence segment plus a matching word stream (text_src == joined words,
    so concatenated EN cues must reproduce text_src exactly)."""
    words, t = [], start
    for i in range(count):
        words.append({"start": round(t, 3), "end": round(t + dur, 3), "word": f"w{i}"})
        t += dur + gap
    seg = Segment(index=0, start=words[0]["start"], end=words[-1]["end"],
                  text_src=" ".join(w["word"] for w in words))
    return seg, words


def test_to_srt_blocks():
    segments = [
        Segment(index=0, start=0.0, end=2.5, text_src="Hello there."),
        Segment(index=1, start=3.25, end=7.8, text_src="Welcome to the show."),
    ]
    out = srt.to_srt(segments, side="source")
    assert "1\n00:00:00,000 --> 00:00:02,500\nHello there." in out
    assert "2\n00:00:03,250 --> 00:00:07,800\nWelcome to the show." in out


def test_fmt_time():
    assert srt.fmt_time(3661.042) == "01:01:01,042"
    assert srt.parse_time("01:01:01,042") == 3661.042


def test_vi_text_used_for_vi_lang():
    seg = Segment(index=0, start=0, end=1, text_src="Hi", text_target="Chào")
    assert "Chào" in srt.to_srt([seg], side="target")
    assert "Hi" in srt.to_srt([seg], side="source")


def test_target_side_renders_target_text_for_any_language():
    # U10/R16: the branch keys on side, not "vi" — a French target (text_target in
    # French) must render the French target text on the target side, not fall
    # through to the English source. Guards the to_srt_wrapped branch restructure.
    seg = Segment(index=0, start=0.0, end=2.0, text_src="Hello world",
                  text_target="Bonjour le monde")
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], [], side="target"))
    rendered = " ".join(c.text for c in cues)
    assert "Bonjour le monde" in rendered
    assert "Hello world" not in rendered
    # Plain to_srt agrees.
    assert "Bonjour le monde" in srt.to_srt([seg], side="target")


# --- to_srt_wrapped: sub-style cues from whole-sentence segments (U2) ---

def test_wrapped_en_splits_long_sentence_at_word_timestamps():
    seg, words = _en_segment(count=30)
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], words, side="source",
                                             max_chars=20, max_dur=6.0))
    assert len(cues) > 1
    word_starts = {round(w["start"], 3) for w in words}
    word_ends = {round(w["end"], 3) for w in words}
    for c in cues:
        assert len(c.text) <= 20                       # within the char limit
        assert c.duration <= 6.0 + 1e-6                # within the duration limit
        assert round(c.start, 3) in word_starts        # boundaries at real words
        assert round(c.end, 3) in word_ends
    assert " ".join(c.text for c in cues) == seg.text_src  # nothing dropped


def test_wrapped_en_breaks_on_duration_when_chars_fit():
    # Generous char budget, tight duration budget: cues must still break.
    seg, words = _en_segment(count=20, dur=0.4, gap=0.1)  # ~0.5s/word
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], words, side="source",
                                             max_chars=1000, max_dur=2.0))
    assert len(cues) > 1
    for c in cues:
        assert c.duration <= 2.0 + 1e-6


def test_wrapped_en_short_sentence_is_one_cue():
    seg, words = _en_segment(count=3)
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], words, side="source"))
    assert len(cues) == 1
    assert cues[0].text == seg.text_src


def test_wrapped_en_segment_without_words_is_single_full_span_cue():
    seg = Segment(index=0, start=1.0, end=5.0, text_src="Hello there world.")
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], [], side="source"))
    assert len(cues) == 1
    assert cues[0].text == "Hello there world."
    assert cues[0].start == 1.0 and cues[0].end == 5.0


def test_wrapped_vi_tiles_span_proportionally_without_gaps():
    seg = Segment(index=0, start=2.0, end=12.0, text_src="x",
                  text_target=" ".join(f"tu{i}" for i in range(20)))
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], side="target",
                                             max_chars=20, max_dur=6.0))
    assert len(cues) > 1
    assert cues[0].start == 2.0           # tiles the full span
    assert cues[-1].end == 12.0
    for a, b in zip(cues, cues[1:]):
        assert abs(a.end - b.start) < 1e-3   # no gap, no overlap
    assert " ".join(c.text for c in cues) == seg.text_target


def test_wrapped_vi_prefers_clause_punctuation_breaks():
    # The first cue should end at the comma, not be packed to the char limit
    # mid-phrase, so a multi-syllable VI word isn't split across cues.
    seg = Segment(index=0, start=0.0, end=10.0, text_src="x",
                  text_target="Đôi khi bạn cần một agent, nhưng lúc khác cần cả đội ngũ phối hợp.")
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], side="target", max_chars=40, max_dur=6.0))
    assert len(cues) > 1
    assert cues[0].text.endswith(",")              # broke at the clause boundary
    # tiling invariants still hold
    assert cues[0].start == 0.0 and cues[-1].end == 10.0
    assert " ".join(c.text for c in cues) == seg.text_target


# --- U1: VI cue times anchored to the real EN word-timestamp curve (R1-R3) ---

def test_wrapped_vi_anchors_cue_boundary_to_en_word_times():
    # R1: a mid-sentence pause in the EN words pulls the VI cue boundary onto
    # the real post-pause anchor (2.0s), not the uniform proportional midpoint
    # (3.5s = seg.start + 0.5 * span).
    words = [
        {"start": 0.0, "end": 1.0, "word": "one"},
        {"start": 1.0, "end": 2.0, "word": "two"},
        # a long pause spoken-time 2.0 -> 5.0 with no words
        {"start": 5.0, "end": 6.0, "word": "three"},
        {"start": 6.0, "end": 7.0, "word": "four"},
    ]
    seg = Segment(index=0, start=0.0, end=7.0, text_src="one two three four",
                  text_target="aa bb cc dd")  # 2-char tokens force a split at index 2
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], words, side="target",
                                             max_chars=5, max_dur=6.0))
    assert len(cues) == 2
    assert abs(cues[0].end - 2.0) < 1e-3            # real anchor, not 3.5 uniform
    assert cues[0].start == 0.0 and cues[-1].end == 7.0
    assert " ".join(c.text for c in cues) == seg.text_target


def test_wrapped_vi_without_words_matches_proportional_tiling():
    # R2: with no covered EN words, VI cue times fall back to uniform
    # proportional tiling — passing words=[] is byte-identical to omitting them.
    seg = Segment(index=0, start=2.0, end=12.0, text_src="x",
                  text_target=" ".join(f"tu{i}" for i in range(20)))
    no_words = srt.to_srt_wrapped([seg], side="target", max_chars=20, max_dur=6.0)
    empty_words = srt.to_srt_wrapped([seg], [], side="target", max_chars=20, max_dur=6.0)
    assert no_words == empty_words
    cues = srt.parse_cues(no_words)
    assert cues[0].start == 2.0 and cues[-1].end == 12.0


def test_wrapped_vi_with_words_preserves_tiling_invariants():
    # R3: anchoring must keep the span fully tiled with no gaps/overlaps and no
    # negative durations when real words are supplied.
    seg, words = _en_segment(count=12)
    seg.text_target = " ".join(f"tu{i}" for i in range(25))
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], words, side="target",
                                             max_chars=20, max_dur=6.0))
    assert len(cues) > 1
    assert cues[0].start == seg.start
    assert cues[-1].end == seg.end
    for a, b in zip(cues, cues[1:]):
        assert abs(a.end - b.start) < 1e-3          # abut: no gap, no overlap
        assert b.end >= b.start                      # monotone, no negative dur
    assert " ".join(c.text for c in cues) == seg.text_target


def test_wrapped_vi_clause_break_holds_with_words_supplied():
    # The clause-boundary preference is independent of timing, so supplying
    # words must not regress test_wrapped_vi_prefers_clause_punctuation_breaks.
    words = [{"start": i * 0.5, "end": i * 0.5 + 0.4, "word": f"w{i}"} for i in range(20)]
    seg = Segment(index=0, start=0.0, end=words[-1]["end"], text_src="x",
                  text_target="Đôi khi bạn cần một agent, nhưng lúc khác cần cả đội ngũ phối hợp.")
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], words, side="target",
                                             max_chars=40, max_dur=6.0))
    assert len(cues) > 1
    assert cues[0].text.endswith(",")
    assert cues[0].start == 0.0 and cues[-1].end == words[-1]["end"]
    assert " ".join(c.text for c in cues) == seg.text_target


def test_wrapped_vi_fewer_words_than_cues_stays_monotone_in_span():
    # Edge: one covered EN word but many VI cues — boundaries stay monotone and
    # in-span with no division-by-zero or negative durations.
    words = [{"start": 1.0, "end": 4.0, "word": "solo"}]
    seg = Segment(index=0, start=1.0, end=4.0, text_src="solo",
                  text_target=" ".join(f"tu{i}" for i in range(12)))
    cues = srt.parse_cues(srt.to_srt_wrapped([seg], words, side="target",
                                             max_chars=12, max_dur=6.0))
    assert len(cues) > 1
    assert cues[0].start == 1.0 and cues[-1].end == 4.0
    prev = -1.0
    for c in cues:
        assert c.end >= c.start                      # no negative duration
        assert 1.0 - 1e-9 <= c.start <= 4.0 + 1e-9   # in span
        assert c.start >= prev - 1e-9                # monotone non-decreasing
        prev = c.start


def test_no_cross_check_path_writes_wrapped_en_srt(tmp_path):
    from loro.nodes.crosscheck import crosscheck
    seg, words = _en_segment(count=30)
    state = {"workdir": str(tmp_path), "segments": [seg], "words": words}
    result = crosscheck(state, Config(enable_cross_check=False, srt_max_cue_chars=20))
    cues = srt.parse_cues(Path(result["srt_src"]).read_text(encoding="utf-8"))
    assert len(cues) > 1   # sub-style cues, not a single sentence-wall block
