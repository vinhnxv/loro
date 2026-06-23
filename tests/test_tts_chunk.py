"""The TTS clip builder splits long text across Higgs calls and concatenates.

Guards the regression that motivated chunking: a whole paragraph sent as one
request came back truncated (81s of audio for ~186s of text) yet passed QA.
After chunking, long text makes several calls and the concatenated clip carries
the full duration; short text still makes exactly one call.
"""

import numpy as np
import soundfile as sf

from loro.config import Config
from loro.harness import qa
from loro.nodes.tts import _synthesize_clip
from loro.utils.textchunk import HARDWRAP, chunk_for_tts, chunk_for_tts_typed

SR = 24000


class FakeHiggs:
    """Writes a sine clip whose length matches the text's expected speech
    duration, so each chunk lands inside the QA window."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.calls: list[str] = []

    def synthesize(self, text, out, voice=None):
        self.calls.append(text)
        dur = max(0.4, qa.syllable_count(text) / self.cfg.language_profile.rate)
        t = np.linspace(0, dur, int(SR * dur), endpoint=False)
        sf.write(str(out), (0.3 * np.sin(2 * np.pi * 220 * t)).astype("float32"), SR)


def _duration(path):
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    return len(audio) / sr


def test_short_text_makes_one_call(tmp_path):
    cfg = Config()
    fake = FakeHiggs(cfg)
    out = tmp_path / "seg_0000.wav"
    _synthesize_clip(fake, "Xin chào các bạn.", out, cfg)
    assert fake.calls == ["Xin chào các bạn."]
    assert out.exists() and _duration(out) > 0


def test_long_text_chunks_and_concatenates(tmp_path):
    cfg = Config()
    text = " ".join(f"Đây là câu thử nghiệm số {i} nhé." for i in range(20))
    expected_chunks = chunk_for_tts(text, cfg.tts_max_chunk_syllables, qa.syllable_count)
    assert len(expected_chunks) > 1  # the paragraph must actually split

    fake = FakeHiggs(cfg)
    out = tmp_path / "seg_0005.wav"
    _synthesize_clip(fake, text, out, cfg)

    # One Higgs call per chunk, in order
    assert fake.calls == expected_chunks

    # The concatenated clip carries (close to) the full content — not a single
    # truncated clip. Sum the per-chunk durations the fake would emit.
    per_chunk = [max(0.4, qa.syllable_count(c) / cfg.language_profile.rate)
                 for c in expected_chunks]
    gap = (len(expected_chunks) - 1) * cfg.tts_chunk_gap_ms / 1000
    total = _duration(out)
    assert total >= 0.8 * sum(per_chunk)          # nothing dropped
    assert total <= sum(per_chunk) + gap + 0.3    # and nothing fabricated
    assert total > max(per_chunk)                 # genuinely concatenated

    # The whole-clip QA gate accepts the assembled duration
    qa.check_clip(out, text, cfg)

    # Chunk temp files are cleaned up
    assert not list(tmp_path.glob(".chunk.*"))


def test_hardwrapped_sentence_has_no_internal_gap(tmp_path):
    # One long clause with no sentence/clause punctuation: it is hard-wrapped, so
    # every chunk join is a mid-clause cut and (tts_hardwrap_gap_ms=0) no silence
    # is inserted — the clip reads as continuous audio (U4).
    cfg = Config()
    assert cfg.tts_hardwrap_gap_ms == 0.0
    text = " ".join(f"tu{i}" for i in range(120))  # no punctuation -> hard-wrapped
    chunks, breaks = chunk_for_tts_typed(text, cfg.tts_max_chunk_syllables, qa.syllable_count)
    assert len(chunks) > 1 and all(b == HARDWRAP for b in breaks)

    fake = FakeHiggs(cfg)
    out = tmp_path / "seg_0009.wav"
    _synthesize_clip(fake, text, out, cfg)

    per_chunk = [max(0.4, qa.syllable_count(c) / cfg.language_profile.rate) for c in chunks]
    total = _duration(out)
    # No gaps: total tracks the summed chunk durations (edge-trim only), with no
    # (n-1)*chunk_gap_ms silence added between hard-wrap cuts.
    assert total <= sum(per_chunk) + 0.05
    assert total >= 0.8 * sum(per_chunk)


def test_sentence_joins_still_insert_gaps(tmp_path):
    # Contrast: real sentence boundaries keep the natural inter-chunk pause.
    cfg = Config()
    text = " ".join(f"Đây là câu thử nghiệm số {i} nhé." for i in range(20))
    chunks, breaks = chunk_for_tts_typed(text, cfg.tts_max_chunk_syllables, qa.syllable_count)
    assert len(chunks) > 1 and all(b != HARDWRAP for b in breaks)

    fake = FakeHiggs(cfg)
    out = tmp_path / "seg_0010.wav"
    _synthesize_clip(fake, text, out, cfg)

    per_chunk = [max(0.4, qa.syllable_count(c) / cfg.language_profile.rate) for c in chunks]
    gaps = (len(chunks) - 1) * cfg.tts_chunk_gap_ms / 1000
    total = _duration(out)
    # The boundary gaps measurably extend the clip beyond the bare chunk sum.
    assert total >= sum(per_chunk) + 0.5 * gaps
