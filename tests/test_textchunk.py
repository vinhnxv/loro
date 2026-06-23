"""Unit tests for the TTS sentence chunker (loro.utils.textchunk).

Sizing uses the real qa.syllable_count so the tests pin the same syllable model
the QA gate and TTS path use in production.
"""

import pytest

from loro.config import Config
from loro.harness.qa import syllable_count
from loro.utils.textchunk import BOUNDARY, HARDWRAP, chunk_for_tts, chunk_for_tts_typed


def chunk(text, cap):
    return chunk_for_tts(text, cap, syllable_count)


def typed(text, cap):
    return chunk_for_tts_typed(text, cap, syllable_count)


def test_short_text_is_one_chunk():
    assert chunk("Xin chào các bạn.", 60) == ["Xin chào các bạn."]


def test_no_terminal_punctuation_still_one_chunk():
    # The whole string is one atom when nothing splits it
    assert chunk("bản dịch số 0 đây", 60) == ["bản dịch số 0 đây"]


def test_empty_or_whitespace_yields_no_chunks():
    assert chunk("", 60) == []
    assert chunk("   \n  ", 60) == []


def test_packs_sentences_up_to_budget():
    sents = ["Câu một hai ba.", "Câu bốn năm sáu.", "Câu bảy tám chín.",
             "Câu mười mười một."]  # 4 syllables each
    chunks = chunk(" ".join(sents), cap=8)
    # 4-syllable sentences pack two-per-chunk under an 8-syllable cap
    assert all(syllable_count(c) <= 8 for c in chunks)
    assert len(chunks) == 2
    # Every sentence survives, in order, with its punctuation
    assert " ".join(chunks) == " ".join(sents)


def test_oversized_sentence_splits_on_clauses():
    text = "alpha, beta gamma delta epsilon, zeta eta."  # one sentence, 7 syl
    chunks = chunk(text, cap=4)
    assert all(syllable_count(c) <= 4 for c in chunks)
    # split at the commas, content preserved
    assert "".join(chunks).replace(" ", "") == text.replace(" ", "")


def test_oversized_clause_hard_wraps_on_words():
    text = "a b c d e f g h i j"  # ten one-syllable words, no punctuation
    chunks = chunk(text, cap=3)
    assert all(syllable_count(c) <= 3 for c in chunks)
    assert " ".join(chunks).split() == text.split()  # no word lost or reordered


def test_decimal_is_not_split_at_its_dot():
    # The dot in "3.5" is followed by a digit, not whitespace, so it is not a
    # sentence boundary
    assert chunk("Mất 3.5 giây để chạy.", 100) == ["Mất 3.5 giây để chạy."]


def test_invalid_budget_raises():
    with pytest.raises(ValueError):
        chunk("bất kỳ.", 0)


# --- break-type-aware chunking (U4) ---

def test_typed_sentence_joins_are_boundaries():
    sents = ["Câu một hai ba.", "Câu bốn năm sáu.", "Câu bảy tám chín.",
             "Câu mười mười một."]  # 4 syllables each, pack two-per-chunk at cap 8
    chunks, breaks = typed(" ".join(sents), cap=8)
    assert len(chunks) == 2
    assert len(breaks) == len(chunks) - 1
    assert all(b == BOUNDARY for b in breaks)


def test_typed_hardwrapped_clause_joins_are_hardwrap():
    # No sentence/clause punctuation: one clause hard-wrapped on words, so every
    # internal join is a mid-clause cut.
    chunks, breaks = typed("a b c d e f g h i j", cap=3)
    assert len(chunks) > 1
    assert breaks and all(b == HARDWRAP for b in breaks)


def test_typed_mixed_boundary_then_hardwrap():
    # A short sentence, then a long unpunctuated clause that must hard-wrap.
    chunks, breaks = typed("Ngắn. " + " ".join(["w"] * 9), cap=3)
    assert breaks[0] == BOUNDARY        # join after the sentence is a real pause
    assert HARDWRAP in breaks[1:]       # cuts inside the long clause are hardwraps


def test_typed_short_text_single_chunk_no_joins():
    chunks, breaks = typed("Xin chào các bạn.", 60)
    assert chunks == ["Xin chào các bạn."]
    assert breaks == []


def test_chunk_for_tts_return_shape_unchanged():
    # The thin wrapper must keep the list-of-str contract existing callers rely on.
    assert chunk("Xin chào các bạn.", 60) == ["Xin chào các bạn."]
    sents = " ".join(["Câu một hai ba."] * 4)
    assert chunk(sents, 8) == chunk_for_tts_typed(sents, 8, syllable_count)[0]


def test_cps_profile_does_not_over_fragment_a_passage():
    # U5/R5: a non-VI passage chunked by the CPS counter against the FR profile's
    # character chunk_budget (240) must NOT fragment ~4x the way feeding a char
    # counter a syllable-valued budget (60) would. Comparable-duration chunks.
    fr = Config(target_lang="fr").language_profile
    passage = " ".join(["Bonjour tout le monde, comment allez-vous."] * 8)

    correct = chunk_for_tts(passage, fr.chunk_budget, fr.counter)      # 240 chars
    fragmented = chunk_for_tts(passage, 60, fr.counter)                # syllable budget, char counter
    assert len(correct) < len(fragmented)
    # Every correct chunk stays within the character budget.
    assert all(fr.counter(c) <= fr.chunk_budget for c in correct)
