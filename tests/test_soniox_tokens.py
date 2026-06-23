"""The pure _group_soniox_tokens transform in the soniox ASR provider (KTD3).

Soniox emits sub-word tokens carrying ms timestamps and (with diarization) a
string speaker; the token that begins a new word carries a leading space. These
pin the grouping on that whitespace boundary, the ms->s conversion, speaker
propagation, marker/empty skipping, and the collapse guard that fails loud when
a wrong boundary rule would fold the whole transcript into one word."""

import pytest

from loro.providers.asr import soniox as soniox_provider

group = soniox_provider._group_soniox_tokens


def _tok(text, start_ms, end_ms, speaker=None):
    t = {"text": text, "start_ms": start_ms, "end_ms": end_ms}
    if speaker is not None:
        t["speaker"] = speaker
    return t


def test_subword_tokens_group_into_one_word():
    # "Beau"/"ti"/"ful" with no leading spaces are continuations of one word.
    words = group([_tok("Beau", 1000, 1200), _tok("ti", 1200, 1350),
                   _tok("ful", 1350, 1600)])
    assert len(words) == 1
    assert words[0]["word"] == "Beautiful"
    assert words[0]["start"] == 1.0   # first token start_ms / 1000
    assert words[0]["end"] == 1.6     # last token end_ms / 1000


def test_leading_space_token_starts_a_new_word():
    words = group([_tok("Hello", 100, 480), _tok(" world", 500, 900)])
    assert [w["word"] for w in words] == ["Hello", "world"]
    assert words[1]["start"] == 0.5 and words[1]["end"] == 0.9


def test_ms_to_seconds_rounds_to_three_dp():
    words = group([_tok("test.", 6399, 6800)])
    assert words[0]["start"] == 6.399  # round(6399 / 1000, 3)
    assert words[0]["end"] == 6.8


def test_speaker_propagates_from_first_token():
    words = group([_tok("Hi", 0, 100, speaker="1"), _tok("there", 150, 400, speaker="1"),
                   _tok(" you", 450, 700, speaker="2")])
    assert words[0]["speaker"] == "1"   # first token of the word
    assert words[1]["speaker"] == "2"


def test_no_diarization_yields_none_speaker():
    words = group([_tok("Hi", 0, 100), _tok(" there", 150, 400)])
    assert all(w["speaker"] is None for w in words)


def test_whitespace_only_and_empty_tokens_are_skipped():
    words = group([_tok("Hi", 0, 100), _tok(" ", 100, 110), _tok("", 110, 110),
                   _tok(" there", 150, 400)])
    assert [w["word"] for w in words] == ["Hi", "there"]


def test_end_marker_token_is_skipped():
    words = group([_tok("Done", 0, 300), _tok("<end>", 300, 300)])
    assert [w["word"] for w in words] == ["Done"]


def test_punctuation_token_attaches_to_preceding_word():
    # A comma with no leading space is a continuation of the prior word.
    words = group([_tok("Hello", 0, 300), _tok(",", 300, 320), _tok(" world", 350, 700)])
    assert [w["word"] for w in words] == ["Hello,", "world"]


def test_word_dict_shape_is_exactly_start_end_word_speaker():
    words = group([_tok("Hi", 0, 100, speaker="1")])
    assert set(words[0]) == {"start", "end", "word", "speaker"}


def test_empty_token_stream_returns_empty():
    assert group([]) == []


def test_collapse_into_one_word_trips_the_guard():
    # A long stream where (wrong convention) nothing carries a leading space
    # folds into a single word — the silent-corruption failure mode. The guard
    # must raise rather than return one transcript-spanning word.
    tokens = [_tok(f"x{i}", i * 100, i * 100 + 80) for i in range(20)]
    with pytest.raises((RuntimeError, ValueError)):
        group(tokens)


def test_normal_token_density_does_not_trip_the_guard():
    # Many tokens that group into a plausible number of words must pass.
    tokens = []
    for i in range(20):
        # each word is two sub-word tokens; the first carries a leading space
        tokens.append(_tok(f" w{i}a", i * 100, i * 100 + 40))
        tokens.append(_tok(f"{i}b", i * 100 + 40, i * 100 + 80))
    words = group(tokens)
    assert len(words) == 20  # one word per leading-space boundary
