"""Voice-clone reference selection over the sentence backbone (U5).

After sentence_seg, the reference candidates are whole sentences rather than
acoustic units, so the 3–12s preferred band has fewer members. These pin that
a sensible reference clip is still chosen, and that the relaxed band and the
clamp behave as before.
"""

import json

import pytest

from loro.config import Config
from loro.harness import artifacts
from loro.nodes.voice import MAX_REF_SECONDS, _pick_reference, _voice_cast, voice_ref
from loro.state import Segment


def _seg(index, dur, start=0.0):
    return Segment(index=index, start=start, end=start + dur, text_src=f"s{index}")


def test_prefers_longest_clip_in_3_to_12s_band():
    # Sentence-sized durations; the longest inside [3, 12] wins (not the 15s one).
    segs = [_seg(0, 2.0), _seg(1, 5.0), _seg(2, 8.0), _seg(3, 11.0), _seg(4, 15.0)]
    assert _pick_reference(segs).index == 3


def test_falls_back_to_relaxed_band_when_no_preferred_member():
    # All sentences short (a plausible monologue of clipped sentences): the
    # relaxed [1.5, 20] band still yields a reference.
    segs = [_seg(0, 1.6), _seg(1, 2.2), _seg(2, 1.8)]
    assert _pick_reference(segs).index == 1   # longest within the relaxed band


def test_max_segment_duration_sentences_stay_within_relaxed_band():
    # sentence_seg caps units at max_segment_duration (18s) < relaxed max (20s),
    # so a pool of long sentences always has a valid reference.
    segs = [_seg(0, 17.0), _seg(1, 18.0)]
    chosen = _pick_reference(segs)
    assert chosen.index == 1 and chosen.duration <= 20.0


def test_no_usable_segment_raises():
    with pytest.raises(RuntimeError):
        _pick_reference([_seg(0, 0.5), _seg(1, 30.0)])  # too short / too long


def test_max_ref_seconds_is_the_preferred_upper_bound():
    assert MAX_REF_SECONDS == 12.0


# --- preset voice casting (U3) ---

POOL = ["Adrian", "Maya", "Noah", "Nina"]


def _spk_seg(index, speaker):
    return Segment(index=index, start=float(index), end=float(index) + 1.0,
                   text_src=f"s{index}", speaker=speaker)


def _soniox_cfg(**kw):
    # Explicit pool/map/default so the cast is deterministic regardless of the
    # developer's SONIOX_* environment; any field is overridable via kw.
    base = {"tts_engine": "soniox", "soniox_voice_pool": list(POOL),
            "soniox_default_voice": "Adrian"}
    base.update(kw)
    return Config(**base)


def test_two_speakers_get_distinct_pool_voices(tmp_path):
    # R4: A and B each get a distinct pool voice by sorted index; cast.json
    # round-trips the map.
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, "A"), _spk_seg(1, "B"), _spk_seg(2, "A")]}
    cast = _voice_cast(state, _soniox_cfg())["voice_cast"]
    assert cast == {"A": "Adrian", "B": "Maya"}
    assert cast["A"] != cast["B"]
    assert set(cast.values()) <= set(POOL)
    on_disk = json.loads((tmp_path / "voice" / "cast.json").read_text())
    assert on_disk == cast


def test_pin_does_not_reshuffle_other_speaker(tmp_path):
    # R5: pinning A=Grace does not change B's pool-assigned voice (B's slot is a
    # fixed function of its own sorted index, KTD6).
    segs = [_spk_seg(0, "A"), _spk_seg(1, "B")]
    unpinned = _voice_cast({"workdir": str(tmp_path / "u"), "segments": segs},
                           _soniox_cfg())["voice_cast"]
    pinned = _voice_cast({"workdir": str(tmp_path / "p"), "segments": segs},
                         _soniox_cfg(soniox_voice_map={"A": "Grace"}))["voice_cast"]
    assert pinned["A"] == "Grace"
    assert pinned["B"] == unpinned["B"]


def test_all_empty_speakers_map_to_default_only(tmp_path):
    # R6: no diarization -> the "" sentinel maps to the default voice and there
    # are no per-speaker pool assignments.
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, ""), _spk_seg(1, "")]}
    cast = _voice_cast(state, _soniox_cfg())["voice_cast"]
    assert cast == {"": "Adrian"}


def test_mixed_speaker_and_empty_casts_both(tmp_path):
    # A real speaker plus a segment with no label: the speaker draws from the
    # pool and the "" sentinel still maps to the default.
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, "A"), _spk_seg(1, "")]}
    cast = _voice_cast(state, _soniox_cfg())["voice_cast"]
    assert cast == {"A": "Adrian", "": "Adrian"}


def test_more_speakers_than_pool_wraps_deterministically(tmp_path):
    # pool of 4; speakers A..F -> E wraps to pool[4 % 4], F to pool[5 % 4].
    segs = [_spk_seg(i, chr(ord("A") + i)) for i in range(6)]
    cast = _voice_cast({"workdir": str(tmp_path), "segments": segs},
                       _soniox_cfg())["voice_cast"]
    assert cast["E"] == POOL[4 % 4]
    assert cast["F"] == POOL[5 % 4]


def test_cast_is_deterministic_regardless_of_segment_order(tmp_path):
    # Unsorted segment order still casts by sorted speaker id (no reassignment
    # risk from set iteration order).
    segs = [_spk_seg(0, "B"), _spk_seg(1, "A"), _spk_seg(2, "C")]
    a = _voice_cast({"workdir": str(tmp_path / "a"), "segments": segs},
                    _soniox_cfg())["voice_cast"]
    b = _voice_cast({"workdir": str(tmp_path / "b"), "segments": segs},
                    _soniox_cfg())["voice_cast"]
    assert a == b == {"A": "Adrian", "B": "Maya", "C": "Noah"}


def test_rerun_reuses_cached_cast_artifact(tmp_path):
    # Identical inputs -> the cached cast.json is reused (the sidecar is not
    # rewritten), so no speaker is silently reassigned.
    segs = [_spk_seg(0, "A"), _spk_seg(1, "B")]
    state = {"workdir": str(tmp_path), "segments": segs}
    cfg = _soniox_cfg()
    _voice_cast(state, cfg)
    cast_path = tmp_path / "voice" / "cast.json"
    written_at = artifacts.read_meta(cast_path)["written_at"]
    _voice_cast(state, cfg)
    assert artifacts.read_meta(cast_path)["written_at"] == written_at


def test_empty_pool_falls_back_to_default_voice(tmp_path):
    # An empty pool (e.g. SONIOX_VOICE_POOL set to whitespace) must not
    # ZeroDivisionError on pool[i % len(pool)] — it falls back to the default.
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, "A"), _spk_seg(1, "B")]}
    cast = _voice_cast(state, _soniox_cfg(soniox_voice_pool=[]))["voice_cast"]
    assert cast == {"A": "Adrian", "B": "Adrian"}


# --- preset_voices accessor characterization + Gemini cast (U2) ---

# Golden values captured from the pre-refactor Soniox _voice_cast for the
# A/B/A fixture with pool [Adrian,Maya,Noah,Nina], empty map, default Adrian.
# The U2 refactor routes _voice_cast through cfg.preset_voices; these pin that
# Soniox's resolved cast AND its persisted input_fingerprint stay byte-identical
# (KTD6) — any drift would invalidate every existing Soniox cast.json.
_SONIOX_GOLDEN_CAST = {"A": "Adrian", "B": "Maya"}
_SONIOX_GOLDEN_CAST_JSON = '{\n "A": "Adrian",\n "B": "Maya"\n}'
_SONIOX_GOLDEN_FINGERPRINT = "a8b7af6e05b76be42c736da4fa87976cbc4bd4626ba2b448bfe25b39232d4fd5"


def test_soniox_cast_is_byte_identical_after_refactor(tmp_path):
    segs = [_spk_seg(0, "A"), _spk_seg(1, "B"), _spk_seg(2, "A")]
    state = {"workdir": str(tmp_path), "segments": segs}
    cast = _voice_cast(state, _soniox_cfg())["voice_cast"]
    assert cast == _SONIOX_GOLDEN_CAST
    cast_path = tmp_path / "voice" / "cast.json"
    assert cast_path.read_text() == _SONIOX_GOLDEN_CAST_JSON
    # The persisted fingerprint pins the inputs dict's key NAMES and shape
    # (speakers/pool/map/default), not just the resolved voices.
    assert artifacts.read_meta(cast_path)["input_fingerprint"] == _SONIOX_GOLDEN_FINGERPRINT


GEMINI_POOL = ["Kore", "Puck", "Aoede", "Charon"]


def _gemini_cfg(**kw):
    base = {"tts_engine": "gemini", "gemini_voice_pool": list(GEMINI_POOL),
            "gemini_default_voice": "Kore"}
    base.update(kw)
    return Config(**base)


def test_gemini_two_speakers_cast_from_gemini_pool(tmp_path):
    # A,B cast deterministically by sorted index from the Gemini pool.
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, "A"), _spk_seg(1, "B")]}
    cast = _voice_cast(state, _gemini_cfg())["voice_cast"]
    assert cast == {"A": "Kore", "B": "Puck"}
    assert set(cast.values()) <= set(GEMINI_POOL)


def test_gemini_pin_overrides_without_reshuffling_other(tmp_path):
    segs = [_spk_seg(0, "A"), _spk_seg(1, "B")]
    unpinned = _voice_cast({"workdir": str(tmp_path / "u"), "segments": segs},
                           _gemini_cfg())["voice_cast"]
    pinned = _voice_cast({"workdir": str(tmp_path / "p"), "segments": segs},
                         _gemini_cfg(gemini_voice_map={"A": "Charon"}))["voice_cast"]
    assert pinned["A"] == "Charon"
    assert pinned["B"] == unpinned["B"]


def test_gemini_empty_speaker_maps_to_default(tmp_path):
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, ""), _spk_seg(1, "")]}
    cast = _voice_cast(state, _gemini_cfg())["voice_cast"]
    assert cast == {"": "Kore"}


def test_cloning_engine_returns_ref_not_voice_cast(tmp_path):
    # The cloning path is unchanged: a vieneu run returns ref_audio/ref_text and
    # never a voice_cast key (uses the preset --ref-audio branch, no ffmpeg).
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")
    state = {"workdir": str(tmp_path), "segments": [_spk_seg(0, "A")]}
    cfg = Config(tts_engine="vieneu", ref_audio=str(ref), ref_text="hello")
    out = voice_ref(state, cfg)
    assert out["ref_audio"].endswith("ref.wav")
    assert out["ref_text"] == "hello"
    assert "voice_cast" not in out


def test_preset_engine_voice_ref_casts_via_capability_flag(tmp_path):
    # AE3 (U4): a preset engine routes voice_ref to casting (voice_cast), selected
    # by the provider's clones flag (False) through the cfg.tts_uses_cloning
    # adapter — no engine-name check in voice casting or config.
    from loro import providers
    assert providers.tts("soniox").clones is False
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, "A"), _spk_seg(1, "B")]}
    out = voice_ref(state, _soniox_cfg())
    assert out["voice_cast"] == {"A": "Adrian", "B": "Maya"}
    assert (tmp_path / "voice" / "cast.json").exists()
