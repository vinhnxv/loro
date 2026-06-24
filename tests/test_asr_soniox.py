"""The soniox ASR path and the three-way engine dispatch in the asr node.

The Soniox STT client is mocked (a canned completed token transcript), so these
pin the mapping (tokens -> words, ms -> s, speaker capture, R2/R3/R4), the
punct_presplit segment fallback, the no-speech guard, the fingerprinted cache
(R7/KTD9, incl. context-change invalidation), and that the dispatch routes
`assemblyai`/`local` away from the Soniox client entirely (R8)."""

import json
from pathlib import Path

import pytest

from loro.config import Config
from loro.nodes import asr as asr_mod
from loro.providers.base import AsrResult
from loro.services import soniox_stt
from loro.state import Segment


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("ASR_ENGINE", "SONIOX_API_KEY", "SONIOX_STT_MODEL",
                 "SONIOX_STT_LANGUAGE_HINTS", "SONIOX_STT_SPEAKER_DIARIZATION",
                 "SONIOX_STT_CONTEXT_TERMS", "SONIOX_STT_CONTEXT_TEXT"):
        monkeypatch.delenv(name, raising=False)


# "Hello world. This is a test." as Soniox sub-word tokens (ms units). The token
# that begins a new word carries a leading space; two speakers are diarized.
CANNED = {
    "tokens": [
        {"text": "Hello", "start_ms": 100, "end_ms": 480, "speaker": "1"},
        {"text": " world.", "start_ms": 500, "end_ms": 900, "speaker": "1"},
        {"text": " This", "start_ms": 1000, "end_ms": 1400, "speaker": "2"},
        {"text": " is", "start_ms": 1450, "end_ms": 1700, "speaker": "2"},
        {"text": " a", "start_ms": 1750, "end_ms": 1950, "speaker": "2"},
        {"text": " test.", "start_ms": 6399, "end_ms": 6800, "speaker": "2"},
    ],
}


@pytest.fixture
def state(tmp_path):
    audio = tmp_path / "audio_16k.wav"
    audio.write_bytes(b"RIFFfake-wav")
    return {"workdir": str(tmp_path), "audio_16k": str(audio)}


def _cfg(**kw):
    return Config(asr_engine="soniox", soniox_api_key="k", **kw)


def _mock_transcribe(monkeypatch, payload=CANNED):
    calls = {"n": 0}

    def fake(cfg, audio, **kw):
        calls["n"] += 1
        calls["kw"] = kw
        return payload

    # The soniox provider calls soniox_stt.transcribe module-qualified (U5), so
    # patching the service module is visible to it.
    monkeypatch.setattr(soniox_stt, "transcribe", fake)
    return calls


def test_default_source_lang_reaches_state_without_lid(state, monkeypatch):
    # R20: the default (configured) source is en, LID stays off (byte-identical),
    # and the configured source reaches state.
    calls = _mock_transcribe(monkeypatch)
    result = asr_mod.asr(state, _cfg())
    assert result["source_lang"] == "en"
    assert calls["kw"]["enable_language_identification"] is False


def test_auto_enables_lid_and_detects_source_language(state, monkeypatch):
    # R9: source_lang=auto turns LID on and the detected per-token language
    # populates state["source_lang"].
    payload = {"tokens": [{**t, "language": "fr"} for t in CANNED["tokens"]]}
    calls = _mock_transcribe(monkeypatch, payload)
    result = asr_mod.asr(state, _cfg(source_lang="auto"))
    assert result["source_lang"] == "fr"
    assert calls["kw"]["enable_language_identification"] is True
    assert calls["kw"]["language_hints"] != ["en"]  # widened


def test_auto_mixed_detection_warns_and_uses_majority(state, monkeypatch, caplog):
    # R12: a mixed LID result is surfaced (warned), not silently mis-targeted.
    import logging
    toks = CANNED["tokens"]
    payload = {"tokens": [{**t, "language": ("fr" if i else "de")}
                          for i, t in enumerate(toks)]}
    _mock_transcribe(monkeypatch, payload)
    with caplog.at_level(logging.WARNING):
        result = asr_mod.asr(state, _cfg(source_lang="auto"))
    assert result["source_lang"] == "fr"  # majority
    assert any("mixed" in r.message.lower() for r in caplog.records)


def test_auto_missing_language_field_falls_back_loudly(state, monkeypatch, caplog):
    # KTD5: the LID response shape is unverified; an absent language field must
    # degrade to a fallback with a loud warning, never crash.
    import logging
    _mock_transcribe(monkeypatch)  # CANNED has no `language` field
    with caplog.at_level(logging.WARNING):
        result = asr_mod.asr(state, _cfg(source_lang="auto"))
    assert result["source_lang"] == "en"
    assert any("UNVERIFIED" in r.message or "no per-token" in r.message
               for r in caplog.records)


# --- U9: auto-LID mixed/low-confidence surfaced via asr/lid.json marker (B7/R9) ---

def test_auto_mixed_detection_writes_lid_marker(state, monkeypatch):
    toks = CANNED["tokens"]
    payload = {"tokens": [{**t, "language": ("fr" if i else "de")}
                          for i, t in enumerate(toks)]}
    _mock_transcribe(monkeypatch, payload)
    asr_mod.asr(state, _cfg(source_lang="auto"))
    marker = json.loads((Path(state["workdir"]) / "asr" / "lid.json").read_text())
    assert marker["degraded"] is True
    assert marker["detected"] == "fr"   # majority


def test_auto_missing_language_field_writes_lid_marker(state, monkeypatch):
    _mock_transcribe(monkeypatch)        # CANNED has no per-token `language` field
    asr_mod.asr(state, _cfg(source_lang="auto"))
    marker = json.loads((Path(state["workdir"]) / "asr" / "lid.json").read_text())
    assert marker["degraded"] is True
    assert marker["detected"] == "en"    # loud fallback


def test_auto_clean_detection_writes_no_lid_marker(state, monkeypatch):
    payload = {"tokens": [{**t, "language": "fr"} for t in CANNED["tokens"]]}
    _mock_transcribe(monkeypatch, payload)
    asr_mod.asr(state, _cfg(source_lang="auto"))
    assert not (Path(state["workdir"]) / "asr" / "lid.json").exists()


def test_non_auto_run_writes_no_lid_marker(state, monkeypatch):
    _mock_transcribe(monkeypatch)
    asr_mod.asr(state, _cfg())           # default en, LID off, detector not called
    assert not (Path(state["workdir"]) / "asr" / "lid.json").exists()


def test_non_auto_run_clears_stale_lid_marker(state, monkeypatch):
    # An earlier auto run leaves a degraded marker; a later non-auto run on the
    # same workdir must clear it, never leave a stale agent-facing signal.
    _mock_transcribe(monkeypatch)        # CANNED has no per-token language field
    asr_mod.asr(state, _cfg(source_lang="auto"))
    assert (Path(state["workdir"]) / "asr" / "lid.json").exists()
    asr_mod.asr(state, _cfg())           # non-auto rerun, same workdir
    assert not (Path(state["workdir"]) / "asr" / "lid.json").exists()


def test_maps_tokens_to_words_in_seconds_with_speaker(state, monkeypatch):
    # R2/R3/R4: tokens -> words, ms -> s, joined word text, speaker captured.
    _mock_transcribe(monkeypatch)
    words = asr_mod.asr(state, _cfg())["words"]
    assert [w["word"] for w in words] == ["Hello", "world.", "This", "is", "a", "test."]
    assert words[0] == {"start": 0.1, "end": 0.48, "word": "Hello", "speaker": "1"}
    assert words[-1]["start"] == 6.399  # round(6399 / 1000, 3)
    assert words[2]["speaker"] == "2"


def test_segments_from_punct_presplit_carry_text(state, monkeypatch):
    # Soniox returns no utterances -> punctuation split over the grouped words.
    _mock_transcribe(monkeypatch)
    segs = asr_mod.asr(state, _cfg())["segments"]
    assert len(segs) == 2
    assert segs[0].text_src == "Hello world."
    assert segs[1].text_src == "This is a test."
    assert segs[0].start == 0.1


def test_persists_raw_transcript_and_srt_no_utterances_artifact(state, monkeypatch):
    _mock_transcribe(monkeypatch)
    result = asr_mod.asr(state, _cfg())
    workdir = Path(state["workdir"])
    raw = json.loads((workdir / "asr" / "soniox.json").read_text())
    assert raw["tokens"][0]["text"] == "Hello"
    # Soniox has no utterance grouping and no consumer reads utterances.json on
    # this path, so it must not be written (dead weight).
    assert not (workdir / "asr" / "utterances.json").exists()
    assert result["srt_src"].endswith("transcript.en.srt")
    assert (workdir / "transcript.en.srt").exists()


def test_no_speech_raises_runtime_error(state, monkeypatch):
    _mock_transcribe(monkeypatch, {"tokens": []})
    with pytest.raises(RuntimeError):
        asr_mod.asr(state, _cfg())


def test_cached_transcript_not_refetched(state, monkeypatch):
    calls = _mock_transcribe(monkeypatch)
    asr_mod.asr(state, _cfg())
    assert calls["n"] == 1
    asr_mod.asr(state, _cfg())  # valid asr/soniox.json + unchanged inputs
    assert calls["n"] == 1


def test_context_change_invalidates_cache_and_refetches(state, monkeypatch):
    calls = _mock_transcribe(monkeypatch)
    asr_mod.asr(state, _cfg())
    assert calls["n"] == 1
    # Changing the recognition context changes the fingerprint -> re-call.
    asr_mod.asr(state, _cfg(soniox_stt_context_terms=["LangGraph"]))
    assert calls["n"] == 2


_SENTINEL = AsrResult(
    segments=[Segment(index=0, start=0.0, end=1.0, text_src="x")],
    words=[{"start": 0.0, "end": 1.0, "word": "x", "speaker": None}],
)


def test_dispatch_assemblyai_never_calls_soniox_client(state, monkeypatch):
    # Selecting assemblyai routes to its provider via the registry (U5); the
    # soniox STT client is never constructed.
    calls = _mock_transcribe(monkeypatch)
    from loro.providers.asr.assemblyai import AssemblyaiAsrProvider
    monkeypatch.setattr(AssemblyaiAsrProvider, "transcribe",
                        lambda self, st, cfg, asr_dir: _SENTINEL)
    out = asr_mod.asr(state, Config(asr_engine="assemblyai"))
    assert out["segments"] == _SENTINEL.segments
    assert calls["n"] == 0


def test_dispatch_local_never_calls_soniox_client(state, monkeypatch):
    calls = _mock_transcribe(monkeypatch)
    from loro.providers.asr.local import LocalAsrProvider
    monkeypatch.setattr(LocalAsrProvider, "transcribe",
                        lambda self, st, cfg, asr_dir: _SENTINEL)
    out = asr_mod.asr(state, Config(asr_engine="local"))
    assert out["segments"] == _SENTINEL.segments
    assert calls["n"] == 0
