"""The assemblyai ASR path and the engine dispatch in the asr node.

The AssemblyAI client is mocked (a canned completed transcript), so these pin
the mapping (ms->s, text->word, speaker capture), the utterance->segment
fallback, the no-speech guard, the fingerprinted cache (R6), and that the
dispatch routes `local` away from the cloud client entirely."""

import json
from pathlib import Path

import pytest

from loro.config import Config
from loro.nodes import asr as asr_mod
from loro.providers.base import AsrResult
from loro.services import assemblyai
from loro.state import Segment


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("ASR_ENGINE", "ASSEMBLYAI_API_KEY", "ASSEMBLYAI_SPEECH_MODELS",
                 "ASSEMBLYAI_SPEAKER_LABELS", "ASSEMBLYAI_LANGUAGE_DETECTION",
                 "ASSEMBLYAI_LANGUAGE_CODE"):
        monkeypatch.delenv(name, raising=False)


CANNED = {
    "status": "completed",
    "text": "Hello world. This is a test.",
    "words": [
        {"start": 100, "end": 480, "text": "Hello", "speaker": "A"},
        {"start": 500, "end": 900, "text": "world.", "speaker": "A"},
        {"start": 1000, "end": 1400, "text": "This", "speaker": "B"},
        {"start": 1450, "end": 1700, "text": "is", "speaker": "B"},
        {"start": 1750, "end": 1950, "text": "a", "speaker": "B"},
        {"start": 6399, "end": 6800, "text": "test.", "speaker": "B"},
    ],
    "utterances": [
        {"start": 100, "end": 900, "text": "Hello world.", "speaker": "A"},
        {"start": 1000, "end": 6800, "text": "This is a test.", "speaker": "B"},
    ],
}


@pytest.fixture
def state(tmp_path):
    audio = tmp_path / "audio_16k.wav"
    audio.write_bytes(b"RIFFfake-wav")
    return {"workdir": str(tmp_path), "audio_16k": str(audio)}


def _cfg(**kw):
    return Config(asr_engine="assemblyai", assemblyai_api_key="k", **kw)


def _mock_transcribe(monkeypatch, payload=CANNED):
    calls = {"n": 0}

    def fake(cfg, audio):
        calls["n"] += 1
        return payload

    # The assemblyai provider calls assemblyai.transcribe module-qualified (U5),
    # so patching the service module is visible to it.
    monkeypatch.setattr(assemblyai, "transcribe", fake)
    return calls


def test_maps_words_ms_to_seconds_with_text_and_speaker(state, monkeypatch):
    # R2/KTD4: ms -> s rounded to 3dp, AssemblyAI `text` -> `word`, speaker kept.
    _mock_transcribe(monkeypatch)
    result = asr_mod.asr(state, _cfg())
    words = result["words"]
    assert len(words) == len(CANNED["words"])
    assert words[0] == {"start": 0.1, "end": 0.48, "word": "Hello", "speaker": "A"}
    assert words[-1]["start"] == 6.399  # round(6399 / 1000, 3)
    assert all("speaker" in w for w in words)


def test_segments_from_utterances_carry_speaker(state, monkeypatch):
    # R3: utterance-derived raw segments carry their speaker.
    _mock_transcribe(monkeypatch)
    segs = asr_mod.asr(state, _cfg())["segments"]
    assert [s.speaker for s in segs] == ["A", "B"]
    assert segs[0].text_src == "Hello world."
    assert segs[0].start == 0.1 and segs[0].end == 0.9


def test_persists_raw_transcript_and_utterances_and_srt(state, monkeypatch):
    _mock_transcribe(monkeypatch)
    result = asr_mod.asr(state, _cfg())
    workdir = Path(state["workdir"])
    raw = json.loads((workdir / "asr" / "assemblyai.json").read_text())
    assert raw["status"] == "completed"
    utt = json.loads((workdir / "asr" / "utterances.json").read_text())
    assert [u["speaker"] for u in utt["utterances"]] == ["A", "B"]
    assert utt["utterances"][0]["start"] == 0.1  # seconds
    assert result["srt_src"].endswith("transcript.en.srt")
    assert (workdir / "transcript.en.srt").exists()


def test_no_speech_raises_runtime_error(state, monkeypatch):
    _mock_transcribe(monkeypatch, {"status": "completed", "words": [], "utterances": []})
    with pytest.raises(RuntimeError):
        asr_mod.asr(state, _cfg())


def test_cached_transcript_not_refetched(state, monkeypatch):
    calls = _mock_transcribe(monkeypatch)
    asr_mod.asr(state, _cfg())
    assert calls["n"] == 1
    # Second call with the valid asr/assemblyai.json + unchanged inputs reuses it.
    asr_mod.asr(state, _cfg())
    assert calls["n"] == 1


def test_dispatch_local_never_calls_cloud_client(state, monkeypatch):
    # Selecting local routes to its provider via the registry (U5); the
    # AssemblyAI client is never constructed.
    calls = _mock_transcribe(monkeypatch)
    from loro.providers.asr.local import LocalAsrProvider
    sentinel = AsrResult(
        segments=[Segment(index=0, start=0.0, end=1.0, text_src="x")],
        words=[{"start": 0.0, "end": 1.0, "word": "x", "speaker": None}],
    )
    monkeypatch.setattr(LocalAsrProvider, "transcribe",
                        lambda self, st, cfg, asr_dir: sentinel)
    out = asr_mod.asr(state, Config(asr_engine="local"))
    assert out["segments"] == sentinel.segments
    assert calls["n"] == 0


def test_utterances_null_falls_back_without_raising(state, monkeypatch):
    payload = {
        "status": "completed",
        "text": "Hello world. This is fine.",
        "words": [
            {"start": 100, "end": 480, "text": "Hello", "speaker": None},
            {"start": 500, "end": 900, "text": "world.", "speaker": None},
            {"start": 1000, "end": 1400, "text": "This", "speaker": None},
            {"start": 1450, "end": 1700, "text": "is", "speaker": None},
            {"start": 1750, "end": 2100, "text": "fine.", "speaker": None},
        ],
        "utterances": None,
    }
    _mock_transcribe(monkeypatch, payload)
    result = asr_mod.asr(state, _cfg())
    assert len(result["segments"]) >= 1
    # null utterances persist as an empty list (U3 owns the artifact).
    utt = json.loads((Path(state["workdir"]) / "asr" / "utterances.json").read_text())
    assert utt["utterances"] == []
