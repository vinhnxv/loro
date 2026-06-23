"""services/gemini batched path (U4): the multi-speaker request body, and the
silence splitter over synthetic tone/silence audio (real ffmpeg silencedetect,
`requests` mocked). Pins multi- vs single-speaker config, the per-line pause
directive (KTD9), the n-1-deepest-gap cut, speech-span trimming (A1), and the
SplitError fallback signal."""

import base64
import json

import numpy as np
import pytest

from loro.config import Config
from loro.services import gemini
from loro.services.gemini import SplitError

SR = 24000


def _tone(dur, amp=0.5, freq=220):
    t = np.arange(int(dur * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype("float32")


def _silence(dur):
    return np.zeros(int(dur * SR), dtype="float32")


def _pcm(*parts):
    audio = np.concatenate(parts)
    return (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()


def _audio_payload(pcm):
    return {"candidates": [{"content": {"parts": [
        {"inlineData": {"data": base64.b64encode(pcm).decode("ascii")}}
    ]}}]}


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text or (json.dumps(payload) if payload else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json})
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("TTS_ENGINE", "GEMINI_API_KEY", "GEMINI_DEFAULT_VOICE",
                 "GEMINI_VOICE_POOL", "GEMINI_SPLIT_MIN_GAP_MS", "GEMINI_STYLE_PROMPT"):
        monkeypatch.delenv(name, raising=False)


def _client(monkeypatch, pcm, **cfg_kw):
    fake = _FakeHTTP([_Resp(200, _audio_payload(pcm))])
    monkeypatch.setattr(gemini, "requests", fake)
    cfg = Config(tts_engine="gemini", gemini_api_key="k", retry_base_delay=0.0,
                 gemini_split_min_gap_ms=200.0, **cfg_kw)
    return gemini.GeminiClient(cfg), fake


# --- request body shape ---

def test_two_distinct_speakers_build_multi_speaker_body(monkeypatch):
    # n=2, one clear gap so the split succeeds and we can read back the body.
    pcm = _pcm(_tone(0.4), _silence(0.4), _tone(0.4))
    client, fake = _client(monkeypatch, pcm)
    turns = [("A", "Xin chào", "Kore"), ("B", "Tạm biệt", "Puck")]
    with client as c:
        pieces, sr = c.synthesize_batch(turns)
    assert len(pieces) == 2 and sr == SR
    sc = fake.calls[0]["json"]["generationConfig"]["speechConfig"]
    cfgs = sc["multiSpeakerVoiceConfig"]["speakerVoiceConfigs"]
    assert len(cfgs) == 2
    assert {c["speaker"]: c["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
            for c in cfgs} == {"A": "Kore", "B": "Puck"}
    prompt = fake.calls[0]["json"]["contents"][0]["parts"][0]["text"]
    assert "A: Xin chào" in prompt and "B: Tạm biệt" in prompt
    assert prompt.index("A: Xin chào") < prompt.index("B: Tạm biệt")


def test_single_distinct_speaker_builds_single_voice_config(monkeypatch):
    pcm = _pcm(_tone(0.4), _silence(0.4), _tone(0.4))
    client, fake = _client(monkeypatch, pcm)
    turns = [("A", "Xin chào", "Kore"), ("A", "Tạm biệt", "Kore")]
    with client as c:
        c.synthesize_batch(turns)
    sc = fake.calls[0]["json"]["generationConfig"]["speechConfig"]
    assert "multiSpeakerVoiceConfig" not in sc
    assert sc["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Kore"


def test_pause_directive_present_for_same_speaker_batch(monkeypatch):
    # KTD9: a same-speaker batch still requests an inter-line pause for every
    # line so the splitter has boundaries to cut on.
    pcm = _pcm(_tone(0.4), _silence(0.4), _tone(0.4), _silence(0.4), _tone(0.4))
    client, fake = _client(monkeypatch, pcm)
    turns = [("A", "một", "Kore"), ("A", "hai", "Kore"), ("A", "ba", "Kore")]
    with client as c:
        pieces, _ = c.synthesize_batch(turns)
    assert len(pieces) == 3
    prompt = fake.calls[0]["json"]["contents"][0]["parts"][0]["text"]
    assert "pause between every line" in prompt
    assert "một" in prompt and "hai" in prompt and "ba" in prompt


# --- splitter ---

def test_split_happy_path_yields_n_pieces(monkeypatch):
    pcm = _pcm(_tone(0.4), _silence(0.4), _tone(0.4), _silence(0.4), _tone(0.4))
    client, _ = _client(monkeypatch, pcm)
    turns = [("A", "a", "Kore"), ("B", "b", "Puck"), ("A", "c", "Kore")]
    with client as c:
        pieces, sr = c.synthesize_batch(turns)
    assert len(pieces) == 3
    assert all(len(p) > 0 for p in pieces)


def test_each_piece_trimmed_to_speech_span(monkeypatch):
    # A1: a cut inside a 0.4s gap must not leave ~0.2s of silence on a neighbor —
    # each piece's duration reflects its ~0.4s speech span, not the gap share.
    pcm = _pcm(_tone(0.4), _silence(0.4), _tone(0.4), _silence(0.4), _tone(0.4))
    client, _ = _client(monkeypatch, pcm)
    turns = [("A", "a", "Kore"), ("B", "b", "Puck"), ("A", "c", "Kore")]
    with client as c:
        pieces, sr = c.synthesize_batch(turns)
    for p in pieces:
        dur = len(p) / sr
        assert 0.3 < dur < 0.55   # ~0.4s speech, not 0.4 + 0.2 gap share


def test_split_failure_too_few_gaps_raises(monkeypatch):
    # n=3 but only one gap in the audio -> fewer than n-1 -> SplitError (fallback).
    pcm = _pcm(_tone(0.4), _silence(0.4), _tone(0.4))
    client, _ = _client(monkeypatch, pcm)
    turns = [("A", "a", "Kore"), ("B", "b", "Puck"), ("A", "c", "Kore")]
    with client as c:
        with pytest.raises(SplitError):
            c.synthesize_batch(turns)


def test_within_span_dip_under_min_gap_is_not_a_boundary(monkeypatch):
    # A 0.1s dip (< 200ms min gap) inside a turn must not be treated as a cut;
    # only the one real 0.4s gap qualifies, so n=2 splits cleanly into 2.
    pcm = _pcm(_tone(0.4), _silence(0.4),
               _tone(0.2), _silence(0.1), _tone(0.2))
    client, _ = _client(monkeypatch, pcm)
    turns = [("A", "a", "Kore"), ("B", "b", "Puck")]
    with client as c:
        pieces, sr = c.synthesize_batch(turns)
    assert len(pieces) == 2
    # The second piece keeps its internal dip (~0.2 + 0.1 + 0.2 of speech+dip).
    assert len(pieces[1]) / sr > 0.4


def test_leading_silence_not_chosen_as_cut(monkeypatch):
    # TTS lead-in/out padding must NOT be picked as the cut for a 2-turn batch,
    # even when it is longer than the real interior gap — otherwise the two turns
    # merge into one piece and the other piece is empty. The interior gap is the
    # boundary; both pieces carry ~0.4s of speech.
    pcm = _pcm(_silence(0.8), _tone(0.4), _silence(0.3), _tone(0.4), _silence(0.6))
    client, _ = _client(monkeypatch, pcm)
    turns = [("A", "a", "Kore"), ("B", "b", "Puck")]
    with client as c:
        pieces, sr = c.synthesize_batch(turns)
    assert len(pieces) == 2
    assert all(0.25 < len(p) / sr < 0.55 for p in pieces)


def test_same_speaker_too_few_gaps_falls_back(monkeypatch):
    # KTD9: a same-speaker batch where the model emitted too few pauses raises
    # SplitError (the node will fall back), not a wrong-count result.
    pcm = _pcm(_tone(0.4), _silence(0.4), _tone(0.4))   # only 1 gap
    client, _ = _client(monkeypatch, pcm)
    turns = [("A", "một", "Kore"), ("A", "hai", "Kore"), ("A", "ba", "Kore")]
    with client as c:
        with pytest.raises(SplitError):
            c.synthesize_batch(turns)
