"""services/gemini single-speaker client: one generateContent POST per clip,
`requests` mocked (no network). Pins the request body shape, the header-only
x-goog-api-key credential (never logged), the base64-PCM → WAV wrap, the
text-instead-of-audio retry (KTD8), and the retry/error classification."""

import base64
import json

import numpy as np
import pytest
import requests
import soundfile as sf

from loro.config import Config
from loro.harness.retry import StageError
from loro.services import gemini


def _pcm_bytes(n_samples=2400):
    # s16le mono PCM (what Gemini returns base64-encoded in inlineData.data).
    return (np.arange(n_samples, dtype="<i2") % 1000).tobytes()


def _audio_payload(pcm):
    return {"candidates": [{"content": {"parts": [
        {"inlineData": {"mimeType": "audio/L16;codec=pcm;rate=24000",
                        "data": base64.b64encode(pcm).decode("ascii")}}
    ]}}]}


def _text_payload(text="xin lỗi, tôi không thể"):
    # The documented text-instead-of-audio quirk (KTD8): a part with no
    # inlineData audio.
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _FakeHTTP:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json,
                           "timeout": timeout})
        resp = self.responses.pop(0)
        if isinstance(resp, BaseException):
            raise resp
        return resp


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("TTS_ENGINE", "GEMINI_API_KEY", "GEMINI_BASE_URL", "GEMINI_MODEL",
                 "GEMINI_SAMPLE_RATE", "GEMINI_DEFAULT_VOICE", "GEMINI_VOICE_POOL",
                 "GEMINI_VOICE_MAP", "GEMINI_STYLE_PROMPT", "GEMINI_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


def _cfg(**kw):
    return Config(tts_engine="gemini", gemini_api_key="secret-key",
                  retry_base_delay=0.0, **kw)


def _client(monkeypatch, responses, **cfg_kw):
    fake = _FakeHTTP(responses)
    monkeypatch.setattr(gemini, "requests", fake)
    return gemini.GeminiClient(_cfg(**cfg_kw)), fake


def test_request_body_shape_and_url(monkeypatch, tmp_path):
    client, fake = _client(monkeypatch, [_Resp(200, _audio_payload(_pcm_bytes()))])
    with client as c:
        c.synthesize("Xin chào", tmp_path / "seg.wav", "Puck")
    body = fake.calls[0]["json"]
    gen = body["generationConfig"]
    assert gen["responseModalities"] == ["AUDIO"]
    assert gen["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"] == "Puck"
    assert body["contents"][0]["parts"][0]["text"] == "Xin chào"
    assert fake.calls[0]["url"].endswith(
        "/models/gemini-3.1-flash-tts-preview:generateContent")
    assert "multiSpeakerVoiceConfig" not in gen["speechConfig"]


def test_style_prompt_prepended_when_set(monkeypatch, tmp_path):
    client, fake = _client(monkeypatch, [_Resp(200, _audio_payload(_pcm_bytes()))],
                           gemini_style_prompt="Speak warmly.")
    with client as c:
        c.synthesize("Xin chào", tmp_path / "seg.wav", "Kore")
    assert fake.calls[0]["json"]["contents"][0]["parts"][0]["text"] == "Speak warmly.\nXin chào"


def test_pcm_decoded_to_mono_24k_wav(monkeypatch, tmp_path):
    pcm = _pcm_bytes(2400)
    client, fake = _client(monkeypatch, [_Resp(200, _audio_payload(pcm))])
    out = tmp_path / "seg.wav"
    with client as c:
        c.synthesize("Xin chào", out, "Kore")
    audio, sr = sf.read(str(out), always_2d=False)
    assert sr == 24000
    assert audio.ndim == 1
    assert len(audio) == 2400   # 2 bytes/sample, mono


def test_missing_inline_data_retries_then_succeeds(monkeypatch, tmp_path):
    # KTD8: a text-token response raises retryable; with_retry resamples and the
    # second (audio) response writes the clip.
    client, fake = _client(
        monkeypatch,
        [_Resp(200, _text_payload()), _Resp(200, _audio_payload(_pcm_bytes()))],
    )
    out = tmp_path / "seg.wav"
    with client as c:
        c.synthesize("Xin chào", out, "Kore")
    assert len(fake.calls) == 2
    assert out.exists()


def test_voice_defaults_to_preset_default(monkeypatch, tmp_path):
    client, fake = _client(monkeypatch, [_Resp(200, _audio_payload(_pcm_bytes()))])
    with client as c:
        c.synthesize("Xin chào", tmp_path / "seg.wav")  # no voice arg
    voice = fake.calls[0]["json"]["generationConfig"]["speechConfig"][
        "voiceConfig"]["prebuiltVoiceConfig"]["voiceName"]
    assert voice == "Kore"   # gemini_default_voice


def test_401_raises_content_stage_error_no_retry(monkeypatch, tmp_path, caplog):
    payload = {"error": {"code": 401, "status": "UNAUTHENTICATED",
                         "message": "API key not valid"}}
    client, fake = _client(monkeypatch, [_Resp(401, payload)])
    with caplog.at_level("ERROR", logger="loro.gemini"):
        with pytest.raises(StageError) as exc_info:
            with client as c:
                c.synthesize("Xin chào", tmp_path / "seg.wav", "Kore")
    assert exc_info.value.signature == ("tts", "content", "UNAUTHENTICATED")
    assert "API key not valid" in exc_info.value.detail
    assert len(fake.calls) == 1   # 4xx is not retried
    assert "secret-key" not in caplog.text


def test_429_and_503_retry_then_succeed(monkeypatch, tmp_path):
    for status in (429, 503):
        client, fake = _client(
            monkeypatch,
            [_Resp(status, {"error": {"status": "X", "message": "transient"}}),
             _Resp(200, _audio_payload(_pcm_bytes()))],
        )
        with client as c:
            c.synthesize("Xin chào", tmp_path / f"seg_{status}.wav", "Kore")
        assert len(fake.calls) == 2
        assert (tmp_path / f"seg_{status}.wav").exists()


def test_api_key_in_header_only_never_logged(monkeypatch, tmp_path, caplog):
    client, fake = _client(monkeypatch, [_Resp(200, _audio_payload(_pcm_bytes()))])
    with caplog.at_level("DEBUG", logger="loro.gemini"):
        with client as c:
            c.synthesize("Xin chào", tmp_path / "seg.wav", "Kore")
    assert fake.calls[0]["headers"]["x-goog-api-key"] == "secret-key"
    assert "Authorization" not in fake.calls[0]["headers"]
    assert "secret-key" not in caplog.text


def test_error_body_bounded_and_redacted(monkeypatch, tmp_path, caplog):
    # S2: on an error the log carries error.status/message bounded to 500 chars
    # and never the key or any request header.
    long_msg = "x" * 2000
    payload = {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                         "message": long_msg}}
    client, fake = _client(monkeypatch, [_Resp(400, payload)])
    with caplog.at_level("ERROR", logger="loro.gemini"):
        with pytest.raises(StageError) as exc_info:
            with client as c:
                c.synthesize("Xin chào", tmp_path / "seg.wav", "Kore")
    assert exc_info.value.signature[:2] == ("tts", "content")
    assert exc_info.value.code == "INVALID_ARGUMENT"
    # The logged body line is capped at 500 chars of the response text.
    for rec in caplog.records:
        assert "secret-key" not in rec.getMessage()
        assert long_msg not in rec.getMessage()   # bounded, not the full 2000-char body


def test_request_timeout_raises_infra(monkeypatch, tmp_path):
    client, fake = _client(monkeypatch, [requests.Timeout("timed out")] * 3)
    with pytest.raises(StageError) as exc_info:
        with client as c:
            c.synthesize("Xin chào", tmp_path / "seg.wav", "Kore")
    assert exc_info.value.signature == ("tts", "infra", "timeout")


def test_voice_pool_default_members_are_known_voices():
    cfg = Config(tts_engine="gemini")
    assert set(cfg.gemini_voice_pool) <= gemini.GEMINI_VOICES
    assert cfg.gemini_default_voice in gemini.GEMINI_VOICES
    assert len(gemini.GEMINI_VOICES) == 30
