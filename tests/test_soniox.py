"""services/soniox client: one synchronous POST per clip, `requests` mocked
(no network). Pins the request payload shape, the header-only Bearer credential
(never logged), the WAV write, and the retry/error classification."""

import io

import numpy as np
import pytest
import requests
import soundfile as sf

from loro.config import Config
from loro.harness.retry import StageError
from loro.services import soniox


def _wav_bytes(sr=24000, seconds=0.1):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype="float32"), sr, format="WAV")
    return buf.getvalue()


class _Resp:
    def __init__(self, status_code=200, content=b"", payload=None, text=""):
        self.status_code = status_code
        self.content = content
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class _FakeHTTP:
    """Pops queued responses in order; an Exception in the queue is raised
    (to simulate a connection/timeout error)."""

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
    for name in ("TTS_ENGINE", "SONIOX_API_KEY", "SONIOX_BASE_URL", "SONIOX_MODEL",
                 "SONIOX_LANGUAGE", "SONIOX_SAMPLE_RATE", "SONIOX_AUDIO_FORMAT",
                 "SONIOX_TIMEOUT", "SONIOX_DEFAULT_VOICE", "SONIOX_VOICE_POOL",
                 "SONIOX_VOICE_MAP"):
        monkeypatch.delenv(name, raising=False)


def _cfg(**kw):
    # base_delay 0 keeps the retry backoff instant in tests.
    return Config(tts_engine="soniox", soniox_api_key="secret-key",
                  retry_base_delay=0.0, **kw)


def _client(monkeypatch, responses, **cfg_kw):
    fake = _FakeHTTP(responses)
    monkeypatch.setattr(soniox, "requests", fake)
    return soniox.SonioxClient(_cfg(**cfg_kw)), fake


def test_happy_path_writes_decodable_wav_with_payload(monkeypatch, tmp_path):
    client, fake = _client(monkeypatch, [_Resp(200, content=_wav_bytes())])
    out = tmp_path / "seg.wav"
    with client as c:
        c.synthesize("Xin chào", out, "Maya")

    # A real, decodable WAV landed at the output path.
    audio, sr = sf.read(str(out))
    assert sr == 24000 and len(audio) > 0

    payload = fake.calls[0]["json"]
    assert payload["voice"] == "Maya"
    assert payload["language"] == "vi"
    assert payload["model"] == "tts-rt-v1"
    assert payload["sample_rate"] == 24000
    assert payload["audio_format"] == "wav"
    assert payload["text"] == "Xin chào"
    assert fake.calls[0]["url"] == "https://tts-rt.soniox.com/tts"


def test_voice_defaults_to_configured_default(monkeypatch, tmp_path):
    client, fake = _client(monkeypatch, [_Resp(200, content=_wav_bytes())])
    with client as c:
        c.synthesize("Xin chào", tmp_path / "seg.wav")  # no voice arg
    assert fake.calls[0]["json"]["voice"] == "Adrian"  # soniox_default_voice


def test_authorization_header_is_bearer_and_key_never_logged(monkeypatch, tmp_path, caplog):
    client, fake = _client(monkeypatch, [_Resp(200, content=_wav_bytes())])
    with caplog.at_level("DEBUG", logger="loro.soniox"):
        with client as c:
            c.synthesize("Xin chào", tmp_path / "seg.wav", "Maya")
    assert fake.calls[0]["headers"]["Authorization"] == "Bearer secret-key"
    assert "secret-key" not in caplog.text


def test_401_raises_content_stage_error_with_error_type(monkeypatch, tmp_path, caplog):
    body = '{"error_type":"unauthorized","error_message":"Invalid API key"}'
    resp = _Resp(401, payload={"error_type": "unauthorized",
                               "error_message": "Invalid API key"}, text=body)
    client, fake = _client(monkeypatch, [resp])
    with caplog.at_level("ERROR", logger="loro.soniox"):
        with pytest.raises(StageError) as exc_info:
            with client as c:
                c.synthesize("Xin chào", tmp_path / "seg.wav", "Maya")
    assert exc_info.value.signature == ("tts", "content", "unauthorized")
    assert "Invalid API key" in exc_info.value.detail
    # 4xx is not retried — exactly one POST.
    assert len(fake.calls) == 1
    assert "secret-key" not in caplog.text


def test_transient_5xx_retries_then_succeeds(monkeypatch, tmp_path):
    client, fake = _client(
        monkeypatch,
        [_Resp(503, text="upstream error"), _Resp(200, content=_wav_bytes())],
    )
    with client as c:
        c.synthesize("Xin chào", tmp_path / "seg.wav", "Maya")
    assert len(fake.calls) == 2  # one 503 retry, then success
    assert (tmp_path / "seg.wav").exists()


def test_429_rate_limit_retries_then_succeeds(monkeypatch, tmp_path):
    # A 429 is transient for a metered per-segment API: it must back off and
    # retry (infra), not skip the clip as a content failure.
    client, fake = _client(
        monkeypatch,
        [_Resp(429, text="rate limited"), _Resp(200, content=_wav_bytes())],
    )
    with client as c:
        c.synthesize("Xin chào", tmp_path / "seg.wav", "Maya")
    assert len(fake.calls) == 2
    assert (tmp_path / "seg.wav").exists()


def test_persistent_5xx_non_json_raises_infra_signature(monkeypatch, tmp_path):
    # A 5xx with a non-JSON body exhausts retries and surfaces an infra
    # StageError keyed on the HTTP status (the except-ValueError detail path).
    client, fake = _client(monkeypatch, [_Resp(503, text="upstream error")] * 3)
    with pytest.raises(StageError) as exc_info:
        with client as c:
            c.synthesize("Xin chào", tmp_path / "seg.wav", "Maya")
    assert exc_info.value.signature == ("tts", "infra", "http_503")
    assert len(fake.calls) == 3


def test_request_timeout_raises_infra_stage_error(monkeypatch, tmp_path):
    client, fake = _client(
        monkeypatch,
        [requests.Timeout("timed out")] * 3,  # exhaust the 3 retry attempts
    )
    with pytest.raises(StageError) as exc_info:
        with client as c:
            c.synthesize("Xin chào", tmp_path / "seg.wav", "Maya")
    assert exc_info.value.signature == ("tts", "infra", "timeout")


def test_voice_pool_default_members_are_known_voices():
    # The shipped default pool + default voice must all be real Soniox voices,
    # or U5 preflight would reject our own defaults.
    cfg = Config(tts_engine="soniox")
    assert set(cfg.soniox_voice_pool) <= soniox.SONIOX_VOICES
    assert cfg.soniox_default_voice in soniox.SONIOX_VOICES
    assert len(soniox.SONIOX_VOICES) == 28
