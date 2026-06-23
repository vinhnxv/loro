"""services/assemblyai client: upload -> create -> poll, all with `requests`
mocked (no network). Pins the retry classification, the language-pin payload
shape, the error/timeout signatures, and that the API key is never logged."""

import pytest

from loro.config import Config
from loro.harness.retry import StageError
from loro.services import assemblyai as aai


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("ASR_ENGINE", "ASSEMBLYAI_API_KEY", "ASSEMBLYAI_BASE_URL",
                 "ASSEMBLYAI_SPEECH_MODELS", "ASSEMBLYAI_SPEAKER_LABELS",
                 "ASSEMBLYAI_LANGUAGE_DETECTION", "ASSEMBLYAI_LANGUAGE_CODE",
                 "ASSEMBLYAI_POLL_INTERVAL"):
        monkeypatch.delenv(name, raising=False)


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _FakeHTTP:
    def __init__(self):
        self.calls = []
        self.upload_responses = [_Resp(200, {"upload_url": "https://cdn/up.wav"})]
        self.create_responses = [_Resp(200, {"id": "tid-1"})]
        self.poll_responses = []
        self.create_payloads = []

    def post(self, url, headers=None, data=None, json=None, timeout=None):
        self.calls.append(("post", url))
        if url.endswith("/upload"):
            return self.upload_responses.pop(0)
        if url.endswith("/transcript"):
            self.create_payloads.append(json)
            return self.create_responses.pop(0)
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("get", url))
        return self.poll_responses.pop(0)


@pytest.fixture
def http(monkeypatch, tmp_path):
    fake = _FakeHTTP()
    monkeypatch.setattr(aai, "requests", fake)
    monkeypatch.setattr(aai.ffmpeg, "probe_duration", lambda p: 5.0)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFfake-wav-bytes")
    fake.audio = audio
    return fake


def _cfg(**kw):
    # base_delay 0 keeps the retry backoff instant in tests.
    return Config(assemblyai_api_key="secret-key", retry_base_delay=0.0,
                  assemblyai_poll_interval=0.0, **kw)


def test_happy_path_returns_completed_dict(http):
    completed = {"status": "completed", "text": "hello world",
                 "words": [{"start": 100, "end": 500, "text": "hello", "speaker": "A"}],
                 "utterances": [{"start": 100, "end": 500, "text": "hello world", "speaker": "A"}]}
    http.poll_responses = [_Resp(200, {"status": "queued"}), _Resp(200, completed)]

    result = aai.transcribe(_cfg(), http.audio)

    assert result == completed
    payload = http.create_payloads[0]
    assert payload["speech_models"] == ["universal-3-pro", "universal-2"]
    assert payload["speaker_labels"] is True
    assert payload["language_detection"] is True
    assert "language_code" not in payload
    # one upload, one create, two polls
    assert [c[0] for c in http.calls] == ["post", "post", "get", "get"]


def test_language_pin_sets_code_and_omits_detection(http):
    http.poll_responses = [_Resp(200, {"status": "completed", "words": []})]
    aai.transcribe(_cfg(assemblyai_language_code="en"), http.audio)
    payload = http.create_payloads[0]
    assert payload["language_code"] == "en"
    assert "language_detection" not in payload


def test_error_status_raises_non_retryable_stage_error(http):
    http.poll_responses = [_Resp(200, {"status": "error", "error": "no audio in file"})]
    with pytest.raises(StageError) as exc_info:
        aai.transcribe(_cfg(), http.audio)
    assert exc_info.value.signature == ("asr", "content", "assemblyai_error")
    assert "no audio in file" in exc_info.value.detail
    # the poll loop does not retry an error status (one GET only)
    assert [c[0] for c in http.calls].count("get") == 1


def test_auth_failure_on_create_is_content_class_no_retry(http):
    http.create_responses = [_Resp(401, text='{"error":"Authentication failed"}')]
    with pytest.raises(StageError) as exc_info:
        aai.transcribe(_cfg(), http.audio)
    assert exc_info.value.signature == ("asr", "content", "http_401")
    # create attempted exactly once (4xx is not retried)
    assert [c[0] for c in http.calls].count("post") == 2  # 1 upload + 1 create


def test_poll_timeout_raises_infra_stage_error(http):
    # zero budget: the first non-terminal poll exceeds the deadline immediately.
    http.poll_responses = [_Resp(200, {"status": "processing"})]
    cfg = _cfg(assemblyai_poll_timeout_base=0.0, assemblyai_poll_timeout_per_sec=0.0)
    with pytest.raises(StageError) as exc_info:
        aai.transcribe(cfg, http.audio)
    assert exc_info.value.signature == ("asr", "infra", "poll_timeout")


def test_transient_5xx_on_upload_retries_then_succeeds(http):
    http.upload_responses = [_Resp(503, text="upstream error"),
                             _Resp(200, {"upload_url": "https://cdn/up.wav"})]
    http.poll_responses = [_Resp(200, {"status": "completed", "words": []})]
    result = aai.transcribe(_cfg(), http.audio)
    assert result["status"] == "completed"
    # upload posted twice (one 503 retry), then create once
    assert [c[0] for c in http.calls].count("post") == 3


def test_api_key_never_logged_on_error(http, caplog):
    http.create_responses = [_Resp(401, text='{"error":"Authentication failed"}')]
    with caplog.at_level("ERROR", logger="loro.assemblyai"):
        with pytest.raises(StageError):
            aai.transcribe(_cfg(), http.audio)
    assert "secret-key" not in caplog.text
