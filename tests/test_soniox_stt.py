"""services/soniox_stt client: upload -> create -> poll -> retrieve -> cleanup,
all with `requests` mocked (no network). Pins the async lifecycle, the create
payload shape (model/diarization/file_id/context), the ~8000-token context cap,
the retry/error/timeout signatures, the best-effort cleanup, and that the API
key (shared with TTS) is never logged."""

import logging

import pytest

from loro.config import Config
from loro.harness.retry import StageError
from loro.services import soniox_stt as stt


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("ASR_ENGINE", "SONIOX_API_KEY", "SONIOX_STT_BASE_URL",
                 "SONIOX_STT_MODEL", "SONIOX_STT_LANGUAGE_HINTS",
                 "SONIOX_STT_ENABLE_LANGUAGE_IDENTIFICATION",
                 "SONIOX_STT_SPEAKER_DIARIZATION", "SONIOX_STT_CONTEXT_TERMS",
                 "SONIOX_STT_CONTEXT_TEXT", "SONIOX_STT_CLEANUP",
                 "SONIOX_STT_POLL_INTERVAL"):
        monkeypatch.delenv(name, raising=False)


class _Resp:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload


class _FakeHTTP:
    def __init__(self):
        self.calls = []  # (method, url)
        self.upload_responses = [_Resp(200, {"id": "file-1"})]
        self.create_responses = [_Resp(200, {"id": "tr-1"})]
        self.poll_responses = []
        self.transcript_responses = []
        self.delete_responses = []  # default 200 when empty
        self.create_payloads = []
        self.upload_files = []

    def post(self, url, headers=None, data=None, json=None, files=None, timeout=None):
        self.calls.append(("post", url))
        if url.endswith("/v1/files"):
            self.upload_files.append(files)
            return self.upload_responses.pop(0)
        if url.endswith("/v1/transcriptions"):
            self.create_payloads.append(json)
            return self.create_responses.pop(0)
        raise AssertionError(f"unexpected POST {url}")

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("get", url))
        if url.endswith("/transcript"):
            return self.transcript_responses.pop(0)
        return self.poll_responses.pop(0)

    def delete(self, url, headers=None, timeout=None):
        self.calls.append(("delete", url))
        if self.delete_responses:
            resp = self.delete_responses.pop(0)
            if isinstance(resp, BaseException):
                raise resp
            return resp
        return _Resp(200, {})


@pytest.fixture
def http(monkeypatch, tmp_path):
    fake = _FakeHTTP()
    monkeypatch.setattr(stt, "requests", fake)
    monkeypatch.setattr(stt.ffmpeg, "probe_duration", lambda p: 5.0)
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"RIFFfake-wav-bytes")
    fake.audio = audio
    return fake


def _cfg(**kw):
    # base_delay 0 keeps the retry backoff instant in tests.
    return Config(asr_engine="soniox", soniox_api_key="secret-key",
                  retry_base_delay=0.0, soniox_stt_poll_interval=0.0, **kw)


def _tokens():
    return {"tokens": [{"text": "Hello", "start_ms": 100, "end_ms": 480, "speaker": "1"}]}


def test_happy_path_returns_tokens_and_create_payload(http):
    http.poll_responses = [_Resp(200, {"status": "processing"}),
                           _Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]

    result = stt.transcribe(_cfg(), http.audio)

    assert result == _tokens()
    payload = http.create_payloads[0]
    assert payload["model"] == "stt-async-v5"
    assert payload["enable_speaker_diarization"] is True
    assert payload["file_id"] == "file-1"
    assert payload["language_hints"] == ["en"]
    assert "context" not in payload  # empty by default (R5)
    # upload, create, two polls, one transcript fetch, two deletes (cleanup on)
    methods = [c[0] for c in http.calls]
    assert methods[:5] == ["post", "post", "get", "get", "get"]
    # the multipart upload carried the audio bytes under "file"
    assert "file" in http.upload_files[0]


def test_context_included_only_when_non_empty(http):
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]
    stt.transcribe(_cfg(soniox_stt_context_terms=["LangGraph"]), http.audio)
    assert http.create_payloads[0]["context"] == {"terms": ["LangGraph"]}


def test_context_omitted_when_terms_and_text_empty(http):
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]
    stt.transcribe(_cfg(), http.audio)
    assert "context" not in http.create_payloads[0]


def test_context_text_only_is_sent(http):
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]
    stt.transcribe(_cfg(soniox_stt_context_text="A talk about ML."), http.audio)
    assert http.create_payloads[0]["context"] == {"text": "A talk about ML."}


def test_context_over_cap_raises_content_before_any_call(http):
    huge = "x" * (stt.CONTEXT_CHAR_CAP + 1)
    with pytest.raises(StageError) as exc_info:
        stt.transcribe(_cfg(soniox_stt_context_text=huge), http.audio)
    assert exc_info.value.error_class == "content"
    assert "context" in exc_info.value.code or "context" in exc_info.value.detail.lower()
    # No network at all — the cap is enforced before upload/create.
    assert http.calls == []


def test_error_status_raises_non_retryable_with_message(http):
    http.poll_responses = [_Resp(200, {"status": "error",
                                       "error_message": "no audio in file"})]
    with pytest.raises(StageError) as exc_info:
        stt.transcribe(_cfg(), http.audio)
    assert exc_info.value.error_class == "content"
    assert "no audio in file" in exc_info.value.detail
    # the poll loop does not retry an error status (one poll GET only)
    assert [c[0] for c in http.calls].count("get") == 1


def test_auth_failure_on_upload_is_content_no_retry(http):
    http.upload_responses = [_Resp(401, payload={"error_type": "unauthorized",
                                                 "error_message": "Invalid API key"},
                                   text='{"error_type":"unauthorized"}')]
    with pytest.raises(StageError) as exc_info:
        stt.transcribe(_cfg(), http.audio)
    assert exc_info.value.error_class == "content"
    assert exc_info.value.code == "unauthorized"
    # 4xx is not retried — upload attempted exactly once, nothing after it.
    assert http.calls == [("post", "https://api.soniox.com/v1/files")]


def test_poll_timeout_raises_infra(http):
    http.poll_responses = [_Resp(200, {"status": "processing"})]
    cfg = _cfg(soniox_stt_poll_timeout_base=0.0, soniox_stt_poll_timeout_per_sec=0.0)
    with pytest.raises(StageError) as exc_info:
        stt.transcribe(cfg, http.audio)
    assert exc_info.value.signature == ("asr", "infra", "poll_timeout")


def test_transient_5xx_on_upload_retries_then_succeeds(http):
    http.upload_responses = [_Resp(503, text="upstream error"),
                             _Resp(200, {"id": "file-1"})]
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]
    result = stt.transcribe(_cfg(), http.audio)
    assert result == _tokens()
    # upload posted twice (one 503 retry), then create once
    assert [c[1] for c in http.calls if c[0] == "post"].count(
        "https://api.soniox.com/v1/files") == 2


def test_transient_5xx_on_transcript_fetch_retries_fetch_not_job(http):
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(503, text="upstream error"),
                                 _Resp(200, _tokens())]
    result = stt.transcribe(_cfg(), http.audio)
    assert result == _tokens()
    # exactly one upload + one create (the job is not re-created), two fetches
    posts = [c[1] for c in http.calls if c[0] == "post"]
    assert posts.count("https://api.soniox.com/v1/files") == 1
    assert posts.count("https://api.soniox.com/v1/transcriptions") == 1
    assert [c for c in http.calls if c[0] == "get"
            and c[1].endswith("/transcript")] == [
        ("get", "https://api.soniox.com/v1/transcriptions/tr-1/transcript")] * 2


def test_cleanup_deletes_file_and_transcription_when_on(http):
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]
    stt.transcribe(_cfg(soniox_stt_cleanup=True), http.audio)
    deleted = {c[1] for c in http.calls if c[0] == "delete"}
    assert "https://api.soniox.com/v1/transcriptions/tr-1" in deleted
    assert "https://api.soniox.com/v1/files/file-1" in deleted


def test_cleanup_off_fires_no_delete(http):
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]
    stt.transcribe(_cfg(soniox_stt_cleanup=False), http.audio)
    assert [c for c in http.calls if c[0] == "delete"] == []


def test_failed_delete_does_not_fail_transcribe(http, caplog):
    import requests as _requests
    http.poll_responses = [_Resp(200, {"status": "completed"})]
    http.transcript_responses = [_Resp(200, _tokens())]
    http.delete_responses = [_requests.ConnectionError("refused"), _Resp(200, {})]
    with caplog.at_level(logging.WARNING, logger="loro.soniox_stt"):
        result = stt.transcribe(_cfg(soniox_stt_cleanup=True), http.audio)
    assert result == _tokens()  # cleanup failure is non-fatal
    assert "secret-key" not in caplog.text


def test_api_key_never_logged_on_error(http, caplog):
    http.upload_responses = [_Resp(401, payload={"error_type": "unauthorized",
                                                 "error_message": "bad"},
                                   text='{"error_type":"unauthorized"}')]
    with caplog.at_level(logging.DEBUG, logger="loro.soniox_stt"):
        with pytest.raises(StageError):
            stt.transcribe(_cfg(), http.audio)
    assert "secret-key" not in caplog.text


def test_build_context_returns_none_when_empty():
    assert stt._build_context(_cfg()) is None


def test_build_context_omits_empty_subfields():
    ctx = stt._build_context(_cfg(soniox_stt_context_terms=["A", "B"]))
    assert ctx == {"terms": ["A", "B"]}
