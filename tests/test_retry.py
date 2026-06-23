import subprocess

import pytest
import requests

from loro.harness.retry import StageError, classify, with_retry


class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code


def _http_error(status):
    err = requests.HTTPError(f"http {status}")
    err.response = _FakeResponse(status)
    return err


class _FakeOpenAITimeout(Exception):
    """Mimics openai.APITimeoutError without importing the SDK."""


_FakeOpenAITimeout.__name__ = "APITimeoutError"


class TestClassify:
    def test_timeout_is_infra(self):
        assert classify(TimeoutError()) == ("infra", "timeout")
        assert classify(requests.Timeout()) == ("infra", "timeout")
        assert classify(subprocess.TimeoutExpired("cmd", 5)) == ("infra", "timeout")
        assert classify(_FakeOpenAITimeout()) == ("infra", "timeout")

    def test_connection_is_infra(self):
        assert classify(ConnectionError()) == ("infra", "connection")
        assert classify(requests.ConnectionError()) == ("infra", "connection")

    def test_5xx_is_infra_4xx_is_content(self):
        assert classify(_http_error(503)) == ("infra", "http_503")
        assert classify(_http_error(500)) == ("infra", "http_500")
        assert classify(_http_error(422)) == ("content", "http_422")
        assert classify(_http_error(404)) == ("content", "http_404")

    def test_parse_error_is_content(self):
        assert classify(ValueError("no JSON found"))[0] == "content"

    def test_unknown_is_content(self):
        assert classify(RuntimeError("weird"))[0] == "content"


class TestWithRetry:
    def test_retries_infra_then_succeeds(self):
        delays = []
        calls = []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("refused")
            return "ok"

        result = with_retry("tts", flaky, attempts=4, base_delay=1.0, sleep=delays.append)
        assert result == "ok"
        assert len(calls) == 3
        assert len(delays) == 2
        # Exponential backoff with jitter: delay 2 grows from delay 1's base
        assert 0.5 <= delays[0] <= 1.5
        assert 1.0 <= delays[1] <= 3.0

    def test_exhausted_raises_signature(self):
        def always_down():
            raise ConnectionError("refused")

        with pytest.raises(StageError) as exc_info:
            with_retry("translate", always_down, attempts=2, sleep=lambda _: None)
        err = exc_info.value
        assert err.signature == ("translate", "infra", "connection")

    def test_content_error_not_retried(self):
        calls = []

        def bad_request():
            calls.append(1)
            raise _http_error(422)

        with pytest.raises(StageError) as exc_info:
            with_retry("translate", bad_request, attempts=3, sleep=lambda _: None)
        assert len(calls) == 1
        assert exc_info.value.signature == ("translate", "content", "http_422")

    def test_503_is_retried(self):
        calls = []

        def degraded():
            calls.append(1)
            raise _http_error(503)

        with pytest.raises(StageError) as exc_info:
            with_retry("tts", degraded, attempts=3, sleep=lambda _: None)
        assert len(calls) == 3
        assert exc_info.value.signature == ("tts", "infra", "http_503")

    def test_timeout_signature(self):
        with pytest.raises(StageError) as exc_info:
            with_retry("asr", lambda: (_ for _ in ()).throw(TimeoutError()),
                       attempts=1, sleep=lambda _: None)
        assert exc_info.value.signature == ("asr", "infra", "timeout")

    def test_qa_class_retryable_when_requested(self):
        calls = []

        def flaky_quality():
            calls.append(1)
            if len(calls) < 2:
                raise StageError("tts", "qa", "too_short")
            return "ok"

        result = with_retry("tts", flaky_quality, attempts=3,
                            retry_classes=("infra", "qa"), sleep=lambda _: None)
        assert result == "ok"
        assert len(calls) == 2

    def test_preclassified_stage_error_passes_through(self):
        with pytest.raises(StageError) as exc_info:
            with_retry("tts", lambda: (_ for _ in ()).throw(StageError("tts", "qa", "silent")),
                       attempts=1, sleep=lambda _: None)
        assert exc_info.value.signature == ("tts", "qa", "silent")
