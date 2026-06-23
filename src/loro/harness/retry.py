"""Unified retry/backoff policy and error signatures for all external calls.

Every failure is classified into a signature (stage, class, code) where class
is one of:
  infra   — connection refused, timeout, HTTP 5xx: the server's fault, retried
  content — HTTP 4xx, parse failures, unexpected errors: retrying verbatim is
            pointless at this layer (callers may have their own fallback)
  qa      — mechanical QA-gate failures, raised pre-classified by the caller

Signatures feed the skip ledger's abort window (R5a): many segments failing
with the same signature means systemic degradation, not flaky segments.
"""

import random
import subprocess
import time
from typing import Callable, TypeVar

import requests

T = TypeVar("T")

INFRA = "infra"
CONTENT = "content"
QA = "qa"


class StageError(Exception):
    def __init__(self, stage: str, error_class: str, code: str, detail: str = ""):
        self.stage = stage
        self.error_class = error_class
        self.code = code
        self.detail = detail
        super().__init__(f"[{stage}] {error_class}/{code}" + (f": {detail}" if detail else ""))

    @property
    def signature(self) -> tuple[str, str, str]:
        return (self.stage, self.error_class, self.code)


def _status_code(exc: BaseException) -> int | None:
    code = getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    return code if isinstance(code, int) else None


def classify(exc: BaseException) -> tuple[str, str]:
    """Map an arbitrary exception to (error_class, code)."""
    status = _status_code(exc)
    if status is not None:
        return (INFRA if status >= 500 else CONTENT, f"http_{status}")
    name = type(exc).__name__
    if isinstance(exc, (TimeoutError, subprocess.TimeoutExpired, requests.Timeout)) or "Timeout" in name:
        return (INFRA, "timeout")
    if isinstance(exc, (ConnectionError, requests.ConnectionError)) or "Connection" in name:
        return (INFRA, "connection")
    return (CONTENT, name)


def with_retry(
    stage: str,
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retry_classes: tuple[str, ...] = (INFRA,),
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `fn`, retrying retryable classes with exponential backoff + jitter.

    Raises StageError carrying the (stage, class, code) signature once
    exhausted or on a non-retryable class.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except StageError as err:
            if err.error_class not in retry_classes or attempt == attempts - 1:
                raise
        except Exception as exc:
            error_class, code = classify(exc)
            if error_class not in retry_classes or attempt == attempts - 1:
                raise StageError(stage, error_class, code, str(exc)[:500]) from exc
        delay = min(max_delay, base_delay * (2 ** attempt)) * (0.5 + random.random())
        sleep(delay)
    raise AssertionError("unreachable")
