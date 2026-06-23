"""Client for AssemblyAI pre-recorded transcription (universal-3-pro).

A thin HTTP transport mirroring services/llm.py and services/higgs.py: it
uploads the audio, creates a transcript, polls to completion, and returns the
raw API dict (milliseconds, AssemblyAI field names). The ms->s conversion and
field mapping live in the asr node (single responsibility) so this stays pure
transport.

Retry policy lives here, in one place (R2): the upload and create calls route
through `with_retry`, so a transient 5xx/connection error backs off like every
other external call while a 4xx (auth/validation) fails fast and clearly. The
poll loop is not wrapped — it owns a duration-scaled wall-clock budget instead,
so a transcript stuck "processing" eventually surfaces as `poll_timeout` rather
than retrying the whole upload.

The API key is a credential (R8): it is sent only in the `authorization`
header and is never logged. On error we log the HTTP status and the response
body (AssemblyAI's bodies don't echo the key), never the request headers.
"""

import logging
import time
from pathlib import Path

import requests

from loro.config import Config
from loro.harness.retry import CONTENT, INFRA, StageError, with_retry
from loro.utils import ffmpeg

log = logging.getLogger("loro.assemblyai")

STAGE = "asr"


def _headers(cfg: Config) -> dict:
    # AssemblyAI uses the raw key in `authorization` (no "Bearer" prefix).
    return {"authorization": cfg.assemblyai_api_key}


def _raise_for_status(resp: requests.Response, what: str) -> None:
    """Turn a non-2xx into a classified StageError. >=500 is infra (retryable);
    a 4xx (e.g. 401 bad key, 422 bad request) is content (non-retryable). The
    body is surfaced as the detail and logged; the key/headers never are."""
    if resp.status_code < 400:
        return
    body = resp.text[:500]
    log.error("AssemblyAI %s failed: HTTP %s — %s", what, resp.status_code, body)
    error_class = INFRA if resp.status_code >= 500 else CONTENT
    raise StageError(STAGE, error_class, f"http_{resp.status_code}", body)


def _upload(cfg: Config, audio_path: str | Path) -> str:
    data = Path(audio_path).read_bytes()

    def call() -> str:
        resp = requests.post(
            f"{cfg.assemblyai_base_url}/upload",
            headers=_headers(cfg),
            data=data,
            timeout=cfg.assemblyai_request_timeout,
        )
        _raise_for_status(resp, "upload")
        return resp.json()["upload_url"]

    return with_retry(STAGE, call, attempts=cfg.retry_attempts,
                      base_delay=cfg.retry_base_delay)


def _create(cfg: Config, audio_url: str) -> str:
    payload: dict = {
        "audio_url": audio_url,
        "speech_models": cfg.assemblyai_speech_models,
        "speaker_labels": cfg.assemblyai_speaker_labels,
    }
    # A pinned language_code fixes the language and replaces detection (KTD/U1).
    if cfg.assemblyai_language_code:
        payload["language_code"] = cfg.assemblyai_language_code
    else:
        payload["language_detection"] = cfg.assemblyai_language_detection

    def call() -> str:
        resp = requests.post(
            f"{cfg.assemblyai_base_url}/transcript",
            headers=_headers(cfg),
            json=payload,
            timeout=cfg.assemblyai_request_timeout,
        )
        _raise_for_status(resp, "create")
        return resp.json()["id"]

    return with_retry(STAGE, call, attempts=cfg.retry_attempts,
                      base_delay=cfg.retry_base_delay)


def _poll(cfg: Config, transcript_id: str, duration: float) -> dict:
    # Wall-clock ceiling scales with audio length (mirrors asr_timeout_*); a
    # transcript that never reaches completed/error within it is a poll_timeout.
    budget = cfg.assemblyai_poll_timeout_base + cfg.assemblyai_poll_timeout_per_sec * duration
    deadline = time.monotonic() + budget
    url = f"{cfg.assemblyai_base_url}/transcript/{transcript_id}"
    while True:
        resp = requests.get(url, headers=_headers(cfg),
                            timeout=cfg.assemblyai_request_timeout)
        _raise_for_status(resp, "poll")
        payload = resp.json()
        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "error":
            raise StageError(STAGE, CONTENT, "assemblyai_error",
                             payload.get("error", "transcription failed"))
        if time.monotonic() >= deadline:
            raise StageError(STAGE, INFRA, "poll_timeout",
                             f"status={status!r} after {budget:.0f}s budget")
        time.sleep(cfg.assemblyai_poll_interval)


def transcribe(cfg: Config, audio_path: str | Path) -> dict:
    """Upload, create, and poll one AssemblyAI transcript; return the raw
    completed dict (ms units, AssemblyAI field names)."""
    duration = ffmpeg.probe_duration(audio_path)
    upload_url = _upload(cfg, audio_path)
    transcript_id = _create(cfg, upload_url)
    log.info("AssemblyAI transcript %s created; polling", transcript_id)
    return _poll(cfg, transcript_id, duration)
