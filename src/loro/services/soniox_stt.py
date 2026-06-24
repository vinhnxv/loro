"""Client for Soniox async pre-recorded transcription (stt-async-v5).

A thin HTTP transport mirroring services/assemblyai.py, but for Soniox's
async lifecycle: upload the file (POST /v1/files), create a transcription
(POST /v1/transcriptions), poll to completion (GET /v1/transcriptions/{id}),
retrieve the token transcript (GET /v1/transcriptions/{id}/transcript), then
best-effort delete the server-side artifacts. It returns the raw transcript
dict (milliseconds, Soniox sub-word `tokens`); the token->word grouping and the
ms->s conversion live in the asr node (single responsibility) so this stays
pure transport.

Retry policy lives here, in one place (KTD2): the upload, create, AND the
transcript-retrieval calls route through `with_retry`, so a transient
5xx/connection error backs off like every other external call while a 4xx
(auth/validation) fails fast and clearly. The poll loop is NOT wrapped — it
owns a duration-scaled wall-clock budget, so a job stuck "processing" surfaces
as `poll_timeout` rather than re-creating the whole job; a transient failure on
the separate transcript fetch retries the fetch alone.

The API key is reused from the Soniox TTS account (KTD4, cfg.soniox_api_key):
it rides only in the `Authorization: Bearer` header and is never logged. On
error we log the HTTP status and the response body (Soniox bodies key on
error_type/error_message and don't echo the key), never the request headers.
"""

import logging
import time
from pathlib import Path

import requests
from requests.exceptions import RequestException

from loro.config import Config
from loro.harness.retry import CONTENT, INFRA, StageError, with_retry
from loro.utils import ffmpeg

log = logging.getLogger("loro.soniox_stt")

STAGE = "asr"

# Soniox caps recognition context at ~8000 tokens (docs/stt/concepts/context).
# We approximate a token at ~4 characters and reject an over-cap context
# client-side with a clear error, rather than letting Soniox reject the create
# with an opaque 4xx (KTD3/U2). ~8000 tokens * 4 chars/token.
CONTEXT_CHAR_CAP = 8000 * 4


def _headers(cfg: Config) -> dict:
    return {"Authorization": f"Bearer {cfg.soniox_api_key}"}


def _raise_for_status(resp: requests.Response, what: str) -> None:
    """Turn a non-2xx into a classified StageError. >=500 is infra (retryable);
    a 4xx (e.g. 401 bad key, 422 bad request) is content (non-retryable),
    matching the assemblyai template. Soniox keys error bodies on
    error_type/error_message — surface error_type as the code (more specific
    than http_NNN) and error_message as the detail when present; both are
    logged, the key/headers never are."""
    if resp.status_code < 400:
        return
    body = resp.text[:500]
    log.error("Soniox STT %s failed: HTTP %s — %s", what, resp.status_code, body)
    error_class = INFRA if resp.status_code >= 500 else CONTENT
    code = f"http_{resp.status_code}"
    detail = body
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        code = payload.get("error_type") or code
        detail = payload.get("error_message") or detail
    raise StageError(STAGE, error_class, code, detail)


def _build_context(cfg: Config) -> dict | None:
    """Assemble the Soniox recognition-context object from configured terms and
    background text, omitting empty sub-fields. Returns None when both are empty
    so the create payload omits `context` entirely (R5). Raises a content
    StageError naming the context when it exceeds the ~8000-token cap (KTD3)."""
    terms = [t for t in cfg.soniox_stt_context_terms if t]
    text = cfg.soniox_stt_context_text.strip()
    context: dict = {}
    if terms:
        context["terms"] = terms
    if text:
        context["text"] = text
    if not context:
        return None
    char_count = sum(len(t) for t in terms) + len(text)
    if char_count > CONTEXT_CHAR_CAP:
        raise StageError(
            STAGE, CONTENT, "context_too_large",
            f"recognition context ~{char_count // 4} tokens exceeds the ~8000-token "
            f"Soniox cap; trim SONIOX_STT_CONTEXT_TERMS / SONIOX_STT_CONTEXT_TEXT",
        )
    return context


def _upload(cfg: Config, audio_path: str | Path) -> str:
    data = Path(audio_path).read_bytes()

    def call() -> str:
        resp = requests.post(
            f"{cfg.soniox_stt_base_url}/v1/files",
            headers=_headers(cfg),
            files={"file": data},
            timeout=cfg.soniox_stt_request_timeout,
        )
        _raise_for_status(resp, "upload")
        return resp.json()["id"]

    return with_retry(STAGE, call, attempts=cfg.retry_attempts,
                      base_delay=cfg.retry_base_delay)


def _create(cfg: Config, file_id: str, context: dict | None,
            language_hints=None, enable_language_identification=None) -> str:
    # language_hints / enable_language_identification override the cfg defaults
    # when the caller resolved effective values (source_lang="auto" widens hints
    # and turns LID on, U7); None keeps the configured value (EN default
    # byte-identical, R20).
    payload: dict = {
        "model": cfg.soniox_stt_model,
        "file_id": file_id,
        "language_hints": (cfg.soniox_stt_language_hints if language_hints is None
                           else language_hints),
        "enable_language_identification": (
            cfg.soniox_stt_enable_language_identification
            if enable_language_identification is None else enable_language_identification),
        "enable_speaker_diarization": cfg.soniox_stt_speaker_diarization,
    }
    if context is not None:
        payload["context"] = context

    def call() -> str:
        resp = requests.post(
            f"{cfg.soniox_stt_base_url}/v1/transcriptions",
            headers=_headers(cfg),
            json=payload,
            timeout=cfg.soniox_stt_request_timeout,
        )
        _raise_for_status(resp, "create")
        return resp.json()["id"]

    return with_retry(STAGE, call, attempts=cfg.retry_attempts,
                      base_delay=cfg.retry_base_delay)


def _poll(cfg: Config, transcription_id: str, duration: float) -> None:
    """Block until the transcription is completed. The wall-clock ceiling scales
    with audio length (mirrors assemblyai); a job that never reaches
    completed/error within it is a poll_timeout, and the loop is not retried.

    A TRANSIENT status-GET failure must not discard the already-created (paid)
    job (B3/R5/KTD3): a 5xx (infra-class StageError from _raise_for_status) OR a
    raw connection/timeout error (the caffeinate/sleep socket-reset mode, which
    surfaces as a requests.RequestException, NOT a StageError) is logged and
    polling continues until the existing deadline. Only a 4xx/content error, an
    explicit status=="error", or the deadline ends the loop — the single
    wall-clock ceiling is preserved, no new retry wrapper."""
    budget = cfg.soniox_stt_poll_timeout_base + cfg.soniox_stt_poll_timeout_per_sec * duration
    deadline = time.monotonic() + budget
    url = f"{cfg.soniox_stt_base_url}/v1/transcriptions/{transcription_id}"
    while True:
        status = None
        try:
            resp = requests.get(url, headers=_headers(cfg),
                                timeout=cfg.soniox_stt_request_timeout)
            _raise_for_status(resp, "poll")
            payload = resp.json()
            status = payload.get("status")
        except StageError as exc:
            if exc.error_class != INFRA:
                raise  # 4xx / content (e.g. bad key) -> fail fast, no retry
            log.warning("Soniox STT poll transient error (%s) — continuing to poll "
                        "within the %.0fs budget", exc.code, budget)
        except RequestException as exc:
            log.warning("Soniox STT poll connection error (%s) — continuing to poll "
                        "within the %.0fs budget", type(exc).__name__, budget)
        else:
            if status == "completed":
                return
            if status == "error":
                raise StageError(STAGE, CONTENT, "soniox_error",
                                 payload.get("error_message", "transcription failed"))
        if time.monotonic() >= deadline:
            raise StageError(STAGE, INFRA, "poll_timeout",
                             f"status={status!r} after {budget:.0f}s budget")
        time.sleep(cfg.soniox_stt_poll_interval)


def _fetch_transcript(cfg: Config, transcription_id: str) -> dict:
    def call() -> dict:
        resp = requests.get(
            f"{cfg.soniox_stt_base_url}/v1/transcriptions/{transcription_id}/transcript",
            headers=_headers(cfg),
            timeout=cfg.soniox_stt_request_timeout,
        )
        _raise_for_status(resp, "transcript")
        return resp.json()

    return with_retry(STAGE, call, attempts=cfg.retry_attempts,
                      base_delay=cfg.retry_base_delay)


def _delete(cfg: Config, url: str, what: str) -> None:
    """Best-effort server-side delete. A non-2xx or a transport error is logged
    at warning level (so an operator notices audio/transcript left on Soniox)
    and is non-fatal — the local cache is the source of truth for reruns (KTD7).
    The key rides only in the header and is never logged."""
    try:
        resp = requests.delete(url, headers=_headers(cfg),
                               timeout=cfg.soniox_stt_request_timeout)
        if resp.status_code >= 400:
            log.warning("Soniox STT cleanup of %s left server-side (HTTP %s); "
                        "delete it manually if retention matters", what, resp.status_code)
    except Exception as exc:
        log.warning("Soniox STT cleanup of %s failed (%s); delete it manually if "
                    "retention matters", what, type(exc).__name__)


def transcribe(cfg: Config, audio_path: str | Path, language_hints=None,
               enable_language_identification=None) -> dict:
    """Run one Soniox async transcription: upload -> create -> poll -> retrieve,
    then best-effort delete the server-side artifacts. Returns the raw transcript
    dict (ms units, Soniox `tokens`). The context cap is enforced up front so an
    over-cap context fails before anything is uploaded. language_hints /
    enable_language_identification override the cfg defaults when the caller
    resolved effective values for source_lang="auto" (U7)."""
    context = _build_context(cfg)
    duration = ffmpeg.probe_duration(audio_path)
    file_id = _upload(cfg, audio_path)
    transcription_id = _create(cfg, file_id, context, language_hints,
                               enable_language_identification)
    log.info("Soniox STT transcription %s created; polling", transcription_id)
    _poll(cfg, transcription_id, duration)
    transcript = _fetch_transcript(cfg, transcription_id)
    if cfg.soniox_stt_cleanup:
        _delete(cfg, f"{cfg.soniox_stt_base_url}/v1/transcriptions/{transcription_id}",
                "transcription")
        _delete(cfg, f"{cfg.soniox_stt_base_url}/v1/files/{file_id}", "uploaded file")
    return transcript
