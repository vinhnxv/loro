"""Client for the Soniox cloud TTS server (tts-rt-v1).

A thin HTTP transport mirroring services/higgs.py and services/assemblyai.py:
one synchronous POST per clip returns raw WAV bytes (no polling). Unlike the
cloning clients there is no reference clip and no temporary HTTP server —
Soniox selects one of 28 preset voices by name, so the voice is passed per
call and __enter__/__exit__ are trivial (KTD1/KTD3).

Retry policy lives here, in one place (R7): the call routes through
`with_retry`, so a transient 5xx/connection/timeout backs off like every other
external call while a 4xx (auth/validation) fails fast and clearly.

The API key is a credential (R11): it is sent only in the `Authorization`
header and is never logged. On error we log the HTTP status and the JSON error
body (Soniox bodies key on error_type/error_message and don't echo the key),
never the request headers.
"""

import logging
import os
import uuid
from pathlib import Path

import requests

from loro.config import Config
from loro.harness.retry import CONTENT, INFRA, StageError, with_retry

log = logging.getLogger("loro.soniox")

STAGE = "tts"

# The 28 documented Soniox preset voices (14 female, 14 male). The client owns
# its domain constants (mirroring assemblyai.py); preflight imports this set to
# validate voice names before a run so a typo fails fast (U5). Source:
# https://soniox.com/docs/tts/concepts/voices
SONIOX_VOICES = frozenset({
    # female
    "Maya", "Nina", "Emma", "Claire", "Grace", "Mina", "Lucia", "Sofia",
    "Isla", "Victoria", "Ruby", "Elise", "Priya", "Meera",
    # male
    "Daniel", "Noah", "Jack", "Adrian", "Owen", "Kenji", "Rafael", "Mateo",
    "Oliver", "Arthur", "Cooper", "Mason", "Arjun", "Rohan",
})


def _raise_for_status(resp: requests.Response) -> None:
    """Turn a non-2xx into a classified StageError. >=500 and 429 are infra
    (retryable — a server fault or a transient rate-limit that backoff clears);
    a 4xx (e.g. 401 bad key, 422 bad request) is content (non-retryable). 429 is
    treated as infra (unlike the strict 5xx-only rule) because Soniox bills and
    synthesizes one request per segment, so a rate-limit burst should back off
    and retry rather than permanently skip the clip. Soniox keys pre-stream
    errors on error_type/error_message — surface error_type as the code (more
    specific than http_NNN, and stable for the abort window) and error_message
    as the detail; both are logged, the key/headers never are."""
    if resp.status_code < 400:
        return
    body = resp.text[:500]
    log.error("Soniox TTS failed: HTTP %s — %s", resp.status_code, body)
    error_class = INFRA if (resp.status_code >= 500 or resp.status_code == 429) else CONTENT
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


class SonioxClient:
    """Synthesizes Vietnamese speech in a named preset voice. Use as a context
    manager so the tts node stays engine-agnostic; the cloning clients take the
    same surface but ignore the per-call voice (KTD1)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synthesize(self, text: str, output: Path, voice: str | None = None) -> None:
        voice = voice or self.cfg.soniox_default_voice
        output = Path(output)

        def call() -> None:
            resp = requests.post(
                f"{self.cfg.soniox_base_url}/tts",
                headers={
                    "Authorization": f"Bearer {self.cfg.soniox_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.cfg.soniox_model,
                    # Profile-derived spoken language (U9): matches the clip
                    # fingerprint's `language` so the synthesized audio is in the
                    # target language, not the engine's vi default. Generic
                    # fallback resolves to the target tag, never empty (#3).
                    "language": self.cfg.effective_tts_language,
                    "voice": voice,
                    "text": text,
                    "audio_format": self.cfg.soniox_audio_format,
                    "sample_rate": self.cfg.soniox_sample_rate,
                },
                timeout=self.cfg.soniox_timeout,
            )
            _raise_for_status(resp)
            tmp = output.with_name(f".tmp.{output.name}.{uuid.uuid4().hex}")
            try:
                tmp.write_bytes(resp.content)
                os.replace(tmp, output)
            finally:
                Path(tmp).unlink(missing_ok=True)

        with_retry(STAGE, call, attempts=self.cfg.retry_attempts,
                   base_delay=self.cfg.retry_base_delay)
