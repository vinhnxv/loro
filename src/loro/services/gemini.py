"""Client for the Gemini cloud TTS API (generateContent).

A thin HTTP transport mirroring services/soniox.py: a preset-voice engine with
no reference clip and no temporary server, so the voice is passed per call and
__enter__/__exit__ are trivial (KTD1/KTD3). One synchronous POST returns one
continuous audio blob.

Two Gemini-specific differences from the Soniox path live here. First, Gemini
returns raw PCM (s16le / 24 kHz / mono, base64 in inlineData.data), not a WAV
container, so the client wraps it with the stdlib `wave` module (KTD4) — exactly
the user's wave_file() shape — and downstream soundfile.read works unchanged.
Second, the batched path (synthesize_batch, U4) carries a multi-speaker prompt
and splits the returned audio at inter-turn silences; it reuses the single-call
machinery here.

Retry policy lives here, in one place: the call routes through `with_retry`, so
a transient 5xx/429/connection backs off while a 4xx fails fast. Google
documents that the model occasionally returns text tokens instead of audio
(KTD8); that surfaces as a retryable INFRA error so with_retry resamples rather
than crashing or writing an empty clip.

The API key is a credential (R10): it rides only in the `x-goog-api-key` header
and is never logged. On error we log only the bounded JSON body (resp.text[:500])
— never the request headers or the key. The 500-char bound also caps how much
submitted dub text a Gemini content-policy error can echo back into the log.
"""

import base64
import logging
import os
import tempfile
import uuid
import wave
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

from loro.config import Config
from loro.harness.retry import CONTENT, INFRA, StageError, with_retry
from loro.utils import ffmpeg
from loro.utils.audio import trim_silence_edges

log = logging.getLogger("loro.gemini")

STAGE = "tts"

# Prepended to a batched prompt so the model leaves a detectable pause between
# EVERY line — not only at speaker changes. Distinct-speaker turns get a pause
# for free at the turn boundary, but consecutive same-speaker segments (the
# monologue case) have none, so the splitter would find too few gaps and fall
# back. Asking for a pause per line is what makes same-speaker batching even
# attempt to split (KTD9); whether the model honors it is an open question the
# fallback makes safe.
_PAUSE_DIRECTIVE = (
    "Read each of the following lines as a separate turn. Leave a clear, "
    "audible pause between every line so the turns can be told apart."
)

# A silence within this many seconds of the clip's start/end is lead-in/out
# padding, not an inter-turn boundary — excluded from split cut candidates.
_EDGE_SILENCE_S = 0.02

# The 30 documented Gemini prebuilt voices. The client owns its domain constants
# (mirroring soniox.SONIOX_VOICES); preflight imports this set to validate voice
# names before a run so a typo fails fast (U6). Source:
# https://ai.google.dev/gemini-api/docs/speech-generation
GEMINI_VOICES = frozenset({
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus", "Aoede",
    "Callirrhoe", "Autonoe", "Enceladus", "Iapetus", "Umbriel", "Algieba",
    "Despina", "Erinome", "Algenib", "Rasalgethi", "Laomedeia", "Achernar",
    "Alnilam", "Schedar", "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
})


class SplitError(StageError):
    """A batch's audio could not be split into the expected per-segment count —
    fewer than len(batch)-1 qualifying inter-turn silences were detected (KTD9).
    A distinct, catchable type so the tts node falls back to per-segment
    synthesis; for same-speaker batches this is an expected path, not an error.
    Carries a content-class StageError signature for uniformity, but the node
    catches it before it ever reaches the ledger."""

    def __init__(self, detail: str = ""):
        super().__init__(STAGE, CONTENT, "split_failed", detail)


def _raise_for_status(resp: requests.Response) -> None:
    """Turn a non-2xx into a classified StageError. >=500 and 429 are infra
    (retryable — a server fault or a transient rate-limit that backoff clears);
    a 4xx (e.g. 401 bad key, 400 bad request) is content (non-retryable). 429 is
    treated as infra so a rate-limit burst against Gemini's tight RPM budget
    backs off and retries rather than permanently skipping the clip. Gemini keys
    its error body on error.status/error.message — surface error.status as the
    code (stable for the abort window) and error.message as the detail; both are
    logged bounded to 500 chars, the key/headers never are."""
    if resp.status_code < 400:
        return
    body = resp.text[:500]
    log.error("Gemini TTS failed: HTTP %s — %s", resp.status_code, body)
    error_class = INFRA if (resp.status_code >= 500 or resp.status_code == 429) else CONTENT
    code = f"http_{resp.status_code}"
    detail = body
    try:
        payload = resp.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict):
            code = err.get("status") or code
            detail = err.get("message") or detail
    raise StageError(STAGE, error_class, code, detail)


def _extract_pcm(resp_json: dict) -> bytes:
    """Pull the base64 PCM out of the first inlineData audio part and decode it.
    Google documents that the model sometimes returns text tokens instead of
    audio (KTD8); a response with no inlineData part raises a retryable INFRA
    error so with_retry resamples rather than writing an empty clip."""
    for cand in resp_json.get("candidates") or []:
        parts = (cand.get("content") or {}).get("parts") or []
        for part in parts:
            inline = part.get("inlineData") or part.get("inline_data") or {}
            data = inline.get("data")
            if data:
                return base64.b64decode(data)
    raise StageError(STAGE, INFRA, "no_audio",
                     "Gemini response carried no inlineData audio part")


def _pcm_to_wav(pcm: bytes, out: Path, sample_rate: int) -> None:
    """Wrap raw s16le mono PCM in a WAV via the stdlib `wave` module (KTD4),
    written to a tmp file then atomically renamed (like soniox.py)."""
    out = Path(out)
    tmp = out.with_name(f".tmp.{out.name}.{uuid.uuid4().hex}")
    try:
        with wave.open(str(tmp), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)        # s16le
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        os.replace(tmp, out)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _split_on_silence(wav_path: Path, n: int, min_gap_ms: float,
                      threshold_db: float) -> list[np.ndarray]:
    """Split the batch WAV into exactly `n` per-turn arrays at the deepest
    inter-turn silences (KTD2). Runs the codebase's windowed silencedetect (the
    same primitive crosscheck uses), keeping only gaps >= min_gap_ms, then cuts
    inside the n-1 LONGEST qualifying gaps' midpoints — the way split_bounds
    cuts at silence midpoints. Raises SplitError when fewer than n-1 gaps
    qualify (-> the node falls back to per-segment). Each piece is trimmed to its
    speech span so its duration drives `fit` the same as a per-segment clip (A1).

    The trim is only per-piece edge cleanup AFTER the cut (trim_silence_edges);
    cut points come from silencedetect's duration-windowed runs, never from
    instantaneous amplitude (speech crosses zero constantly)."""
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr if sr else 0.0
    # detect_silences' min_duration filter does the >= min_gap_ms qualification:
    # a within-turn dip shorter than the floor is never returned as a boundary.
    silences = ffmpeg.detect_silences(wav_path, threshold_db, min_gap_ms / 1000.0)
    # Only INTERIOR silences (speech on both sides) are turn boundaries. A
    # leading or trailing silence — common as TTS lead-in/out padding — is not a
    # cut point: selecting it (it can be longer than a real inter-turn gap, so
    # the deepest-gap pick would grab it) merges two turns into one piece and
    # leaves an empty edge piece, which QA then rejects, forcing a wasteful full
    # fallback. Edge silence is dropped per-piece by trim_silence_edges instead.
    interior = [(s, e) for s, e in silences
                if s > _EDGE_SILENCE_S and e < duration - _EDGE_SILENCE_S]
    if len(interior) < n - 1:
        raise SplitError(
            f"{len(interior)} interior gap(s) for {n} turns "
            f"(need {n - 1}); falling back to per-segment")
    # The n-1 longest (deepest) gaps, cut at each gap's midpoint, in time order.
    deepest = sorted(interior, key=lambda se: se[1] - se[0], reverse=True)[:n - 1]
    cut_samples = sorted(int(round((s + e) / 2 * sr)) for s, e in deepest)
    bounds = [0, *cut_samples, len(audio)]
    return [trim_silence_edges(audio[bounds[i]:bounds[i + 1]], threshold_db)
            for i in range(n)]


class GeminiClient:
    """Synthesizes Vietnamese speech in a named prebuilt voice. Use as a context
    manager so the tts node stays engine-agnostic; the preset surface matches
    SonioxClient (per-call voice, trivial enter/exit, no reference)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _url(self) -> str:
        return f"{self.cfg.gemini_base_url}/models/{self.cfg.gemini_model}:generateContent"

    def _compose_prompt(self, text: str) -> str:
        """Prepend the configured style/audio-tag directive to the text (R7);
        an empty style_prompt leaves the text unchanged."""
        return (self.cfg.gemini_style_prompt + "\n" + text).strip()

    def _generate(self, body: dict) -> bytes:
        """One generateContent call -> decoded PCM bytes. Raises a classified
        StageError on HTTP error (infra/content) or on the text-instead-of-audio
        quirk (retryable infra, KTD8). The caller wraps this in with_retry."""
        resp = requests.post(
            self._url(),
            headers={"x-goog-api-key": self.cfg.gemini_api_key,
                     "Content-Type": "application/json"},
            json=body,
            timeout=self.cfg.gemini_timeout,
        )
        _raise_for_status(resp)
        return _extract_pcm(resp.json())

    def synthesize(self, text: str, output: Path, voice: str | None = None) -> None:
        """Single-speaker synthesis (R3): one call -> PCM -> WAV. This is the
        per-segment / GEMINI_BATCH_SEGMENTS=1 path and the batched fallback."""
        voice = voice or self.cfg.preset_voices.default
        body = {
            "contents": [{"parts": [{"text": self._compose_prompt(text)}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}
                },
            },
        }

        def call() -> None:
            pcm = self._generate(body)
            _pcm_to_wav(pcm, output, self.cfg.gemini_sample_rate)

        with_retry(STAGE, call, attempts=self.cfg.retry_attempts,
                   base_delay=self.cfg.retry_base_delay)

    def synthesize_batch(self, turns: list[tuple[str, str, str]]
                         ) -> tuple[list[np.ndarray], int]:
        """Batched multi-speaker synthesis (R4): build one prompt of labelled
        turns, make ONE call, and split the returned audio into per-turn arrays.

        `turns` is an ordered list of (speaker_label, text, voice). The call is
        retried for infra/text-token failures like synthesize(); the split runs
        AFTER (and outside) the retry, because re-calling Gemini would not change
        a deterministic split and would waste the batch budget the engine exists
        to save — a SplitError instead signals the node to fall back per-segment
        (KTD2). Returns (per-turn arrays in order, sample_rate); raises SplitError
        when the audio can't be cut into len(turns) pieces."""
        distinct = self._distinct_speakers(turns)
        body = self._batch_body(turns, distinct)

        def call() -> bytes:
            return self._generate(body)

        pcm = with_retry(STAGE, call, attempts=self.cfg.retry_attempts,
                         base_delay=self.cfg.retry_base_delay)

        sr = self.cfg.gemini_sample_rate
        fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        tmp_wav = Path(tmp_wav)
        try:
            _pcm_to_wav(pcm, tmp_wav, sr)
            pieces = _split_on_silence(tmp_wav, len(turns),
                                       self.cfg.gemini_split_min_gap_ms,
                                       self.cfg.silence_threshold_db)
        finally:
            tmp_wav.unlink(missing_ok=True)
        return pieces, sr

    @staticmethod
    def _distinct_speakers(turns: list[tuple[str, str, str]]) -> list[tuple[str, str]]:
        """Ordered distinct (label, voice) pairs in the batch. A batch is
        pre-bounded to <= 2 distinct diarized speakers in U5, so this is <= 2."""
        distinct: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for label, _text, voice in turns:
            key = (label, voice)
            if key not in seen:
                seen.add(key)
                distinct.append(key)
        return distinct

    def _batch_body(self, turns: list[tuple[str, str, str]],
                    distinct: list[tuple[str, str]]) -> dict:
        """One generateContent body for the batch. >= 2 distinct speakers send a
        multiSpeakerVoiceConfig (<= 2 voices) and label each transcript line with
        its speaker so Gemini voices each turn; a single distinct speaker sends a
        plain voiceConfig and unlabelled lines (R6). Either way the pause
        directive asks for an inter-line gap so the splitter has boundaries."""
        multi = len(distinct) >= 2
        if multi:
            lines = [f"{label}: {text}" for label, text, _voice in turns]
            speech_config = {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [
                {"speaker": label,
                 "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
                for label, voice in distinct
            ]}}
        else:
            lines = [text for _label, text, _voice in turns]
            voice = distinct[0][1] if distinct else self.cfg.preset_voices.default
            speech_config = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
        prompt = self._compose_prompt(_PAUSE_DIRECTIVE + "\n\n" + "\n".join(lines))
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": speech_config,
            },
        }
