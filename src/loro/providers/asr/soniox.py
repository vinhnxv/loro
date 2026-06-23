"""Soniox cloud ASR provider (stt-async-v5).

Wraps the existing `services.soniox_stt` module function (unchanged, KTD2): one
async job, group the sub-word tokens into the word-level stream the pipeline
consumes, derive raw segments from a punctuation split (Soniox returns no
utterances), and cache the raw token transcript. The asr node owns the shared
tail (EN SRT + return); this provider owns the engine-specific mapping and its
own no-speech guard so the trigger/message stay byte-identical (KTD3).
"""

import logging
from collections import Counter

import requests

from loro.config import Config
from loro.harness import artifacts
from loro.providers.base import PROBE_TIMEOUT, AsrResult
from loro.services import soniox_stt
from loro.state import Segment
from loro.utils import sentences

log = logging.getLogger("loro.asr")

# Language hints sent when source_lang="auto" widens detection beyond the
# configured (EN) default (KTD5). A bias list, not a whitelist — the tier-1
# source/target languages plus the most common dub source. Calibrate in U11.
_AUTO_LANGUAGE_HINTS = ["en", "fr", "es", "vi"]


def _detect_source_language(raw: dict, fallback: str) -> tuple[str, bool]:
    """Resolve the detected source language from a LID-enabled Soniox response.

    UNVERIFIED RESPONSE SHAPE (KTD5): the plan calls for one cheap short-clip
    probe with enable_language_identification=true to confirm whether the detected
    language is carried per-token (a `language` field) or per-response, BEFORE
    relying on auto-detection in production. This reads the documented per-token
    `language` field defensively: it takes the majority, flags a mixed result, and
    falls back (loudly) to `fallback` when the field is absent — so a wrong
    assumption degrades to the configured source rather than crashing. Returns
    (resolved_language, mixed_or_lowconfidence)."""
    tokens = raw.get("tokens") or []
    langs = [t.get("language") for t in tokens if t.get("language")]
    if not langs:
        log.warning(
            "Soniox language identification was requested but the response carried "
            "no per-token `language` field (UNVERIFIED shape, KTD5) — falling back to "
            "%r; run the LID probe and fix _detect_source_language before trusting "
            "--source-lang auto", fallback)
        return fallback, True
    counts = Counter(langs)
    top, n = counts.most_common(1)[0]
    mixed = len(counts) > 1 and n / len(langs) < 0.9
    if mixed:
        log.warning("Soniox detected mixed languages %s — using majority %r; the "
                    "translation may be mis-targeted", dict(counts), top)
    return top, mixed


# The probe transcription id must be a syntactically valid UUID: stt-async-v5
# validates the path `transcription_id` as a UUID *before* evaluating auth or
# existence, so a non-UUID id (e.g. "preflight-nonexistent") returns 400
# invalid_request (uuid_parsing) regardless of the key — which the probe below
# would misread as a transient failure and fail an otherwise-good key. An
# all-zeros UUID is well-formed yet can never be a real transcription id, so a
# good key gets a clean 404 (transcription_not_found).
_PROBE_TRANSCRIPTION_ID = "00000000-0000-0000-0000-000000000000"


def _probe_soniox_stt(cfg: Config) -> int:
    """Liveness + auth probe for the Soniox ASR cloud engine. A GET on a
    nonexistent (but UUID-shaped) transcription id returns 404 for a good key,
    401 for a bad key, and 403 for an authenticated-but-unauthorized key — so the
    key authenticates without spending a transcription. A 429/5xx is
    transient/unknown and is NOT proof of a good key (kept consistent with the STT
    client's 4xx-vs-5xx classification). Returns the HTTP status; raises on a
    connection/timeout (endpoint down). The key rides only in the header and is
    never logged."""
    resp = requests.get(
        f"{cfg.soniox_stt_base_url}/v1/transcriptions/{_PROBE_TRANSCRIPTION_ID}",
        headers={"Authorization": f"Bearer {cfg.soniox_api_key}"},
        timeout=PROBE_TIMEOUT,
    )
    return resp.status_code


# Soniox token-grouping collapse guard (KTD3). The dangerous failure mode of a
# wrong whitespace convention is silent: every token folds into one word, which
# is non-empty and so slips past the no-speech guard. Once the stream is long
# enough to judge (>= _GROUP_GUARD_MIN_TOKENS), an average above
# _GROUP_GUARD_MAX_TOKENS_PER_WORD sub-word tokens per word is implausible (real
# English averages ~1.3, even long words rarely exceed a handful) and signals
# the convention is wrong — fail loud on the first real run instead of emitting
# one transcript-spanning word.
_GROUP_GUARD_MIN_TOKENS = 10
_GROUP_GUARD_MAX_TOKENS_PER_WORD = 8


def _is_skippable_token(text: str) -> bool:
    """Soniox marker/special tokens (e.g. an `<end>` marker) and whitespace-only
    or empty tokens must neither create empty words nor corrupt timings."""
    stripped = text.strip()
    if not stripped:
        return True
    return stripped.startswith("<") and stripped.endswith(">")


def _group_soniox_tokens(tokens: list[dict]) -> list[dict]:
    """Group Soniox sub-word tokens into the word-level stream the pipeline
    consumes (R3/R4), converting ms -> s. A token whose `text` begins with
    whitespace starts a new word (KTD3); others append to the current word. Each
    emitted word is exactly `{"start", "end", "word", "speaker"}` so the SRT
    writers, sentence_seg._words_sha, and the assemblyai path stay byte-compatible
    (speaker is additive). Raises when the grouping collapses implausibly (a wrong
    boundary rule) rather than returning a corrupted single-word backbone."""
    words: list[dict] = []
    cur: list[dict] = []
    n_content = 0

    def emit(group: list[dict]) -> None:
        text = "".join(t.get("text", "") for t in group).strip()
        if not text:
            return
        words.append({
            "start": round(group[0]["start_ms"] / 1000, 3),
            "end": round(group[-1]["end_ms"] / 1000, 3),
            "word": text,
            "speaker": group[0].get("speaker"),
        })

    for tok in tokens:
        text = tok.get("text", "")
        if _is_skippable_token(text):
            continue
        n_content += 1
        if cur and not text[:1].isspace():
            cur.append(tok)
        else:
            emit(cur)
            cur = [tok]
    emit(cur)

    if (n_content >= _GROUP_GUARD_MIN_TOKENS and words
            and n_content / len(words) > _GROUP_GUARD_MAX_TOKENS_PER_WORD):
        raise RuntimeError(
            f"Soniox token grouping collapsed: {n_content} tokens -> {len(words)} "
            "word(s); the leading-whitespace word-boundary convention is likely "
            "wrong for this stt-async-v5 response — inspect a raw transcript and "
            "fix _group_soniox_tokens before trusting the output"
        )
    return words


class SonioxAsrProvider:
    name = "soniox"
    # Cloud engine: its own source of truth, so the graph omits the Granite
    # ensemble crosscheck (KTD8).
    wants_crosscheck = False
    detects_language = True  # supports source_lang="auto" via language identification

    def transcribe(self, state: dict, cfg: Config, asr_dir) -> AsrResult:
        """One Soniox async job: group the sub-word tokens into words (+ speaker),
        derive raw segments from a punctuation split, cache the raw token
        transcript. The word stream is the primary backbone input; Segment.speaker
        is assigned downstream by sentence_seg's majority vote (KTD8)."""
        audio = state["audio_16k"]
        # source_lang="auto" turns Soniox language identification on and widens the
        # hints — a deliberate ASR-fingerprint change / re-bill (KTD5/R20). Any
        # other source_lang leaves both at their configured values, so the EN
        # default fingerprint is byte-identical. Both the fingerprint inputs and
        # the request read these one set of effective values.
        auto = cfg.source_lang == "auto"
        lid = cfg.soniox_stt_enable_language_identification or auto
        hints = _AUTO_LANGUAGE_HINTS if auto else cfg.soniox_stt_language_hints
        inputs = {
            "audio_sha": artifacts.cached_file_sha256(audio),
            "engine": "soniox",
            "model": cfg.soniox_stt_model,
            "language_hints": hints,
            "enable_language_identification": lid,
            "enable_speaker_diarization": cfg.soniox_stt_speaker_diarization,
            # Context fingerprint (KTD9): a biasing change invalidates the cache.
            "context_terms": cfg.soniox_stt_context_terms,
            "context_text": cfg.soniox_stt_context_text,
        }
        # Cache the raw token transcript so a resumed/re-run job never re-uploads
        # or re-pays (R7/KTD9); transcribe() is only called when stale.
        raw = artifacts.produce_json(
            asr_dir / "soniox.json", inputs, "asr",
            lambda: soniox_stt.transcribe(cfg, audio, language_hints=hints,
                                          enable_language_identification=lid),
        )
        source_lang = (_detect_source_language(raw, "en")[0] if auto
                       else cfg.source_lang)

        words = _group_soniox_tokens(raw.get("tokens") or [])
        if not words:
            raise RuntimeError("ASR produced no segments — is there speech in the video?")

        # Soniox returns no utterances, so segments come from a punctuation split
        # over the words; a single span when punctuation yields none.
        raw_segments = [
            Segment(index=i, start=span[0]["start"], end=span[-1]["end"],
                    text_src=" ".join(w["word"] for w in span))
            for i, span in enumerate(sentences.punct_presplit(words))
        ]
        if not raw_segments:
            raw_segments = [Segment(index=0, start=words[0]["start"], end=words[-1]["end"],
                                    text_src=" ".join(w["word"] for w in words))]

        log.info("Soniox: %d word(s), %d raw segment(s), source=%s",
                 len(words), len(raw_segments), source_lang)
        return AsrResult(segments=raw_segments, words=words, source_lang=source_lang)

    def preflight(self, cfg: Config) -> list[str]:
        """Key presence + a liveness/auth probe (R9). Only a 404 is a clean
        authenticated probe; 401 = bad key, 403 = authenticated but insufficient
        scope/quota (report each distinctly); a 429/5xx is transient/unknown and
        is NOT proof of a good key, so it must not pass preflight."""
        problems: list[str] = []
        if not cfg.soniox_api_key:
            problems.append(
                "missing SONIOX_API_KEY (the soniox ASR engine needs a key, SHARED with "
                "the soniox TTS engine; set it in .env, or switch to --asr-engine local)"
            )
        else:
            try:
                status = _probe_soniox_stt(cfg)
                if status == 401:
                    problems.append(
                        "SONIOX_API_KEY unauthenticated (HTTP 401) — recheck the "
                        "key in .env"
                    )
                elif status == 403:
                    problems.append(
                        "SONIOX_API_KEY authenticates but lacks permission/quota "
                        "for STT (HTTP 403) — check the key's permission/quota in .env"
                    )
                elif status != 404:
                    problems.append(
                        f"Soniox STT could not confirm the key (HTTP {status}, not "
                        f"404) — may be overloaded/rate-limited, retry "
                        f"or check {cfg.soniox_stt_base_url}"
                    )
            except Exception as exc:
                problems.append(
                    f"Soniox STT unreachable ({cfg.soniox_stt_base_url}): {exc}"
                )
        return problems
