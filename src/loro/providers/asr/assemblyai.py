"""AssemblyAI cloud ASR provider (universal-3-pro).

Wraps the existing `services.assemblyai` module function (unchanged, KTD2): one
cloud call mapped into words (+ speaker) and raw segments, the raw response
cached, and the speaker-grouped utterances persisted as utterances.json. The asr
node owns the shared tail (EN SRT + return); this provider owns the engine
mapping, its utterances.json artifact, and its own no-speech guard (KTD3).
"""

import logging

import requests

from loro.config import Config
from loro.harness import artifacts
from loro.providers.base import PROBE_TIMEOUT, AsrResult
from loro.services import assemblyai
from loro.state import Segment
from loro.utils import sentences

log = logging.getLogger("loro.asr")


def _probe_assemblyai(cfg: Config) -> int:
    """Liveness + auth probe for the cloud engine. A GET on a nonexistent
    transcript id returns 401 for a bad key and a reachable non-401 (e.g. 404)
    for a good one, so the key authenticates without spending a transcription.
    Returns the HTTP status; raises on a connection/timeout (endpoint down).
    The key rides only in the header and is never logged."""
    resp = requests.get(
        f"{cfg.assemblyai_base_url}/transcript/preflight-nonexistent",
        headers={"authorization": cfg.assemblyai_api_key},
        timeout=PROBE_TIMEOUT,
    )
    return resp.status_code


class AssemblyaiAsrProvider:
    name = "assemblyai"
    # Cloud engine: its own source of truth, so the graph omits the crosscheck.
    wants_crosscheck = False
    detects_language = True  # supports source_lang="auto" via language_detection

    def transcribe(self, state: dict, cfg: Config, asr_dir) -> AsrResult:
        """One AssemblyAI cloud call: map the transcript into words (+ speaker) and
        raw segments, cache the raw response, and persist utterances.json. The word
        stream is the primary backbone input; utterance-derived segments are
        sentence_seg's fallback (KTD1/KTD8)."""
        audio = state["audio_16k"]
        inputs = {
            "audio_sha": artifacts.cached_file_sha256(audio),
            "engine": "assemblyai",
            "speech_models": cfg.assemblyai_speech_models,
            "speaker_labels": cfg.assemblyai_speaker_labels,
            "language_code": cfg.assemblyai_language_code,
        }
        # Mirror the request branch in services/assemblyai._create exactly
        # (B4/R7/KTD5): the request sends language_code XOR language_detection, so
        # a pinned code OMITS language_detection from the request — and the
        # fingerprint must omit it too. Otherwise toggling detection on a pinned
        # run changes the cache key for an identical request, forcing a needless
        # recompute and re-bill. The unpinned (detection) path is unchanged, so the
        # default fingerprint stays byte-identical.
        if not cfg.assemblyai_language_code:
            inputs["language_detection"] = cfg.assemblyai_language_detection
        # Cache the raw API response so a resumed/re-run job never re-uploads or
        # re-pays (R6/KTD8); transcribe() is only called when stale.
        raw = artifacts.produce_json(
            asr_dir / "assemblyai.json", inputs, "asr",
            lambda: assemblyai.transcribe(cfg, audio),
        )

        # ms -> s (KTD4); keep start/end/word so sentence_seg._words_sha and the
        # SRT writers are unaffected. speaker is additive (not part of words hash).
        words = [
            {"start": round(w["start"] / 1000, 3), "end": round(w["end"] / 1000, 3),
             "word": w["text"], "speaker": w.get("speaker")}
            for w in (raw.get("words") or [])
        ]
        if not words:
            raise RuntimeError("ASR produced no segments — is there speech in the video?")

        # Raw segments: one per utterance when diarization returns them, else a
        # punctuation split over the words (single span if neither yields text).
        utterances = raw.get("utterances") or []
        if utterances:
            raw_segments = [
                Segment(index=i, start=round(u["start"] / 1000, 3),
                        end=round(u["end"] / 1000, 3), text_src=u["text"].strip(),
                        speaker=u.get("speaker") or "")
                for i, u in enumerate(u for u in utterances if u.get("text", "").strip())
            ]
        else:
            raw_segments = [
                Segment(index=i, start=span[0]["start"], end=span[-1]["end"],
                        text_src=" ".join(w["word"] for w in span))
                for i, span in enumerate(sentences.punct_presplit(words))
            ]
        if not raw_segments:
            raw_segments = [Segment(index=0, start=words[0]["start"], end=words[-1]["end"],
                                    text_src=(raw.get("text") or "").strip())]

        # Persist the speaker-grouped utterances in seconds (R3); null utterances
        # persist as an empty list.
        utt_out = [
            {"speaker": u.get("speaker"), "text": u.get("text", ""),
             "start": round(u["start"] / 1000, 3), "end": round(u["end"] / 1000, 3)}
            for u in utterances
        ]
        artifacts.produce_json(asr_dir / "utterances.json", inputs, "asr",
                               lambda: {"utterances": utt_out})

        # source_lang="auto" reads AssemblyAI's detected top-level language_code
        # (it already runs language_detection by default, so the request — and the
        # fingerprint — are unchanged either way, U7). Any other source uses the
        # configured value.
        if cfg.source_lang == "auto":
            source_lang = raw.get("language_code") or "en"
            log.info("AssemblyAI detected source language: %s", source_lang)
        else:
            source_lang = cfg.source_lang

        log.info("AssemblyAI: %d word(s), %d raw segment(s), source=%s",
                 len(words), len(raw_segments), source_lang)
        return AsrResult(segments=raw_segments, words=words, source_lang=source_lang)

    def preflight(self, cfg: Config) -> list[str]:
        """Key presence + a liveness/auth probe (R9). A 401 means a bad key; a
        reachable non-401 (e.g. 404) authenticates without spending a job."""
        problems: list[str] = []
        if not cfg.assemblyai_api_key:
            problems.append(
                "missing ASSEMBLYAI_API_KEY (the assemblyai engine needs a key; set it in "
                ".env, or switch to --asr-engine local)"
            )
        else:
            try:
                if _probe_assemblyai(cfg) == 401:
                    problems.append(
                        "ASSEMBLYAI_API_KEY unauthenticated (HTTP 401) — recheck "
                        "the key in .env"
                    )
            except Exception as exc:
                problems.append(
                    f"AssemblyAI unreachable ({cfg.assemblyai_base_url}): {exc}"
                )
        return problems
