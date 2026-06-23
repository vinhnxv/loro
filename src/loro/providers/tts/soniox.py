"""Soniox cloud preset-voice provider (tts-rt-v1).

Wraps the existing `services.soniox.SonioxClient` (unchanged, KTD2); this module
adds the engine-specific orchestration surface the tts node used to inline. Being
a preset engine, `client` ignores the reference args, and `preflight` warns on a
dead --ref-audio. The preflight key/probe/voice-name checks live here (U6).
"""

import logging

import requests

from loro.config import Config, PresetVoices
from loro.providers.base import PROBE_TIMEOUT
from loro.services.soniox import SONIOX_VOICES, SonioxClient

log = logging.getLogger("loro.preflight")


def _probe_soniox(cfg: Config) -> int:
    """Liveness + auth probe for the preset cloud engine. POST a body that
    deliberately omits the required `text` field, so a good key fails on
    validation (a non-401 4xx) while a bad key fails on auth (401) — the key
    authenticates without synthesizing (and being billed for) any audio.
    Returns the HTTP status; raises on a connection/timeout (endpoint down).
    The key rides only in the header and is never logged."""
    resp = requests.post(
        f"{cfg.soniox_base_url}/tts",
        headers={"Authorization": f"Bearer {cfg.soniox_api_key}"},
        # `text` is omitted on purpose: the request must fail validation before
        # any audio is synthesized, so the probe can never trigger a billed call.
        json={
            "model": cfg.soniox_model,
            "language": cfg.effective_tts_language,
            "voice": cfg.soniox_default_voice,
            "audio_format": cfg.soniox_audio_format,
            "sample_rate": cfg.soniox_sample_rate,
        },
        timeout=PROBE_TIMEOUT,
    )
    return resp.status_code


def _unknown_soniox_voices(cfg: Config) -> list[str]:
    """Every configured voice name (pool + map pins + default) that is not a
    documented Soniox voice, so a typo fails preflight instead of 40 minutes
    into the run."""
    names = (set(cfg.soniox_voice_pool) | set(cfg.soniox_voice_map.values())
             | {cfg.soniox_default_voice})
    return sorted(n for n in names if n not in SONIOX_VOICES)


class SonioxTtsProvider:
    name = "soniox"
    clones = False           # preset voices, no cloning
    batches = False
    native_long_text = False

    def client(self, cfg: Config, ref_audio=None, ref_text=None) -> SonioxClient:
        # Preset engine: no reference clip, so ref_audio/ref_text are ignored.
        return SonioxClient(cfg)

    def clones_in(self, locale: str) -> bool:
        return self.clones  # preset engine: never clones (KTD7)

    def engine_inputs(self, cfg: Config) -> dict:
        return {
            "engine": "soniox",
            "model": cfg.soniox_model,
            # The spoken-language param is profile-derived (U9): one --target-lang
            # fr makes the engine synthesize French, not the engine's vi default.
            # VI resolves "vi", so the clip fingerprint stays byte-identical (R19).
            # The generic fallback (empty profile code) falls back to the target
            # tag so a --allow-fallback run never sends an empty language (#3).
            "language": cfg.effective_tts_language,
            "sample_rate": cfg.soniox_sample_rate,
            # audio_format shapes the emitted bytes (wav vs another container),
            # so it is part of clip identity: changing it resynthesizes.
            "audio_format": cfg.soniox_audio_format,
        }

    def chunk_budget(self, cfg: Config) -> int:
        return cfg.tts_chunk_budget

    def preset_voices(self, cfg: Config) -> PresetVoices:
        return PresetVoices(cfg.soniox_voice_pool, cfg.soniox_voice_map,
                            cfg.soniox_default_voice)

    def preflight(self, cfg: Config) -> list[str]:
        """Key presence + a liveness/auth probe + voice-name validation (R10).
        A 401/403 means the key won't synthesize at run time — flag here rather
        than fail every segment 40 minutes in; a transient 5xx/429 is left to the
        run's own retry."""
        problems: list[str] = []
        # The preset engine casts preset voices, so an explicit clone reference
        # is dead config — warn rather than silently ignore the flag.
        if cfg.ref_audio:
            log.warning("--ref-audio/--ref-text ignored with the soniox engine "
                        "(preset voice, no cloning); use --tts-engine vieneu "
                        "or higgs to clone the source voice")
        if not cfg.soniox_api_key:
            problems.append(
                "missing SONIOX_API_KEY (the soniox engine needs a key; set it in .env, "
                "or switch to --tts-engine vieneu)"
            )
        else:
            try:
                status = _probe_soniox(cfg)
                if status in (401, 403):
                    problems.append(
                        f"SONIOX_API_KEY unauthenticated or lacks permission "
                        f"(HTTP {status}) — recheck the key/quota in .env"
                    )
            except Exception as exc:
                problems.append(
                    f"Soniox unreachable ({cfg.soniox_base_url}): {exc}"
                )
        # Voice-name validation runs regardless of connectivity (config check):
        # a typo in the pool/map/default fails fast rather than mid-run.
        bad_voices = _unknown_soniox_voices(cfg)
        if bad_voices:
            problems.append(
                f"invalid Soniox voice name(s): {', '.join(bad_voices)} "
                "(check SONIOX_VOICE_POOL / SONIOX_VOICE_MAP / "
                "SONIOX_DEFAULT_VOICE; voice list: "
                "soniox.com/docs/tts/concepts/voices)"
            )
        return problems
