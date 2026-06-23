"""Gemini cloud preset-voice provider (generateContent).

Wraps the existing `services.gemini.GeminiClient` (unchanged, KTD2); this module
adds the engine-specific orchestration surface the tts node used to inline. The
Gemini batch/split/fallback orchestration stays in the node (R7); only
`synthesize_batch` lives on the client this returns. The preflight key/models.list
probe + model + voice-name checks live here (U6).
"""

import logging

import requests

from loro.config import Config, PresetVoices
from loro.providers.base import PROBE_TIMEOUT
from loro.services.gemini import GEMINI_VOICES, GeminiClient

log = logging.getLogger("loro.preflight")


def _probe_gemini(cfg: Config) -> tuple[int, list[str]]:
    """Liveness + auth probe for the Gemini preset engine. A GET on models.list
    authenticates the key (401 bad key, 403 no permission/quota, 200 good)
    WITHOUT spending any tokens, and returns the served model ids so a wrong or
    renamed model id is caught at preflight rather than 404-ing every segment
    mid-run. The key rides only in the `x-goog-api-key` header — never as a
    `?key=` query string, which would land in access/proxy logs (R10/S1) — and
    is never logged. Returns (status, served model ids); raises on a
    connection/timeout (endpoint down)."""
    resp = requests.get(
        f"{cfg.gemini_base_url}/models",
        headers={"x-goog-api-key": cfg.gemini_api_key},
        timeout=PROBE_TIMEOUT,
    )
    served: list[str] = []
    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            data = {}
        served = [m.get("name", "").split("/")[-1]
                  for m in data.get("models", []) if isinstance(m, dict)]
    return resp.status_code, served


def _unknown_gemini_voices(cfg: Config) -> list[str]:
    """Every configured voice name (pool + map pins + default) that is not a
    documented Gemini voice, so a typo fails preflight instead of mid-run."""
    names = (set(cfg.gemini_voice_pool) | set(cfg.gemini_voice_map.values())
             | {cfg.gemini_default_voice})
    return sorted(n for n in names if n not in GEMINI_VOICES)


class GeminiTtsProvider:
    name = "gemini"
    clones = False           # preset voices, no cloning
    batches = True           # batches consecutive segments into one multi-speaker call
    native_long_text = True  # handles long text natively (no per-segment chunking)

    def client(self, cfg: Config, ref_audio=None, ref_text=None) -> GeminiClient:
        # Preset engine: no reference clip, so ref_audio/ref_text are ignored.
        return GeminiClient(cfg)

    def clones_in(self, locale: str) -> bool:
        return self.clones  # preset engine: never clones (KTD7)

    def engine_inputs(self, cfg: Config) -> dict:
        return {
            "engine": "gemini",
            "model": cfg.gemini_model,
            "sample_rate": cfg.gemini_sample_rate,
            # style_prompt is prepended to the synthesized text, so it shapes the
            # audio and is identity-bearing. batch_max_syllables is Gemini's
            # effective chunk budget (the per-segment path splits on it), so it
            # belongs in the fingerprint — but the BATCH COMPOSITION (which
            # neighbors a segment was synthesized with) deliberately does not, so
            # a neighbor edit re-batches without re-synthesizing this clip (KTD5).
            "style_prompt": cfg.gemini_style_prompt,
            "batch_max_syllables": cfg.gemini_batch_max_syllables,
        }

    def chunk_budget(self, cfg: Config) -> int:
        # Gemini handles long text natively, so a normal segment is one call; the
        # raised budget guards only a pathologically long SINGLE segment (A4).
        return cfg.gemini_batch_max_syllables

    def preset_voices(self, cfg: Config) -> PresetVoices:
        return PresetVoices(cfg.gemini_voice_pool, cfg.gemini_voice_map,
                            cfg.gemini_default_voice)

    def preflight(self, cfg: Config) -> list[str]:
        """Key presence + a models.list probe (auth + served-model check) + voice-
        name validation (R10). 401/403 won't synthesize at run time; a served-model
        mismatch would 404 every segment — flag both here."""
        problems: list[str] = []
        # Preset engine: an explicit clone reference is dead config — warn,
        # don't fail (same shape as the Soniox branch).
        if cfg.ref_audio:
            log.warning("--ref-audio/--ref-text ignored with the gemini engine "
                        "(preset voice, no cloning); use --tts-engine vieneu "
                        "or higgs to clone the source voice")
        if not cfg.gemini_api_key:
            problems.append(
                "missing GEMINI_API_KEY (the gemini engine needs a key; set it in .env, "
                "or switch to --tts-engine vieneu)"
            )
        else:
            try:
                status, served = _probe_gemini(cfg)
                if status in (401, 403):
                    problems.append(
                        f"GEMINI_API_KEY unauthenticated or lacks permission "
                        f"(HTTP {status}) — recheck the key/quota in .env"
                    )
                elif status == 200 and cfg.gemini_model not in served:
                    problems.append(
                        f"Gemini model unavailable: `{cfg.gemini_model}` "
                        f"(set via GEMINI_MODEL; available: "
                        f"{', '.join(served) or 'no models'})"
                    )
            except Exception as exc:
                problems.append(
                    f"Gemini unreachable ({cfg.gemini_base_url}): {exc}"
                )
        # Voice-name validation runs regardless of connectivity (config check).
        bad_voices = _unknown_gemini_voices(cfg)
        if bad_voices:
            problems.append(
                f"invalid Gemini voice name(s): {', '.join(bad_voices)} "
                "(check GEMINI_VOICE_POOL / GEMINI_VOICE_MAP / "
                "GEMINI_DEFAULT_VOICE; voice list: "
                "ai.google.dev/gemini-api/docs/speech-generation)"
            )
        return problems
