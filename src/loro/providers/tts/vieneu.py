"""VieNeu-TTS on-device cloning provider (warm subprocess worker).

Wraps the existing `services.vieneu.VieNeuClient` (unchanged, KTD2); this module
adds the engine-specific orchestration surface the tts node used to inline.
"""

from loro.config import Config, PresetVoices
from loro.services.vieneu import VieNeuClient


class VieNeuTtsProvider:
    name = "vieneu"
    clones = True            # clones a reference voice
    batches = False
    native_long_text = False

    def client(self, cfg: Config, ref_audio=None, ref_text=None) -> VieNeuClient:
        return VieNeuClient(cfg, ref_audio, ref_text)

    def clones_in(self, locale: str) -> bool:
        # VieNeu is intrinsically Vietnamese-only (sea-g2p VI front end), so it
        # clones in vi alone — never for a non-VI target (KTD7, R13).
        return self.clones and locale.split("-")[0].lower() == "vi"

    def engine_inputs(self, cfg: Config) -> dict:
        return {
            "engine": "vieneu",
            "model": cfg.vieneu_model,
            "temperature": cfg.vieneu_temperature,
            "emotion": cfg.vieneu_emotion,
        }

    def chunk_budget(self, cfg: Config) -> int:
        return cfg.tts_chunk_budget

    def preset_voices(self, cfg: Config) -> PresetVoices | None:
        return None  # cloning engine: no preset voices to cast

    def preflight(self, cfg: Config) -> list[str]:
        # The vieneu venv is checked lazily in VieNeuClient.__enter__ (like the
        # NeMo interpreter), so a vieneu run adds no preflight TTS check.
        return []
