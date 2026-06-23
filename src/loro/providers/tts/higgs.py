"""Higgs Audio v3 remote cloning provider (sglang-omni server).

Wraps the existing `services.higgs.HiggsClient` (unchanged, KTD2); this module
adds the engine-specific orchestration surface the tts node used to inline. Its
preflight probes the server's /health for liveness (U6).
"""

import requests

from loro.config import Config, PresetVoices
from loro.services.higgs import HiggsClient


def _probe_higgs(cfg: Config) -> None:
    """Any HTTP response at all proves the server process is alive."""
    requests.get(f"{cfg.higgs_host}/health", timeout=10)


class HiggsTtsProvider:
    name = "higgs"
    clones = True            # clones a reference voice
    batches = False
    native_long_text = False

    def client(self, cfg: Config, ref_audio=None, ref_text=None) -> HiggsClient:
        return HiggsClient(cfg, ref_audio, ref_text)

    def clones_in(self, locale: str) -> bool:
        # Higgs Audio v3 is multilingual, so the static capability allows cloning
        # in any locale (back-compat default); cross-lingual clone QUALITY is
        # uneven and validated at tier-1 calibration (U11), not gated here (KTD7).
        return self.clones

    def engine_inputs(self, cfg: Config) -> dict:
        return {"engine": "higgs", "model": cfg.higgs_model}

    def chunk_budget(self, cfg: Config) -> int:
        return cfg.tts_chunk_budget

    def preset_voices(self, cfg: Config) -> PresetVoices | None:
        return None  # cloning engine: no preset voices to cast

    def preflight(self, cfg: Config) -> list[str]:
        try:
            _probe_higgs(cfg)
        except Exception as exc:
            return [f"Higgs unreachable ({cfg.higgs_host}): {exc}"]
        return []
