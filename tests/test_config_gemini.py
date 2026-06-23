"""Gemini engine config surface (U1): env-sourced gemini_* fields, the
comma-split voice pool, _parse_voice_map reuse, and numeric-parse failure
behaviour matching the rest of the namespaced engine blocks."""

import pytest

from loro.config import Config

_GEMINI_ENV = (
    "TTS_ENGINE", "GEMINI_API_KEY", "GEMINI_BASE_URL", "GEMINI_MODEL",
    "GEMINI_SAMPLE_RATE", "GEMINI_DEFAULT_VOICE", "GEMINI_VOICE_POOL",
    "GEMINI_VOICE_MAP", "GEMINI_BATCH_SEGMENTS", "GEMINI_BATCH_MAX_SYLLABLES",
    "GEMINI_SPLIT_MIN_GAP_MS", "GEMINI_STYLE_PROMPT", "GEMINI_TIMEOUT",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _GEMINI_ENV:
        monkeypatch.delenv(name, raising=False)


def test_gemini_defaults_populated():
    cfg = Config()
    assert cfg.gemini_api_key == ""
    assert cfg.gemini_base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert cfg.gemini_model == "gemini-3.1-flash-tts-preview"
    assert cfg.gemini_sample_rate == 24000
    assert cfg.gemini_default_voice == "Kore"
    assert cfg.gemini_voice_pool == ["Kore", "Puck", "Aoede", "Charon", "Leda", "Orus"]
    assert cfg.gemini_voice_map == {}
    assert cfg.gemini_batch_segments == 8
    assert cfg.gemini_batch_max_syllables == 360
    assert cfg.gemini_split_min_gap_ms == 200.0
    assert cfg.gemini_style_prompt == ""
    assert cfg.gemini_timeout == 300.0


def test_env_overrides_flow_into_fields(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-preview-tts")
    monkeypatch.setenv("GEMINI_DEFAULT_VOICE", "Puck")
    monkeypatch.setenv("GEMINI_BATCH_SEGMENTS", "4")
    cfg = Config()
    assert cfg.gemini_model == "gemini-2.5-flash-preview-tts"
    assert cfg.gemini_default_voice == "Puck"
    assert cfg.gemini_batch_segments == 4


def test_voice_pool_parses_trimmed_and_drops_blanks(monkeypatch):
    monkeypatch.setenv("GEMINI_VOICE_POOL", "Kore, Puck, Leda,")
    assert Config().gemini_voice_pool == ["Kore", "Puck", "Leda"]


def test_voice_map_parses_via_shared_helper(monkeypatch):
    # "A" has no "=", "=Puck" has no speaker — both skipped, not fatal.
    monkeypatch.setenv("GEMINI_VOICE_MAP", "A=Kore,B=Puck,A,=Puck")
    assert Config().gemini_voice_map == {"A": "Kore", "B": "Puck"}


def test_non_numeric_batch_segments_raises(monkeypatch):
    # Consistent with vieneu_temperature: a bad numeric env raises at construction.
    monkeypatch.setenv("GEMINI_BATCH_SEGMENTS", "lots")
    with pytest.raises(ValueError):
        Config()


def test_capability_adapters_delegate_to_provider():
    # U4: Config.tts_uses_cloning / preset_voices are thin adapters over the
    # gemini provider, not engine-name branches in config.
    from loro import providers
    cfg = Config(tts_engine="gemini")
    assert cfg.tts_uses_cloning is providers.tts("gemini").clones is False
    assert cfg.preset_voices == providers.tts("gemini").preset_voices(cfg)
    assert cfg.preset_voices.default == "Kore"
