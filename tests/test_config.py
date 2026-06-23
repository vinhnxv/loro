"""U4: language selection on Config — target/source defaults, env + the lazy
LanguageProfile accessor, with no import cycle at module load (R1, R4, R19)."""

import ast
import pathlib

from loro.config import Config
from loro.profiles import GENERIC


class TestLanguageDefaults:
    def test_defaults_are_vi_target_en_source(self):
        cfg = Config()
        assert cfg.target_lang == "vi"
        assert cfg.source_lang == "en"
        assert cfg.allow_fallback is False

    def test_default_profile_is_vietnamese(self):
        # R1/R19: an empty config resolves the VI profile (the byte-identical path).
        p = Config().language_profile
        assert p.locale == "vi"
        assert p.length_model == "syllable"

    def test_env_overrides_target_and_source(self, monkeypatch):
        monkeypatch.setenv("TARGET_LANG", "fr")
        monkeypatch.setenv("SOURCE_LANG", "de")
        cfg = Config()
        assert cfg.target_lang == "fr"
        assert cfg.source_lang == "de"
        assert cfg.language_profile.locale == "fr"

    def test_kwarg_overrides_select_es(self):
        assert Config(target_lang="es").language_profile.locale == "es"

    def test_region_variant_resolves_base_profile(self):
        assert Config(target_lang="es-MX").language_profile.locale == "es"

    def test_unprofiled_target_resolves_generic(self):
        # Config does not reject — preflight gates --allow-fallback (R4).
        assert Config(target_lang="xx").language_profile is GENERIC

    def test_allow_fallback_env_truthy(self, monkeypatch):
        monkeypatch.setenv("ALLOW_FALLBACK", "1")
        assert Config().allow_fallback is True


class TestLanguageNormalization:
    def test_target_source_lowercased(self):
        # #1/R19: case is normalized so a non-canonical spelling resolves to the
        # same profile AND the same translate fingerprint as the canonical default.
        cfg = Config(target_lang="VI", source_lang="EN")
        assert cfg.target_lang == "vi"
        assert cfg.source_lang == "en"

    def test_auto_source_lowercased(self):
        # #1: "AUTO" must normalize to "auto" or it silently disables detection.
        assert Config(source_lang="AUTO").source_lang == "auto"

    def test_effective_tts_language_uses_profile_code(self):
        # Profiled target: the spoken-language code is the profile's, byte-identical.
        assert Config(target_lang="vi").effective_tts_language == "vi"
        assert Config(target_lang="fr").effective_tts_language == "fr"

    def test_effective_tts_language_falls_back_to_target_tag_for_generic(self):
        # #3: the generic fallback profile has an empty tts_language_code; the
        # effective language falls back to the target's base tag, never empty, so a
        # --allow-fallback run never sends an empty `language` to the engine.
        assert GENERIC.tts_language_code == ""
        assert Config(target_lang="de").effective_tts_language == "de"
        assert Config(target_lang="zh-Hans").effective_tts_language == "zh"


def test_config_module_does_not_import_nodes_or_providers_at_load():
    """R4 scenario: importing config must not pull nodes/providers (the profile
    accessor and provider adapters are all lazy). Scans config's top-level import
    AST so a docstring mention does not false-positive."""
    path = pathlib.Path(__import__("loro.config", fromlist=["x"]).__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    top_level_imports = []
    for node in tree.body:  # only module-level statements, not function bodies
        if isinstance(node, ast.Import):
            top_level_imports += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top_level_imports.append(node.module or "")
    for m in top_level_imports:
        assert not m.startswith("loro.nodes"), f"config imports {m} at load"
        assert not m.startswith("loro.providers"), f"config imports {m} at load"
        assert not m.startswith("loro.profiles"), f"config imports {m} at load"
