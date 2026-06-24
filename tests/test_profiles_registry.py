"""U3: the LanguageProfile registry resolves BCP-47 tags to the right profile and
carries the per-language length/framing/voice/font constants (R2/R3/R4).

Mirrors test_providers_registry.py: pins the contract surface + the fallback
chain the rest of the multi-language refactor reads instead of `if lang == "vi"`.
"""

import ast
import dataclasses
import pathlib

import pytest

from loro.config import Config
from loro.harness import qa
from loro.nodes import translate
from loro.profiles import GENERIC, is_profiled, registered_tags, resolve
from loro.profiles.base import LanguageProfile, char_count, vi_syllable_count


class TestResolve:
    @pytest.mark.parametrize("tag,locale", [
        ("vi", "vi"), ("en", "en"), ("fr", "fr"), ("es", "es"),
    ])
    def test_base_tag_resolves_to_its_profile(self, tag, locale):
        assert resolve(tag).locale == locale

    def test_region_variant_falls_back_to_base_language(self):
        # R3: es-MX -> es -> ES; fr-FR -> fr; case/separator-insensitive.
        assert resolve("es-MX") is resolve("es")
        assert resolve("fr-FR") is resolve("fr")
        assert resolve("FR") is resolve("fr")
        assert resolve("es_419") is resolve("es")

    def test_unknown_tag_falls_back_to_generic(self):
        assert resolve("xx") is GENERIC
        assert resolve("zz-ZZ") is GENERIC

    def test_is_profiled_gates_allow_fallback(self):
        # R4: profiled tags (and their region variants) are True; the unknown is
        # False so Config/preflight can demand --allow-fallback.
        for tag in ("vi", "en", "fr", "es", "es-MX", "fr-FR"):
            assert is_profiled(tag) is True
        assert is_profiled("xx") is False
        assert is_profiled("klingon") is False

    def test_registered_tags(self):
        assert registered_tags() == ["en", "es", "fr", "vi"]


class TestLengthModel:
    def test_vi_is_the_only_syllable_model(self):
        assert resolve("vi").length_model == "syllable"
        for tag in ("en", "fr", "es", "xx"):
            assert resolve(tag).length_model == "cps"

    def test_vi_counter_matches_legacy_syllable_count(self):
        # R19: the VI profile's counter must equal qa.syllable_count verbatim, so
        # the VI translation budget fingerprint never drifts.
        for text in ("Xin chào các bạn", "22 con mèo", "một hai ba bốn năm", "a"):
            assert resolve("vi").counter(text) == qa.syllable_count(text)

    def test_cps_counter_is_character_based(self):
        fr = resolve("fr")
        assert fr.counter is char_count
        assert fr.counter("bonjour") == 7
        assert fr.counter("  hi  ") == 2  # edges stripped, internal counted

    def test_vi_rate_and_chunk_budget_are_legacy_values(self):
        from loro.config import Config
        vi = resolve("vi")
        assert vi.rate == 4.3
        assert vi.chunk_budget == 60  # == legacy tts_max_chunk_syllables
        # The VI profile is now the single source of the rate (the dead
        # vi_syllables_per_sec config knob was removed, U5/#10); the chunk-budget
        # knob still exists, so guard the profile/knob agreement that survives.
        assert vi.chunk_budget == Config().tts_max_chunk_syllables

    def test_cps_chunk_budget_is_character_sized(self):
        # CPS chunk budget is in characters (comparable duration to VI's 60 syl),
        # not the 60-syllable count — feeding 60 to a char counter over-fragments.
        for tag in ("fr", "es", "xx"):
            assert resolve(tag).chunk_budget >= 200


class TestFramingAndVoice:
    def test_vi_system_prompt_is_the_legacy_vietnamese_prompt(self):
        # KTD2/R19: U8 sources the system prompt from the profile; the VI prompt
        # must stay the legacy Vietnamese string (the translate-budget fingerprint
        # golden, which runs the real node, is the hard byte-identity guard).
        sp = resolve("vi").system_prompt
        assert sp.startswith("Bạn là biên dịch viên lồng tiếng chuyên nghiệp Anh-Việt")
        assert sp.endswith("Không thêm chú thích.")

    def test_vi_output_key_preserves_legacy_json_contract(self):
        assert resolve("vi").output_key == "vi"

    def test_tier1_profiles_have_font_glyph_and_pool(self):
        for tag in ("en", "vi", "fr", "es"):
            p = resolve(tag)
            assert p.font and p.glyph_sample and p.preset_pool
            assert p.src_lang_label and p.tgt_lang_label

    def test_generic_and_unprofiled_default_to_preset_voice(self):
        # KTD7: an unvalidated cross-lingual clone is never the silent default.
        assert GENERIC.voice_strategy == "preset"
        assert resolve("xx").voice_strategy == "preset"

    def test_vieneu_is_never_a_default_engine_for_non_vi(self):
        for tag in ("en", "fr", "es", "xx"):
            assert resolve(tag).tts_engine != "vieneu"

    def test_iso639_2_tags(self):
        assert resolve("vi").iso639_2 == "vie"
        assert resolve("fr").iso639_2 == "fra"
        assert resolve("es").iso639_2 == "spa"


class TestCalibratedConstants:
    """U11/R3: every tier-1 profile (+ generic) resolves and carries real,
    non-placeholder length/voice/subtitle constants. FR/ES values are SEEDS
    pending empirical calibration from real runs (documented), but must still be
    plausible and present, never zero/empty placeholders."""

    @pytest.mark.parametrize("tag", ["en", "vi", "fr", "es"])
    def test_tier1_constants_present_and_plausible(self, tag):
        p = resolve(tag)
        assert p.rate > 0
        assert p.cps_max > 0
        assert p.expansion_factor >= 1.0
        assert p.chunk_budget > 0
        assert len(p.iso639_2) == 3          # ISO 639-2/B is a 3-letter tag
        assert p.tts_language_code           # the spoken-language param is set
        assert p.system_prompt and p.preset_pool and p.font and p.glyph_sample

    def test_generic_fallback_has_constants(self):
        assert GENERIC.rate > 0 and GENERIC.cps_max > 0 and GENERIC.chunk_budget > 0
        assert GENERIC.length_model == "cps"

    def test_cps_profiles_have_localization_expansion(self):
        # FR/ES expand vs English source (seed values, calibrated in U11).
        assert resolve("fr").expansion_factor > 1.0
        assert resolve("es").expansion_factor > 1.0


def test_profiles_module_is_pure_no_nodes_import():
    """Verification: the profiles package imports nothing from loro.nodes/providers
    (it is resolved from config through a lazy accessor; either would cycle). Scans
    the import AST so docstrings that name a module don't false-positive."""
    pkg = pathlib.Path(translate.__file__).resolve().parents[1] / "profiles"
    forbidden = ("loro.nodes", "loro.providers")
    offenders = []
    for src in pkg.glob("*.py"):
        tree = ast.parse(src.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            for m in mods:
                if any(m == f or m.startswith(f + ".") for f in forbidden):
                    offenders.append(f"{src.name}: {m}")
    assert offenders == [], f"profiles must stay pure; found imports: {offenders}"


# --- U14/B11: per-language SRT cue-char budget wiring (phase-1, surfaced inert) ---

class TestSrtTargetBudget:
    def test_vi_keeps_global_budget(self):
        # R19 key guard: VI (syllable model) uses the global cue-char budget, so
        # its subtitle bytes never move.
        cfg = Config()
        assert cfg.srt_target_max_cue_chars == cfg.srt_max_cue_chars

    def test_cps_budget_is_reading_speed_clamped_and_inert_today(self):
        # CPS profiles derive min(global, cps_max * max_dur). With the shipped
        # cps_max=17.0 and max_dur=6.0 -> 102 >= global 84, so the global binds and
        # the wiring is INERT for FR/ES today (no behavior change) — it needs a
        # calibrated lower cps_max to bite (deferred follow-up). Surfaced, not
        # silently merged.
        cfg = Config(target_lang="fr")
        assert cfg.srt_target_max_cue_chars == min(
            cfg.srt_max_cue_chars, int(17.0 * cfg.srt_max_cue_dur))
        assert cfg.srt_target_max_cue_chars == cfg.srt_max_cue_chars  # inert (84)

    def test_calibrated_lower_cps_max_tightens_budget(self, monkeypatch):
        # Proves the wiring will bite once FR/ES cps_max is calibrated: a synthetic
        # profile with cps_max=10 -> 10*6=60 < global 84 -> the budget tightens.
        cfg = Config(target_lang="fr")
        tight = dataclasses.replace(cfg.language_profile, cps_max=10.0)
        monkeypatch.setattr(type(cfg), "language_profile",
                            property(lambda self: tight))
        assert cfg.srt_target_max_cue_chars == 60
        assert cfg.srt_target_max_cue_chars < cfg.srt_max_cue_chars
