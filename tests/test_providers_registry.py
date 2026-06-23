"""U2: the providers registry routes engine names to the right provider and
surfaces each engine's capability flags in one place (R1-R4). Behavioral methods
are added in U3 (TTS) / U5 (ASR) / U6 (preflight); this pins the contract surface
and the capability flags the rest of the refactor reads instead of name checks.
"""

import ast
import pathlib

import pytest

from loro.providers import UnknownEngineError, asr, tts


def test_no_engine_name_dispatch_outside_providers():
    """AE1: adding an engine is one provider module + one registry entry, so no
    `cfg.tts_engine == "x"` / `asr_engine != "y"` dispatch may survive in the
    nodes, graph, preflight, or config — only the per-engine providers and the
    registry name engines, and the two Config capability adapters delegate via a
    call (not a comparison). Scans the AST so comments/docstrings that mention an
    engine name don't false-positive (U8)."""
    root = pathlib.Path(__file__).resolve().parents[1] / "src" / "loro"
    targets = [root / "nodes", root / "graph.py",
               root / "harness" / "preflight.py", root / "config.py"]
    files: list[pathlib.Path] = []
    for t in targets:
        files += sorted(t.rglob("*.py")) if t.is_dir() else [t]

    offenders = []
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for operand in [node.left, *node.comparators]:
                    if (isinstance(operand, ast.Attribute)
                            and operand.attr in ("tts_engine", "asr_engine")):
                        offenders.append(f"{f.relative_to(root)}:{node.lineno}")
    assert offenders == [], (
        "engine-name dispatch must live only in providers/registry; found: "
        + ", ".join(offenders))


def test_unknown_engine_error_is_not_keyerror():
    # The plan requires a clear NAMED error, not the bare KeyError a dict lookup
    # would raise (R2/R4).
    assert not issubclass(UnknownEngineError, KeyError)


class TestAsrRegistry:
    @pytest.mark.parametrize("name", ["soniox", "assemblyai", "local"])
    def test_returns_provider_with_matching_name(self, name):
        assert asr(name).name == name

    def test_unknown_engine_raises_named_error(self):
        with pytest.raises(UnknownEngineError):
            asr("nope")

    def test_wants_crosscheck_true_only_for_local(self):
        assert asr("local").wants_crosscheck is True
        assert asr("soniox").wants_crosscheck is False
        assert asr("assemblyai").wants_crosscheck is False


class TestTtsRegistry:
    @pytest.mark.parametrize("name", ["vieneu", "higgs", "soniox", "gemini"])
    def test_returns_provider_with_matching_name(self, name):
        assert tts(name).name == name

    def test_unknown_engine_raises_named_error(self):
        with pytest.raises(UnknownEngineError):
            tts("nope")

    def test_clones_true_only_for_cloning_engines(self):
        assert tts("vieneu").clones is True
        assert tts("higgs").clones is True
        assert tts("soniox").clones is False
        assert tts("gemini").clones is False

    def test_batches_and_native_long_text_true_only_for_gemini(self):
        for name in ("vieneu", "higgs", "soniox"):
            assert tts(name).batches is False
            assert tts(name).native_long_text is False
        assert tts("gemini").batches is True
        assert tts("gemini").native_long_text is True

    def test_clones_in_is_language_aware(self):
        # U9/KTD7: VieNeu clones only Vietnamese; Higgs clones in any locale
        # (multilingual); the preset engines never clone.
        assert tts("vieneu").clones_in("vi") is True
        assert tts("vieneu").clones_in("vi-VN") is True
        assert tts("vieneu").clones_in("fr") is False
        assert tts("higgs").clones_in("fr") is True
        assert tts("higgs").clones_in("vi") is True
        for name in ("soniox", "gemini"):
            assert tts(name).clones_in("vi") is False
            assert tts(name).clones_in("fr") is False


class TestCloningResolution:
    # R13: the cloning-vs-preset decision is language-aware via tts_uses_cloning.
    def test_vieneu_clones_vi_but_not_fr(self):
        from loro.config import Config
        assert Config(tts_engine="vieneu", target_lang="vi").tts_uses_cloning is True
        assert Config(tts_engine="vieneu", target_lang="fr").tts_uses_cloning is False

    def test_higgs_clones_fr(self):
        from loro.config import Config
        assert Config(tts_engine="higgs", target_lang="fr").tts_uses_cloning is True

    def test_preset_engines_never_clone(self):
        from loro.config import Config
        assert Config(tts_engine="soniox", target_lang="fr").tts_uses_cloning is False
        assert Config(tts_engine="gemini", target_lang="vi").tts_uses_cloning is False
