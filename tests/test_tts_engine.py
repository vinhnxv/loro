"""Engine selection: TTS_ENGINE/--tts-engine routing and VieNeu config
defaults (U1), plus the tts-node client factory and the engine-aware clip
fingerprint (U4)."""

import numpy as np
import pytest
import soundfile as sf

from loro import __main__ as cli
from loro import providers
from loro.config import DEFAULT_VIENEU_PYTHON, Config
from loro.harness import artifacts
from loro.harness.artifacts import fingerprint
from loro.harness.ledger import SkipLedger
from loro.nodes import tts as tts_mod
from loro.nodes.tts import _chunk_budget, _engine_inputs, _tts_client, tts
from loro.services.gemini import GeminiClient, SplitError
from loro.services.higgs import HiggsClient
from loro.services.soniox import SonioxClient
from loro.services.vieneu import VieNeuClient
from loro.state import Segment


_SONIOX_ENV = (
    "TTS_ENGINE", "SONIOX_API_KEY", "SONIOX_BASE_URL", "SONIOX_MODEL",
    "SONIOX_LANGUAGE", "SONIOX_SAMPLE_RATE", "SONIOX_AUDIO_FORMAT",
    "SONIOX_TIMEOUT", "SONIOX_DEFAULT_VOICE", "SONIOX_VOICE_POOL",
    "SONIOX_VOICE_MAP",
)


def _clean_soniox_env(monkeypatch):
    for name in _SONIOX_ENV:
        monkeypatch.delenv(name, raising=False)


class TestEngineConfig:
    def test_default_engine_is_soniox(self, monkeypatch):
        monkeypatch.delenv("TTS_ENGINE", raising=False)
        assert Config().tts_engine == "soniox"

    def test_vieneu_defaults_populated(self, monkeypatch):
        for var in ("VIENEU_PYTHON", "VIENEU_MODEL", "VIENEU_TEMPERATURE", "VIENEU_EMOTION"):
            monkeypatch.delenv(var, raising=False)
        cfg = Config()
        assert cfg.vieneu_python == str(DEFAULT_VIENEU_PYTHON)
        assert cfg.vieneu_model == "pnnbao-ump/VieNeu-TTS-v3-Turbo"
        assert cfg.vieneu_temperature == 0.8
        assert cfg.vieneu_emotion == "natural"
        assert cfg.vieneu_timeout == 600.0

    def test_higgs_fields_unchanged(self, monkeypatch):
        monkeypatch.delenv("HIGGS_HOST", raising=False)
        monkeypatch.delenv("HIGGS_MODEL", raising=False)
        cfg = Config()
        assert cfg.higgs_host == "http://localhost:8000"
        assert cfg.higgs_model == "bosonai/higgs-audio-v3-tts-4b"

    def test_env_selects_higgs(self, monkeypatch):
        monkeypatch.setenv("TTS_ENGINE", "higgs")
        assert Config().tts_engine == "higgs"

    def test_env_overrides_vieneu_params(self, monkeypatch):
        monkeypatch.setenv("VIENEU_MODEL", "me/custom")
        monkeypatch.setenv("VIENEU_TEMPERATURE", "0.5")
        cfg = Config()
        assert cfg.vieneu_model == "me/custom"
        assert cfg.vieneu_temperature == 0.5


class TestSonioxConfig:
    def test_soniox_defaults_populated(self, monkeypatch):
        _clean_soniox_env(monkeypatch)
        cfg = Config()
        assert cfg.tts_engine == "soniox"
        assert cfg.soniox_api_key == ""
        assert cfg.soniox_base_url == "https://tts-rt.soniox.com"
        assert cfg.soniox_model == "tts-rt-v1"
        assert cfg.soniox_sample_rate == 24000
        assert cfg.soniox_audio_format == "wav"
        assert cfg.soniox_timeout == 120.0
        assert cfg.soniox_default_voice == "Adrian"
        assert cfg.soniox_voice_pool == ["Adrian", "Maya", "Noah", "Nina", "Jack", "Emma"]
        assert cfg.soniox_voice_map == {}

    def test_voice_map_parses_with_whitespace(self, monkeypatch):
        _clean_soniox_env(monkeypatch)
        monkeypatch.setenv("SONIOX_VOICE_MAP", "A=Adrian, B=Maya")
        assert Config().soniox_voice_map == {"A": "Adrian", "B": "Maya"}

    def test_voice_map_skips_malformed_entries(self, monkeypatch):
        _clean_soniox_env(monkeypatch)
        # "C" has no "=", "=Nina" has no speaker, "D=" has no voice — all skipped.
        monkeypatch.setenv("SONIOX_VOICE_MAP", "A=Adrian,C,=Nina,D=")
        assert Config().soniox_voice_map == {"A": "Adrian"}

    def test_voice_pool_parses_to_ordered_list(self, monkeypatch):
        _clean_soniox_env(monkeypatch)
        monkeypatch.setenv("SONIOX_VOICE_POOL", "Maya,Noah")
        assert Config().soniox_voice_pool == ["Maya", "Noah"]

    def test_soniox_params_env_overridable(self, monkeypatch):
        _clean_soniox_env(monkeypatch)
        monkeypatch.setenv("SONIOX_SAMPLE_RATE", "48000")
        monkeypatch.setenv("SONIOX_DEFAULT_VOICE", "Maya")
        cfg = Config()
        assert cfg.soniox_sample_rate == 48000
        assert cfg.soniox_default_voice == "Maya"

    def test_tts_uses_cloning_by_engine(self):
        assert Config(tts_engine="vieneu").tts_uses_cloning is True
        assert Config(tts_engine="higgs").tts_uses_cloning is True
        assert Config(tts_engine="soniox").tts_uses_cloning is False
        assert Config(tts_engine="gemini").tts_uses_cloning is False


class TestEngineCli:
    def _run(self, monkeypatch, tmp_path, argv_extra, env=None):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00" * 64)
        workdir = tmp_path / "work"
        captured = {}
        monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)

        class _FakeGraph:
            def invoke(self, state, config):
                return {"output_path": "o", "srt_src": "a", "srt_target": "b"}

        def capture(cfg, timings=None):
            captured["cfg"] = cfg
            return _FakeGraph()

        monkeypatch.setattr(cli, "build_graph", capture)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        monkeypatch.setattr("sys.argv", ["loro", str(video), "-w", str(workdir), *argv_extra])
        with pytest.raises(SystemExit):
            cli.main()
        return captured["cfg"]

    def test_cli_overrides_env(self, monkeypatch, tmp_path):
        cfg = self._run(monkeypatch, tmp_path, ["--tts-engine", "higgs"],
                        env={"TTS_ENGINE": "vieneu"})
        assert cfg.tts_engine == "higgs"

    def test_cli_selects_soniox_over_env(self, monkeypatch, tmp_path):
        cfg = self._run(monkeypatch, tmp_path, ["--tts-engine", "soniox"],
                        env={"TTS_ENGINE": "vieneu"})
        assert cfg.tts_engine == "soniox"

    def test_cli_selects_gemini(self, monkeypatch, tmp_path):
        # argparse accepts the new choice and engine_override carries it to Config.
        cfg = self._run(monkeypatch, tmp_path, ["--tts-engine", "gemini"],
                        env={"TTS_ENGINE": "soniox"})
        assert cfg.tts_engine == "gemini"

    def test_cli_absent_uses_env(self, monkeypatch, tmp_path):
        cfg = self._run(monkeypatch, tmp_path, [], env={"TTS_ENGINE": "higgs"})
        assert cfg.tts_engine == "higgs"

    def test_invalid_engine_rejected_by_argparse(self, monkeypatch, tmp_path):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00")
        monkeypatch.setattr("sys.argv", ["loro", str(video), "--tts-engine", "nope"])
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        assert exc_info.value.code == 2  # argparse usage error


class TestTtsProviderSurface:
    """U3: per-engine client construction, capability flags, and chunk budget now
    live on the provider; the node reads them instead of branching on the engine
    name. The node's _tts_client/_engine_inputs/_chunk_budget are thin shims."""

    def test_client_type_matches_old_factory(self, tmp_path):
        ref = tmp_path / "ref.wav"
        assert isinstance(providers.tts("vieneu").client(
            Config(tts_engine="vieneu"), ref, "t"), VieNeuClient)
        assert isinstance(providers.tts("higgs").client(
            Config(tts_engine="higgs"), ref, "t"), HiggsClient)
        assert isinstance(providers.tts("soniox").client(
            Config(tts_engine="soniox")), SonioxClient)
        assert isinstance(providers.tts("gemini").client(
            Config(tts_engine="gemini")), GeminiClient)

    def test_preset_client_ignores_reference_args(self, tmp_path):
        # The preset engines share the client(cfg, ref, text) surface but ignore
        # the reference (KTD1) — passing one still yields a preset client.
        ref = tmp_path / "ref.wav"
        assert isinstance(providers.tts("soniox").client(
            Config(tts_engine="soniox"), ref, "t"), SonioxClient)
        assert isinstance(providers.tts("gemini").client(
            Config(tts_engine="gemini"), ref, "t"), GeminiClient)

    def test_chunk_budget_per_engine(self):
        gcfg = Config(tts_engine="gemini")
        assert providers.tts("gemini").chunk_budget(gcfg) == gcfg.gemini_batch_max_syllables
        for eng in ("vieneu", "higgs", "soniox"):
            cfg = Config(tts_engine=eng)
            assert providers.tts(eng).chunk_budget(cfg) == cfg.tts_max_chunk_syllables

    def test_node_shims_delegate_to_active_provider(self):
        for eng in ("vieneu", "higgs", "soniox", "gemini"):
            cfg = Config(tts_engine=eng)
            assert _engine_inputs(cfg) == providers.tts(eng).engine_inputs(cfg)
            assert _chunk_budget(cfg) == providers.tts(eng).chunk_budget(cfg)


class TestClientFactory:
    def test_factory_returns_vieneu(self, tmp_path):
        client = _tts_client(Config(tts_engine="vieneu"), tmp_path / "ref.wav", "t")
        assert isinstance(client, VieNeuClient)

    def test_factory_returns_higgs(self, tmp_path):
        client = _tts_client(Config(tts_engine="higgs"), tmp_path / "ref.wav", "t")
        assert isinstance(client, HiggsClient)

    def test_factory_returns_soniox(self):
        # The preset engine takes no reference args (KTD1).
        client = _tts_client(Config(tts_engine="soniox"))
        assert isinstance(client, SonioxClient)

    def test_factory_returns_gemini(self):
        client = _tts_client(Config(tts_engine="gemini"))
        assert isinstance(client, GeminiClient)


class TestEngineFingerprint:
    def test_engine_changes_fingerprint(self):
        vi = fingerprint(_engine_inputs(Config(tts_engine="vieneu")))
        hi = fingerprint(_engine_inputs(Config(tts_engine="higgs")))
        assert vi != hi

    def test_vieneu_temperature_changes_fingerprint(self):
        a = fingerprint(_engine_inputs(Config(tts_engine="vieneu", vieneu_temperature=0.8)))
        b = fingerprint(_engine_inputs(Config(tts_engine="vieneu", vieneu_temperature=0.5)))
        assert a != b

    def test_vieneu_model_changes_fingerprint(self):
        a = fingerprint(_engine_inputs(Config(tts_engine="vieneu", vieneu_model="me/a")))
        b = fingerprint(_engine_inputs(Config(tts_engine="vieneu", vieneu_model="me/b")))
        assert a != b

    def test_higgs_only_field_does_not_change_vieneu_fingerprint(self):
        a = fingerprint(_engine_inputs(Config(tts_engine="vieneu", higgs_model="m1")))
        b = fingerprint(_engine_inputs(Config(tts_engine="vieneu", higgs_model="m2")))
        assert a == b

    def test_engine_switch_invalidates_existing_clip(self, tmp_path):
        # R4 / AE2: a clip synthesized under one engine must not be reused under
        # the other — the fingerprint mismatch drives a resynthesis.
        art = tmp_path / "seg_0000.wav"
        base = {"text_vi": "Xin chào", "ref_sha": "abc", "ref_text": "hi"}
        vi_inputs = {**base, **_engine_inputs(Config(tts_engine="vieneu"))}
        hi_inputs = {**base, **_engine_inputs(Config(tts_engine="higgs"))}
        artifacts.produce(art, vi_inputs, "tts", lambda tmp: tmp.write_bytes(b"RIFFfake"))
        assert artifacts.is_valid(art, vi_inputs)        # same engine reuses
        assert not artifacts.is_valid(art, hi_inputs)    # other engine re-synthesizes

    # --- preset (soniox) fingerprint (U4) ---

    def test_soniox_engine_inputs_shape(self):
        # The preset engine identity carries model/language/sample_rate and no
        # ref or voice fields (the per-segment voice is folded in by the node).
        ei = _engine_inputs(Config(tts_engine="soniox"))
        assert ei["engine"] == "soniox"
        assert ei["model"] == "tts-rt-v1"
        assert ei["language"] == "vi"
        assert ei["sample_rate"] == 24000
        assert "ref_sha" not in ei and "ref_text" not in ei and "voice" not in ei

    def test_soniox_sample_rate_changes_fingerprint(self):
        a = fingerprint(_engine_inputs(Config(tts_engine="soniox", soniox_sample_rate=24000)))
        b = fingerprint(_engine_inputs(Config(tts_engine="soniox", soniox_sample_rate=48000)))
        assert a != b

    def test_soniox_language_changes_fingerprint(self):
        # The spoken language is now profile-derived from --target-lang (U9), so a
        # different target language (-> different tts_language_code) changes the
        # clip fingerprint; the VI default keeps "vi" (byte-identical).
        a = fingerprint(_engine_inputs(Config(tts_engine="soniox", target_lang="vi")))
        b = fingerprint(_engine_inputs(Config(tts_engine="soniox", target_lang="fr")))
        assert a != b

    def test_soniox_synthesizes_in_the_target_language(self):
        # R15/U9: the preset engine's spoken-language param is profile-derived, so
        # one --target-lang fr makes the soniox engine_inputs carry language="fr"
        # (not the engine's vi default).
        assert _engine_inputs(Config(tts_engine="soniox"))["language"] == "vi"
        assert _engine_inputs(
            Config(tts_engine="soniox", target_lang="fr"))["language"] == "fr"
        assert _engine_inputs(
            Config(tts_engine="soniox", target_lang="es-MX"))["language"] == "es"

    def test_soniox_audio_format_changes_fingerprint(self):
        # audio_format shapes the emitted bytes, so it must be part of clip
        # identity (else changing it silently reuses the wrong-format clip).
        a = fingerprint(_engine_inputs(Config(tts_engine="soniox", soniox_audio_format="wav")))
        b = fingerprint(_engine_inputs(Config(tts_engine="soniox", soniox_audio_format="mp3")))
        assert a != b

    def test_soniox_model_changes_fingerprint(self):
        a = fingerprint(_engine_inputs(Config(tts_engine="soniox", soniox_model="tts-rt-v1")))
        b = fingerprint(_engine_inputs(Config(tts_engine="soniox", soniox_model="tts-rt-v2")))
        assert a != b

    def test_soniox_differs_from_cloning_engines(self):
        so = fingerprint(_engine_inputs(Config(tts_engine="soniox")))
        vi = fingerprint(_engine_inputs(Config(tts_engine="vieneu")))
        hi = fingerprint(_engine_inputs(Config(tts_engine="higgs")))
        assert so != vi and so != hi

    def test_voice_changes_preset_clip_fingerprint(self):
        base = {"text_vi": "Xin chào", **_engine_inputs(Config(tts_engine="soniox"))}
        assert fingerprint({**base, "voice": "Adrian"}) != fingerprint({**base, "voice": "Maya"})

    # --- gemini fingerprint (U5) ---

    def test_gemini_engine_inputs_shape(self):
        ei = _engine_inputs(Config(tts_engine="gemini"))
        assert ei["engine"] == "gemini"
        assert ei["model"] == "gemini-3.1-flash-tts-preview"
        assert ei["sample_rate"] == 24000
        assert "batch_max_syllables" in ei and "style_prompt" in ei
        assert "ref_sha" not in ei and "voice" not in ei

    def test_gemini_model_and_style_change_fingerprint(self):
        base = fingerprint(_engine_inputs(Config(tts_engine="gemini")))
        model = fingerprint(_engine_inputs(Config(tts_engine="gemini", gemini_model="gemini-2.5-flash-preview-tts")))
        style = fingerprint(_engine_inputs(Config(tts_engine="gemini", gemini_style_prompt="warm")))
        assert base != model and base != style

    def test_gemini_differs_from_other_engines(self):
        ge = fingerprint(_engine_inputs(Config(tts_engine="gemini")))
        so = fingerprint(_engine_inputs(Config(tts_engine="soniox")))
        vi = fingerprint(_engine_inputs(Config(tts_engine="vieneu")))
        hi = fingerprint(_engine_inputs(Config(tts_engine="higgs")))
        assert ge != so and ge != vi and ge != hi

    def _gemini_clip_fp(self, **cfg_kw):
        from loro.nodes.tts import _seg_inputs
        cfg = Config(tts_engine="gemini", **cfg_kw)
        seg = Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="Xin chào", speaker="A")
        inputs, _ = _seg_inputs(seg, cfg, False, None, None, {"A": "Kore"})
        return fingerprint(inputs)

    def test_gemini_clip_excludes_chunking_keys(self):
        # Sc3: changing the Higgs/VieNeu chunk knobs must NOT change a Gemini
        # clip's fingerprint (those keys are omitted for Gemini, KTD5)...
        base = self._gemini_clip_fp()
        assert base == self._gemini_clip_fp(tts_max_chunk_syllables=999)
        assert base == self._gemini_clip_fp(tts_chunk_gap_ms=999.0)
        assert base == self._gemini_clip_fp(tts_hardwrap_gap_ms=999.0)

    def test_gemini_clip_batch_max_syllables_changes_fingerprint(self):
        # ...while changing Gemini's own effective chunk budget DOES.
        assert self._gemini_clip_fp(gemini_batch_max_syllables=360) != \
            self._gemini_clip_fp(gemini_batch_max_syllables=120)

    def test_soniox_clip_invalid_under_vieneu(self, tmp_path):
        # A clip made under soniox must resynthesize under a cloning engine for
        # identical text (engine in the key), and the preset key carries no ref.
        art = tmp_path / "seg_0000.wav"
        soniox_inputs = {"text_vi": "Xin chào", "voice": "Adrian",
                         **_engine_inputs(Config(tts_engine="soniox"))}
        vieneu_inputs = {"text_vi": "Xin chào", "ref_sha": "abc", "ref_text": "hi",
                         **_engine_inputs(Config(tts_engine="vieneu"))}
        artifacts.produce(art, soniox_inputs, "tts", lambda tmp: tmp.write_bytes(b"RIFFfake"))
        assert artifacts.is_valid(art, soniox_inputs)
        assert not artifacts.is_valid(art, vieneu_inputs)


class TestPresetVoices:
    def test_soniox_triple(self):
        cfg = Config(tts_engine="soniox", soniox_voice_pool=["Adrian", "Maya"],
                     soniox_voice_map={"A": "Maya"}, soniox_default_voice="Adrian")
        pv = cfg.preset_voices
        assert pv.pool == ["Adrian", "Maya"]
        assert pv.voice_map == {"A": "Maya"}
        assert pv.default == "Adrian"

    def test_gemini_triple(self):
        cfg = Config(tts_engine="gemini", gemini_voice_pool=["Kore", "Puck"],
                     gemini_voice_map={"A": "Puck"}, gemini_default_voice="Kore")
        pv = cfg.preset_voices
        assert pv.pool == ["Kore", "Puck"]
        assert pv.voice_map == {"A": "Puck"}
        assert pv.default == "Kore"


class _StubClient:
    """Records synthesize(text, voice) and writes placeholder bytes."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synthesize(self, text, output, voice=None):
        self.calls.append((text, voice))
        from pathlib import Path
        Path(output).write_bytes(b"RIFFfakewavbytes")


class TestSonioxTtsNode:
    def _run(self, monkeypatch, tmp_path, segments, voice_cast, cfg, stub):
        # Inject the stub client and neutralize the QA gate (its own tests cover
        # it); the node test isolates voice threading + fingerprint identity.
        monkeypatch.setattr(tts_mod, "_tts_client", lambda c, *a, **k: stub)
        monkeypatch.setattr(tts_mod.qa, "check_clip", lambda *a, **k: None)
        state = {"workdir": str(tmp_path), "segments": segments, "voice_cast": voice_cast}
        return tts(state, cfg)

    def _cfg(self):
        return Config(tts_engine="soniox", soniox_default_voice="Adrian",
                      retry_base_delay=0.0)

    def _segs(self):
        return [Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="Xin chào", speaker="A"),
                Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="Tạm biệt", speaker="B")]

    def test_each_segment_synthesized_in_its_cast_voice(self, monkeypatch, tmp_path):
        stub = _StubClient()
        out = self._run(monkeypatch, tmp_path, self._segs(),
                        {"A": "Adrian", "B": "Maya"}, self._cfg(), stub)
        assert stub.calls == [("Xin chào", "Adrian"), ("Tạm biệt", "Maya")]
        assert all(seg.tts_wav for seg in out["segments"])

    def test_repinning_one_speaker_invalidates_only_its_clip(self, monkeypatch, tmp_path):
        # R8: re-pinning A's voice resynthesizes only A; B's clip stays cached.
        cfg = self._cfg()
        stub = _StubClient()
        self._run(monkeypatch, tmp_path, self._segs(),
                  {"A": "Adrian", "B": "Maya"}, cfg, stub)
        stub.calls.clear()
        self._run(monkeypatch, tmp_path, self._segs(),
                  {"A": "Grace", "B": "Maya"}, cfg, stub)
        assert stub.calls == [("Xin chào", "Grace")]

    def test_unmapped_and_empty_speaker_use_default_voice(self, monkeypatch, tmp_path):
        # A "" (no-diarization) segment and a speaker missing from the cast both
        # fall back to soniox_default_voice; a run without ref_audio/ref_text in
        # state never KeyErrors (the cloning-only reads are gated).
        segs = [Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="Chào", speaker=""),
                Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="Hế lô", speaker="Z")]
        stub = _StubClient()
        out = self._run(monkeypatch, tmp_path, segs, {"": "Adrian"}, self._cfg(), stub)
        assert stub.calls == [("Chào", "Adrian"), ("Hế lô", "Adrian")]
        assert all(seg.tts_wav for seg in out["segments"])

    def test_gemini_default_voice_fallback_via_preset_voices(self, monkeypatch, tmp_path):
        # Under the Gemini engine the node's default-voice fallback resolves to
        # gemini_default_voice through cfg.preset_voices (KTD6), not the Soniox
        # default. (batch_segments=1 keeps this on the per-segment path.)
        cfg = Config(tts_engine="gemini", gemini_default_voice="Kore",
                     gemini_batch_segments=1, retry_base_delay=0.0)
        segs = [Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="Chào", speaker="")]
        stub = _StubClient()
        out = self._run(monkeypatch, tmp_path, segs, {}, cfg, stub)
        assert stub.calls == [("Chào", "Kore")]
        assert all(seg.tts_wav for seg in out["segments"])


def _audio(dur=0.5, sr=24000):
    return np.full(int(dur * sr), 0.1, dtype="float32")


def _write_wav(path, dur=0.5, sr=24000):
    sf.write(str(path), _audio(dur, sr), sr)


class _GeminiStub:
    """Records single-call and batch-call usage. synthesize writes a real short
    WAV; synthesize_batch returns one real array per turn (or raises SplitError
    to force fallback)."""

    def __init__(self, split_error=False, piece_dur=0.5):
        self.synth_calls = []
        self.batch_calls = []
        self.split_error = split_error
        self.piece_dur = piece_dur

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def synthesize(self, text, output, voice=None):
        self.synth_calls.append((text, voice))
        _write_wav(output, self.piece_dur)

    def synthesize_batch(self, turns):
        self.batch_calls.append(list(turns))
        if self.split_error:
            raise SplitError("forced split failure")
        return [_audio(self.piece_dur) for _ in turns], 24000


def _gseg(index, speaker, text_target):
    return Segment(index=index, start=float(index), end=float(index) + 1.0,
                   text_src=f"s{index}", text_target=text_target, speaker=speaker)


class TestGeminiBatchedTtsNode:
    def _run(self, monkeypatch, tmp_path, segments, voice_cast, cfg, stub,
             neutralize_qa=True):
        monkeypatch.setattr(tts_mod, "_tts_client", lambda c, *a, **k: stub)
        if neutralize_qa:
            monkeypatch.setattr(tts_mod.qa, "check_clip", lambda *a, **k: None)
        state = {"workdir": str(tmp_path), "segments": segments,
                 "voice_cast": voice_cast}
        return tts(state, cfg)

    def _cfg(self, **kw):
        base = {"tts_engine": "gemini", "gemini_default_voice": "Kore",
                "gemini_batch_segments": 8, "retry_base_delay": 0.0}
        base.update(kw)
        return Config(**base)

    def test_split_failure_falls_back_to_per_segment(self, monkeypatch, tmp_path):
        # FALLBACK CONTRACT (execution note): a batch whose split returns the
        # wrong count must produce the same per-segment artifacts and ledger
        # state as a pure per-segment run — never a skipped or missing clip.
        segs = [_gseg(0, "A", "một"), _gseg(1, "A", "hai"), _gseg(2, "A", "ba")]
        stub = _GeminiStub(split_error=True)
        out = self._run(monkeypatch, tmp_path, segs, {"A": "Kore"}, self._cfg(), stub)
        assert len(stub.batch_calls) == 1   # one batch attempt
        assert stub.synth_calls == [("một", "Kore"), ("hai", "Kore"), ("ba", "Kore")]
        assert all(s.tts_wav for s in out["segments"])
        assert not any(s.skipped for s in out["segments"])

    def test_happy_batch_one_call_for_n_segments(self, monkeypatch, tmp_path):
        # A batch of N to-do segments yields N artifacts from ONE synthesize_batch
        # call (the whole point — call minimization), each recorded ok.
        segs = [_gseg(0, "A", "một"), _gseg(1, "B", "hai"), _gseg(2, "A", "ba")]
        stub = _GeminiStub()
        out = self._run(monkeypatch, tmp_path, segs, {"A": "Kore", "B": "Puck"},
                        self._cfg(), stub)
        assert len(stub.batch_calls) == 1
        assert len(stub.batch_calls[0]) == 3        # all three turns in one call
        assert stub.synth_calls == []               # no per-segment calls
        assert all((tmp_path / "tts" / f"seg_{s.index:04d}.wav").exists()
                   for s in out["segments"])
        assert all(s.tts_wav and not s.skipped for s in out["segments"])

    def test_qa_failure_on_one_clip_falls_back(self, monkeypatch, tmp_path):
        # R5: one split clip failing QA fails the whole batch's atomic gate, so
        # it falls back to per-segment (where each clip is QA'd on its own and
        # the good per-segment audio passes). QA is REAL here.
        class _BadPieceStub(_GeminiStub):
            def synthesize_batch(self, turns):
                self.batch_calls.append(list(turns))
                pieces = [_audio(self.piece_dur) for _ in turns]
                pieces[1] = np.zeros(int(0.5 * 24000), dtype="float32")  # silent -> QA fail
                return pieces, 24000

        segs = [_gseg(0, "A", "một"), _gseg(1, "A", "hai"), _gseg(2, "A", "ba")]
        stub = _BadPieceStub()
        out = self._run(monkeypatch, tmp_path, segs, {"A": "Kore"}, self._cfg(), stub,
                        neutralize_qa=False)
        assert len(stub.batch_calls) == 1
        assert stub.synth_calls == [("một", "Kore"), ("hai", "Kore"), ("ba", "Kore")]
        assert all(s.tts_wav and not s.skipped for s in out["segments"])

    def test_batch_segments_one_uses_per_segment_path(self, monkeypatch, tmp_path):
        # GEMINI_BATCH_SEGMENTS=1 never batches: each segment is a single call.
        cfg = self._cfg(gemini_batch_segments=1)
        segs = [_gseg(0, "A", "một"), _gseg(1, "B", "hai")]
        stub = _GeminiStub()
        out = self._run(monkeypatch, tmp_path, segs, {"A": "Kore", "B": "Puck"}, cfg, stub)
        assert stub.batch_calls == []
        assert stub.synth_calls == [("một", "Kore"), ("hai", "Puck")]
        assert all(s.tts_wav for s in out["segments"])

    def test_skipped_and_empty_segments_excluded_from_batch(self, monkeypatch, tmp_path):
        # An upstream-skipped segment and one with empty text_target are never
        # batched or synthesized.
        s0 = _gseg(0, "A", "một")
        s1 = _gseg(1, "A", "")           # empty translation
        s2 = _gseg(2, "A", "ba")
        s3 = _gseg(3, "A", "bốn")
        s3.skipped = True                 # upstream skip
        stub = _GeminiStub()
        self._run(monkeypatch, tmp_path, [s0, s1, s2, s3], {"A": "Kore"},
                  self._cfg(), stub)
        # Only s0 and s2 reach the one batch call.
        assert len(stub.batch_calls) == 1
        assert [t[1] for t in stub.batch_calls[0]] == ["một", "ba"]

    def test_cache_reuse_batches_only_invalid_segments(self, monkeypatch, tmp_path):
        # R11: a rerun where 2 of 4 clips are still valid batches only the 2
        # changed segments; the valid clips are reused without any call.
        cfg = self._cfg()
        segs1 = [_gseg(0, "A", "một"), _gseg(1, "A", "hai"),
                 _gseg(2, "A", "ba"), _gseg(3, "A", "bốn")]
        stub1 = _GeminiStub()
        self._run(monkeypatch, tmp_path, segs1, {"A": "Kore"}, cfg, stub1)
        assert len(stub1.batch_calls) == 1 and len(stub1.batch_calls[0]) == 4

        # Rerun: segments 1 and 3 get new text (invalid); 0 and 2 unchanged.
        segs2 = [_gseg(0, "A", "một"), _gseg(1, "A", "HAI-moi"),
                 _gseg(2, "A", "ba"), _gseg(3, "A", "BON-moi")]
        stub2 = _GeminiStub()
        out = self._run(monkeypatch, tmp_path, segs2, {"A": "Kore"}, cfg, stub2)
        assert len(stub2.batch_calls) == 1
        assert [t[1] for t in stub2.batch_calls[0]] == ["HAI-moi", "BON-moi"]
        assert stub2.synth_calls == []
        assert all(s.tts_wav for s in out["segments"])


class TestGeminiBatchGrouping:
    def _todo(self, *segs):
        return [(s, None, None, None, "Kore") for s in segs]

    def test_respects_segment_count_cap(self):
        from loro.nodes.tts import _group_batches
        cfg = Config(tts_engine="gemini", gemini_batch_segments=2,
                     gemini_batch_max_syllables=10_000)
        segs = [_gseg(i, "A", "một") for i in range(5)]
        batches = _group_batches(self._todo(*segs), cfg)
        assert [len(b) for b in batches] == [2, 2, 1]

    def test_respects_syllable_budget(self):
        from loro.nodes.tts import _group_batches
        cfg = Config(tts_engine="gemini", gemini_batch_segments=99,
                     gemini_batch_max_syllables=3)
        # 2-syllable segments; budget 3 -> only one per batch (2+2 > 3).
        segs = [_gseg(i, "A", "xin chào") for i in range(3)]
        batches = _group_batches(self._todo(*segs), cfg)
        assert [len(b) for b in batches] == [1, 1, 1]

    def test_caps_distinct_speakers_at_two(self):
        from loro.nodes.tts import _group_batches
        cfg = Config(tts_engine="gemini", gemini_batch_segments=99,
                     gemini_batch_max_syllables=10_000)
        # A, A, B fit (2 distinct); C would be the 3rd -> new batch.
        segs = [_gseg(0, "A", "a"), _gseg(1, "A", "b"),
                _gseg(2, "B", "c"), _gseg(3, "C", "d")]
        batches = _group_batches(self._todo(*segs), cfg)
        assert [[t[0].speaker for t in b] for b in batches] == [["A", "A", "B"], ["C"]]

    def test_preserves_order(self):
        from loro.nodes.tts import _group_batches
        cfg = Config(tts_engine="gemini", gemini_batch_segments=2,
                     gemini_batch_max_syllables=10_000)
        segs = [_gseg(i, "A", f"t{i}") for i in range(4)]
        batches = _group_batches(self._todo(*segs), cfg)
        flat = [t[0].index for b in batches for t in b]
        assert flat == [0, 1, 2, 3]


class TestMeasuredDurationGate:
    """U6: the measured-clip-duration-vs-slot gate for non-VI profiles, its
    length_overflow recording, the re-translation escalation, and its cache-first
    determinism. The TTS client + ffmpeg.probe_duration are mocked so the tests do
    not depend on real audio or calibrated constants."""

    def _segs(self, n=1, end=2.0):
        return [Segment(index=i, start=0.0, end=end, text_src=f"s{i}",
                        text_target=f"Bonjour le monde {i}", speaker="A")
                for i in range(n)]

    def _state(self, tmp_path, segs):
        return {"workdir": str(tmp_path), "segments": segs,
                "voice_cast": {"A": "Adrian"}, "words": []}

    def _setup(self, monkeypatch, stub, probe):
        monkeypatch.setattr(tts_mod, "_tts_client", lambda c, *a, **k: stub)
        monkeypatch.setattr(tts_mod.qa, "check_clip", lambda *a, **k: None)
        monkeypatch.setattr(tts_mod.ffmpeg, "probe_duration", probe)

    def _fr_cfg(self, **kw):
        return Config(tts_engine="soniox", target_lang="fr", soniox_api_key="k",
                      retry_base_delay=0.0, **kw)

    def test_vi_clip_never_records_length_overflow(self, monkeypatch, tmp_path):
        # R19: the gate is inactive for VI even for a wildly over-slot clip.
        self._setup(monkeypatch, _StubClient(), lambda p: 99.0)
        tts(self._state(tmp_path, self._segs()),
            Config(tts_engine="soniox", retry_base_delay=0.0))  # target vi
        assert SkipLedger(tmp_path).entries() == {}

    def test_fr_over_slot_records_length_overflow_keeps_clip(self, monkeypatch, tmp_path):
        # R8/R7 baseline: an over-slot FR clip is KEPT best-effort + length_overflow
        # (escalation off by default).
        self._setup(monkeypatch, _StubClient(), lambda p: 10.0)  # 10s clip, 2s slot
        out = tts(self._state(tmp_path, self._segs()), self._fr_cfg())
        seg = out["segments"][0]
        assert not seg.skipped and seg.tts_wav
        assert SkipLedger(tmp_path).entries()["seg_0000"]["status"] == "length_overflow"

    def test_fr_in_slot_clip_no_overflow(self, monkeypatch, tmp_path):
        self._setup(monkeypatch, _StubClient(), lambda p: 2.0)  # fits the 2s slot
        tts(self._state(tmp_path, self._segs()), self._fr_cfg())
        assert SkipLedger(tmp_path).entries() == {}

    def test_escalation_retranslates_shorter_and_regenerates_srt(self, monkeypatch, tmp_path):
        # R6: with escalation on, an over-slot segment re-translates shorter and
        # converges; the target SRT is regenerated so subtitles match the audio.
        from loro.nodes import translate as tr
        seen = {"n": 0}
        monkeypatch.setattr(tr, "translate_segment", lambda c, s, ctx, b: "Court")

        def probe(p):
            seen["n"] += 1
            return 10.0 if seen["n"] == 1 else 2.0  # over once, then fits

        self._setup(monkeypatch, _StubClient(), probe)
        out = tts(self._state(tmp_path, self._segs()),
                  self._fr_cfg(enable_budget_retry=True, budget_retry_max=2))
        assert out["segments"][0].text_target == "Court"
        srt_txt = (tmp_path / "transcript.fr.srt").read_text(encoding="utf-8")
        assert "Court" in srt_txt and "Bonjour le monde 0" not in srt_txt

    def test_never_converging_caps_and_is_muxable(self, monkeypatch, tmp_path):
        # R7: a segment that never fits terminates at the cap, is length_overflow,
        # stays muxable, and its attempts never trip the abort window (no AbortRun).
        from loro.nodes import translate as tr
        n = {"i": 0}
        monkeypatch.setattr(tr, "translate_segment",
                            lambda c, s, ctx, b: f"tres tres long {(n.__setitem__('i', n['i']+1)) or n['i']}")
        self._setup(monkeypatch, _StubClient(), lambda p: 10.0)  # always over
        out = tts(self._state(tmp_path, self._segs()),
                  self._fr_cfg(enable_budget_retry=True, budget_retry_max=2))
        seg = out["segments"][0]
        assert not seg.skipped and seg.tts_wav  # still muxable
        assert SkipLedger(tmp_path).entries()["seg_0000"]["status"] == "length_overflow"

    def test_determinism_second_run_no_resynth_no_remeasure(self, monkeypatch, tmp_path):
        # Determinism: run 2 takes the cached converged-text path BEFORE any
        # synthesis or measurement, and hits the clip cache (no re-bill).
        from loro.nodes import translate as tr
        monkeypatch.setattr(tr, "translate_segment", lambda c, s, ctx, b: "Court")
        cfg = self._fr_cfg(enable_budget_retry=True, budget_retry_max=2)
        seen = {"n": 0}
        self._setup(monkeypatch, _StubClient(),
                    lambda p: 10.0 if (seen.__setitem__("n", seen["n"] + 1) or seen["n"]) == 1 else 2.0)
        tts(self._state(tmp_path, self._segs()), cfg)  # run 1 converges + caches

        stub2 = _StubClient()
        probe_calls = {"n": 0}
        self._setup(monkeypatch, stub2,
                    lambda p: (probe_calls.__setitem__("n", probe_calls["n"] + 1)) or 2.0)
        tts(self._state(tmp_path, self._segs()), cfg)  # run 2
        assert stub2.calls == []        # clip cache hit -> no re-synthesis / re-bill
        assert probe_calls["n"] == 0    # converged-text cache hit -> no re-measure


def test_length_overflow_excluded_from_abort_window(tmp_path):
    # R7: best-effort length overflows never count toward the systemic-failure
    # abort window, no matter how many fire.
    led = SkipLedger(tmp_path, window=20, abort_threshold=3)
    for i in range(10):
        led.record_length_overflow(f"seg_{i:04d}")  # must never raise AbortRun
    entry = led.entries()["seg_0000"]
    assert entry["status"] == "length_overflow" and entry["reason"] == "length_overflow"
    assert led.should_attempt("seg_0000", "anyhash") is True  # not a skip
