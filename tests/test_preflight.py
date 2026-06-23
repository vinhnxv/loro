import pytest

from loro.config import Config
from loro.harness import preflight as pf
from loro.providers.asr import assemblyai as assemblyai_asr
from loro.providers.asr import soniox as soniox_asr
from loro.providers.tts import gemini as gemini_tts
from loro.providers.tts import higgs as higgs_tts
from loro.providers.tts import soniox as soniox_tts


@pytest.fixture
def video(tmp_path):
    v = tmp_path / "in.mp4"
    v.write_bytes(b"\x00" * 64)
    return v


@pytest.fixture
def ok_env(monkeypatch, tmp_path):
    """Monkeypatch every external dependency to a healthy state."""
    monkeypatch.setattr(pf.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(pf.llm, "list_models",
                        lambda cfg, **kw: [cfg.llm_model, "other"])
    monkeypatch.setattr(pf.llm, "chat", lambda cfg, messages, **kw: "ok")
    monkeypatch.setattr(higgs_tts, "_probe_higgs", lambda cfg: None)
    # Audio probe extracts a real clip via ffmpeg; the dummy video has no real
    # audio, so return its path and let the mocked llm.chat carry the probe.
    monkeypatch.setattr(pf, "_extract_probe_clip", lambda video, workdir: video)
    nemo = tmp_path / "nemo-python"
    nemo.write_text("#!/bin/sh\n")
    granite = tmp_path / "granite-python"
    granite.write_text("#!/bin/sh\n")
    # The default engines are now cloud (asr=assemblyai, tts=soniox); these
    # checks exercise the local-ASR + on-device-TTS prerequisites, so pin
    # asr_engine=local and tts_engine=vieneu (vieneu adds no preflight TTS
    # check, the way the suite assumed before the default flip).
    return {"nemotron_python": str(nemo), "granite_python": str(granite),
            "asr_engine": "local", "tts_engine": "vieneu"}


def test_missing_granite_reported_when_cross_check_enabled(ok_env, video, tmp_path):
    env = {**ok_env, "granite_python": "/nonexistent/granite/python"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env, enable_cross_check=True), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "Granite" in msg
    assert "pyenv virtualenv" in msg


def test_missing_granite_ignored_when_cross_check_disabled(ok_env, video, tmp_path):
    env = {**ok_env, "granite_python": "/nonexistent/granite/python"}
    pf.preflight(Config(**env, enable_cross_check=False), video, tmp_path / "work")  # no raise


def test_all_good_passes(ok_env, video, tmp_path):
    cfg = Config(**ok_env)
    pf.preflight(cfg, video, tmp_path / "work")


def test_unprofiled_target_without_allow_fallback_fails(ok_env, video, tmp_path):
    # R4/AE3: a typo'd / unprofiled target must fail preflight before any billing.
    cfg = Config(**ok_env, target_lang="xx")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "unprofiled target language 'xx'" in msg
    assert "--allow-fallback" in msg


def test_unprofiled_target_with_allow_fallback_warns_and_passes(
        ok_env, video, tmp_path, caplog, monkeypatch):
    # R4/AE3: --allow-fallback proceeds on the generic profile with a loud warning.
    # Use a preset engine so the non-VI target exercises the profile gate, not the
    # VieNeu cloning gate (_soniox_env defined below; resolved at call time).
    import logging
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    cfg = Config(**_soniox_env(ok_env), target_lang="xx", allow_fallback=True)
    with caplog.at_level(logging.WARNING):
        pf.preflight(cfg, video, tmp_path / "work")  # no raise
    assert any("unprofiled" in r.message and "xx" in r.message for r in caplog.records)


def test_profiled_region_variant_passes(ok_env, video, tmp_path, monkeypatch):
    # es-MX resolves to the ES profile, so it is profiled (no flag needed).
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    cfg = Config(**_soniox_env(ok_env), target_lang="es-MX")
    pf.preflight(cfg, video, tmp_path / "work")


def test_local_engine_rejects_source_lang_auto(ok_env, video, tmp_path):
    # R12: the local engine has no language identification, so `auto` must fail
    # preflight (it requires an explicit --source-lang). ok_env is asr=local.
    cfg = Config(**ok_env, source_lang="auto")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    assert "--source-lang auto" in str(exc_info.value)


def test_cloud_engine_allows_source_lang_auto(video, tmp_path, monkeypatch):
    # The soniox/assemblyai engines detect language, so `auto` passes preflight.
    monkeypatch.setattr(pf.shutil, "which", lambda tool: f"/usr/bin/{tool}")
    monkeypatch.setattr(pf.llm, "list_models", lambda cfg, **kw: [cfg.llm_model, "other"])
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 404)
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 200)
    cfg = Config(asr_engine="soniox", tts_engine="soniox",
                 soniox_api_key="k", source_lang="auto")
    pf.preflight(cfg, video, tmp_path / "work")  # no raise


def test_vieneu_clone_in_non_vi_target_fails_preflight(ok_env, video, tmp_path):
    # R14/AE6: VieNeu can only clone Vietnamese; an FR target must be rejected
    # before billing (it has no preset fallback).
    cfg = Config(**ok_env, target_lang="fr")  # ok_env is tts_engine="vieneu"
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "cannot clone" in msg and "fr" in msg


def test_vieneu_vi_target_passes(ok_env, video, tmp_path):
    # VieNeu clones Vietnamese, so the default VI target is fine.
    pf.preflight(Config(**ok_env, target_lang="vi"), video, tmp_path / "work")


def test_target_switch_in_workdir_warns_stale_overrides(ok_env, video, tmp_path, caplog):
    # R21: work dirs are single-target; a prior target subtitle in the work dir +
    # a new target warns about stale per-target overrides/artifacts.
    import logging
    wd = tmp_path / "work"
    wd.mkdir()
    (wd / "transcript.en.srt").write_text("1\n", encoding="utf-8")   # source, ignored
    (wd / "transcript.vi.srt").write_text("1\n", encoding="utf-8")   # a PRIOR target
    cfg = Config(**ok_env, target_lang="vi")  # same target -> no warning
    with caplog.at_level(logging.WARNING):
        pf.preflight(cfg, video, wd)
    assert not any("single-target" in r.message for r in caplog.records)

    caplog.clear()
    # Switch the target to fr (soniox preset so fr clears the cloning gate).
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
        cfg_fr = Config(**_soniox_env(ok_env), target_lang="fr")
        with caplog.at_level(logging.WARNING):
            pf.preflight(cfg_fr, video, wd)
    assert any("single-target" in r.message and "vi" in r.message for r in caplog.records)


def test_non_english_source_warns_about_crosscheck(ok_env, video, tmp_path, caplog):
    # R3: a non-EN configured source with the English-tuned crosscheck warns.
    import logging
    cfg = Config(**ok_env, source_lang="de", enable_cross_check=True)
    with caplog.at_level(logging.WARNING):
        pf.preflight(cfg, video, tmp_path / "work")  # warns, no raise
    assert any("cross-check" in r.message and "de" in r.message for r in caplog.records)


def test_collects_all_problems_in_one_failure(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(pf.shutil, "which", lambda tool: None)
    cfg = Config(nemotron_python="/nonexistent/python", asr_engine="local")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "ffmpeg" in msg
    assert "ffprobe" in msg
    assert "/nonexistent/python" in msg


def test_dead_llm_reported(ok_env, video, tmp_path, monkeypatch):
    def down(cfg, **kw):
        raise ConnectionError("refused")
    monkeypatch.setattr(pf.llm, "list_models", down)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env), video, tmp_path / "work")
    assert "Model server" in str(exc_info.value)


def test_missing_model_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(pf.llm, "list_models", lambda cfg, **kw: ["some-other-model"])
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env), video, tmp_path / "work")
    assert Config().llm_model in str(exc_info.value)


def test_translate_model_unserved_reported(ok_env, video, tmp_path):
    # ok_env serves [llm_model, "other"]; a distinct translate model that is
    # not served must be reported by name (R38).
    cfg = Config(**ok_env, llm_model_translate="qwen3-14b-4bit")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    assert "qwen3-14b-4bit" in str(exc_info.value)


def test_translate_model_served_passes(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(pf.llm, "list_models",
                        lambda cfg, **kw: [cfg.llm_model, "qwen3-14b-4bit"])
    pf.preflight(Config(**ok_env, llm_model_translate="qwen3-14b-4bit"),
                 video, tmp_path / "work")  # no raise


def test_sentence_seg_model_unserved_reported(ok_env, video, tmp_path):
    # ok_env serves [llm_model, "other"]; a distinct sentence_seg model that
    # is not served must be reported by name (mirrors llm_model_translate).
    cfg = Config(**ok_env, llm_model_seg="gemma-seg-only")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    assert "gemma-seg-only" in str(exc_info.value)


def test_sentence_seg_model_served_passes(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(pf.llm, "list_models",
                        lambda cfg, **kw: [cfg.llm_model, "gemma-seg-only"])
    pf.preflight(Config(**ok_env, llm_model_seg="gemma-seg-only"),
                 video, tmp_path / "work")  # no raise


def test_translate_model_equal_gemma_not_double_checked(ok_env, video, tmp_path, monkeypatch):
    # Default config: llm_model_translate == llm_model. Even when the server
    # lists ONLY gemma, the equal-model branch must not raise a duplicate.
    monkeypatch.setattr(pf.llm, "list_models", lambda cfg, **kw: [cfg.llm_model])
    pf.preflight(Config(**ok_env), video, tmp_path / "work")  # no raise


def test_audio_probe_runs_when_cross_check_enabled(ok_env, video, tmp_path, monkeypatch):
    probes = []
    monkeypatch.setattr(pf, "_probe_audio_input", lambda cfg, video, workdir: probes.append(1))
    pf.preflight(Config(**ok_env, enable_cross_check=True), video, tmp_path / "work")
    assert probes == [1]


def test_audio_probe_skipped_when_cross_check_disabled(ok_env, video, tmp_path, monkeypatch):
    probes = []
    monkeypatch.setattr(pf, "_probe_audio_input", lambda cfg, video, workdir: probes.append(1))
    pf.preflight(Config(**ok_env, enable_cross_check=False), video, tmp_path / "work")
    assert probes == []


def test_audio_probe_failure_reported(ok_env, video, tmp_path, monkeypatch):
    def no_audio(cfg, messages, **kw):
        raise ValueError("audio content part rejected")
    monkeypatch.setattr(pf.llm, "chat", no_audio)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env, enable_cross_check=True), video, tmp_path / "work")
    assert "audio" in str(exc_info.value).lower()


# --- model split: llm_model_audio is served and actually hears audio (U3) ---

def test_audio_probe_uses_audio_model(ok_env, video, tmp_path, monkeypatch):
    # On the split profile the probe must target llm_model_audio, not llm_model.
    captured = []
    def fake_chat(cfg, messages, **kw):
        # The audio probe now names the audio role (U7); read its resolved model.
        captured.append(kw["role"].model if "role" in kw else kw.get("model"))
        return "ok"
    monkeypatch.setattr(pf.llm, "list_models",
                        lambda cfg, **kw: [cfg.llm_model, "gemma-4-12B"])
    monkeypatch.setattr(pf.llm, "chat", fake_chat)
    pf.preflight(Config(**ok_env, llm_model_audio="gemma-4-12B", enable_cross_check=True),
                 video, tmp_path / "work")
    assert captured == ["gemma-4-12B"]


def test_audio_model_unserved_reported(ok_env, video, tmp_path):
    # ok_env serves [llm_model, "other"]; a distinct audio model that is not
    # served must be reported by name (mirrors translate/sentence_seg).
    cfg = Config(**ok_env, llm_model_audio="gemma-4-12B", enable_cross_check=True)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    assert "gemma-4-12B" in str(exc_info.value)


def test_audio_model_equal_gemma_not_double_checked(ok_env, video, tmp_path, monkeypatch):
    # Default profile: llm_model_audio == llm_model. Even when the server lists
    # ONLY gemma, the equal-model branch must not raise a duplicate problem.
    monkeypatch.setattr(pf.llm, "list_models", lambda cfg, **kw: [cfg.llm_model])
    pf.preflight(Config(**ok_env, enable_cross_check=True), video, tmp_path / "work")  # no raise


def test_audio_model_not_checked_when_cross_check_disabled(ok_env, video, tmp_path):
    # Cross-check off: the audio model is never used, so an unserved one must
    # not block preflight.
    cfg = Config(**ok_env, llm_model_audio="gemma-4-12B", enable_cross_check=False)
    pf.preflight(cfg, video, tmp_path / "work")  # no raise


def test_audio_model_rejecting_audio_reported(ok_env, video, tmp_path, monkeypatch):
    # LLM_MODEL_AUDIO wrongly pointed at a served but no-audio model (e.g. the 26B
    # vision model): the probe catches it and names the model.
    monkeypatch.setattr(pf.llm, "list_models",
                        lambda cfg, **kw: [cfg.llm_model, "vision-26B"])
    def no_audio(cfg, messages, **kw):
        raise ValueError("audio content part rejected")
    monkeypatch.setattr(pf.llm, "chat", no_audio)
    cfg = Config(**ok_env, llm_model_audio="vision-26B", enable_cross_check=True)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "audio" in msg.lower()
    assert "vision-26B" in msg


# --- per-role hosts: a split endpoint is contacted independently ---

def test_audio_on_separate_host_checked_independently(ok_env, video, tmp_path, monkeypatch):
    # Audio split onto its own host: that host is queried separately and its
    # model checked there, not against the base host's served list.
    def list_by_host(cfg, host=None, **kw):
        if host == "http://audio:1234/v1":
            return ["gemma-4-12B-it-8bit"]
        return [cfg.llm_model, "other"]
    monkeypatch.setattr(pf.llm, "list_models", list_by_host)
    cfg = Config(**ok_env, enable_cross_check=True,
                 llm_host_audio="http://audio:1234/v1",
                 llm_model_audio="gemma-4-12B-it-8bit")
    pf.preflight(cfg, video, tmp_path / "work")  # no raise


def test_audio_host_down_reported(ok_env, video, tmp_path, monkeypatch):
    # The audio host being unreachable surfaces by host, and the audio probe is
    # skipped (not run against a dead endpoint).
    def list_by_host(cfg, host=None, **kw):
        if host == "http://audio:1234/v1":
            raise ConnectionError("refused")
        return [cfg.llm_model, "other"]
    monkeypatch.setattr(pf.llm, "list_models", list_by_host)
    cfg = Config(**ok_env, enable_cross_check=True,
                 llm_host_audio="http://audio:1234/v1",
                 llm_model_audio="gemma-4-12B-it-8bit")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(cfg, video, tmp_path / "work")
    assert "http://audio:1234/v1" in str(exc_info.value)


def test_higgs_probed_when_engine_higgs(ok_env, video, tmp_path, monkeypatch):
    # With Higgs selected, a down server must surface as a preflight problem.
    def down(cfg):
        raise ConnectionError("refused")
    monkeypatch.setattr(higgs_tts, "_probe_higgs", down)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**{**ok_env, "tts_engine": "higgs"}), video, tmp_path / "work")
    assert "Higgs" in str(exc_info.value)


def test_higgs_not_probed_when_engine_vieneu(ok_env, video, tmp_path, monkeypatch):
    # R11: a vieneu run must not probe (or fail on) a down Higgs, nor Soniox.
    called = []
    def down(cfg):
        called.append(1)
        raise ConnectionError("Higgs is down")
    monkeypatch.setattr(higgs_tts, "_probe_higgs", down)
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: called.append(1))
    pf.preflight(Config(**ok_env), video, tmp_path / "work")  # vieneu: no raise
    assert called == []


def test_higgs_reachable_passes(ok_env, video, tmp_path, monkeypatch):
    # Happy path: a reachable Higgs server (probe returns) passes preflight.
    monkeypatch.setattr(higgs_tts, "_probe_higgs", lambda cfg: None)
    pf.preflight(Config(**{**ok_env, "tts_engine": "higgs"}), video, tmp_path / "work")  # no raise


# --- engine-conditional Soniox preflight (U5/R10/R11) ---


def _soniox_env(ok_env, **over):
    # Deterministic Soniox config (explicit voices) so the developer's SONIOX_*
    # environment can't perturb the voice-name validation.
    base = {**ok_env, "tts_engine": "soniox", "soniox_api_key": "good-key",
            "soniox_voice_pool": ["Adrian", "Maya"], "soniox_voice_map": {},
            "soniox_default_voice": "Adrian"}
    base.update(over)
    return base


def test_soniox_missing_key_reported(ok_env, video, tmp_path, monkeypatch):
    # soniox + empty key -> a problem naming the env var; Higgs/vieneu absence
    # (no Higgs server, no vieneu venv) must NOT add problems.
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    env = _soniox_env(ok_env, soniox_api_key="", higgs_host="http://nonexistent:9")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "SONIOX_API_KEY" in msg
    assert "Higgs" not in msg


def test_soniox_reachable_key_passes(ok_env, video, tmp_path, monkeypatch):
    # A reachable, authenticating key (non-401 validation error) + valid voices
    # passes without needing Higgs or the vieneu venv.
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    pf.preflight(Config(**_soniox_env(ok_env)), video, tmp_path / "work")  # no raise


def test_soniox_bad_key_401_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 401)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**_soniox_env(ok_env)), video, tmp_path / "work")
    assert "unauthenticated" in str(exc_info.value)


def test_soniox_forbidden_key_403_reported(ok_env, video, tmp_path, monkeypatch):
    # 403 (no quota/permission) won't synthesize at run time -> flag in preflight.
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 403)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**_soniox_env(ok_env)), video, tmp_path / "work")
    assert "HTTP 403" in str(exc_info.value)


def test_soniox_ref_audio_warns_but_does_not_fail(ok_env, video, tmp_path, monkeypatch, caplog):
    # An explicit --ref-audio under the preset engine is dead config: warn, but
    # do not fail preflight (and never echo the key into logs).
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")
    env = _soniox_env(ok_env, ref_audio=str(ref), ref_text="hi")
    with caplog.at_level("WARNING", logger="loro.preflight"):
        pf.preflight(Config(**env), video, tmp_path / "work")  # no raise
    assert "ref-audio" in caplog.text


def test_soniox_probe_does_not_log_key(monkeypatch, caplog):
    # R11: the auth probe carries the key only in the header and never logs it.
    class _Fake:
        def post(self, url, headers=None, json=None, timeout=None):
            class _R:
                status_code = 422
            return _R()

    monkeypatch.setattr(soniox_tts, "requests", _Fake())
    with caplog.at_level("DEBUG", logger="loro.preflight"):
        soniox_tts._probe_soniox(Config(tts_engine="soniox", soniox_api_key="secret-key"))
    assert "secret-key" not in caplog.text


def test_soniox_unreachable_endpoint_reported(ok_env, video, tmp_path, monkeypatch):
    def down(cfg):
        raise ConnectionError("refused")
    monkeypatch.setattr(soniox_tts, "_probe_soniox", down)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**_soniox_env(ok_env)), video, tmp_path / "work")
    assert "Soniox" in str(exc_info.value)


def test_soniox_unknown_voice_in_map_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    env = _soniox_env(ok_env, soniox_voice_map={"A": "Nonexistent"})
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    assert "Nonexistent" in str(exc_info.value)


def test_soniox_unknown_voice_in_pool_or_default_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    env = _soniox_env(ok_env, soniox_voice_pool=["Adrian", "Bogus"],
                      soniox_default_voice="AlsoBogus")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "Bogus" in msg and "AlsoBogus" in msg


def test_soniox_probe_omits_text_field(monkeypatch):
    # The auth probe must never carry a `text` field, so it can never trigger a
    # billed synthesis; the key rides in the Authorization header only.
    captured = {}

    class _Fake:
        def post(self, url, headers=None, json=None, timeout=None):
            captured.update(url=url, headers=headers, json=json)

            class _R:
                status_code = 422
            return _R()

    monkeypatch.setattr(soniox_tts, "requests", _Fake())
    status = soniox_tts._probe_soniox(Config(tts_engine="soniox", soniox_api_key="k"))
    assert status == 422
    assert "text" not in captured["json"]
    assert captured["headers"]["Authorization"] == "Bearer k"


def test_soniox_does_not_probe_higgs(ok_env, video, tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(higgs_tts, "_probe_higgs", lambda cfg: called.append(1))
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    pf.preflight(Config(**_soniox_env(ok_env)), video, tmp_path / "work")
    assert called == []


def test_soniox_problems_aggregate_into_one_failure(ok_env, video, tmp_path, monkeypatch):
    # R10 fail-once: missing ffmpeg + missing key collected into one error.
    monkeypatch.setattr(pf.shutil, "which", lambda tool: None)
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    env = _soniox_env(ok_env, soniox_api_key="")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "ffmpeg" in msg and "SONIOX_API_KEY" in msg


# --- Gemini preflight (U6): key, models.list probe, model + voice validation ---


def _gemini_env(ok_env, **over):
    base = {**ok_env, "tts_engine": "gemini", "gemini_api_key": "good-key",
            "gemini_voice_pool": ["Kore", "Puck"], "gemini_voice_map": {},
            "gemini_default_voice": "Kore",
            "gemini_model": "gemini-3.1-flash-tts-preview"}
    base.update(over)
    return base


def _ok_probe(cfg):
    return 200, ["gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]


def test_gemini_missing_key_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini_tts, "_probe_gemini", _ok_probe)
    env = _gemini_env(ok_env, gemini_api_key="", higgs_host="http://nonexistent:9")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "GEMINI_API_KEY" in msg
    assert "Higgs" not in msg


def test_gemini_reachable_key_and_model_passes(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini_tts, "_probe_gemini", _ok_probe)
    pf.preflight(Config(**_gemini_env(ok_env)), video, tmp_path / "work")  # no raise


def test_gemini_bad_key_401_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini_tts, "_probe_gemini", lambda cfg: (401, []))
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**_gemini_env(ok_env)), video, tmp_path / "work")
    assert "unauthenticated" in str(exc_info.value)


def test_gemini_forbidden_403_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini_tts, "_probe_gemini", lambda cfg: (403, []))
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**_gemini_env(ok_env)), video, tmp_path / "work")
    assert "HTTP 403" in str(exc_info.value)


def test_gemini_unreachable_endpoint_reported(ok_env, video, tmp_path, monkeypatch):
    def down(cfg):
        raise ConnectionError("refused")
    monkeypatch.setattr(gemini_tts, "_probe_gemini", down)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**_gemini_env(ok_env)), video, tmp_path / "work")
    assert "Gemini" in str(exc_info.value)


def test_gemini_model_unavailable_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini_tts, "_probe_gemini", lambda cfg: (200, ["some-other-model"]))
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**_gemini_env(ok_env)), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "unavailable" in msg and "gemini-3.1-flash-tts-preview" in msg


def test_gemini_unknown_voice_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(gemini_tts, "_probe_gemini", _ok_probe)
    env = _gemini_env(ok_env, gemini_voice_pool=["Kore", "Bogus"],
                      gemini_default_voice="AlsoBogus")
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "Bogus" in msg and "AlsoBogus" in msg


def test_gemini_ref_audio_warns_but_does_not_fail(ok_env, video, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(gemini_tts, "_probe_gemini", _ok_probe)
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFFfake")
    env = _gemini_env(ok_env, ref_audio=str(ref), ref_text="hi")
    with caplog.at_level("WARNING", logger="loro.preflight"):
        pf.preflight(Config(**env), video, tmp_path / "work")  # no raise
    assert "ref-audio" in caplog.text


def test_gemini_probe_key_in_header_not_url(monkeypatch):
    # S1/R10: the probe sends the key in the x-goog-api-key header, never as a
    # ?key= query string (which lands in access/proxy logs), and never logs it.
    captured = {}

    class _Fake:
        def get(self, url, headers=None, timeout=None):
            captured.update(url=url, headers=headers)

            class _R:
                status_code = 200

                def json(self):
                    return {"models": [{"name": "models/gemini-3.1-flash-tts-preview"}]}
            return _R()

    monkeypatch.setattr(gemini_tts, "requests", _Fake())
    status, served = gemini_tts._probe_gemini(
        Config(tts_engine="gemini", gemini_api_key="secret-key"))
    assert status == 200
    assert "secret-key" not in captured["url"]
    assert "key=" not in captured["url"]
    assert captured["headers"]["x-goog-api-key"] == "secret-key"
    assert "gemini-3.1-flash-tts-preview" in served


def test_gemini_not_probed_when_engine_soniox(ok_env, video, tmp_path, monkeypatch):
    # Branch isolation: a soniox run adds no Gemini checks.
    called = []
    monkeypatch.setattr(gemini_tts, "_probe_gemini", lambda cfg: called.append(1) or (200, []))
    monkeypatch.setattr(soniox_tts, "_probe_soniox", lambda cfg: 422)
    pf.preflight(Config(**_soniox_env(ok_env)), video, tmp_path / "work")
    assert called == []


def test_new_provider_preflight_surfaces_without_editing_preflight(ok_env, video, tmp_path, monkeypatch):
    # AE1: a new engine added as a provider with its own preflight contributes its
    # checks purely through the registry — preflight.py has no per-engine branch
    # to touch. Register a stub TTS provider and confirm its problem surfaces.
    from loro import providers as preg

    class _StubTts:
        name = "stubtts"
        clones = False
        batches = False
        native_long_text = False

        def preflight(self, cfg):
            return ["stub-tts-preflight-problem"]

    monkeypatch.setitem(preg._TTS, "stubtts", _StubTts())
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**{**ok_env, "tts_engine": "stubtts"}), video, tmp_path / "work")
    assert "stub-tts-preflight-problem" in str(exc_info.value)


# --- burn-in preflight (U4/R9): subtitles filter + Vietnamese glyph coverage ---


def test_burn_off_skips_subtitle_probes(ok_env, video, tmp_path, monkeypatch):
    # Default (no --burn-subs): neither the filter nor the font is probed.
    calls = []
    monkeypatch.setattr(pf, "_has_subtitles_filter", lambda: calls.append("filter") or True)
    monkeypatch.setattr(pf, "_renders_glyphs", lambda wd, *a: calls.append("glyph") or True)
    pf.preflight(Config(**ok_env), video, tmp_path / "work")
    assert calls == []


def test_burn_on_subtitles_filter_missing_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_has_subtitles_filter", lambda: False)
    monkeypatch.setattr(pf, "_renders_glyphs", lambda wd, *a: True)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env, subtitle_burn=True), video, tmp_path / "work")
    msg = str(exc_info.value).lower()
    assert "subtitles" in msg and "libass" in msg


def test_burn_on_glyphs_missing_reported(ok_env, video, tmp_path, monkeypatch):
    # The filter is present but no font (profile or bundled) covers the target
    # glyphs (trial render empty/tofu) — presence-only would falsely pass (R18).
    monkeypatch.setattr(pf, "_has_subtitles_filter", lambda: True)
    monkeypatch.setattr(pf, "_renders_glyphs", lambda wd, *a: False)
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env, subtitle_burn=True), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "glyph coverage" in msg and "assets/fonts/" in msg


def test_burn_on_all_present_passes(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(pf, "_has_subtitles_filter", lambda: True)
    monkeypatch.setattr(pf, "_renders_glyphs", lambda wd, *a: True)
    pf.preflight(Config(**ok_env, subtitle_burn=True), video, tmp_path / "work")  # no raise


def test_glyph_probe_uses_profile_font_and_sample(ok_env, video, tmp_path, monkeypatch):
    # R17: the glyph probe trial-renders the PROFILE's font + glyph sample (not a
    # hardcoded Arial/Đường), so each target language is checked against its own.
    captured = {}
    monkeypatch.setattr(pf, "_has_subtitles_filter", lambda: True)
    monkeypatch.setattr(pf, "_renders_glyphs",
                        lambda wd, font, sample: captured.update(font=font, sample=sample) or True)
    pf.preflight(Config(**ok_env, subtitle_burn=True, target_lang="vi"), video,
                 tmp_path / "work")
    assert captured == {"font": "Arial", "sample": "Đường"}  # the VI profile's


def test_subtitles_filter_detected_from_filters_listing(monkeypatch):
    # The probe matches the filter-name column, not a description mention.
    monkeypatch.setattr(pf.ffmpeg, "run",
                        lambda args: " ..C subtitles         V->V       Render text subtitles.\n")
    assert pf._has_subtitles_filter() is True
    monkeypatch.setattr(pf.ffmpeg, "run",
                        lambda args: " TS asubboost        A->A       Boost subwoofer.\n")
    assert pf._has_subtitles_filter() is False


# --- engine-conditional ASR prerequisites (U6/R7) ---


def test_assemblyai_missing_key_reported(ok_env, video, tmp_path, monkeypatch):
    # assemblyai engine + empty key -> a problem naming the env var. Missing
    # NeMo/Granite must NOT add problems (those are local-engine-only).
    monkeypatch.setattr(assemblyai_asr, "_probe_assemblyai", lambda cfg: 404)
    env = {**ok_env, "asr_engine": "assemblyai", "assemblyai_api_key": "",
           "nemotron_python": "/nonexistent/nemo", "granite_python": "/nonexistent/granite"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "ASSEMBLYAI_API_KEY" in msg
    assert "NeMo" not in msg and "Granite" not in msg


def test_assemblyai_bad_key_401_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(assemblyai_asr, "_probe_assemblyai", lambda cfg: 401)
    env = {**ok_env, "asr_engine": "assemblyai", "assemblyai_api_key": "bad-key"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    assert "unauthenticated" in str(exc_info.value)


def test_assemblyai_reachable_key_passes(ok_env, video, tmp_path, monkeypatch):
    # Reachable, authenticating key + no NeMo/Granite venvs -> passes; only the
    # vision/translate/seg model-server checks remain (mocked healthy by ok_env).
    monkeypatch.setattr(assemblyai_asr, "_probe_assemblyai", lambda cfg: 404)
    env = {**ok_env, "asr_engine": "assemblyai", "assemblyai_api_key": "good-key",
           "nemotron_python": "/nonexistent/nemo", "granite_python": "/nonexistent/granite"}
    pf.preflight(Config(**env), video, tmp_path / "work")  # no raise


def test_assemblyai_unreachable_endpoint_reported(ok_env, video, tmp_path, monkeypatch):
    def down(cfg):
        raise ConnectionError("refused")
    monkeypatch.setattr(assemblyai_asr, "_probe_assemblyai", down)
    env = {**ok_env, "asr_engine": "assemblyai", "assemblyai_api_key": "good-key"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    assert "AssemblyAI" in str(exc_info.value)


def test_assemblyai_engine_skips_audio_probe(ok_env, video, tmp_path, monkeypatch):
    # The audio probe is the local-engine cross-check check; assemblyai never runs it.
    probes = []
    monkeypatch.setattr(assemblyai_asr, "_probe_assemblyai", lambda cfg: 404)
    monkeypatch.setattr(pf, "_probe_audio_input", lambda cfg, v, w: probes.append(1))
    env = {**ok_env, "asr_engine": "assemblyai", "assemblyai_api_key": "good-key",
           "enable_cross_check": True}
    pf.preflight(Config(**env), video, tmp_path / "work")
    assert probes == []


def test_assemblyai_problems_aggregate_into_one_failure(ok_env, video, tmp_path, monkeypatch):
    # R7 fail-once on the cloud branch: missing ffmpeg + missing key collected together.
    monkeypatch.setattr(pf.shutil, "which", lambda tool: None)
    monkeypatch.setattr(assemblyai_asr, "_probe_assemblyai", lambda cfg: 404)
    env = {**ok_env, "asr_engine": "assemblyai", "assemblyai_api_key": ""}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "ffmpeg" in msg and "ASSEMBLYAI_API_KEY" in msg


# --- soniox ASR engine preflight (U5/R9) ---

def test_soniox_stt_missing_key_reported(ok_env, video, tmp_path, monkeypatch):
    # soniox engine + empty key -> a problem naming the (shared) env var. Missing
    # NeMo/Granite must NOT add problems (those are local-engine-only).
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 404)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "",
           "nemotron_python": "/nonexistent/nemo", "granite_python": "/nonexistent/granite"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "SONIOX_API_KEY" in msg
    assert "NeMo" not in msg and "Granite" not in msg


def test_soniox_stt_bad_key_401_reported(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 401)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "bad-key"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "unauthenticated" in msg and "401" in msg


def test_soniox_stt_forbidden_403_reported_distinctly(ok_env, video, tmp_path, monkeypatch):
    # 403 (authenticated but insufficient scope/quota) is reported distinctly
    # from the 401 bad-key case.
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 403)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "scoped-key"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "403" in msg and "permission" in msg
    assert "unauthenticated" not in msg  # not conflated with the 401 message


def test_soniox_stt_transient_5xx_does_not_pass(ok_env, video, tmp_path, monkeypatch):
    # A 5xx is not proof of a good key — it must not pass preflight.
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 503)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "good-key"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    assert "503" in str(exc_info.value)


def test_soniox_stt_rate_limit_429_does_not_pass(ok_env, video, tmp_path, monkeypatch):
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 429)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "good-key"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    assert "429" in str(exc_info.value)


def test_soniox_stt_reachable_key_404_passes(ok_env, video, tmp_path, monkeypatch):
    # 404 = clean authenticated probe; no NeMo/Granite venvs needed -> passes.
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 404)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "good-key",
           "nemotron_python": "/nonexistent/nemo", "granite_python": "/nonexistent/granite"}
    pf.preflight(Config(**env), video, tmp_path / "work")  # no raise


def test_soniox_stt_engine_skips_audio_probe(ok_env, video, tmp_path, monkeypatch):
    # The audio probe is the local-engine cross-check check; soniox never runs it.
    probes = []
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 404)
    monkeypatch.setattr(pf, "_probe_audio_input", lambda cfg, v, w: probes.append(1))
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "good-key",
           "enable_cross_check": True}
    pf.preflight(Config(**env), video, tmp_path / "work")
    assert probes == []


def test_soniox_stt_unreachable_endpoint_reported(ok_env, video, tmp_path, monkeypatch):
    def down(cfg):
        raise ConnectionError("refused")
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", down)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": "good-key"}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    assert "Soniox STT" in str(exc_info.value)


def test_soniox_stt_problems_aggregate_into_one_failure(ok_env, video, tmp_path, monkeypatch):
    # R9 fail-once: missing ffmpeg + missing key collected into one failure.
    monkeypatch.setattr(pf.shutil, "which", lambda tool: None)
    monkeypatch.setattr(soniox_asr, "_probe_soniox_stt", lambda cfg: 404)
    env = {**ok_env, "asr_engine": "soniox", "soniox_api_key": ""}
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**env), video, tmp_path / "work")
    msg = str(exc_info.value)
    assert "ffmpeg" in msg and "SONIOX_API_KEY" in msg


def test_missing_video_reported(ok_env, tmp_path):
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env), tmp_path / "missing.mp4", tmp_path / "work")
    assert "missing.mp4" in str(exc_info.value)


def test_malformed_overrides_reported(ok_env, video, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "overrides.json").write_text('{"seg_0001": "ok", broken')
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env), video, workdir)
    assert "overrides.json" in str(exc_info.value)


def test_non_string_override_values_reported(ok_env, video, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "overrides.json").write_text('{"seg_0001": 42}')
    with pytest.raises(pf.PreflightError) as exc_info:
        pf.preflight(Config(**ok_env), video, workdir)
    assert "overrides.json" in str(exc_info.value)


def test_valid_overrides_pass(ok_env, video, tmp_path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "overrides.json").write_text('{"seg_0001": "bản sửa"}')
    pf.preflight(Config(**ok_env), video, workdir)  # no raise


# --- post-segmentation override range check (U5) ---

def test_in_range_override_keys_are_accepted():
    overrides = {"seg_0000": "a", "seg_0009": "b"}
    assert pf.out_of_range_override_keys(overrides, num_segments=10) == []


def test_out_of_range_override_key_is_flagged():
    overrides = {"seg_0001": "ok", "seg_0050": "stale"}
    assert pf.out_of_range_override_keys(overrides, num_segments=10) == ["seg_0050"]


def test_unrecognized_override_key_is_flagged():
    # A key that is not a seg_NNNN id would map to no segment — surface it too.
    overrides = {"seg_0001": "ok", "garbage": "x"}
    assert pf.out_of_range_override_keys(overrides, num_segments=10) == ["garbage"]
