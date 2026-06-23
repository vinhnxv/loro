"""CLI lifecycle: exit codes and report behavior of loro.__main__.main()."""

import json

import pytest

from loro import __main__ as cli
from loro.config import Config
from loro.harness.artifacts import LockError
from loro.harness.ledger import AbortRun
from loro.harness.preflight import PreflightError


class _FakeGraph:
    def __init__(self, effect):
        self._effect = effect

    def invoke(self, state, config):
        return self._effect(state)


@pytest.fixture
def env(tmp_path, monkeypatch):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"\x00" * 64)
    workdir = tmp_path / "work"
    monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)

    def run(effect, argv_extra=()):
        monkeypatch.setattr(
            cli, "build_graph",
            lambda cfg, timings=None: _FakeGraph(effect),
        )
        monkeypatch.setattr(
            "sys.argv",
            ["loro", str(video), "-w", str(workdir), "-o", str(tmp_path / "out.mp4"),
             *argv_extra],
        )
        with pytest.raises(SystemExit) as exc_info:
            cli.main()
        return exc_info.value.code

    return {"run": run, "workdir": workdir, "monkeypatch": monkeypatch}


def test_context_flags_default_on_and_toggle_off(env):
    captured = {}

    def capture_graph(cfg, timings=None):
        captured["cfg"] = cfg
        return _FakeGraph(lambda s: {"output_path": "o", "srt_src": "a", "srt_target": "b"})

    env["monkeypatch"].setattr(cli, "build_graph", capture_graph)
    env["monkeypatch"].setattr(
        "sys.argv", ["loro", "x.mp4", "-w", str(env["workdir"])])
    with pytest.raises(SystemExit):
        cli.main()
    assert captured["cfg"].enable_seg_visual is True
    assert captured["cfg"].enable_summary is True

    env["monkeypatch"].setattr(
        "sys.argv",
        ["loro", "x.mp4", "-w", str(env["workdir"]), "--no-seg-visual", "--no-summary"])
    with pytest.raises(SystemExit):
        cli.main()
    assert captured["cfg"].enable_seg_visual is False
    assert captured["cfg"].enable_summary is False


def test_clean_run_exits_zero_and_writes_report(env):
    def ok(state):
        return {"output_path": "out.mp4", "srt_src": "a.srt", "srt_target": "b.srt"}

    assert env["run"](ok) == 0
    report = json.loads((env["workdir"] / "report.json").read_text())
    assert report["status"] == "completed"


def test_abort_exits_three_with_report(env):
    def abort(state):
        raise AbortRun(("tts", "infra", "http_503"), 5, 20)

    assert env["run"](abort) == 3
    report = json.loads((env["workdir"] / "report.json").read_text())
    assert report["status"] == "aborted"
    assert report["abort"]["signature"] == {"stage": "tts", "class": "infra",
                                            "code": "http_503"}
    assert report["abort"]["count"] == 5


def test_fatal_error_exits_one_with_report(env):
    def boom(state):
        raise RuntimeError("ffmpeg exploded")

    assert env["run"](boom) == 1
    report = json.loads((env["workdir"] / "report.json").read_text())
    assert report["status"] == "failed"


def test_preflight_failure_exits_one_without_report(env):
    def fail_preflight(cfg, video, wd):
        raise PreflightError("missing ffmpeg")

    env["monkeypatch"].setattr(cli, "preflight", fail_preflight)
    assert env["run"](lambda s: {}) == 1
    assert not (env["workdir"] / "report.json").exists()


# --- ASR engine selection (U1) ---

_ASR_ENV = (
    "ASR_ENGINE", "ASSEMBLYAI_API_KEY", "ASSEMBLYAI_BASE_URL",
    "ASSEMBLYAI_SPEECH_MODELS", "ASSEMBLYAI_SPEAKER_LABELS",
    "ASSEMBLYAI_LANGUAGE_DETECTION", "ASSEMBLYAI_LANGUAGE_CODE",
    "SONIOX_STT_BASE_URL", "SONIOX_STT_MODEL", "SONIOX_STT_LANGUAGE_HINTS",
    "SONIOX_STT_ENABLE_LANGUAGE_IDENTIFICATION", "SONIOX_STT_SPEAKER_DIARIZATION",
    "SONIOX_STT_CONTEXT_TERMS", "SONIOX_STT_CONTEXT_TEXT", "SONIOX_STT_CLEANUP",
)


def _clean_asr_env(monkeypatch):
    for name in _ASR_ENV:
        monkeypatch.delenv(name, raising=False)


def test_asr_engine_defaults_to_soniox(monkeypatch):
    # U1/KTD5: the default ASR engine is now the Soniox cloud path with the
    # documented stt-async-v5 defaults.
    _clean_asr_env(monkeypatch)
    cfg = Config()
    assert cfg.asr_engine == "soniox"
    assert cfg.soniox_stt_model == "stt-async-v5"
    assert cfg.soniox_stt_base_url == "https://api.soniox.com"
    assert cfg.soniox_stt_language_hints == ["en"]
    assert cfg.soniox_stt_enable_language_identification is False
    assert cfg.soniox_stt_speaker_diarization is True
    assert cfg.soniox_stt_context_terms == []
    assert cfg.soniox_stt_context_text == ""
    assert cfg.soniox_stt_cleanup is True


def test_assemblyai_still_selectable_with_its_defaults(monkeypatch):
    # The prior cloud engine stays a first-class choice with its own defaults.
    _clean_asr_env(monkeypatch)
    monkeypatch.setenv("ASR_ENGINE", "assemblyai")
    cfg = Config()
    assert cfg.asr_engine == "assemblyai"
    assert cfg.assemblyai_speech_models == ["universal-3-pro", "universal-2"]
    assert cfg.assemblyai_base_url == "https://api.assemblyai.com/v2"


def test_soniox_stt_reuses_soniox_api_key_no_dedicated_field(monkeypatch):
    # KTD4: the STT path reads cfg.soniox_api_key; there is no soniox_stt_api_key.
    _clean_asr_env(monkeypatch)
    monkeypatch.setenv("SONIOX_API_KEY", "shared-key")
    cfg = Config()
    assert cfg.soniox_api_key == "shared-key"
    assert not hasattr(cfg, "soniox_stt_api_key")


def test_soniox_stt_language_hints_and_context_terms_parse_as_lists(monkeypatch):
    _clean_asr_env(monkeypatch)
    monkeypatch.setenv("SONIOX_STT_LANGUAGE_HINTS", "en,vi")
    monkeypatch.setenv("SONIOX_STT_CONTEXT_TERMS", "LangGraph,Nemotron")
    cfg = Config()
    assert cfg.soniox_stt_language_hints == ["en", "vi"]
    assert cfg.soniox_stt_context_terms == ["LangGraph", "Nemotron"]


def test_asr_speech_models_env_parses_to_list(monkeypatch):
    _clean_asr_env(monkeypatch)
    monkeypatch.setenv("ASSEMBLYAI_SPEECH_MODELS", "universal-3-pro")
    assert Config().assemblyai_speech_models == ["universal-3-pro"]
    monkeypatch.setenv("ASSEMBLYAI_SPEECH_MODELS", "a, b ,c")
    assert Config().assemblyai_speech_models == ["a", "b", "c"]


def test_asr_language_code_env_surfaced(monkeypatch):
    _clean_asr_env(monkeypatch)
    monkeypatch.setenv("ASSEMBLYAI_LANGUAGE_CODE", "en")
    assert Config().assemblyai_language_code == "en"


def test_asr_engine_env_selects_local(monkeypatch):
    _clean_asr_env(monkeypatch)
    monkeypatch.setenv("ASR_ENGINE", "local")
    assert Config().asr_engine == "local"


def _capture_cfg(env, argv_extra=(), setenv=None):
    captured = {}
    mp = env["monkeypatch"]
    for name in _ASR_ENV:
        mp.delenv(name, raising=False)
    for k, v in (setenv or {}).items():
        mp.setenv(k, v)

    def capture(cfg, timings=None):
        captured["cfg"] = cfg
        return _FakeGraph(lambda s: {"output_path": "o", "srt_src": "a", "srt_target": "b"})

    mp.setattr(cli, "build_graph", capture)
    mp.setattr("sys.argv", ["loro", "x.mp4", "-w", str(env["workdir"]), *argv_extra])
    with pytest.raises(SystemExit):
        cli.main()
    return captured["cfg"]


def test_asr_engine_cli_overrides_env(env):
    # CLI wins over $ASR_ENGINE (mirrors --tts-engine).
    cfg = _capture_cfg(env, ["--asr-engine", "local"], setenv={"ASR_ENGINE": "assemblyai"})
    assert cfg.asr_engine == "local"


def test_asr_engine_cli_absent_uses_env(env):
    cfg = _capture_cfg(env, [], setenv={"ASR_ENGINE": "local"})
    assert cfg.asr_engine == "local"


def test_asr_engine_assemblyai_cli_overrides_soniox_env(env):
    # New-default scenario: --asr-engine assemblyai wins over ASR_ENGINE=soniox.
    cfg = _capture_cfg(env, ["--asr-engine", "assemblyai"], setenv={"ASR_ENGINE": "soniox"})
    assert cfg.asr_engine == "assemblyai"


def test_asr_engine_soniox_cli_accepted(env):
    cfg = _capture_cfg(env, ["--asr-engine", "soniox"])
    assert cfg.asr_engine == "soniox"


def test_invalid_asr_engine_rejected_by_argparse(monkeypatch, tmp_path):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"\x00")
    monkeypatch.setattr("sys.argv", ["loro", str(video), "--asr-engine", "nope"])
    with pytest.raises(SystemExit) as exc_info:
        cli.main()
    assert exc_info.value.code == 2  # argparse usage error


# --- burn-in flag wiring (U3) ---

def test_burn_subs_flag_sets_config(env):
    cfg = _capture_cfg(env, ["--burn-subs"])
    assert cfg.subtitle_burn is True


def test_burn_subs_absent_defaults_false(env):
    cfg = _capture_cfg(env, [])
    assert cfg.subtitle_burn is False


def test_lock_loss_exits_one_without_touching_report(env):
    class _BusyLock:
        def __init__(self, workdir):
            pass

        def __enter__(self):
            raise LockError("workdir đang được dùng bởi process 123")

        def __exit__(self, *exc):
            pass

    env["monkeypatch"].setattr(cli, "WorkdirLock", _BusyLock)
    assert env["run"](lambda s: {}) == 1
    # The live run's workdir must not be written to
    assert not (env["workdir"] / "report.json").exists()
