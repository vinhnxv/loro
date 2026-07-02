"""CLI lifecycle: exit codes and report behavior of loro.__main__.main()."""

import json
import logging
from pathlib import Path

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


def test_verbose_deepens_only_loro_logger(env):
    # -v puts loro's own loggers at DEBUG, but must NOT flip the root logger to
    # DEBUG: third-party libraries (urllib3/openai/httpcore) would then dump
    # base64 audio request/response bodies, bloating the log to hundreds of KB
    # per line. Regression guard for that fix.
    ok = lambda s: {"output_path": "o", "srt_src": "a", "srt_target": "b"}
    env["run"](ok, ["-v"])
    assert logging.getLogger("loro").level == logging.DEBUG
    env["run"](ok, [])
    assert logging.getLogger("loro").level == logging.INFO


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


# --- URL input support (U3) ---

@pytest.fixture
def url_env(tmp_path, monkeypatch):
    """Fixture for URL input tests: mocks preflight, download, and build_graph."""
    workdir = tmp_path / "work"
    downloaded_file = tmp_path / "downloaded" / "source.mp4"
    downloaded_file.parent.mkdir(parents=True, exist_ok=True)
    downloaded_file.write_bytes(b"\x00" * 64)

    monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)

    def mock_download(url, dest_dir, cfg=None):
        # Ensure the file exists at the expected path
        p = Path(dest_dir) / "source.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 64)
        return {"path": str(p), "title": "Test Video", "video_id": "abc123"}

    monkeypatch.setattr(cli, "ytdl_download", mock_download)

    return {"workdir": workdir, "downloaded_file": downloaded_file, "monkeypatch": monkeypatch}


def test_url_input_graph_receives_downloaded_path(url_env, monkeypatch):
    """The graph_state['video_path'] points to the downloaded file, not the URL."""
    captured_state = {}

    class _StateGraph:
        def __init__(self, cfg, timings=None):
            pass
        def invoke(self, state, config):
            captured_state.update(state)
            return {"output_path": "o", "srt_src": "a", "srt_target": "b"}

    monkeypatch.setattr(cli, "build_graph", _StateGraph)
    monkeypatch.setattr(
        "sys.argv",
        ["loro", "https://example.com/watch?v=abc123", "-w", str(url_env["workdir"])],
    )
    with pytest.raises(SystemExit):
        cli.main()

    assert captured_state["video_path"] != "https://example.com/watch?v=abc123"
    assert "source.mp4" in captured_state["video_path"]


def test_file_path_input_backward_compat_no_download(url_env, monkeypatch):
    """File path input does not trigger download; video_path is unchanged."""
    download_called = {"n": 0}

    def mock_download(url, dest_dir, cfg=None):
        download_called["n"] += 1
        return {"path": "x", "title": "x", "video_id": "x"}

    monkeypatch.setattr(cli, "ytdl_download", mock_download)

    video_path = url_env["downloaded_file"]  # reuse as a local file
    class _Ok:
        def __init__(self, cfg, timings=None):
            pass
        def invoke(self, state, config):
            assert state["video_path"] == str(video_path)
            return {"output_path": "o", "srt_src": "a", "srt_target": "b"}

    monkeypatch.setattr(cli, "build_graph", _Ok)
    monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)
    monkeypatch.setattr(
        "sys.argv", ["loro", str(video_path), "-w", str(url_env["workdir"])],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert download_called["n"] == 0


def test_url_input_workdir_derived_from_url(tmp_path, monkeypatch):
    """Without -w, the workdir is cfg.workdir / derive_workdir_stem(url)."""
    from loro.utils.url import derive_workdir_stem

    cfg_workdir = tmp_path / "cfgwork"
    monkeypatch.setattr("loro.config.Config.workdir", cfg_workdir, raising=False)
    # We need a Config whose .workdir returns cfg_workdir
    # Simpler: patch the Config class

    captured_state = {}

    class _StateGraph:
        def __init__(self, cfg, timings=None):
            self.cfg = cfg
        def invoke(self, state, config):
            captured_state.update(state)
            return {"output_path": "o", "srt_src": "a", "srt_target": "b"}

    monkeypatch.setattr(cli, "build_graph", _StateGraph)
    monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)
    monkeypatch.setattr(
        cli, "ytdl_download",
        lambda url, dest_dir, cfg=None: {"path": str(dest_dir / "source.mp4"),
                                         "title": "T", "video_id": "v"},
    )
    # Create the source.mp4 so preflight/pipeline doesn't choke on missing file
    monkeypatch.setattr(
        "sys.argv",
        ["loro", "https://www.youtube.com/watch?v=l6KeLCuB90o"],
    )

    # Patch Config to return our tmp_path as workdir
    import loro.config as config_mod
    original_init = config_mod.Config.__init__

    def patched_init(self, **kwargs):
        original_init(self, **kwargs)
        self.workdir = cfg_workdir
    monkeypatch.setattr(config_mod.Config, "__init__", patched_init)

    with pytest.raises(SystemExit):
        cli.main()

    expected_stem = derive_workdir_stem("https://www.youtube.com/watch?v=l6KeLCuB90o")
    expected_workdir = str(cfg_workdir / expected_stem)
    assert captured_state["workdir"] == expected_workdir


def test_url_input_with_workdir_override(url_env, monkeypatch):
    """With -w, the workdir is the override; download goes to workdir/ingest/."""
    download_dest = {"dir": None}

    def mock_download(url, dest_dir, cfg=None):
        download_dest["dir"] = str(dest_dir)
        p = Path(dest_dir) / "source.mp4"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00" * 64)
        return {"path": str(p), "title": "T", "video_id": "v"}

    monkeypatch.setattr(cli, "ytdl_download", mock_download)

    class _Ok:
        def __init__(self, cfg, timings=None):
            pass
        def invoke(self, state, config):
            return {"output_path": "o", "srt_src": "a", "srt_target": "b"}

    monkeypatch.setattr(cli, "build_graph", _Ok)
    custom_wd = url_env["workdir"]
    monkeypatch.setattr(
        "sys.argv", ["loro", "https://example.com/watch?v=abc123", "-w", str(custom_wd)],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert download_dest["dir"] == str(custom_wd / "ingest")


def test_url_input_output_naming_from_title(url_env, monkeypatch):
    """Without -o, URL input output defaults to workdir/sanitized_title.<tag>.mp4."""
    captured_state = {}

    class _StateGraph:
        def __init__(self, cfg, timings=None):
            self.cfg = cfg
        def invoke(self, state, config):
            captured_state.update(state)
            return {"output_path": "o", "srt_src": "a", "srt_target": "b"}

    monkeypatch.setattr(cli, "build_graph", _StateGraph)
    monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)
    monkeypatch.setattr(
        cli, "ytdl_download",
        lambda url, dest_dir, cfg=None: {"path": str(dest_dir / "source.mp4"),
                                         "title": "My Cool Video", "video_id": "v1"},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["loro", "https://example.com/watch?v=v1", "-w", str(url_env["workdir"])],
    )
    with pytest.raises(SystemExit):
        cli.main()

    # Output should be derived from title: workdir/My Cool Video.vi.mp4
    expected = str(url_env["workdir"] / "My Cool Video.vi.mp4")
    assert captured_state.get("output_path") == expected


def test_url_input_output_falls_back_to_video_id(url_env, monkeypatch):
    """Empty title falls back to video_id for output naming."""
    captured_state = {}

    class _StateGraph:
        def __init__(self, cfg, timings=None):
            self.cfg = cfg
        def invoke(self, state, config):
            captured_state.update(state)
            return {"output_path": "o", "srt_src": "a", "srt_target": "b"}

    monkeypatch.setattr(cli, "build_graph", _StateGraph)
    monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)
    monkeypatch.setattr(
        cli, "ytdl_download",
        lambda url, dest_dir, cfg=None: {"path": str(dest_dir / "source.mp4"),
                                         "title": "", "video_id": "vid123"},
    )
    monkeypatch.setattr(
        "sys.argv",
        ["loro", "https://example.com/watch?v=vid123", "-w", str(url_env["workdir"])],
    )
    with pytest.raises(SystemExit):
        cli.main()

    expected = str(url_env["workdir"] / "vid123.vi.mp4")
    assert captured_state.get("output_path") == expected


def test_url_input_error_propagates_clearly(url_env, monkeypatch):
    """Download error surfaces clearly and the pipeline does not start."""
    pipeline_started = {"n": 0}

    class _Graph:
        def __init__(self, cfg, timings=None):
            pass
        def invoke(self, state, config):
            pipeline_started["n"] += 1
            return {}

    monkeypatch.setattr(cli, "build_graph", _Graph)
    monkeypatch.setattr(cli, "preflight", lambda cfg, video, wd: None)
    monkeypatch.setattr(
        cli, "ytdl_download",
        lambda url, dest_dir, cfg=None: (_ for _ in ()).throw(
            RuntimeError("Video unavailable")),
    )
    monkeypatch.setattr(
        "sys.argv",
        ["loro", "https://example.com/err", "-w", str(url_env["workdir"])],
    )
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1
    assert pipeline_started["n"] == 0  # pipeline never started
