"""Granite worker protocol (R27), tested offline with a fake worker script:
NDJSON lines persist one artifact per clip the moment they arrive, stdout
noise is skipped, and worker death / timeout surfaces as an infra-signature
StageError instead of hanging."""

import json
import sys
import textwrap

import pytest

from loro.config import Config
from loro.harness import artifacts
from loro.harness.retry import StageError
from loro.nodes import crosscheck as xck

FAKE_OK = textwrap.dedent('''
    import json, os, sys
    assert os.environ["GRANITE_PROMPT"], "prompt must arrive via env"
    for path in sys.argv[1:]:
        print(json.dumps({"path": path, "text": f"reading of {path}"}), flush=True)
''')

FAKE_NOISY = textwrap.dedent('''
    import json, sys
    print("loading checkpoint shards: 100%")   # library noise on stdout
    paths = sys.argv[1:]
    print(json.dumps({"path": paths[0], "text": "first clip"}), flush=True)
    print("some more noise")
''')

FAKE_CRASH = textwrap.dedent('''
    import sys
    print("dying before any output", file=sys.stderr)
    sys.exit(3)
''')

FAKE_HANG = textwrap.dedent('''
    import time
    time.sleep(60)
''')


@pytest.fixture
def env(tmp_path, monkeypatch):
    xdir = tmp_path / "crosscheck"
    xdir.mkdir()

    def make_jobs(n, parts=1):
        jobs = []
        for i in range(n):
            wavs = []
            for p in range(parts):
                wav = tmp_path / f"clip_{i:04d}_p{p}.wav"
                wav.write_bytes(b"fake-wav")
                wavs.append(str(wav))
            jobs.append({
                "wavs": wavs,
                "duration": 4.0,
                "artifact": xdir / f"seg_{i:04d}.granite.json",
                "inputs": {"clip": i},
            })
        return jobs

    def use_worker(source):
        worker = tmp_path / "fake_worker.py"
        worker.write_text(source)
        monkeypatch.setattr(xck, "GRANITE_WORKER", worker)

    return {"xdir": xdir, "make_jobs": make_jobs, "use_worker": use_worker,
            "cfg": Config(granite_python=sys.executable)}


class TestGraniteWorkerProtocol:
    def test_each_line_persists_one_artifact(self, env):
        env["use_worker"](FAKE_OK)
        jobs = env["make_jobs"](3)
        xck._run_granite_worker(env["cfg"], env["xdir"], jobs, PROMPT_X)
        for job in jobs:
            assert artifacts.is_valid(job["artifact"], job["inputs"])
            data = json.loads(job["artifact"].read_text())
            assert data["text"] == f"reading of {job['wavs'][0]}"

    def test_multi_part_segment_rejoins_in_order(self, env):
        env["use_worker"](FAKE_OK)
        jobs = env["make_jobs"](1, parts=3)
        xck._run_granite_worker(env["cfg"], env["xdir"], jobs, PROMPT_X)
        data = json.loads(jobs[0]["artifact"].read_text())
        assert data["text"] == " ".join(f"reading of {w}" for w in jobs[0]["wavs"])

    def test_stdout_noise_skipped_missing_clip_detected(self, env):
        env["use_worker"](FAKE_NOISY)
        jobs = env["make_jobs"](2)
        with pytest.raises(StageError) as exc_info:
            xck._run_granite_worker(env["cfg"], env["xdir"], jobs, PROMPT_X)
        assert exc_info.value.code == "missing_clips"
        # The clip that did arrive was persisted before the failure surfaced
        assert artifacts.is_valid(jobs[0]["artifact"], jobs[0]["inputs"])
        assert not jobs[1]["artifact"].exists()

    def test_worker_exit_nonzero_is_infra_signature(self, env):
        env["use_worker"](FAKE_CRASH)
        jobs = env["make_jobs"](1)
        with pytest.raises(StageError) as exc_info:
            xck._run_granite_worker(env["cfg"], env["xdir"], jobs, PROMPT_X)
        assert exc_info.value.signature == ("crosscheck", "infra", "worker_exit_3")

    def test_timeout_kills_worker_infra_signature(self, env):
        env["use_worker"](FAKE_HANG)
        jobs = env["make_jobs"](1)
        cfg = Config(granite_python=sys.executable,
                     granite_timeout_base=0.5, granite_timeout_per_sec=0.0)
        with pytest.raises(StageError) as exc_info:
            xck._run_granite_worker(cfg, env["xdir"], jobs, PROMPT_X)
        assert exc_info.value.signature == ("crosscheck", "infra", "timeout")

    def test_missing_interpreter_message_includes_venv_hint(self, env):
        env["use_worker"](FAKE_OK)
        cfg = Config(granite_python="/nonexistent/granite/bin/python")
        with pytest.raises(RuntimeError, match="pyenv virtualenv"):
            xck._run_granite_worker(cfg, env["xdir"], env["make_jobs"](1), PROMPT_X)


PROMPT_X = "transcribe the speech with proper punctuation and capitalization."


# --- U11: MPS dtype selection + unsupported-dtype fallback (B8/R11) ---
# The real worker imports torch only inside main(), so the dtype helpers are
# unit-testable here with lightweight fakes (no torch/transformers needed).

from loro.workers import granite_worker as gw


class _FakeTorch:
    bfloat16 = "bfloat16"
    float16 = "float16"
    float32 = "float32"


class TestGraniteDtype:
    def test_mps_avoids_bfloat16(self):
        torch = _FakeTorch()
        chosen = gw._select_dtype(torch, "mps")
        assert chosen != torch.bfloat16        # bf16 is the unsupported one
        assert chosen == torch.float16         # MPS-safe, memory-efficient

    def test_cpu_dtype_unchanged(self):
        assert gw._select_dtype(_FakeTorch(), "cpu") == _FakeTorch.float32

    def test_load_model_falls_back_on_unsupported_dtype(self):
        calls = []

        class FakeModel:
            def to(self, device):
                return self

        class FakeCls:
            def from_pretrained(self, model_id, dtype=None, low_cpu_mem_usage=None):
                calls.append(dtype)
                if dtype == "float16":
                    raise RuntimeError("dtype float16 not supported on this backend")
                return FakeModel()

        model, used = gw._load_model(FakeCls(), "id", "mps", "float16", "float32")
        assert used == "float32"               # degraded, not crashed
        assert calls == ["float16", "float32"]  # tried preferred, then fell back

    def test_load_model_keeps_preferred_dtype_when_supported(self):
        class FakeModel:
            def to(self, device):
                return self

        class FakeCls:
            def from_pretrained(self, model_id, dtype=None, low_cpu_mem_usage=None):
                return FakeModel()

        _, used = gw._load_model(FakeCls(), "id", "cpu", "float32", "float32")
        assert used == "float32"
