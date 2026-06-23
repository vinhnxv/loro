"""VieNeuClient lifecycle and resilience (R3, R7, R8), tested offline with a
stub `vieneu` package run under sys.executable: a warm worker serves sequential
calls from one process, a crashed worker is respawned and the retry succeeds, a
timeout kills the worker, a worker synth-error surfaces as a non-respawning
qa failure, and a missing interpreter fails fast with an actionable message."""

import json
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

from loro.config import Config
from loro.harness.retry import StageError
from loro.services.vieneu import VieNeuClient

# Stub vieneu.Vieneu with env-driven behaviour so each failure mode can be
# exercised without the real model:
#   text == "RAISE"  -> infer raises (worker emits an error line)
#   text == "SLEEP"  -> infer blocks long past the timeout
#   text == "CRASH"  -> the process exits mid-request until a counter exceeds
#                       STUB_CRASH_UNTIL (so a respawn eventually succeeds)
STUB_VIENEU = textwrap.dedent('''
    import json, os, time
    import numpy as np
    import soundfile as sf

    class Vieneu:
        def __init__(self, *a, **k):
            self._kw = {}

        def infer(self, text, ref_audio=None, ref_text=None,
                  emotion="natural", temperature=0.8):
            if text == "RAISE":
                raise RuntimeError("stub synth failure")
            if text == "SLEEP":
                time.sleep(30)
            if text == "CRASH":
                counter = os.environ["STUB_COUNTER"]
                n = 0
                if os.path.exists(counter):
                    n = int(open(counter).read() or "0")
                n += 1
                open(counter, "w").write(str(n))
                if n <= int(os.environ.get("STUB_CRASH_UNTIL", "0")):
                    os._exit(1)  # die mid-request like a crashed worker
            self._kw = {"ref_audio": ref_audio, "ref_text": ref_text,
                        "emotion": emotion, "temperature": temperature}
            return (0.2 * np.sin(np.linspace(0, 1, 4800))).astype("float32")

        def save(self, audio, path):
            sf.write(path, audio, 48000)
            open(path + ".kw.json", "w").write(json.dumps(self._kw))
''')


@pytest.fixture
def stub(tmp_path, monkeypatch):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir()
    (stub_dir / "vieneu.py").write_text(STUB_VIENEU)
    monkeypatch.setenv("PYTHONPATH", str(stub_dir))
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), np.zeros(1600, dtype="float32"), 16000)
    return {"ref": ref, "dir": tmp_path, "counter": tmp_path / "counter.txt"}


def test_warm_spawn_then_synthesize_writes_decodable_wav(stub):
    cfg = Config(vieneu_python=sys.executable)
    out = stub["dir"] / "clip.wav"
    with VieNeuClient(cfg, stub["ref"], "ref text") as client:
        client.synthesize("Xin chào", out)
    audio, sr = sf.read(str(out))
    assert sr == 48000 and len(audio) > 0


def test_two_calls_reuse_one_process(stub):
    cfg = Config(vieneu_python=sys.executable)
    with VieNeuClient(cfg, stub["ref"], "r") as client:
        client.synthesize("một", stub["dir"] / "a.wav")
        pid1 = client._proc.pid
        client.synthesize("hai", stub["dir"] / "b.wav")
        pid2 = client._proc.pid
    assert pid1 == pid2  # no relaunch between calls


def test_crashed_worker_respawns_and_retry_succeeds(stub, monkeypatch):
    monkeypatch.setenv("STUB_COUNTER", str(stub["counter"]))
    monkeypatch.setenv("STUB_CRASH_UNTIL", "1")  # crash once, then succeed
    cfg = Config(vieneu_python=sys.executable, retry_attempts=3, retry_base_delay=0.0)
    out = stub["dir"] / "clip.wav"
    with VieNeuClient(cfg, stub["ref"], "r") as client:
        client.synthesize("CRASH", out)  # infra retry respawns and succeeds
    assert out.exists()


def test_timeout_kills_worker_infra_signature(stub):
    cfg = Config(vieneu_python=sys.executable, vieneu_timeout=0.5,
                 retry_attempts=1, retry_base_delay=0.0)
    with VieNeuClient(cfg, stub["ref"], "r") as client:
        with pytest.raises(StageError) as exc_info:
            client.synthesize("SLEEP", stub["dir"] / "clip.wav")
    assert exc_info.value.signature == ("tts", "infra", "timeout")


def test_worker_error_is_qa_and_does_not_respawn(stub):
    cfg = Config(vieneu_python=sys.executable, retry_attempts=3, retry_base_delay=0.0)
    with VieNeuClient(cfg, stub["ref"], "r") as client:
        pid = client._proc.pid
        with pytest.raises(StageError) as exc_info:
            client.synthesize("RAISE", stub["dir"] / "bad.wav")
        assert exc_info.value.error_class == "qa"
        # Same process still serving — a synth error must not respawn.
        assert client._proc is not None and client._proc.pid == pid
        client.synthesize("ok now", stub["dir"] / "good.wav")
        assert client._proc.pid == pid
    assert (stub["dir"] / "good.wav").exists()
    assert not (stub["dir"] / "bad.wav").exists()


def test_ref_text_audio_only_by_default(stub):
    # R5: ref_text is plumbed but NOT sent unless explicitly enabled.
    cfg = Config(vieneu_python=sys.executable)
    out = stub["dir"] / "clip.wav"
    with VieNeuClient(cfg, stub["ref"], "english reference transcript") as client:
        client.synthesize("Xin chào", out)
    kw = json.loads((stub["dir"] / "clip.wav.kw.json").read_text())
    assert kw["ref_text"] is None
    assert kw["ref_audio"] == str(stub["ref"])  # audio reference still used


def test_ref_text_sent_when_enabled(stub):
    cfg = Config(vieneu_python=sys.executable, vieneu_ref_text=True)
    out = stub["dir"] / "clip.wav"
    with VieNeuClient(cfg, stub["ref"], "english reference transcript") as client:
        client.synthesize("Xin chào", out)
    kw = json.loads((stub["dir"] / "clip.wav.kw.json").read_text())
    assert kw["ref_text"] == "english reference transcript"


def test_missing_interpreter_actionable_error(stub):
    cfg = Config(vieneu_python="/nonexistent/vieneu/bin/python")
    with pytest.raises(RuntimeError, match="pyenv virtualenv"):
        with VieNeuClient(cfg, stub["ref"], "r"):
            pass
