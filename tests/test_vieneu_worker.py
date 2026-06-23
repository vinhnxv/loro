"""VieNeu worker NDJSON protocol (R5, R7), tested offline against a stub
`vieneu` package on PYTHONPATH so CI needs no model: the worker emits exactly
one `ready` line before serving, writes one clip per request, survives a synth
failure, and exits 0 on stdin EOF."""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import loro
import soundfile as sf

WORKER = Path(loro.__file__).resolve().parent / "workers" / "vieneu_worker.py"

# A fake `vieneu.Vieneu` whose infer() returns a short 48 kHz array and whose
# save() writes a WAV — except for the sentinel text "RAISE", which fails so the
# error path can be exercised.
STUB_VIENEU = textwrap.dedent('''
    import numpy as np
    import soundfile as sf

    class Vieneu:
        def __init__(self, *args, **kwargs):
            pass

        def infer(self, text, ref_audio=None, ref_text=None,
                  emotion="natural", temperature=0.8):
            if text == "RAISE":
                raise RuntimeError("stub synth failure")
            return (0.2 * np.sin(np.linspace(0, 1, 4800))).astype("float32")

        def save(self, audio, path):
            sf.write(path, audio, 48000)
''')


def _run(tmp_path, requests, extra_env=None):
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    (stub_dir / "vieneu.py").write_text(STUB_VIENEU)
    env = {**os.environ,
           "PYTHONPATH": os.pathsep.join(
               [str(stub_dir), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)}
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, str(WORKER)],
        input="".join(json.dumps(r) + "\n" for r in requests),
        capture_output=True, text=True, env=env, timeout=60,
    )
    # The worker keeps stdout for protocol only; every non-empty line is JSON.
    lines = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]
    return proc, lines


def test_ready_emitted_once_before_any_request(tmp_path):
    proc, lines = _run(tmp_path, [])
    assert lines == [{"status": "ready"}]
    assert proc.returncode == 0  # clean exit on stdin EOF


def test_one_request_yields_ok_and_decodable_wav(tmp_path):
    out = tmp_path / "clip.wav"
    proc, lines = _run(tmp_path, [{
        "text": "Xin chào", "out": str(out),
        "ref_audio": None, "ref_text": None,
        "temperature": 0.8, "emotion": "natural",
    }])
    assert lines[0] == {"status": "ready"}
    assert lines[1] == {"out": str(out), "status": "ok"}
    assert proc.returncode == 0
    audio, sr = sf.read(str(out))
    assert sr == 48000 and len(audio) > 0


def test_synth_failure_yields_error_and_loop_survives(tmp_path):
    bad = tmp_path / "bad.wav"
    good = tmp_path / "good.wav"
    proc, lines = _run(tmp_path, [
        {"text": "RAISE", "out": str(bad), "temperature": 0.8, "emotion": "natural"},
        {"text": "ok now", "out": str(good), "temperature": 0.8, "emotion": "natural"},
    ])
    assert lines[0] == {"status": "ready"}
    assert lines[1]["out"] == str(bad)
    assert lines[1]["status"] == "error"
    assert lines[1]["error"]  # carries a message
    # The loop survived to serve the next request:
    assert lines[2] == {"out": str(good), "status": "ok"}
    assert proc.returncode == 0
    assert good.exists() and not bad.exists()


def test_exactly_one_ready_across_multiple_requests(tmp_path):
    reqs = [{"text": f"line {i}", "out": str(tmp_path / f"c{i}.wav"),
             "temperature": 0.8, "emotion": "natural"} for i in range(3)]
    _, lines = _run(tmp_path, reqs)
    assert sum(1 for ln in lines if ln.get("status") == "ready") == 1
    assert lines[0] == {"status": "ready"}
    assert [ln["status"] for ln in lines[1:]] == ["ok", "ok", "ok"]
