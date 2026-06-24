"""Standalone VieNeu-TTS worker, run inside the `vieneu` virtualenv.

VieNeu-TTS pulls a heavy on-device stack (onnxruntime + sea-g2p + the MOSS
audio codec; PyTorch on CUDA) that loro's deliberately thin main env avoids, so
this script runs as a subprocess in its own venv and must not import anything
from the loro package.

Unlike the NeMo/Granite workers (argv batch, then exit), this worker stays
*warm* for the whole segment batch: the model is loaded once, then one JSON
request is read per stdin line and one JSON response is written per stdout line
(NDJSON), so the costly model load is paid once per run, not per clip.

Protocol — one JSON object per line, flushed:

    <- {"status": "ready"}                          once, after the model loads
    -> {"text": str, "out": str,                    one request per stdin line
        "ref_audio": str|null, "ref_text": str|null,
        "temperature": float, "emotion": str}
    <- {"out": str, "status": "ok"}                 clip written to `out`
    <- {"out": str, "status": "error", "error": str}    synth failed; loop lives

The worker exits 0 on stdin EOF. Model-load banners and library chatter go to
stderr; stdout carries only protocol lines.

`Vieneu()` with no arguments loads the default v3 Turbo model (matching the
default VIENEU_MODEL) and auto-selects the CPU-ONNX or CUDA-PyTorch backend.
VIENEU_MODEL is recorded by the caller's clip fingerprint for identity; a
load-time integrity/pin check is deferred (see the plan's Risks).
"""

import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

DEFAULT_MODEL = "pnnbao-ump/VieNeu-TTS-v3-Turbo"


def main() -> None:
    # Keep the real stdout for protocol writes only; redirect sys.stdout to
    # stderr so any library banner/progress that prints to stdout cannot corrupt
    # the NDJSON stream.
    protocol = sys.stdout
    sys.stdout = sys.stderr

    def emit(obj: dict) -> None:
        protocol.write(json.dumps(obj, ensure_ascii=False) + "\n")
        protocol.flush()

    model_id = os.environ.get("VIENEU_MODEL", DEFAULT_MODEL)
    default_emotion = os.environ.get("VIENEU_EMOTION", "natural")

    from vieneu import Vieneu

    print(f"loading VieNeu ({model_id})...", file=sys.stderr)
    tts = Vieneu()
    emit({"status": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # One malformed stdin line (bad JSON, or missing out/text) must not kill
        # the warm worker and force a costly model reload (B13/R10): skip it with a
        # logged warning to stderr and keep serving, mirroring the Nemotron
        # parser's defensive `(JSONDecodeError, KeyError, TypeError) -> continue`.
        try:
            req = json.loads(line)
            out = req["out"]
            text = req["text"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"vieneu worker: skipping malformed request "
                  f"({type(exc).__name__}): {line[:200]}", file=sys.stderr, flush=True)
            continue
        try:
            audio = tts.infer(
                text,
                ref_audio=req.get("ref_audio") or None,
                ref_text=req.get("ref_text") or None,
                emotion=req.get("emotion") or default_emotion,
                temperature=req.get("temperature", 0.8),
            )
            tts.save(audio, out)
            emit({"out": out, "status": "ok"})
        except Exception as exc:  # a bad synth, not a dead process: keep serving
            emit({"out": out, "status": "error", "error": str(exc)[:500]})


if __name__ == "__main__":
    main()
