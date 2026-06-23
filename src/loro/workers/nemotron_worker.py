"""Standalone Nemotron ASR worker, run inside the `nemo` virtualenv.

NeMo pins transformers 4.53.x and cannot live in the same env as the rest of
the pipeline (which needs transformers>=5.4 elsewhere), so this script is
executed as a subprocess and communicates via NDJSON on stdout. It must not
import anything from the loro package.

Usage: python nemotron_worker.py <wav_16k_mono> [<wav_16k_mono> ...]

The model is loaded once; one JSON line is emitted per input file as soon as
its transcription finishes, so the caller can persist each window artifact
immediately (a crash mid-stage loses at most one window):

    {"path": "<input>", "text": "...",
     "segments": [{"start": s, "end": e, "text": "..."}],
     "words": [{"start": s, "end": e, "word": "..."}] | null}

Timestamps are relative to the input file; the caller adds window offsets.
"""

import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "nvidia/nemotron-speech-streaming-en-0.6b"


def main() -> None:
    wav_paths = sys.argv[1:]
    if not wav_paths:
        print("usage: nemotron_worker.py <wav> [<wav> ...]", file=sys.stderr)
        sys.exit(2)

    import torch
    import nemo.collections.asr as nemo_asr

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {MODEL_ID} on {device}...", file=sys.stderr)
    model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    for wav_path in wav_paths:
        result = model.transcribe([wav_path], timestamps=True)[0]
        timestamps = result.timestamp or {}
        segments = [
            {"start": float(s["start"]), "end": float(s["end"]), "text": s["segment"]}
            for s in (timestamps.get("segment") or [])
        ]
        raw_words = timestamps.get("word") or []
        words = [
            {"start": float(w["start"]), "end": float(w["end"]), "word": str(w.get("word", ""))}
            for w in raw_words
        ] or None
        line = {"path": wav_path, "text": result.text, "segments": segments, "words": words}
        print(json.dumps(line, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
