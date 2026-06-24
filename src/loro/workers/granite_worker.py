"""Standalone Granite-speech ASR worker, run inside the `granite` virtualenv.

granite-speech-4.1-2b runs on the torch path with its own transformers
requirement (>=4.52), which must stay isolated from both the main env and
the NeMo env, so this script is executed as a subprocess and communicates
via NDJSON on stdout. It must not import anything from the loro package.

Usage: python granite_worker.py <wav_16k_mono> [<wav_16k_mono> ...]

Environment:
    GRANITE_MODEL_ID  HF model id (default ibm-granite/granite-speech-4.1-2b)
    GRANITE_PROMPT    transcription prompt; the caller controls it entirely,
                      including any keyword-biasing clause (R31)

The model is loaded once; one JSON line is emitted per input file as soon as
its transcription finishes, so the caller can persist each clip artifact
immediately (a crash mid-batch loses at most one clip):

    {"path": "<input>", "text": "..."}

Granite emits no timestamps — it never needs to: Nemotron owns timing and
segment boundaries, this worker only verifies text per pre-cut clip.
"""

import json
import os
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

MODEL_ID = "ibm-granite/granite-speech-4.1-2b"
DEFAULT_PROMPT = "transcribe the speech with proper punctuation and capitalization."


def _select_dtype(torch, device: str):
    """Pick the model + audio-feature dtype (B8/R11). MPS does NOT reliably
    support bfloat16 for the audio-feature cast — it raised on real hosts — so MPS
    uses float16 (same memory as bf16, and MPS-supported) instead; CPU stays
    float32. Takes `torch` as a parameter so the selection is unit-testable
    without importing the heavy torch stack."""
    return torch.float16 if device == "mps" else torch.float32


def _load_model(model_cls, model_id: str, device: str, dtype, fallback_dtype):
    """Load the model at `dtype`, degrading to `fallback_dtype` (with a stderr
    warning) if the device rejects the dtype — e.g. an unsupported-dtype error on
    an older MPS backend — rather than aborting the whole verify run (R11).
    Returns (model, dtype_used) so the feature cast matches the loaded dtype."""
    try:
        model = model_cls.from_pretrained(
            model_id, dtype=dtype, low_cpu_mem_usage=True).to(device)
        return model, dtype
    except (RuntimeError, TypeError) as exc:
        print(f"granite worker: dtype {dtype} unsupported on {device} ({exc}); "
              f"falling back to {fallback_dtype}", file=sys.stderr, flush=True)
        model = model_cls.from_pretrained(
            model_id, dtype=fallback_dtype, low_cpu_mem_usage=True).to(device)
        return model, fallback_dtype


def main() -> None:
    wav_paths = sys.argv[1:]
    if not wav_paths:
        print("usage: granite_worker.py <wav> [<wav> ...]", file=sys.stderr)
        sys.exit(2)

    model_id = os.environ.get("GRANITE_MODEL_ID", MODEL_ID)
    prompt = os.environ.get("GRANITE_PROMPT", DEFAULT_PROMPT)

    import torch
    import torchaudio
    import soundfile as sf
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {model_id} on {device}...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_id)
    # MPS-safe dtype (float16, not bfloat16) with a float32 fallback if the device
    # still rejects it, so a verify run degrades instead of crashing (B8/R11).
    model, dtype = _load_model(AutoModelForSpeechSeq2Seq, model_id, device,
                               _select_dtype(torch, device), torch.float32)
    model.eval()

    messages = [{"role": "user", "content": f"<|audio|> {prompt}"}]
    chat_text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    for wav_path in wav_paths:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        if sr != 16000:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 16000)
        audio_tensor = audio_tensor.squeeze(0)

        inputs = processor(chat_text, audio=audio_tensor, return_tensors="pt")
        inputs = {
            k: v.to(device, dtype=dtype) if v.is_floating_point() else v.to(device)
            for k, v in inputs.items()
        }
        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs, max_new_tokens=512, do_sample=False, use_cache=True
            )
        text = processor.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        if device == "mps":
            torch.mps.empty_cache()
        print(json.dumps({"path": wav_path, "text": text}, ensure_ascii=False),
              flush=True)


if __name__ == "__main__":
    main()
