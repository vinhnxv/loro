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
    # bfloat16: same memory as float16 with a larger exponent range
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    print(f"loading {model_id} on {device}...", file=sys.stderr)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
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
