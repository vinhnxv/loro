"""Cast voices for the dub: clone the original speaker, or assign preset voices.

The node dispatches on engine family (KTD4). The cloning engines (vieneu/higgs)
clone a voice from a few seconds of reference audio plus its transcript, so the
best segment of the original English audio doubles as the reference — the
Vietnamese dub keeps the original speaker's timbre. It runs after cross-check so
the reference transcript is the verified text: a misheard term inside the
reference itself would degrade every cloned clip.

The preset engines (soniox/gemini) cannot clone, so instead they cast each
diarized speaker (Segment.speaker) to one of the engine's preset voices and
persist the speaker->voice map as voice/cast.json. The active engine's pool /
pins / default come from cfg.preset_voices, so the cast is engine-agnostic
(KTD6). The graph topology is unchanged — both paths live behind the same
voice_ref entry point (KTD4).
"""

import logging
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.state import DubState, Segment
from loro.utils import ffmpeg

log = logging.getLogger("loro.voice")

MIN_REF_SECONDS = 3.0
MAX_REF_SECONDS = 12.0
# Relaxed bounds tried when no segment fits the preferred window
RELAXED_MIN = 1.5
RELAXED_MAX = 20.0


def _pick_reference(segments: list[Segment]) -> Segment:
    for lo, hi in ((MIN_REF_SECONDS, MAX_REF_SECONDS), (RELAXED_MIN, RELAXED_MAX)):
        candidates = [s for s in segments if lo <= s.duration <= hi]
        if candidates:
            return max(candidates, key=lambda s: s.duration)
    raise RuntimeError(
        "no 1.5–20s segment found to use as a voice-clone reference — "
        "provide a clip manually with --ref-audio and --ref-text"
    )


def voice_ref(state: DubState, cfg: Config) -> DubState:
    """Cloning engines extract a reference clip; the preset engine casts each
    diarized speaker to a Soniox voice (KTD4). One entry point, two branches —
    the graph topology is identical for both."""
    if cfg.tts_uses_cloning:
        return _voice_ref_clone(state, cfg)
    return _voice_cast(state, cfg)


def _voice_cast(state: DubState, cfg: Config) -> DubState:
    """Assign each distinct diarized speaker a preset voice and persist
    voice/cast.json (R4-R6). Engine-agnostic across the preset family
    (soniox/gemini): the pool/pins/default come from cfg.preset_voices (KTD6).
    Deterministic: speakers are cast in sorted order, each taking its pin from
    the voice_map if present, else pool[i % len] by its stable sorted index i —
    so adding or changing one pin never reshuffles another speaker's voice (the
    property R8/KTD6 rely on). The "" sentinel (single-speaker / local-ASR audio)
    maps to the single default voice. No audio is read, so this never raises the
    clone branch's "no reference" error."""
    segments = state["segments"]
    cast_json = Path(state["workdir"]) / "voice" / "cast.json"

    present = {seg.speaker for seg in segments}
    real_speakers = sorted(spk for spk in present if spk)
    pv = cfg.preset_voices
    pool = pv.pool or [pv.default]
    inputs = {
        # All distinct ids (incl. "" when any segment lacks a speaker) so a
        # single-speaker run and a multi-speaker run never share a cast key.
        "speakers": sorted(present),
        "pool": pv.pool,
        "map": pv.voice_map,
        "default": pv.default,
    }

    def compute() -> dict:
        cast = {
            spk: (pv.voice_map.get(spk) or pool[i % len(pool)])
            for i, spk in enumerate(real_speakers)
        }
        if "" in present:
            cast[""] = pv.default
        return cast

    cast = artifacts.produce_json(cast_json, inputs, "voice_ref", compute)
    log.info("voice cast: %d speaker(s) -> %s", len(real_speakers), cast)
    return {"voice_cast": cast}


def _voice_ref_clone(state: DubState, cfg: Config) -> DubState:
    voice_dir = Path(state["workdir"]) / "voice"
    ref_json = voice_dir / "ref.json"

    if cfg.ref_audio:
        if not cfg.ref_text:
            raise RuntimeError("--ref-audio requires --ref-text (transcript of the reference clip)")
        # Content hash, not path: replacing the preset file invalidates the run
        inputs = {
            "preset_sha": artifacts.file_sha256(cfg.ref_audio),
            "ref_text": cfg.ref_text,
        }
        ref = artifacts.produce_json(
            ref_json, inputs, "voice_ref",
            lambda: {"ref_audio": str(Path(cfg.ref_audio).resolve()), "ref_text": cfg.ref_text},
        )
        log.info("using preset reference voice: %s", ref["ref_audio"])
        return {"ref_audio": ref["ref_audio"], "ref_text": ref["ref_text"]}

    seg = _pick_reference(state["segments"])
    end = min(seg.end, seg.start + MAX_REF_SECONDS)
    ref_wav = voice_dir / "ref_voice.wav"
    inputs = {
        "audio_sha": artifacts.cached_file_sha256(state["audio_16k"]),
        "start": seg.start,
        "end": end,
        "ref_text": seg.text_src,
    }
    artifacts.produce(
        ref_wav, inputs, "voice_ref",
        lambda tmp: ffmpeg.ffmpeg(
            "-i", state["audio_16k"], "-ss", f"{seg.start:.3f}", "-to", f"{end:.3f}", str(tmp)
        ),
    )
    ref = artifacts.produce_json(
        ref_json, inputs, "voice_ref",
        lambda: {"ref_audio": str(ref_wav), "ref_text": seg.text_src},
    )
    log.info("reference clip: segment %d (%.1fs-%.1fs) -> %s", seg.index, seg.start, end, ref_wav)
    return {"ref_audio": ref["ref_audio"], "ref_text": ref["ref_text"]}
