"""Per-shot visual grounding for the translation (R39).

Describing every segment's frames would be ~60-100 Gemma calls on a 5-minute
clip; adjacent segments almost always share a shot. So we detect scene cuts,
describe each shot *once*, and map every segment to the description of the
shot containing its midpoint (KTD2). Each shot description is a durable
artifact keyed on the video, the shot bounds, and the prompt/model — so a
rerun is free and a changed shot busts only the context that consumes it.

Degradation never blocks translation (R43): a shot whose frames can't be
extracted, or whose Gemma call fails, yields an empty description (a durable
degraded artifact, R19) and the pipeline runs on the remaining layers.
"""

import logging
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.harness.retry import StageError
from loro.services import llm
from loro.state import DubState
from loro.utils import ffmpeg

log = logging.getLogger("loro.seg_visual")

def _prompt(cfg: Config) -> str:
    """The per-shot visual grounding prompt, naming the target language from the
    profile (U8/R10). VI resolves english_name="Vietnamese", so the prompt — and
    the seg_visual fingerprint that folds it — stays byte-identical (R19)."""
    return (
        "These frames are from one continuous shot of a video. In 1-2 sentences, "
        "describe what is visually on screen: the setting, any on-screen text, "
        "product names or diagrams, and who is speaking (gender, formality) if "
        f"visible. This grounds a {cfg.language_profile.english_name} dubbing "
        "translation, so mention anything that affects word choice or tone. Be concise."
    )


def _shots(cuts: list[float], duration: float,
           min_dur: float = 0.0) -> list[tuple[float, float]]:
    """Turn scene-cut timestamps into contiguous (start, end) shots covering
    [0, duration]. A cut is kept only when it is at least `min_dur` past the
    previous kept boundary, so a cut-heavy video can't explode into micro-shots
    (one Gemma call each) — shots are bounded to ~duration/min_dur. No cuts (or
    min_dur >= duration) -> one shot spanning the whole video."""
    kept = [0.0]
    for c in sorted(cuts):
        if kept[-1] < c < duration and c - kept[-1] >= min_dur:
            kept.append(c)
    if len(kept) > 1 and duration - kept[-1] < min_dur:
        kept.pop()  # don't leave a sub-min_dur stub shot at the tail
    kept.append(duration)
    return [(kept[i], kept[i + 1]) for i in range(len(kept) - 1)]


def _shot_of(mid: float, shots: list[tuple[float, float]]) -> int:
    """Index of the shot whose half-open span contains `mid`; the last shot
    catches a midpoint sitting exactly on the final boundary."""
    for i, (start, end) in enumerate(shots):
        if start <= mid < end:
            return i
    return len(shots) - 1


def seg_visual(state: DubState, cfg: Config) -> DubState:
    sv_dir = Path(state["workdir"]) / "seg_visual"
    segments = state["segments"]
    duration = state["video_duration"]
    video = state["video_path"]
    video_fp = artifacts.video_fingerprint(video)

    # Scene detection is source-only and cheap; if it fails, degrade to a
    # single shot rather than blocking the whole stage (R43).
    try:
        cuts = ffmpeg.detect_scenes(video, cfg.scene_threshold)
    except Exception as exc:
        log.warning("scene detect failed (%s) — treating video as one shot", exc)
        cuts = []
    shots = _shots(cuts, duration, cfg.min_shot_duration)
    log.info("seg_visual: %d shot(s) from %d cut(s) (min_shot_duration=%.0fs)",
             len(shots), len(cuts), cfg.min_shot_duration)

    shot_desc: dict[int, str] = {}
    for si, (start, end) in enumerate(shots):
        inputs = {
            "video": video_fp,
            "shot": [round(start, 3), round(end, 3)],
            "frames": cfg.seg_visual_frames,
            "prompt": _prompt(cfg),
            "model": cfg.llm_model_vision,
        }
        art = sv_dir / f"shot_{si:04d}.json"

        def compute(si=si, start=start, end=end) -> dict:
            frames = ffmpeg.extract_frames_window(
                video, sv_dir / f"frames_{si:04d}", start, end, cfg.seg_visual_frames)
            if not frames:
                return {"description": "", "degraded": True, "reason": "no_frames"}
            try:
                content = [{"type": "text", "text": _prompt(cfg)}] + [
                    llm.image_part(f, cfg.llm_image_max_bytes, stage="seg_visual")
                    for f in frames]
                desc = llm.chat(cfg, [{"role": "user", "content": content}],
                                 max_tokens=256, stage="seg_visual",
                                 role=cfg.llm_role("vision"),
                                 enable_thinking=False)
            except StageError as exc:
                # Durable degraded artifact (R19): delete shot_XXXX.json to retry
                log.warning("seg_visual shot %d failed (%s) — degraded", si, exc)
                return {"description": "", "degraded": True,
                        "reason": f"{exc.error_class}/{exc.code}"}
            return {"description": desc, "degraded": False}

        data = artifacts.produce_json(art, inputs, "seg_visual", compute)
        if data.get("degraded"):
            log.warning("seg_visual shot %d degraded (%s) — delete %s to retry",
                        si, data.get("reason"), art.name)
        shot_desc[si] = data.get("description", "")

    # Map every segment to its shot's description. Reloaded from the durable
    # artifacts on every run (including cache hits) so a resumed run sees a
    # populated dict, not an empty one (mirrors vision.py always returning its
    # parsed payload).
    seg_visuals = {
        seg.index: shot_desc.get(_shot_of((seg.start + seg.end) / 2.0, shots), "")
        for seg in segments
    }
    return {"seg_visuals": seg_visuals}
