"""Summarize visual context from sampled frames to ground the translation,
and extract an expected-terminology list to bias Granite's verify pass (R31).

Vision depends only on the video (frame sampling), so it runs in parallel
with ASR; cross-check waits on both to receive the keyword list."""

import logging
import re
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.harness.retry import StageError
from loro.services import llm
from loro.state import DubState
from loro.utils import ffmpeg

log = logging.getLogger("loro.vision")

def _prompt(cfg: Config) -> str:
    """The vision grounding prompt, naming the target language from the profile
    (U8/R10). VI resolves english_name="Vietnamese", so the VI prompt — and the
    vision fingerprint that folds it — stays byte-identical (R19)."""
    return (
        "These frames are sampled evenly from one video. Describe in 3-5 sentences: "
        "the setting, who is speaking (gender, age, formality), the topic, and the tone "
        "(casual / formal / technical / promotional). This summary will guide a "
        f"{cfg.language_profile.english_name} dubbing translator, so mention anything "
        "that affects word choice such as on-screen text, product names, or audience.\n"
        "Then end with one machine-readable line listing technical terms, proper "
        "names, acronyms and product names likely to be spoken in this video, "
        "exactly in this format:\n"
        "KEYWORDS: term one; term two; term three"
    )

_KEYWORDS_RE = re.compile(r"^\s*KEYWORDS\s*:\s*(.*)$", re.IGNORECASE)


def parse_keywords(summary: str) -> tuple[str, list[str]]:
    """Split the KEYWORDS line off the model reply. Defensive (R31): a
    missing or malformed line yields an empty list, and the context stays
    usable — never degraded."""
    keywords: list[str] = []
    rest: list[str] = []
    for line in summary.splitlines():
        match = _KEYWORDS_RE.match(line)
        if match and not keywords:
            keywords = [k.strip() for k in match.group(1).split(";") if k.strip()]
        else:
            rest.append(line)
    return "\n".join(rest).strip(), keywords


def vision(state: DubState, cfg: Config) -> DubState:
    vision_dir = Path(state["workdir"]) / "vision"
    inputs = {
        "video": artifacts.video_fingerprint(state["video_path"]),
        "frames": cfg.vision_frames,
        "prompt": _prompt(cfg),
        "model": cfg.llm_model_vision,
    }

    def compute() -> dict:
        frames = ffmpeg.extract_frames(
            state["video_path"], vision_dir / "frames", cfg.vision_frames,
            state["video_duration"],
        )
        if not frames:
            log.warning("no frames extracted — degraded vision context")
            return {"context": "", "keywords": [], "degraded": True,
                    "reason": "no_frames"}
        try:
            content = [{"type": "text", "text": _prompt(cfg)}] + [
                llm.image_part(f, cfg.llm_image_max_bytes, stage="vision") for f in frames]
            summary = llm.chat(cfg, [{"role": "user", "content": content}],
                                max_tokens=512, stage="vision",
                                role=cfg.llm_role("vision"),
                                enable_thinking=False)
        except StageError as exc:
            # Durable degraded artifact (R19): visible in the report, not
            # silently cached as a normal empty context. Delete
            # vision/context.json to retry.
            log.warning("vision failed after retries (%s) — degraded context", exc)
            return {"context": "", "keywords": [], "degraded": True,
                    "reason": f"{exc.error_class}/{exc.code}"}
        context, keywords = parse_keywords(summary)
        return {"context": context, "keywords": keywords, "degraded": False}

    data = artifacts.produce_json(vision_dir / "context.json", inputs, "vision", compute)
    if data.get("degraded"):
        log.warning("vision context degraded (%s) — delete vision/context.json to retry",
                    data.get("reason"))
    else:
        log.info("video context: %s | keywords: %s",
                 data["context"][:160], ", ".join(data.get("keywords", []))[:160])
    return {"video_context": data["context"],
            "video_keywords": data.get("keywords", [])}
