"""Assemble per-batch translation context from deterministic source layers.

`translate` historically saw one global `video_context` string for the whole
video. This node builds a durable context artifact *per translation batch* by
merging source-side signals (R40): the global video_context, the visual
description(s) of the shot(s) the batch covers (deduped), and the raw EN text
of the K neighbour sentences on each side of the batch. The running summary
(U5) folds in as one more layer.

It is strictly source-side (KTD4): it never reads a translation, so it can be
cached and never forms a cycle with `translate`. The prev-VI consistency layer
lives in `translate` itself. KB/glossary is a future plug-in point and is
deliberately NOT stubbed here (KTD7).

Degradation never blocks (R43): a missing video_context or absent seg_visuals
simply drops that layer; the artifact is still built from whatever remains.
"""

import logging
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.harness.retry import StageError
from loro.services import llm
from loro.state import DubState

log = logging.getLogger("loro.context")

# Bump to invalidate every context artifact when the assembled format changes.
PROMPT_VERSION = "ctx-v1"

SUMMARY_PROMPT = (
    "You track the running topic of a video for a dubbing translator. Given the "
    "summary so far and the next lines of dialogue, reply with an updated 1-2 "
    "sentence running summary: what the video is about, the current topic, and "
    "the tone. Keep it rough and topical. Reply with the summary text only."
)


def _summarize(cfg: Config, prev_summary: str, batch_en: list[str]) -> str:
    """One sequential running-summary step (R41), on the translation model with
    thinking disabled (KTD6). Degraded (StageError) -> empty string: the summary
    layer drops but the rest of context and translation run on (R43)."""
    user = (
        (f"Summary so far: {prev_summary}\n\n" if prev_summary else "")
        + "Next lines:\n"
        + "\n".join(batch_en)
    )
    try:
        return llm.chat(
            cfg,
            [{"role": "system", "content": SUMMARY_PROMPT},
             {"role": "user", "content": user}],
            temperature=0.2, max_tokens=160, stage="context",
            role=cfg.llm_role("translate"), enable_thinking=False,
        )
    except StageError as exc:
        log.warning("running summary failed (%s) — dropping summary layer for "
                    "this batch", exc)
        return ""


def _unique(values: list) -> list:
    """Order-preserving dedup: adjacent segments share a shot description, so a
    batch's visual layer collapses to the distinct shot descriptions it spans."""
    out: list = []
    for v in values:
        if v not in out:
            out.append(v)
    return out


def context(state: DubState, cfg: Config) -> DubState:
    cdir = Path(state["workdir"]) / "context"
    segments = state["segments"]
    video_context = state.get("video_context", "")
    seg_visuals = state.get("seg_visuals", {})
    k = cfg.context_neighbors

    prev_summary = ""
    for bi, offset in enumerate(range(0, len(segments), cfg.translate_batch)):
        batch = segments[offset : offset + cfg.translate_batch]
        end = offset + len(batch)
        # Distinct visual descriptions the batch spans, capped so an
        # over-segmented batch can't bloat the prompt with a dozen shot
        # descriptions (KTD2 belt). Empty ("" degraded) / missing shots drop
        # out — they contribute nothing to the prompt either way, and a retried
        # shot reappears here and busts the fingerprint (R43).
        shot_visuals = _unique(
            [d for d in (seg_visuals.get(s.index) for s in batch) if d]
        )[: cfg.context_shot_cap]
        nb_before = [s.text_src for s in segments[max(0, offset - k) : offset]]
        nb_after = [s.text_src for s in segments[end : end + k]]
        batch_en = [s.text_src for s in batch]

        inputs = {
            "range": [batch[0].index, batch[-1].index],
            "video_context": video_context,
            "shot_visuals": shot_visuals,
            "neighbors_before": nb_before,
            "neighbors_after": nb_after,
            "k": k,
            "prompt_version": PROMPT_VERSION,
        }
        if cfg.enable_summary:
            # Byte-level cascade link (KTD5/R41): summary_i = f(summary_{i-1},
            # batch_en_i), so BOTH the prior summary and this batch's own EN
            # text are fingerprint inputs — otherwise an edited line would keep
            # a stale cached summary. An early edit thus busts the whole tail.
            # The summary itself is an output of recompute, not an input.
            inputs["summary_in"] = prev_summary
            inputs["batch_en"] = batch_en
        art = cdir / f"batch_{bi:04d}.json"

        def compute(shot_visuals=shot_visuals, nb_before=nb_before,
                    nb_after=nb_after, batch=batch, batch_en=batch_en,
                    prev_summary=prev_summary) -> dict:
            summary = (_summarize(cfg, prev_summary, batch_en)
                       if cfg.enable_summary else "")
            return {
                "range": [batch[0].index, batch[-1].index],
                "video_context": video_context,
                "shot_visuals": shot_visuals,
                "neighbors_before": nb_before,
                "neighbors_after": nb_after,
                "summary": summary,
            }

        data = artifacts.produce_json(art, inputs, "context", compute)
        # Carry summary_i forward even on a cache hit (compute not called), so
        # the sequential chain stays intact across resumed runs.
        prev_summary = data.get("summary", "")

    log.info("context: built %d batch artifact(s)",
             (len(segments) + cfg.translate_batch - 1) // cfg.translate_batch)
    return {}
