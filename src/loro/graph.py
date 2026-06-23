"""LangGraph wiring for the dubbing pipeline.

    ingest -> { asr, vision } -> [crosscheck] -> voice_ref -> translate
           -> tts -> fit -> mux

vision depends only on the video, so it runs in parallel with ASR and ahead
of cross-check: its keyword list biases Granite's verify reading (R31), and
its context still grounds the translation via state.

crosscheck is included only for the local ASR engine (R4/R5/KTD5): AssemblyAI
is its own single source of truth, so the assemblyai engine skips the ensemble
entirely and wires sentence_seg straight to voice_ref. On the local engine the
node is always present and its internal --no-cross-check toggle governs
ensemble-vs-passthrough exactly as before. Either way voice_ref joins on vision
(when enabled) so video_context is ready before downstream reads it; with
--no-vision the head follows sentence_seg directly and runs without keywords.
"""

import time

from langgraph.graph import END, START, StateGraph

from loro import providers
from loro.config import Config
from loro.nodes.asr import asr
from loro.nodes.crosscheck import crosscheck
from loro.nodes.fit import fit
from loro.nodes.context import context
from loro.nodes.ingest import ingest
from loro.nodes.mux import mux
from loro.nodes.seg_visual import seg_visual
from loro.nodes.sentence_seg import sentence_seg
from loro.nodes.translate import translate
from loro.nodes.tts import tts
from loro.nodes.vision import vision
from loro.nodes.voice import voice_ref
from loro.state import DubState


def build_graph(cfg: Config, timings: dict[str, float] | None = None):
    """`timings` (optional) collects per-stage wall time of this invocation
    for the run report; durable run state lives in artifacts, not here."""

    def timed(name, fn):
        def node(state, cfg=cfg):
            started = time.monotonic()
            try:
                return fn(state, cfg=cfg)
            finally:
                if timings is not None:
                    timings[name] = timings.get(name, 0.0) + time.monotonic() - started
        return node

    # crosscheck belongs to the engines that want it — the local engine only
    # (R4/KTD5/KTD8); the cloud engines are their own source of truth and skip
    # it. Read the provider's capability flag instead of an engine-name check
    # (R5). The node's own --no-cross-check toggle still governs ensemble-vs-
    # passthrough on the local engine, so local behavior is unchanged.
    include_crosscheck = providers.asr(cfg.asr_engine).wants_crosscheck

    g = StateGraph(DubState)
    nodes = [
        ("ingest", ingest), ("asr", asr), ("sentence_seg", sentence_seg),
        ("voice_ref", voice_ref), ("translate", translate),
        ("tts", tts), ("fit", fit), ("mux", mux),
    ]
    if include_crosscheck:
        nodes.insert(3, ("crosscheck", crosscheck))
    for name, fn in nodes:
        g.add_node(name, timed(name, fn))

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "asr")
    # sentence_seg turns asr's word stream into the sentence backbone; it needs
    # only asr.
    g.add_edge("asr", "sentence_seg")
    if cfg.enable_vision:
        g.add_node("vision", timed("vision", vision))
        g.add_edge("ingest", "vision")

    # The downstream head waits on vision's keyword list (R31) when present; it
    # is crosscheck on the local engine, otherwise voice_ref directly.
    head = "crosscheck" if include_crosscheck else "voice_ref"
    if cfg.enable_vision:
        g.add_edge(["sentence_seg", "vision"], head)
    else:
        g.add_edge("sentence_seg", head)
    if include_crosscheck:
        g.add_edge("crosscheck", "voice_ref")
    # Per-shot visual grounding (Gemma) runs after voice_ref; --no-seg-visual
    # skips it and context falls back to whatever layers remain (R43). context
    # assembles the per-batch layered context just before translate; it is
    # source-only and cheap, so it always runs.
    g.add_node("context", timed("context", context))
    if cfg.enable_seg_visual:
        g.add_node("seg_visual", timed("seg_visual", seg_visual))
        g.add_edge("voice_ref", "seg_visual")
        g.add_edge("seg_visual", "context")
    else:
        g.add_edge("voice_ref", "context")
    g.add_edge("context", "translate")
    g.add_edge("translate", "tts")
    g.add_edge("tts", "fit")
    g.add_edge("fit", "mux")
    g.add_edge("mux", END)
    return g.compile()
