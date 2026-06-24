"""End-to-end graph integration: real LangGraph topology, real ffmpeg, real
artifact store — only the model servers (oMLX, Higgs, Nemotron) are stubbed.

Proves the layers work together: a full run produces a playable .vi.mp4 and
all durable artifacts; a second identical run reuses everything (AE1/R2) and
calls no model at all."""

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import soundfile as sf

from loro.config import Config
from loro.graph import build_graph
from loro.harness import report as report_mod
from loro.nodes import asr as asr_mod
from loro.nodes import crosscheck as xck_mod
from loro.nodes import translate as tr_mod
from loro.nodes import tts as tts_mod
from loro.nodes import vision as vision_mod
from loro.providers.asr import local as local_provider
from loro.services import assemblyai, soniox_stt

STUB_WORKER = textwrap.dedent('''
    import json, sys
    MODEL_ID = "stub-model"
    for path in sys.argv[1:]:
        print(json.dumps({"path": path, "text": "hello world",
                          "segments": [
                              {"start": 0.2, "end": 1.8, "text": "hello world this is a test"},
                              {"start": 2.0, "end": 3.6, "text": "we deploy the model today"},
                          ],
                          "words": None}), flush=True)
''')


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=4:size=128x72:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=4",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path)],
        check=True,
    )
    return path


# Granite agrees with Nemotron per segment, so the lazy arbiter (Gemma) is
# never consulted during cross-check and a rerun stays at zero model calls.
GRANITE_READINGS = {0: "hello world this is a test", 1: "we deploy the model today"}


@pytest.fixture
def model_stubs(tmp_path, monkeypatch):
    """Stub every model server; count calls per service."""
    calls = {"llm": 0, "higgs": 0, "granite": 0}

    worker = tmp_path / "stub_worker.py"
    worker.write_text(STUB_WORKER)
    # WORKER moved to the local ASR provider (U5).
    monkeypatch.setattr(local_provider, "WORKER", worker)

    def fake_granite(cfg, xdir, jobs, prompt):
        calls["granite"] += 1
        for job in jobs:
            idx = int(__import__("pathlib").Path(job["artifact"]).stem.split("_")[1].split(".")[0])
            text = GRANITE_READINGS.get(idx, "")
            xck_mod.artifacts.produce(
                job["artifact"], job["inputs"], "crosscheck",
                lambda tmp, text=text: tmp.write_text(
                    json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"),
            )

    monkeypatch.setattr(xck_mod, "_run_granite_worker", fake_granite)

    def fake_chat(cfg, messages, stage="llm", **kw):
        calls["llm"] += 1
        if stage == "vision":
            return "A technical demo video about ML models.\nKEYWORDS: ML; model"
        if stage == "seg_visual":
            return "A speaker addressing the camera on a plain background."
        if stage == "context":
            return "An intro to ML, casual technical tone."  # running summary
        if stage == "crosscheck":
            return "hello world this is a test"  # Gemma fallback (not used here)
        if stage == "translate":
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            return json.dumps([{"i": l["i"], "vi": f"bản dịch số {l['i']} đây"}
                               for l in lines], ensure_ascii=False)
        raise AssertionError(f"unexpected stage {stage}")

    for module in (vision_mod, xck_mod, tr_mod):
        monkeypatch.setattr(module.llm, "chat", fake_chat)

    class FakeHiggs:
        def __init__(self, cfg, ref_audio, ref_text):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def synthesize(self, text, output, voice=None):
            calls["higgs"] += 1
            t = np.linspace(0, 1.2, int(24000 * 1.2), endpoint=False)
            sf.write(str(output),
                     (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), 24000)

    # Client construction now lives on the higgs provider (U3), so patch the
    # client on the provider module, not the node module.
    from loro.providers.tts import higgs as higgs_provider
    monkeypatch.setattr(higgs_provider, "HiggsClient", FakeHiggs)
    return calls


def _invoke(tmp_path, video):
    # Drives the full graph through the patched Higgs client; pin TTS to higgs
    # (default is now on-device vieneu) and ASR to local (default is now the
    # AssemblyAI cloud path) since this run exercises the local Nemotron stack.
    cfg = Config(nemotron_python=sys.executable, tts_engine="higgs", asr_engine="local")
    workdir = tmp_path / "work"
    graph = build_graph(cfg)
    return graph.invoke(
        {"video_path": str(video), "workdir": str(workdir),
         "output_path": str(tmp_path / "out.vi.mp4")},
        {"recursion_limit": 50},
    )


def test_topology_vision_runs_before_crosscheck():
    # R5: the local engine keeps today's topology. cross-check needs vision's
    # keyword list, so it waits on both branches; vision itself only needs
    # ingest (frame sampling). sentence_seg sits between asr and crosscheck (the
    # dub backbone), so the asr branch reaches crosscheck through it.
    edges = {(e.source, e.target)
             for e in build_graph(Config(asr_engine="local")).get_graph().edges}
    assert ("ingest", "vision") in edges
    assert ("vision", "crosscheck") in edges
    assert ("asr", "sentence_seg") in edges
    assert ("sentence_seg", "crosscheck") in edges
    assert ("asr", "crosscheck") not in edges
    assert ("crosscheck", "vision") not in edges


def test_topology_no_vision_wires_asr_through_sentence_seg_to_crosscheck():
    graph = build_graph(Config(enable_vision=False, asr_engine="local")).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("asr", "sentence_seg") in edges
    assert ("sentence_seg", "crosscheck") in edges
    assert "vision" not in {n for pair in edges for n in pair}


def test_topology_assemblyai_skips_crosscheck():
    # R4: the assemblyai engine is its own source of truth — no crosscheck node;
    # vision still runs and voice_ref joins on it so video_context is ready.
    graph = build_graph(Config(asr_engine="assemblyai")).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert "crosscheck" not in {n for pair in edges for n in pair}
    assert ("ingest", "vision") in edges
    assert ("sentence_seg", "voice_ref") in edges
    assert ("vision", "voice_ref") in edges


def test_topology_assemblyai_no_vision_wires_sentence_seg_to_voice_ref():
    # --no-vision + assemblyai: sentence_seg -> voice_ref directly, vision absent.
    graph = build_graph(Config(asr_engine="assemblyai", enable_vision=False)).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("sentence_seg", "voice_ref") in edges
    assert "vision" not in {n for pair in edges for n in pair}
    assert "crosscheck" not in {n for pair in edges for n in pair}


def test_topology_default_config_is_soniox_and_skips_crosscheck(monkeypatch):
    # R1/R8: the default engine is now soniox, a cloud single-source-of-truth — a
    # no-flag run compiles a graph with no crosscheck node that still reaches mux;
    # vision runs and voice_ref joins on it.
    monkeypatch.delenv("ASR_ENGINE", raising=False)
    cfg = Config()
    assert cfg.asr_engine == "soniox"
    edges = {(e.source, e.target) for e in build_graph(cfg).get_graph().edges}
    nodes = {n for pair in edges for n in pair}
    assert "crosscheck" not in nodes
    assert ("ingest", "vision") in edges
    assert ("sentence_seg", "voice_ref") in edges
    assert ("vision", "voice_ref") in edges
    assert ("fit", "mux") in edges  # reaches mux


def test_topology_local_still_has_crosscheck():
    # The local engine keeps the ensemble cross-check node.
    nodes = {n for pair in {(e.source, e.target)
                            for e in build_graph(Config(asr_engine="local")).get_graph().edges}
             for n in pair}
    assert "crosscheck" in nodes


def test_crosscheck_inclusion_follows_provider_capability_flag():
    # KTD8 (U5): include_crosscheck reads the provider's wants_crosscheck flag,
    # not an engine-name check — the crosscheck node is present iff the flag is set.
    from loro import providers
    for engine in ("soniox", "assemblyai", "local"):
        nodes = {n for pair in {(e.source, e.target)
                                for e in build_graph(Config(asr_engine=engine)).get_graph().edges}
                 for n in pair}
        assert ("crosscheck" in nodes) is providers.asr(engine).wants_crosscheck


def test_topology_seg_visual_then_context_then_translate():
    edges = {(e.source, e.target)
             for e in build_graph(Config()).get_graph().edges}
    assert ("voice_ref", "seg_visual") in edges
    assert ("seg_visual", "context") in edges
    assert ("context", "translate") in edges
    assert ("seg_visual", "translate") not in edges  # context is between them
    assert ("voice_ref", "translate") not in edges


def test_topology_no_seg_visual_wires_voice_ref_to_context():
    graph = build_graph(Config(enable_seg_visual=False)).get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("voice_ref", "context") in edges
    assert ("context", "translate") in edges
    assert "seg_visual" not in {n for pair in edges for n in pair}


# --- U13: single-writer-after-join state contract (A1/R13/KTD6) ---

def test_parallel_join_keeps_both_branches_and_post_join_mutation(tmp_path, video, model_stubs):
    # Characterization: the sentence_seg || vision parallel join yields a state
    # carrying BOTH the segment backbone (sentence_seg branch) and the vision
    # context/keywords (vision branch) — neither is dropped — and the post-join
    # linear chain's in-place segment mutation (translate's text_target) is
    # reflected downstream.
    final = _invoke(tmp_path, video)
    assert final["segments"]                               # sentence_seg branch landed
    assert final.get("video_context")                     # vision branch landed
    assert "video_keywords" in final                      # vision branch landed
    assert all(s.text_target for s in final["segments"])  # post-join mutation reflected


def test_single_writer_merge_raises_on_concurrent_non_empty_double_write():
    # Fail-loud: two concurrent non-empty writes to a single-writer channel are
    # the dropped-update bug R13 targets — they raise, never silently pick one.
    from loro.state import Segment, StateContractError, single_writer_merge
    a = [Segment(index=0, start=0.0, end=1.0, text_src="a")]
    b = [Segment(index=0, start=0.0, end=1.0, text_src="b")]
    with pytest.raises(StateContractError):
        single_writer_merge(a, b)


def test_single_writer_merge_prefers_non_empty_side():
    from loro.state import Segment, single_writer_merge
    segs = [Segment(index=0, start=0.0, end=1.0, text_src="a")]
    assert single_writer_merge([], segs) == segs          # empty existing -> update
    assert single_writer_merge(segs, []) == segs          # empty update -> existing
    assert single_writer_merge(None, segs) == segs


def test_full_run_then_instant_rerun(tmp_path, video, model_stubs):
    final = _invoke(tmp_path, video)

    out = tmp_path / "out.vi.mp4"
    assert out.exists() and out.stat().st_size > 0
    workdir = tmp_path / "work"
    for marker in ("ingest/audio_16k.wav", "asr/segments.json",
                   "sentence_seg/segments.json",
                   "crosscheck/segments.json", "seg_visual/shot_0000.json",
                   "context/batch_0000.json", "translate/segments.json",
                   "tts/seg_0000.wav", "fit/dub.vi.wav", "mux.json",
                   "transcript.vi.srt"):
        assert (workdir / marker).exists(), marker

    first_calls = dict(model_stubs)
    # Granite agrees with Nemotron, so cross-check makes zero Gemma calls
    # (lazy arbiter, R29): only vision + translate hit oMLX
    assert first_calls["llm"] >= 2  # vision + translate
    assert first_calls["granite"] == 1  # one batch invocation
    assert first_calls["higgs"] == 2

    # Second identical invocation: everything cached, zero model calls (AE1/R2)
    final2 = _invoke(tmp_path, video)
    assert dict(model_stubs) == first_calls
    assert final2["output_path"] == final["output_path"]


# Short, punctuated transcript so sentence_seg never calls the LLM (the stub
# fake_chat has no "sentence_seg" stage). All one speaker — diarization capture
# is pinned by the unit tests; here we only need the assemblyai topology to run.
AAI_TRANSCRIPT = {
    "status": "completed",
    "text": "hello world. we deploy the model today.",
    "words": [
        {"start": 200, "end": 600, "text": "hello", "speaker": "A"},
        {"start": 650, "end": 1800, "text": "world.", "speaker": "A"},
        {"start": 2000, "end": 2400, "text": "we", "speaker": "A"},
        {"start": 2450, "end": 2900, "text": "deploy", "speaker": "A"},
        {"start": 2950, "end": 3200, "text": "the", "speaker": "A"},
        {"start": 3250, "end": 3450, "text": "model", "speaker": "A"},
        {"start": 3500, "end": 3600, "text": "today.", "speaker": "A"},
    ],
    "utterances": [
        {"start": 200, "end": 1800, "text": "hello world.", "speaker": "A"},
        {"start": 2000, "end": 3600, "text": "we deploy the model today.", "speaker": "A"},
    ],
}


def test_full_run_assemblyai_skips_crosscheck(tmp_path, video, model_stubs, monkeypatch):
    # R4: a full assemblyai-engine run reaches mux without ever invoking the
    # Granite/Gemma ensemble, and the report builds cleanly with empty crosscheck
    # tallies (no crosscheck/ dir).
    monkeypatch.setattr(assemblyai, "transcribe", lambda cfg, audio: AAI_TRANSCRIPT)
    cfg = Config(asr_engine="assemblyai", tts_engine="higgs")
    workdir = tmp_path / "work"
    final = build_graph(cfg).invoke(
        {"video_path": str(video), "workdir": str(workdir),
         "output_path": str(tmp_path / "out.vi.mp4")},
        {"recursion_limit": 50},
    )

    assert (tmp_path / "out.vi.mp4").exists()
    assert (workdir / "asr" / "assemblyai.json").exists()
    assert (workdir / "asr" / "utterances.json").exists()
    assert not (workdir / "crosscheck").exists()  # ensemble never ran
    assert model_stubs["granite"] == 0
    # Segments carry the captured speaker through to the sentence backbone.
    seg_manifest = json.loads((workdir / "sentence_seg" / "segments.json").read_text())
    assert all(s["speaker"] == "A" for s in seg_manifest["segments"])

    report = report_mod.build_report(workdir, {}, "completed", None)
    assert report["crosscheck_replacements"] == []
    assert report["crosscheck_summary"]["tally"]["replace"] == 0


# "hello world. we deploy the model today." as Soniox sub-word tokens (ms units):
# short + punctuated so sentence_seg never calls the LLM, one diarized speaker.
SONIOX_TRANSCRIPT = {
    "tokens": [
        {"text": "hello", "start_ms": 200, "end_ms": 600, "speaker": "1"},
        {"text": " world.", "start_ms": 650, "end_ms": 1800, "speaker": "1"},
        {"text": " we", "start_ms": 2000, "end_ms": 2400, "speaker": "1"},
        {"text": " deploy", "start_ms": 2450, "end_ms": 2900, "speaker": "1"},
        {"text": " the", "start_ms": 2950, "end_ms": 3200, "speaker": "1"},
        {"text": " model", "start_ms": 3250, "end_ms": 3450, "speaker": "1"},
        {"text": " today.", "start_ms": 3500, "end_ms": 3600, "speaker": "1"},
    ],
}


def test_full_run_soniox_skips_crosscheck(tmp_path, video, model_stubs, monkeypatch):
    # R8: a full soniox-engine run reaches mux without ever invoking the
    # Granite/Gemma ensemble; no crosscheck/ dir, no utterances.json (soniox
    # has no utterance grouping), and the captured speaker flows to the backbone.
    monkeypatch.setattr(soniox_stt, "transcribe", lambda cfg, audio, **kw: SONIOX_TRANSCRIPT)
    cfg = Config(asr_engine="soniox", tts_engine="higgs")
    workdir = tmp_path / "work"
    final = build_graph(cfg).invoke(
        {"video_path": str(video), "workdir": str(workdir),
         "output_path": str(tmp_path / "out.vi.mp4")},
        {"recursion_limit": 50},
    )

    assert (tmp_path / "out.vi.mp4").exists()
    assert final["output_path"] == str(tmp_path / "out.vi.mp4")
    assert (workdir / "asr" / "soniox.json").exists()
    assert not (workdir / "asr" / "utterances.json").exists()  # no utterances on this path
    assert not (workdir / "crosscheck").exists()  # ensemble never ran
    assert model_stubs["granite"] == 0
    # Segments carry the captured speaker through to the sentence backbone.
    seg_manifest = json.loads((workdir / "sentence_seg" / "segments.json").read_text())
    assert all(s["speaker"] == "1" for s in seg_manifest["segments"])

    report = report_mod.build_report(workdir, {}, "completed", None)
    assert report["crosscheck_replacements"] == []
