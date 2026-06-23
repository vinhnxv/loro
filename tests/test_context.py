"""context node (U4): deterministic per-batch assembly of the source-side
layers (video_context + shot visuals + EN neighbours), durable and
cascade-correct. The running summary (U5) is added separately."""

import hashlib
import json

import pytest

from loro.config import Config
from loro.harness.retry import StageError
from loro.nodes import context as ctx
from loro.state import Segment


@pytest.fixture(autouse=True)
def summary_stub(monkeypatch):
    """The node makes one running-summary model call per batch (U5). Stub it
    deterministically (a hash of the prompt, so distinct inputs give distinct
    summaries and the cascade is exercised). Returns the per-call kwargs so
    tests can count calls and check flags; failure tests override the chat."""
    calls = []

    def fake(cfg, messages, **kw):
        calls.append(kw)
        user = messages[-1]["content"]
        return "SUM:" + hashlib.sha1(user.encode("utf-8")).hexdigest()[:12]

    monkeypatch.setattr(ctx.llm, "chat", fake)
    return calls


def _segments(n=9):
    return [Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.5, text_src=f"en {i}")
            for i in range(n)]


def _state(tmp_path, segs, video_context="ngữ cảnh video", seg_visuals=None):
    return {
        "workdir": str(tmp_path / "work"),
        "segments": segs,
        "video_context": video_context,
        "seg_visuals": seg_visuals or {},
    }


def _run(tmp_path, segs, *, video_context="ngữ cảnh video", seg_visuals=None,
         **cfg_kw):
    ctx.context(
        _state(tmp_path, segs, video_context=video_context, seg_visuals=seg_visuals),
        Config(translate_batch=3, **cfg_kw),
    )


def _batch(tmp_path, bi):
    return json.loads(
        (tmp_path / "work" / "context" / f"batch_{bi:04d}.json").read_text())


def _meta_fp(tmp_path, bi):
    return json.loads(
        (tmp_path / "work" / "context" / f"batch_{bi:04d}.json.meta.json")
        .read_text())["input_fingerprint"]


class TestNeighbours:
    def test_window_before_and_after(self, tmp_path):
        # R40: batch 1 = segs 3,4,5; K=2 -> before [1,2], after [6,7]
        _run(tmp_path, _segments(9), context_neighbors=2)
        b1 = _batch(tmp_path, 1)
        assert b1["neighbors_before"] == ["en 1", "en 2"]
        assert b1["neighbors_after"] == ["en 6", "en 7"]

    def test_boundary_batches_clamped(self, tmp_path):
        _run(tmp_path, _segments(9), context_neighbors=2)
        b0 = _batch(tmp_path, 0)
        assert b0["neighbors_before"] == []           # nothing before first batch
        assert b0["neighbors_after"] == ["en 3", "en 4"]
        b2 = _batch(tmp_path, 2)
        assert b2["neighbors_after"] == []            # nothing after last batch


class TestShotVisuals:
    def test_deduped_by_shot(self, tmp_path):
        # segs 0,1 share shot "A"; seg 2 is shot "B" -> not repeated
        sv = {0: "shot A", 1: "shot A", 2: "shot B"}
        _run(tmp_path, _segments(3), seg_visuals=sv)
        assert _batch(tmp_path, 0)["shot_visuals"] == ["shot A", "shot B"]

    def test_absent_seg_visuals_drop_layer(self, tmp_path):
        _run(tmp_path, _segments(3), seg_visuals={})
        assert _batch(tmp_path, 0)["shot_visuals"] == []

    def test_capped_per_batch(self, tmp_path):
        # Batch 0 spans 3 distinct shots (A, B, C); a cap of 2 keeps only the
        # first two, so the prompt can't balloon on over-segmented video.
        sv = {0: "A", 1: "B", 2: "C"}
        _run(tmp_path, _segments(3), seg_visuals=sv, context_shot_cap=2)
        assert _batch(tmp_path, 0)["shot_visuals"] == ["A", "B"]


class TestCascade:
    def test_neighbour_change_busts_only_affected_batch(self, tmp_path):
        _run(tmp_path, _segments(9), context_neighbors=2)
        fp1 = _meta_fp(tmp_path, 1)

        # A far segment (index 8) is outside batch 1's [1..7] window -> no change
        far = _segments(9)
        far[8].text_src = "far away change"
        _run(tmp_path, far, context_neighbors=2)
        assert _meta_fp(tmp_path, 1) == fp1

        # Segment 2 is batch 1's neighbour_before -> its context busts
        near = _segments(9)
        near[2].text_src = "neighbour change"
        _run(tmp_path, near, context_neighbors=2)
        assert _meta_fp(tmp_path, 1) != fp1


class TestDegradation:
    def test_empty_video_context_still_builds(self, tmp_path):
        # R43: vision degraded (empty video_context) -> still assemble visual +
        # neighbours, no error.
        _run(tmp_path, _segments(3), video_context="",
             seg_visuals={0: "A", 1: "A", 2: "A"})
        b0 = _batch(tmp_path, 0)
        assert b0["video_context"] == ""
        assert b0["shot_visuals"] == ["A"]


class TestReuse:
    def test_unchanged_inputs_not_rebuilt(self, tmp_path):
        _run(tmp_path, _segments(3))
        meta = tmp_path / "work" / "context" / "batch_0000.json.meta.json"
        written0 = json.loads(meta.read_text())["written_at"]
        _run(tmp_path, _segments(3))
        assert json.loads(meta.read_text())["written_at"] == written0  # is_valid -> reuse


class TestSummary:
    def test_one_call_per_batch(self, tmp_path, summary_stub):
        # R41: one summary call per batch (3 batches of 3), not per segment
        _run(tmp_path, _segments(9))
        assert len(summary_stub) == 3

    def test_thinking_disabled_on_translate_model(self, tmp_path, summary_stub):
        _run(tmp_path, _segments(3))
        assert summary_stub[0]["enable_thinking"] is False  # KTD6
        assert summary_stub[0]["stage"] == "context"
        # The context call now names the translate role (U7); the resolved model
        # is the same llm_model_translate as before.
        assert summary_stub[0]["role"].model == Config().llm_model_translate

    def test_summary_folded_into_artifact(self, tmp_path):
        _run(tmp_path, _segments(3))
        assert _batch(tmp_path, 0)["summary"].startswith("SUM:")

    def test_cascade_early_edit_recomputes_tail(self, tmp_path):
        # R41/KTD5: editing batch 0's text changes summary_0, whose value is the
        # summary_in of batch 1, ... so the last batch's fingerprint busts too.
        _run(tmp_path, _segments(9))
        fp_last = _meta_fp(tmp_path, 2)
        edited = _segments(9)
        edited[0].text_src = "a totally different opening line"
        _run(tmp_path, edited)
        assert _meta_fp(tmp_path, 2) != fp_last

    def test_edited_line_recomputes_its_own_summary(self, tmp_path):
        # Determinism guard: a batch's own EN text is a summary fingerprint
        # input, so an edited line never keeps a stale cached summary.
        _run(tmp_path, _segments(9))
        s1_before = _batch(tmp_path, 1)["summary"]
        edited = _segments(9)
        edited[4].text_src = "changed line inside batch 1"  # seg 4 is in batch 1
        _run(tmp_path, edited)
        assert _batch(tmp_path, 1)["summary"] != s1_before

    def test_summary_failure_degrades_without_blocking(self, tmp_path, monkeypatch):
        # R43: a summary StageError drops only the summary layer; the batch
        # artifact is still finalized with the other layers intact.
        def boom(cfg, messages, **kw):
            raise StageError("context", "infra", "down", "summary server down")

        monkeypatch.setattr(ctx.llm, "chat", boom)
        _run(tmp_path, _segments(6))  # 2 batches, so batch 0 has "after" neighbours
        b0 = _batch(tmp_path, 0)
        assert b0["summary"] == ""
        assert b0["neighbors_after"] == ["en 3", "en 4"]
        assert (tmp_path / "work" / "context" / "batch_0000.json.meta.json").exists()

    def test_disabled_summary_makes_no_calls(self, tmp_path, summary_stub):
        _run(tmp_path, _segments(3), enable_summary=False)
        assert len(summary_stub) == 0
        assert _batch(tmp_path, 0)["summary"] == ""
