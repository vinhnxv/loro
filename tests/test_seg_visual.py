"""seg_visual node (U3): one description per shot, mapped to segments, durable
and degradation-tolerant. ffmpeg + Gemma are stubbed here (real ffmpeg lives
in test_ffmpeg.py); this exercises the shot-derivation, mapping, caching, and
R43 degradation logic."""

import pytest

from loro.config import Config
from loro.harness.retry import StageError
from loro.nodes import seg_visual as sv
from loro.state import Segment


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Stub ffmpeg frame extraction + Gemma; chat returns a per-call-numbered
    description so tests can tell shots apart and count calls."""
    (tmp_path / "v.mp4").write_bytes(b"\x00" * 16)
    calls = []

    def fake_extract(video, out_dir, start, end, count):
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in range(count):
            p = out_dir / f"frame_{i:03d}.jpg"
            p.write_bytes(b"x")
            paths.append(p)
        return paths

    def fake_chat(cfg, messages, **kw):
        calls.append(messages)
        return f"shot description {len(calls)}"

    monkeypatch.setattr(sv.ffmpeg, "extract_frames_window", fake_extract)
    monkeypatch.setattr(sv.llm, "image_part", lambda p, *a, **k: {"image": str(p)})
    monkeypatch.setattr(sv.llm, "chat", fake_chat)
    return {"calls": calls, "tmp_path": tmp_path, "monkeypatch": monkeypatch}


def _state(tmp_path, segments, duration=10.0):
    return {
        "workdir": str(tmp_path / "work"),
        "segments": segments,
        "video_path": str(tmp_path / "v.mp4"),
        "video_duration": duration,
    }


def _cuts(monkeypatch, cuts):
    monkeypatch.setattr(sv.ffmpeg, "detect_scenes", lambda video, threshold: cuts)


def test_one_call_per_shot_shared_description(env):
    _cuts(env["monkeypatch"], [5.0])  # two shots: (0,5), (5,10)
    segs = [Segment(0, 0.0, 2.0, "a"), Segment(1, 2.0, 4.0, "b"),
            Segment(2, 6.0, 8.0, "c")]
    # min_shot_duration=0 keeps the cut: this test is about shot mapping, not
    # the short-shot merge (covered by TestShots).
    out = sv.seg_visual(_state(env["tmp_path"], segs), Config(min_shot_duration=0.0))
    assert len(env["calls"]) == 2  # one Gemma call per shot, not per segment
    m = out["seg_visuals"]
    assert m[0] == m[1]    # segments 0,1 share shot 0
    assert m[0] != m[2]    # segment 2 is a different shot


def test_segment_maps_to_its_shot(env):
    # R39: a segment whose midpoint sits in shot 2 gets shot 2's description.
    _cuts(env["monkeypatch"], [5.0])
    segs = [Segment(0, 0.0, 2.0, "a"), Segment(2, 6.0, 8.0, "c")]
    out = sv.seg_visual(_state(env["tmp_path"], segs), Config(min_shot_duration=0.0))
    assert out["seg_visuals"][2] == "shot description 2"  # second shot processed


def test_rerun_cache_hit_still_populates_state(env):
    _cuts(env["monkeypatch"], [5.0])
    segs = [Segment(0, 0.0, 2.0, "a"), Segment(2, 6.0, 8.0, "c")]
    state = _state(env["tmp_path"], segs)
    sv.seg_visual(dict(state), Config(min_shot_duration=0.0))
    assert len(env["calls"]) == 2
    out2 = sv.seg_visual(dict(state), Config(min_shot_duration=0.0))
    assert len(env["calls"]) == 2          # cached: no new model calls
    assert out2["seg_visuals"][0]          # state reloaded from the artifact
    assert out2["seg_visuals"][2]


def test_no_cuts_single_shot(env):
    _cuts(env["monkeypatch"], [])  # whole video is one shot
    segs = [Segment(0, 0.0, 2.0, "a"), Segment(1, 4.0, 6.0, "b"),
            Segment(2, 8.0, 10.0, "c")]
    out = sv.seg_visual(_state(env["tmp_path"], segs), Config())
    assert len(env["calls"]) == 1
    assert len(set(out["seg_visuals"].values())) == 1  # all share one description


def test_no_frames_degrades_without_exception(env):
    # R43: a shot with no extractable frames -> empty description, never raises.
    _cuts(env["monkeypatch"], [])
    env["monkeypatch"].setattr(sv.ffmpeg, "extract_frames_window",
                               lambda *a, **k: [])
    segs = [Segment(0, 0.0, 2.0, "a")]
    out = sv.seg_visual(_state(env["tmp_path"], segs), Config())
    assert out["seg_visuals"][0] == ""
    assert len(env["calls"]) == 0  # never reached the model


class TestShots:
    def test_no_cuts_is_one_shot(self):
        assert sv._shots([], 100.0, 15.0) == [(0.0, 100.0)]

    def test_min_dur_zero_keeps_every_cut(self):
        assert sv._shots([20.0, 50.0], 100.0, 0.0) == [
            (0.0, 20.0), (20.0, 50.0), (50.0, 100.0)]

    def test_cuts_closer_than_min_dur_merge(self):
        # cuts at 5 and 8 are within 12s of the start -> dropped; 30 survives
        assert sv._shots([5.0, 8.0, 30.0], 60.0, 12.0) == [(0.0, 30.0), (30.0, 60.0)]

    def test_tail_stub_merged_into_previous(self):
        # a cut 5s before the end leaves a sub-min_dur tail -> the cut is dropped
        assert sv._shots([50.0], 55.0, 15.0) == [(0.0, 55.0)]

    def test_min_dur_exceeding_duration_is_one_shot(self):
        assert sv._shots([5.0], 10.0, 15.0) == [(0.0, 10.0)]

    def test_cut_heavy_video_is_bounded(self):
        # a cut every 3s would be ~200 shots; the 15s floor bounds it to ~40
        cuts = [float(t) for t in range(2, 600, 3)]
        shots = sv._shots(cuts, 600.0, 15.0)
        assert len(shots) <= 600 / 15 + 1
        assert all(e - s >= 15.0 - 1e-9 for s, e in shots[:-1])  # each >= the floor


def test_stage_error_degrades_then_retry_changes_description(env):
    # R43: Gemma failure -> durable degraded (empty) artifact; deleting it and
    # rerunning with a healthy model yields a real description (which is what
    # busts the downstream context fingerprint).
    _cuts(env["monkeypatch"], [])
    mp = env["monkeypatch"]

    def failing(cfg, messages, **kw):
        env["calls"].append(messages)
        raise StageError("seg_visual", "infra", "boom", "server down")

    mp.setattr(sv.llm, "chat", failing)
    segs = [Segment(0, 0.0, 2.0, "a")]
    state = _state(env["tmp_path"], segs)
    out = sv.seg_visual(dict(state), Config())
    assert out["seg_visuals"][0] == ""

    art = env["tmp_path"] / "work" / "seg_visual" / "shot_0000.json"
    assert art.exists()  # degraded artifact is durable

    art.unlink()  # "xóa shot_0000.json để thử lại"
    mp.setattr(sv.llm, "chat", lambda cfg, m, **kw: "a real description")
    out2 = sv.seg_visual(dict(state), Config())
    assert out2["seg_visuals"][0] == "a real description"
