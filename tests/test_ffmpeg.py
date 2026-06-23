"""Scene-cut detection and windowed frame extraction (U2).

Like the rest of the suite (test_fit, test_subs) these shell out to real
ffmpeg with tiny lavfi clips — no mocks, so the filter strings are exercised
exactly as production runs them."""

import subprocess

import pytest

from loro.utils import ffmpeg


def _two_shot(path):
    """A hard black->white cut at t≈1.0: the boundary scene-score is ~maximal."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:size=128x72:duration=1:rate=10",
         "-f", "lavfi", "-i", "color=c=white:size=128x72:duration=1:rate=10",
         "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]",
         "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast",
         "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


def _one_shot(path, duration=2):
    """A single static color: no frame-to-frame change, so no scene cut."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=c=blue:size=128x72:duration={duration}:rate=10",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True,
    )


class TestDetectScenes:
    def test_finds_cut_near_boundary(self, tmp_path):
        v = tmp_path / "two.mp4"
        _two_shot(v)
        cuts = ffmpeg.detect_scenes(v, threshold=0.3)
        assert cuts, "expected at least one scene cut"
        assert any(abs(c - 1.0) < 0.3 for c in cuts), f"no cut near 1.0s in {cuts}"

    def test_static_clip_returns_empty(self, tmp_path):
        v = tmp_path / "one.mp4"
        _one_shot(v)
        assert ffmpeg.detect_scenes(v, threshold=0.3) == []

    def test_undecodable_video_raises(self, tmp_path):
        bad = tmp_path / "bad.mp4"
        bad.write_bytes(b"\x00" * 64)
        with pytest.raises(RuntimeError):
            ffmpeg.detect_scenes(bad)


class TestExtractFramesWindow:
    def test_writes_count_frames_in_window(self, tmp_path):
        v = tmp_path / "two.mp4"
        _two_shot(v)
        frames = ffmpeg.extract_frames_window(v, tmp_path / "f", 0.2, 0.8, count=3)
        assert len(frames) == 3
        assert all(p.exists() and p.stat().st_size > 0 for p in frames)

    def test_degenerate_window_yields_one_frame(self, tmp_path):
        v = tmp_path / "two.mp4"
        _two_shot(v)
        # Coincident cuts (start >= end): must not divide by zero or return [].
        frames = ffmpeg.extract_frames_window(v, tmp_path / "g", 1.0, 1.0, count=4)
        assert len(frames) >= 1
        assert frames[0].exists()

    def test_single_frame_default(self, tmp_path):
        v = tmp_path / "two.mp4"
        _two_shot(v)
        frames = ffmpeg.extract_frames_window(v, tmp_path / "h", 0.0, 2.0)
        assert len(frames) == 1
