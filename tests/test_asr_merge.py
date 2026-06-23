import json
import textwrap

import pytest

from loro.config import Config
from loro.nodes import asr as asr_mod
from loro.nodes.asr import merge_windows, window_bounds
from loro.providers.asr import local as local_provider
from loro.utils import ffmpeg


class TestWindowBounds:
    def test_short_audio_single_window(self):
        assert window_bounds(120.0, 600.0, 10.0) == [(0.0, 120.0)]

    def test_deterministic_layout(self):
        bounds = window_bounds(1500.0, 600.0, 10.0)
        assert bounds == [(0.0, 600.0), (590.0, 1190.0), (1180.0, 1500.0)]
        # Same inputs -> same layout
        assert window_bounds(1500.0, 600.0, 10.0) == bounds

    def test_full_coverage_with_overlap(self):
        bounds = window_bounds(2000.0, 600.0, 10.0)
        assert bounds[0][0] == 0.0
        assert bounds[-1][1] == 2000.0
        for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:]):
            assert next_start < prev_end  # overlapping


class TestMergeWindows:
    def test_sentence_across_boundary_not_lost_or_duplicated(self):
        # AE6: window A truncates the sentence at its edge; window B heard it
        # whole and deeper inside its window. Overlap is [600, 610].
        win_a = {
            "start": 0.0, "end": 610.0,
            "segments": [
                {"start": 580.0, "end": 596.0, "text": "we package the service"},
                {"start": 606.0, "end": 609.9, "text": "and then we deploy"},
            ],
        }
        win_b = {
            "start": 600.0, "end": 1200.0,
            "segments": [
                {"start": 606.0, "end": 612.0, "text": "and then we deploy the model"},
                {"start": 613.0, "end": 620.0, "text": "to production"},
            ],
        }
        merged = merge_windows([win_a, win_b])
        texts = [s["text"] for s in merged]
        assert texts == [
            "we package the service",
            "and then we deploy the model",
            "to production",
        ]
        # Absolute timestamps preserved
        assert merged[1]["start"] == 606.0
        assert merged[1]["end"] == 612.0

    def test_differing_overlap_readings_choose_deeper_side_stably(self):
        # Both windows heard the same span slightly differently; result must
        # be deterministic and prefer the deeper-in-window reading.
        win_a = {
            "start": 0.0, "end": 610.0,
            "segments": [{"start": 601.0, "end": 608.0, "text": "kubernetes is great"}],
        }
        win_b = {
            "start": 600.0, "end": 1200.0,
            "segments": [{"start": 601.2, "end": 608.1, "text": "cooper netties is great"}],
        }
        first = merge_windows([win_a, win_b])
        assert merge_windows([win_a, win_b]) == first  # stable
        assert len(first) == 1  # never both readings

    def test_no_overlap_segments_passthrough(self):
        win_a = {"start": 0.0, "end": 610.0,
                 "segments": [{"start": 10.0, "end": 20.0, "text": "hello"}]}
        win_b = {"start": 600.0, "end": 900.0,
                 "segments": [{"start": 700.0, "end": 710.0, "text": "world"}]}
        merged = merge_windows([win_a, win_b])
        assert [s["text"] for s in merged] == ["hello", "world"]

    def test_single_window_identity(self):
        win = {"start": 0.0, "end": 100.0,
               "segments": [{"start": 1.0, "end": 2.0, "text": "hi"}]}
        assert merge_windows([win]) == win["segments"]


STUB_WORKER = textwrap.dedent('''
    """Stub worker emitting canned NDJSON per input wav."""
    import json, sys

    MODEL_ID = "stub-model"

    for path in sys.argv[1:]:
        # Window length is encoded in the wav stub content by the test
        with open(path) as f:
            payload = json.load(f)
        print(json.dumps({"path": path, "text": payload["text"],
                          "segments": payload["segments"],
                          "words": payload.get("words")}), flush=True)
''')


class TestAsrNodeResume:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        import sys

        # Stub worker: reads JSON "wav" files and echoes their transcription. The
        # worker, _run_worker, and ffmpeg helpers now live on the local provider
        # (U5); the windowing toolkit (window_bounds/merge_windows) stays node-side.
        worker = tmp_path / "stub_worker.py"
        worker.write_text(STUB_WORKER)
        monkeypatch.setattr(local_provider, "WORKER", worker)

        audio = tmp_path / "audio_16k.wav"
        audio.write_bytes(b"fake-audio-bytes")
        # Three windows: duration 1500, window 600, overlap 10
        monkeypatch.setattr(ffmpeg, "probe_duration", lambda p: 1500.0)

        def fake_cut(src, out, start, end):
            json.dump(
                {"text": f"win {start:.0f}",
                 "segments": [{"start": 1.0, "end": 5.0, "text": f"window at {start:.0f}"}]},
                open(out, "w"),
            )

        monkeypatch.setattr(ffmpeg, "cut_audio", fake_cut)
        # The default engine is now assemblyai (cloud); these tests exercise the
        # local Nemotron-windows path, so pin it.
        cfg = Config(nemotron_python=sys.executable, asr_engine="local")
        state = {"workdir": str(tmp_path / "work"), "audio_16k": str(audio)}
        return {"cfg": cfg, "state": state, "workdir": tmp_path / "work"}

    def test_windows_transcribed_and_merged(self, env):
        result = asr_mod.asr(env["state"], env["cfg"])
        assert len(result["segments"]) == 3
        # Absolute offsets applied: window 1 starts at 590
        assert result["segments"][1].start == pytest.approx(591.0)
        assert (env["workdir"] / "asr" / "segments.json").exists()
        # AE4: the local provider yields the {segments, words, srt_src} contract the
        # cloud providers do, and the node owns the shared EN SRT write.
        assert "words" in result
        assert result["srt_src"].endswith("transcript.en.srt")
        assert (env["workdir"] / "transcript.en.srt").exists()

    def test_resume_only_missing_windows(self, env, monkeypatch):
        asr_mod.asr(env["state"], env["cfg"])

        # Invalidate only window 1 by deleting its artifact
        win1 = env["workdir"] / "asr" / "win_0001.json"
        win1.unlink()
        (env["workdir"] / "asr" / "win_0001.json.meta.json").unlink()

        ran = []
        original = local_provider._run_worker

        def spying_run(cfg, asr_dir, jobs):
            ran.extend(j["index"] for j in jobs)
            return original(cfg, asr_dir, jobs)

        monkeypatch.setattr(local_provider, "_run_worker", spying_run)
        result = asr_mod.asr(env["state"], env["cfg"])
        assert ran == [1]  # windows 0 and 2 reused from artifacts
        assert len(result["segments"]) == 3
