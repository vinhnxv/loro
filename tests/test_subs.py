"""Subtitle ingestion (R34), the coverage/alignment guard (R35) and the
keyword heuristic (R36). Cue parsing and the guard are pure/offline; the
ingest tests use real ffmpeg to mux and probe a tiny subtitle track."""

import json
import subprocess

import pytest

from loro.config import Config
from loro.nodes import crosscheck as xck
from loro.nodes import ingest as ingest_mod
from loro.state import Segment
from loro.utils import srt

SRT_SAMPLE = """1
00:00:01,000 --> 00:00:04,000
We use Kubernetes here.

2
00:00:04,500 --> 00:00:07,000
It orchestrates the containers.
"""

VTT_SAMPLE = """WEBVTT

00:00.000 --> 00:02.000 align:start position:0%
Hello <c>world</c>

cue-2
00:02.000 --> 00:04.000
<00:00:02.500>Foo bar
"""


class TestCueParsing:
    def test_srt_cues(self):
        cues = srt.parse_cues(SRT_SAMPLE)
        assert len(cues) == 2
        assert (cues[0].start, cues[0].end) == (1.0, 4.0)
        assert cues[0].text == "We use Kubernetes here."
        assert cues[1].start == 4.5

    def test_vtt_header_tags_and_settings(self):
        cues = srt.parse_cues(VTT_SAMPLE)
        assert len(cues) == 2
        # MM:SS.mmm short form, cue settings dropped, inline tags stripped
        assert (cues[0].start, cues[0].end) == (0.0, 2.0)
        assert cues[0].text == "Hello world"
        assert cues[1].text == "Foo bar"  # <00:00:02.500> timestamp tag stripped

    def test_zero_length_and_empty_cues_dropped(self):
        text = "1\n00:00:01,000 --> 00:00:01,000\nzero\n\n2\n00:00:02,000 --> 00:00:03,000\n\n"
        assert srt.parse_cues(text) == []


class TestSubtitleKeywords:
    def test_repeated_term_is_picked(self):
        text = ("We use Kubernetes here. Kubernetes orchestrates pods. "
                "We deploy with CI/CD pipelines.")
        kws = xck.subtitle_keywords(text)
        assert "Kubernetes" in kws
        assert "CI/CD" in kws

    def test_sentence_initial_common_word_excluded(self):
        # "We"/"It" start sentences and are stopwords -> never keywords
        kws = xck.subtitle_keywords("We ship it. It works well.")
        assert "We" not in kws and "It" not in kws

    def test_acronym_picked_once(self):
        assert "API" in xck.subtitle_keywords("The API is fast.")


class TestCueTextForSpan:
    def test_full_containment_returns_all(self):
        cue = srt.Cue(10.0, 14.0, "alpha beta gamma delta")
        text, overlap = xck._cue_text_for_span(cue, 8.0, 16.0)
        assert text == "alpha beta gamma delta"
        assert overlap == pytest.approx(4.0)

    def test_straddling_cue_splits_by_time(self):
        cue = srt.Cue(10.0, 14.0, "alpha beta gamma delta")
        first, _ = xck._cue_text_for_span(cue, 8.0, 12.0)
        second, _ = xck._cue_text_for_span(cue, 12.0, 16.0)
        assert first == "alpha beta"
        assert second == "gamma delta"

    def test_no_overlap_returns_empty(self):
        cue = srt.Cue(10.0, 14.0, "alpha beta")
        assert xck._cue_text_for_span(cue, 0.0, 5.0) == ("", 0.0)


class TestSubtitleGuard:
    def _seg(self):
        return Segment(index=0, start=1.0, end=7.0,
                       text_src="We use Kubernetes here it orchestrates the containers")

    def test_full_coverage_and_alignment_qualifies(self):
        cues = srt.parse_cues(SRT_SAMPLE)
        ev = xck._evaluate_subtitle(self._seg(), cues, Config())
        assert ev["qualified"] is True
        assert "Kubernetes" in ev["sub_text"]

    def test_partial_coverage_rejected(self):
        # Only the first half of the segment has a cue
        cues = [srt.Cue(1.0, 3.0, "We use Kubernetes here")]
        ev = xck._evaluate_subtitle(self._seg(), cues, Config())
        assert ev["qualified"] is False
        assert ev["covered"] is True
        assert ev["reason"] == "low_coverage"

    def test_wrong_language_rejected_below_align(self):
        cues = [srt.Cue(1.0, 7.0, "bonjour le monde ceci est un texte different ici")]
        ev = xck._evaluate_subtitle(self._seg(), cues, Config())
        assert ev["qualified"] is False
        assert ev["reason"] == "below_align"


@pytest.fixture
def make_video(tmp_path):
    """Build a tiny mp4, optionally with an embedded English mov_text track."""
    def build(name="in.mp4", with_subs=False, language="eng"):
        srt_path = tmp_path / "embed.srt"
        srt_path.write_text(SRT_SAMPLE, encoding="utf-8")
        out = tmp_path / name
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "lavfi", "-i", "testsrc=duration=8:size=128x72:rate=10",
               "-f", "lavfi", "-i", "sine=frequency=300:duration=8"]
        if with_subs:
            cmd += ["-i", str(srt_path), "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-c:s", "mov_text",
                    "-map", "0:v", "-map", "1:a", "-map", "2:s",
                    f"-metadata:s:s:0", f"language={language}", str(out)]
        else:
            cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
                    "-map", "0:v", "-map", "1:a", str(out)]
        subprocess.run(cmd, check=True)
        return out
    return build


class TestIngestSubtitles:
    def test_no_subtitles_leaves_subs_path_empty(self, make_video, tmp_path):
        video = make_video(with_subs=False)
        state = {"video_path": str(video), "workdir": str(tmp_path / "work")}
        result = ingest_mod.ingest(state, Config())
        assert result["subs_path"] == ""

    def test_embedded_track_extracted(self, make_video, tmp_path):
        video = make_video(with_subs=True)
        state = {"video_path": str(video), "workdir": str(tmp_path / "work")}
        result = ingest_mod.ingest(state, Config())
        assert result["subs_path"]
        cues = srt.parse_cues(open(result["subs_path"], encoding="utf-8").read())
        assert any("Kubernetes" in c.text for c in cues)

    def test_sidecar_preferred_over_embedded(self, make_video, tmp_path):
        video = make_video(with_subs=True)
        sidecar = video.with_suffix(".srt")
        sidecar.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nSidecar wins here.\n", encoding="utf-8")
        state = {"video_path": str(video), "workdir": str(tmp_path / "work")}
        result = ingest_mod.ingest(state, Config())
        text = open(result["subs_path"], encoding="utf-8").read()
        assert "Sidecar wins" in text

    def test_no_embedded_subs_flag_skips_extraction(self, make_video, tmp_path):
        video = make_video(with_subs=True)
        state = {"video_path": str(video), "workdir": str(tmp_path / "work")}
        result = ingest_mod.ingest(state, Config(enable_embedded_subs=False))
        assert result["subs_path"] == ""

    def test_non_english_track_skipped(self, make_video, tmp_path):
        video = make_video(with_subs=True, language="fra")
        state = {"video_path": str(video), "workdir": str(tmp_path / "work")}
        result = ingest_mod.ingest(state, Config())
        assert result["subs_path"] == ""

    def test_sidecar_change_reextracts(self, make_video, tmp_path):
        video = make_video(with_subs=False)
        sidecar = video.with_suffix(".srt")
        sidecar.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nFirst version.\n", encoding="utf-8")
        state = {"video_path": str(video), "workdir": str(tmp_path / "work")}
        first = ingest_mod.ingest(state, Config())["subs_path"]
        assert "First version" in open(first, encoding="utf-8").read()

        sidecar.write_text(
            "1\n00:00:01,000 --> 00:00:04,000\nSecond version edited.\n", encoding="utf-8")
        second = ingest_mod.ingest(state, Config())["subs_path"]
        assert "Second version" in open(second, encoding="utf-8").read()
