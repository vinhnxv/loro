"""Validate the input video and extract audio tracks for the pipeline.

Also detects an existing English subtitle source (R34): a sidecar `.srt`/
`.vtt` next to the video, or an embedded soft-sub track. When found it is
extracted to the durable artifact `ingest/subs.en.srt`; cross-check later
uses it to short-circuit segments the subtitle already covers (R35)."""

import logging
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.state import DubState
from loro.utils import ffmpeg

log = logging.getLogger("loro.ingest")

# Sidecar extensions tried next to the video, most specific first
SIDECAR_SUFFIXES = (".en.srt", ".en.vtt", ".srt", ".vtt")


def _find_sidecar(video: Path) -> Path | None:
    for suffix in SIDECAR_SUFFIXES:
        candidate = video.with_name(video.stem + suffix)
        if candidate.exists():
            return candidate
    return None


def _pick_english_track(streams: list[dict]) -> dict | None:
    """Prefer an eng/en-tagged track; fall back to the first untagged track;
    never pick a track tagged for another language (R34)."""
    for stream in streams:
        if stream["language"] in ("eng", "en"):
            return stream
    for stream in streams:
        if not stream["language"]:
            return stream
    return None


def _extract_subs(video: Path, ingest_dir: Path, video_fp: dict,
                  cfg: Config) -> str:
    """Return the path to ingest/subs.en.srt, or "" when there is no usable
    English subtitle source. Sidecar files win over embedded tracks (a
    separate file is usually placed deliberately)."""
    if not cfg.enable_embedded_subs:
        return ""

    out = ingest_dir / "subs.en.srt"
    sidecar = _find_sidecar(video)
    if sidecar is not None:
        inputs = {"video": video_fp, "source": f"sidecar:{sidecar.name}",
                  "sidecar_sha": artifacts.file_sha256(sidecar)}
        artifacts.produce(out, inputs, "ingest",
                          lambda tmp: ffmpeg.convert_subtitle_file(str(sidecar), str(tmp)))
        log.info("subtitle sidecar %s -> %s", sidecar.name, out.name)
        return str(out)

    try:
        streams = ffmpeg.probe_subtitle_streams(video)
    except RuntimeError as exc:
        log.warning("could not probe subtitle stream (%s) — skipping subtitles", exc)
        return ""
    track = _pick_english_track(streams)
    if track is None:
        return ""
    inputs = {"video": video_fp,
              "source": f"embedded:{track['s_index']}:{track['codec_name']}"}
    artifacts.produce(out, inputs, "ingest",
                      lambda tmp: ffmpeg.extract_subtitle_track(
                          str(video), str(tmp), track["s_index"]))
    log.info("subtitle embedded track %d (%s) -> %s",
             track["s_index"], track["codec_name"], out.name)
    return str(out)


def ingest(state: DubState, cfg: Config) -> DubState:
    video = Path(state["video_path"]).resolve()
    if not video.exists():
        raise FileNotFoundError(video)

    workdir = Path(state["workdir"])
    ingest_dir = workdir / "ingest"

    duration = ffmpeg.probe_duration(video)
    video_fp = artifacts.video_fingerprint(video)

    audio_16k = ingest_dir / "audio_16k.wav"
    audio_orig = ingest_dir / "audio_orig.wav"
    for out, rate, channels in ((audio_16k, 16000, 1), (audio_orig, 44100, 2)):
        cached = artifacts.produce(
            out,
            {"video": video_fp, "rate": rate, "channels": channels},
            "ingest",
            lambda tmp, rate=rate, channels=channels: ffmpeg.extract_audio(
                str(video), str(tmp), rate=rate, channels=channels
            ),
        )
        log.info("%s %s", out.name, "reused" if cached else "extracted")

    subs_path = _extract_subs(video, ingest_dir, video_fp, cfg)

    log.info("video %s (%.1fs)%s", video.name, duration,
             " + subtitles" if subs_path else "")
    return {
        "video_duration": duration,
        "audio_16k": str(audio_16k),
        "audio_orig": str(audio_orig),
        "subs_path": subs_path,
    }
