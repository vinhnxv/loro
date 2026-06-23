"""Thin wrappers around ffmpeg/ffprobe subprocesses."""

import json
import re
import subprocess
from pathlib import Path


def run(args: list[str]) -> str:
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{args[0]} failed ({' '.join(args)}):\n{proc.stderr[-2000:]}")
    return proc.stdout


def ffmpeg(*args: str) -> None:
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args])


def probe_duration(path: str | Path) -> float:
    out = run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)]
    )
    return float(out.strip())


def probe_video_height(path: str | Path) -> int:
    """Pixel height of the first video stream, used to scale burned-caption
    FontSize to the output resolution (U3)."""
    out = run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=height", "-of", "default=nw=1:nk=1", str(path)]
    )
    return int(out.strip().splitlines()[0])


def escape_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for use as a value inside a -filter_complex
    graph (e.g. `subtitles=<path>`). libavfilter parses the graph in layers: a
    backslash escapes the next char, ':' separates a filter's options, and a
    single quote begins a quoted token — all three must be backslash-escaped or
    the filename is mis-parsed (a known footgun, especially Windows drive
    letters like `C:\\`). Backslash is escaped first so the escapes added for
    ':' and "'" are not themselves re-escaped (U3/KTD4)."""
    s = str(path)
    for ch in ("\\", ":", "'"):
        s = s.replace(ch, "\\" + ch)
    return s


def extract_audio(video: str, out: str, rate: int = 16000, channels: int = 1) -> None:
    ffmpeg("-i", video, "-vn", "-ac", str(channels), "-ar", str(rate), out)


def cut_audio(src: str, out: str, start: float, end: float) -> None:
    ffmpeg("-i", src, "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-c", "copy", out)


def probe_subtitle_streams(path: str | Path) -> list[dict]:
    """List subtitle streams in a container (R34). Each entry carries the
    *subtitle-relative* index (`s_index`, for `-map 0:s:N`), the codec name,
    and the language tag (lowercased, "" when untagged)."""
    out = run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index,codec_name:stream_tags=language",
         "-of", "json", str(path)]
    )
    streams = json.loads(out).get("streams", [])
    result = []
    for s_index, stream in enumerate(streams):
        lang = (stream.get("tags") or {}).get("language", "") or ""
        result.append({
            "s_index": s_index,
            "codec_name": stream.get("codec_name", ""),
            "language": lang.lower(),
        })
    return result


def extract_subtitle_track(video: str, out_srt: str, s_index: int) -> None:
    """Extract one subtitle stream to SRT (ffmpeg converts mov_text/ass/vtt by
    the .srt output extension). `s_index` is the subtitle-relative index."""
    ffmpeg("-i", video, "-map", f"0:s:{s_index}", "-c:s", "srt", out_srt)


def convert_subtitle_file(src: str, out_srt: str) -> None:
    """Normalize a sidecar subtitle file (.srt/.vtt) to SRT."""
    ffmpeg("-i", src, "-c:s", "srt", out_srt)


def extract_frames(video: str, out_dir: Path, count: int, duration: float) -> list[Path]:
    """Sample `count` frames evenly across the video."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = count / max(duration, 1e-6)
    pattern = out_dir / "frame_%03d.jpg"
    ffmpeg("-i", video, "-vf", f"fps={fps:.6f},scale=640:-2", "-frames:v", str(count),
           "-q:v", "4", str(pattern))
    return sorted(out_dir.glob("frame_*.jpg"))


def extract_frames_window(video: str, out_dir: Path, start: float, end: float,
                          count: int = 1) -> list[Path]:
    """Sample up to `count` frames evenly within the time window [start, end].

    Used to describe one shot (U3). A degenerate window — start >= end, e.g.
    two coincident scene cuts — still yields a single frame at `start` rather
    than dividing by zero or returning nothing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = max(1, count)
    span = max(end - start, 0.0)
    pattern = out_dir / "frame_%03d.jpg"
    if span <= 1e-3:
        ffmpeg("-i", video, "-ss", f"{start:.3f}", "-frames:v", "1",
               "-vf", "scale=640:-2", "-q:v", "4", str(pattern))
    else:
        fps = count / span
        ffmpeg("-i", video, "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
               "-vf", f"fps={fps:.6f},scale=640:-2", "-frames:v", str(count),
               "-q:v", "4", str(pattern))
    return sorted(out_dir.glob("frame_*.jpg"))


def detect_silences(path: str | Path, noise_db: float = -40.0,
                    min_duration: float = 0.3) -> list[tuple[float, float]]:
    """Return (start, end) of silent stretches; silencedetect logs to stderr."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"silencedetect failed on {path}:\n{proc.stderr[-2000:]}")
    starts = [float(m) for m in re.findall(r"silence_start:\s*([\d.]+)", proc.stderr)]
    ends = [float(m) for m in re.findall(r"silence_end:\s*([\d.]+)", proc.stderr)]
    return list(zip(starts, ends))


def detect_scenes(path: str | Path, threshold: float = 0.35) -> list[float]:
    """Return timestamps (seconds) of scene cuts, via the scene-score filter.

    The `select='gt(scene,threshold)'` filter passes only frames whose visual
    change exceeds `threshold`; `showinfo` then logs each one's pts_time to
    stderr (parsed like detect_silences). An empty list means no cut was
    detected — the caller treats the whole video as a single shot (U3)."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path),
         "-filter:v", f"select='gt(scene,{threshold})',showinfo",
         "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"scene detect failed on {path}:\n{proc.stderr[-2000:]}")
    return [float(m) for m in re.findall(r"pts_time:\s*([\d.]+)", proc.stderr)]


def atempo(src: str, out: str, factor: float) -> None:
    """Time-stretch audio by `factor` (>1 = faster). Chains filters outside [0.5, 2]."""
    filters = []
    remaining = factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    filters.append(f"atempo={remaining:.6f}")
    ffmpeg("-i", src, "-filter:a", ",".join(filters), out)
