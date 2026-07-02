"""Video download service using yt-dlp's Python API (U2).

Mirrors services/soniox.py — a thin client with clear error logging. The
download is cached via ``artifacts.produce()`` so reruns of the same URL skip
re-downloading (R5). The artifact path is always ``source.mp4`` because yt-dlp
is configured with ``merge_output_format=mp4`` and ``final_ext=mp4``, forcing a
deterministic extension regardless of source format (KTD4).

yt-dlp is imported lazily inside ``download()`` so a missing or incompatible
yt-dlp does not break import of unrelated modules. Download errors
(video unavailable, private, region-locked, network failure) are caught and
re-raised as ``RuntimeError`` with a clear message, not a raw traceback (R8).
"""

import json
import logging
from pathlib import Path

from loro.harness import artifacts

log = logging.getLogger("loro.ytdl")

FORMAT_SELECTOR = "bestvideo*+bestaudio/best"

# Filesystem-unsafe characters to strip from titles for output naming (R7).
_UNSAFE_CHARS = '<>:"/\\|?*\n\r\t'


def sanitize_title(title: str | None) -> str:
    """Sanitize a video title for use in a filename (R7).

    Strips/replaces characters that are invalid in filenames on common
    filesystems. Returns an empty string for None or empty input — the caller
    falls back to the video ID in that case.
    """
    if not title:
        return ""
    return "".join(c if c not in _UNSAFE_CHARS else "_" for c in title).strip()


def download(url: str, dest_dir: str | Path, cfg=None) -> dict:
    """Download a video from ``url`` to ``dest_dir / "source.mp4"`` (R2/R5/R8/R9).

    Uses ``artifacts.produce()`` for caching: on a cache hit (same URL + format
    fingerprint) the existing file is reused and yt-dlp is not called. Returns
    ``{"path": str, "title": str, "video_id": str}`` — title and video_id come
    from yt-dlp metadata, stored in a sidecar JSON so they are available on
    cache hits without a network call.

    Raises ``RuntimeError`` with a clear message on download errors.
    """
    dest_dir = Path(dest_dir)
    artifact = dest_dir / "source.mp4"
    info_file = dest_dir / "source.mp4.info.json"
    fp = {"url": url, "format": FORMAT_SELECTOR}

    def build(tmp_path: Path) -> None:
        # Lazy import so missing yt-dlp doesn't break other modules
        import yt_dlp

        opts = {
            "format": FORMAT_SELECTOR,
            "merge_output_format": "mp4",
            "final_ext": "mp4",
            "outtmpl": {"default": str(tmp_path)},
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Persist metadata for cache-hit retrieval
                meta = {
                    "title": info.get("title", "") if info else "",
                    "video_id": info.get("id", "") if info else "",
                }
                info_file.parent.mkdir(parents=True, exist_ok=True)
                info_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            # Catch yt-dlp's DownloadError and any other exception, re-raise
            # as RuntimeError with a clear message (R8)
            raise RuntimeError(
                f"Failed to download video from {url}: {exc}"
            ) from exc

    cached = artifacts.produce(artifact, fp, "download", build)

    if cached:
        log.info("download cache hit for %s — skipping yt-dlp", url)

    # Read metadata from the sidecar (written during build, or already present
    # from a prior run). Fall back to empty strings if missing.
    title = ""
    video_id = ""
    if info_file.exists():
        try:
            meta = json.loads(info_file.read_text(encoding="utf-8"))
            title = meta.get("title", "")
            video_id = meta.get("video_id", "")
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "path": str(artifact),
        "title": title,
        "video_id": video_id,
    }