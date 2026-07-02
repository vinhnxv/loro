"""URL detection and workdir-stem derivation for URL video inputs.

Thin utility module mirroring utils/ffmpeg.py — focused functions with clear
docstrings. No network calls are made here (KTD3): the workdir stem is derived
purely from the URL string so the pipeline can name a workdir before reaching
out to download the video.
"""

import hashlib
from urllib.parse import urlparse, parse_qs


def is_url(s: str) -> bool:
    """Return True when the string has an http:// or https:// scheme.

    file://, ftp://, and bare paths all return False — only http/https counts
    as a downloadable URL input (R1).
    """
    if not s:
        return False
    parsed = urlparse(s)
    return parsed.scheme in ("http", "https")


def derive_workdir_stem(url: str) -> str:
    """Derive a workdir name from a video URL without a network call (KTD3/R6).

    For YouTube URLs the video ID is extracted from:
    - the ``v`` query parameter (e.g. ``watch?v=l6KeLCuB90o`` -> ``l6KeLCuB90o``)
    - the first path segment of ``youtu.be/<id>`` short URLs
    - the path segment of ``/embed/<id>`` embed URLs

    For all other URLs (or YouTube URLs without a parseable ID), the first 12
    characters of a SHA-256 hash of the URL are used. The function is
    deterministic: the same URL always produces the same stem.
    """
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()

    # YouTube video ID extraction
    if "youtube.com" in host or "youtu.be" in host:
        # youtu.be/<id> short URL
        if "youtu.be" in host and parsed.path:
            vid = parsed.path.strip("/").split("/")[0]
            if vid:
                return vid
        # youtube.com/embed/<id>
        if "/embed/" in parsed.path:
            seg = parsed.path.split("/embed/")[-1].strip("/")
            if seg:
                return seg
        # youtube.com/watch?v=<id>
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"][0]:
            return qs["v"][0]

    # Fallback: 12-char SHA-256 hash of the full URL
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]