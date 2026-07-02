"""Tests for the yt-dlp download service (U2).

All tests mock yt-dlp's Python API — no real network calls are made.
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from loro.services import ytdl
from loro.harness import artifacts


class _FakeYdlCtx:
    """Mock yt-dlp YoutubeDL context manager.

    Simulates a download by writing a dummy file to the outtmpl path.
    Records how many times download/extract_info are invoked so tests can
    assert caching behavior.
    """

    instances = []
    info = {"title": "Test Video", "id": "abc123", "ext": "mp4"}

    def __init__(self, opts):
        self.opts = opts
        self.download_calls = 0
        self.extract_info_calls = 0
        _FakeYdlCtx.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def download(self, urls):
        self.download_calls += 1
        outtmpl = self.opts.get("outtmpl", {})
        if isinstance(outtmpl, dict):
            dest = outtmpl.get("default")
        else:
            dest = outtmpl
        if dest:
            Path(dest).write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 32)

    def extract_info(self, url, download=False):
        self.extract_info_calls += 1
        if download:
            self.download([url])
        return dict(self.info)

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.info = {"title": "Test Video", "id": "abc123", "ext": "mp4"}


def _patch_ytdl(monkeypatch, info=None):
    """Patch the yt-dlp module with _FakeYdlCtx and return the class.

    Since ytdl.download() imports yt_dlp lazily inside the build callback, we
    patch sys.modules['yt_dlp'] so the lazy `import yt_dlp` picks up our mock.
    """
    _FakeYdlCtx.reset()
    if info:
        _FakeYdlCtx.info = info
    mock_mod = MagicMock()
    mock_mod.YoutubeDL = _FakeYdlCtx
    DownloadError = type("DownloadError", (Exception,), {})
    mock_mod.utils.DownloadError = DownloadError
    monkeypatch.setitem(sys.modules, "yt_dlp", mock_mod)
    return _FakeYdlCtx


# --- Happy path ---

class TestDownloadHappyPath:
    def test_download_returns_path_title_and_id(self, tmp_path, monkeypatch):
        _patch_ytdl(monkeypatch, info={"title": "My Video", "id": "vid1", "ext": "mp4"})
        dest_dir = tmp_path / "ingest"

        result = ytdl.download("https://example.com/vid", dest_dir)

        assert result["path"] == str(dest_dir / "source.mp4")
        assert result["title"] == "My Video"
        assert result["video_id"] == "vid1"
        assert Path(result["path"]).exists()

    def test_download_creates_dest_dir(self, tmp_path, monkeypatch):
        _patch_ytdl(monkeypatch)
        dest_dir = tmp_path / "nested" / "ingest"

        ytdl.download("https://example.com/v", dest_dir)

        assert dest_dir.exists()

    def test_cached_file_is_valid_artifact(self, tmp_path, monkeypatch):
        cls = _patch_ytdl(monkeypatch)
        dest_dir = tmp_path / "ingest"

        ytdl.download("https://example.com/valid", dest_dir)

        assert artifacts.is_valid(dest_dir / "source.mp4",
                                  {"url": "https://example.com/valid",
                                   "format": ytdl.FORMAT_SELECTOR})


# --- Caching ---

class TestDownloadCaching:
    def test_second_call_uses_cache_no_redownload(self, tmp_path, monkeypatch):
        cls = _patch_ytdl(monkeypatch)
        dest_dir = tmp_path / "ingest"

        ytdl.download("https://example.com/cached", dest_dir)
        assert len(cls.instances) == 1  # yt-dlp was instantiated for download

        # Second call: cache hit, yt-dlp should not be instantiated at all
        cls.reset()  # clear instance count but keep info
        result2 = ytdl.download("https://example.com/cached", dest_dir)

        assert len(cls.instances) == 0  # no yt-dlp instantiation = cache hit
        assert result2["path"] == str(dest_dir / "source.mp4")
        # Metadata should still be available from the info sidecar
        assert result2["title"] == "Test Video"

    def test_different_url_triggers_redownload(self, tmp_path, monkeypatch):
        cls = _patch_ytdl(monkeypatch)
        dest_dir = tmp_path / "ingest"

        ytdl.download("https://example.com/url1", dest_dir)
        first_instance_count = len(cls.instances)
        assert first_instance_count == 1

        cls.reset()
        ytdl.download("https://example.com/url2", dest_dir)
        assert len(cls.instances) == 1  # new download happened


# --- Error handling ---

class TestDownloadErrors:
    def test_download_error_raises_runtime_error(self, tmp_path, monkeypatch):
        DownloadError = type("DownloadError", (Exception,), {})

        class _ErrCtx:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def download(self, urls):
                raise DownloadError("Video unavailable")
            def extract_info(self, url, download=False):
                if download:
                    raise DownloadError("Video unavailable")
                return {"title": "", "id": "x", "ext": "mp4"}

        mock_mod = MagicMock()
        mock_mod.YoutubeDL = _ErrCtx
        mock_mod.utils.DownloadError = DownloadError
        monkeypatch.setitem(sys.modules, "yt_dlp", mock_mod)

        with pytest.raises(RuntimeError) as exc_info:
            ytdl.download("https://example.com/err", tmp_path / "ingest")

        assert "Video unavailable" in str(exc_info.value)

    def test_network_error_raises_runtime_error(self, tmp_path, monkeypatch):
        DownloadError = type("DownloadError", (Exception,), {})

        class _NetErrCtx:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def download(self, urls):
                raise ConnectionError("Network unreachable")
            def extract_info(self, url, download=False):
                if download:
                    raise ConnectionError("Network unreachable")
                return {"title": "", "id": "x", "ext": "mp4"}

        mock_mod = MagicMock()
        mock_mod.YoutubeDL = _NetErrCtx
        mock_mod.utils.DownloadError = DownloadError
        monkeypatch.setitem(sys.modules, "yt_dlp", mock_mod)

        with pytest.raises(RuntimeError) as exc_info:
            ytdl.download("https://example.com/netfail", tmp_path / "ingest")

        assert "Network unreachable" in str(exc_info.value)

    def test_generic_exception_raises_runtime_error(self, tmp_path, monkeypatch):
        DownloadError = type("DownloadError", (Exception,), {})

        class _GenericErrCtx:
            def __init__(self, opts):
                self.opts = opts
            def __enter__(self):
                return self
            def __exit__(self, *exc):
                return False
            def download(self, urls):
                raise ValueError("unexpected")
            def extract_info(self, url, download=False):
                if download:
                    raise ValueError("unexpected")
                return {"title": "", "id": "x", "ext": "mp4"}

        mock_mod = MagicMock()
        mock_mod.YoutubeDL = _GenericErrCtx
        mock_mod.utils.DownloadError = DownloadError
        monkeypatch.setitem(sys.modules, "yt_dlp", mock_mod)

        with pytest.raises(RuntimeError) as exc_info:
            ytdl.download("https://example.com/generr", tmp_path / "ingest")

        assert "unexpected" in str(exc_info.value)


# --- Title sanitization ---

class TestSanitizeTitle:
    def test_strips_slash(self):
        assert "/" not in ytdl.sanitize_title("a/b")

    def test_strips_colon(self):
        assert ":" not in ytdl.sanitize_title("test:here")

    def test_strips_pipe(self):
        assert "|" not in ytdl.sanitize_title("a|b")

    def test_strips_quotes(self):
        assert '"' not in ytdl.sanitize_title('He said "Hi"')

    def test_empty_returns_empty(self):
        assert ytdl.sanitize_title("") == ""

    def test_none_returns_empty(self):
        assert ytdl.sanitize_title(None) == ""


# --- Output name fallback ---

class TestOutputNameFallback:
    def test_empty_title_provides_video_id_for_fallback(self, tmp_path, monkeypatch):
        _patch_ytdl(monkeypatch, info={"title": "", "id": "vid123", "ext": "mp4"})
        dest_dir = tmp_path / "ingest"

        result = ytdl.download("https://example.com/empty", dest_dir)

        assert result["title"] == ""
        assert result["video_id"] == "vid123"