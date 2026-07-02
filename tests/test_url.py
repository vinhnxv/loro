"""Tests for URL detection and workdir-stem derivation (U1)."""

import hashlib
from urllib.parse import urlparse

import pytest

from loro.utils.url import is_url, derive_workdir_stem


# --- is_url ---

class TestIsUrl:
    def test_https_url_is_url(self):
        assert is_url("https://www.youtube.com/watch?v=l6KeLCuB90o") is True

    def test_http_url_is_url(self):
        assert is_url("http://example.com") is True

    def test_local_path_not_url(self):
        assert is_url("/path/to/video.mp4") is False

    def test_relative_path_not_url(self):
        assert is_url("video.mp4") is False

    def test_empty_string_not_url(self):
        assert is_url("") is False

    def test_file_scheme_not_url(self):
        # file:// is not http/https
        assert is_url("file:///local/path.mp4") is False

    def test_ftp_scheme_not_url(self):
        assert is_url("ftp://example.com/file.mp4") is False

    def test_youtu_be_short_url_is_url(self):
        assert is_url("https://youtu.be/l6KeLCuB90o") is True

    def test_vimeo_url_is_url(self):
        assert is_url("https://vimeo.com/123456789") is True


# --- derive_workdir_stem ---

class TestDeriveWorkdirStem:
    def test_youtube_watch_url_extracts_video_id(self):
        assert derive_workdir_stem("https://www.youtube.com/watch?v=l6KeLCuB90o") == "l6KeLCuB90o"

    def test_youtube_short_url_extracts_video_id(self):
        assert derive_workdir_stem("https://youtu.be/l6KeLCuB90o") == "l6KeLCuB90o"

    def test_youtube_watch_url_with_extra_params_extracts_video_id(self):
        url = "https://www.youtube.com/watch?v=l6KeLCuB90o&t=42s&list=PL123"
        assert derive_workdir_stem(url) == "l6KeLCuB90o"

    def test_youtube_embed_url_extracts_video_id(self):
        assert derive_workdir_stem("https://www.youtube.com/embed/l6KeLCuB90o") == "l6KeLCuB90o"

    def test_youtube_embed_url_with_trailing_path_no_slash_in_stem(self):
        # /embed/abc123/playlist should extract only "abc123", not "abc123/playlist"
        stem = derive_workdir_stem("https://www.youtube.com/embed/abc123/playlist")
        assert stem == "abc123"
        assert "/" not in stem

    def test_non_youtube_url_returns_12_char_hash(self):
        url = "https://vimeo.com/123456789"
        stem = derive_workdir_stem(url)
        assert len(stem) == 12
        expected = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        assert stem == expected

    def test_same_url_produces_same_stem(self):
        url = "https://vimeo.com/123456789"
        assert derive_workdir_stem(url) == derive_workdir_stem(url)

    def test_different_urls_produce_different_stems(self):
        assert derive_workdir_stem("https://vimeo.com/123456789") != derive_workdir_stem("https://vimeo.com/987654321")

    def test_youtube_url_without_v_param_returns_hash(self):
        # A YouTube URL without a v query param falls back to hash
        url = "https://www.youtube.com/feed/trending"
        stem = derive_workdir_stem(url)
        assert len(stem) == 12
        expected = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
        assert stem == expected