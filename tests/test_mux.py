"""mux node: sidecar .vi.srt delivery (U2) and the burn-in branch (U3).

mux was previously exercised only end-to-end via test_graph_integration. Here
ffmpeg is faked (the real binary never runs) so we can assert on the sidecar
behaviour and the constructed argument list directly."""

import pytest

from pathlib import Path

from loro.config import Config
from loro.nodes import mux as mux_mod
from loro.nodes.mux import mux
from loro.state import Segment
from loro.utils import ffmpeg, srt


@pytest.fixture
def mux_env(tmp_path, monkeypatch):
    video = tmp_path / "in.mp4"
    video.write_bytes(b"\x00" * 64)
    workdir = tmp_path / "work"
    workdir.mkdir()
    dub = workdir / "dub.wav"
    dub.write_bytes(b"\x00" * 32)
    srt_target = workdir / "transcript.vi.srt"
    srt_target.write_text("1\n00:00:00,000 --> 00:00:01,000\nXin chào\n", encoding="utf-8")

    calls = {"count": 0, "args": None}

    def fake_ffmpeg(*args):
        calls["count"] += 1
        calls["args"] = list(args)
        Path(args[-1]).write_bytes(b"FAKEMP4")  # the tmp output path is last

    monkeypatch.setattr(ffmpeg, "ffmpeg", fake_ffmpeg)

    def state(output_path=None):
        s = {"video_path": str(video), "workdir": str(workdir),
             "dub_wav": str(dub), "srt_target": str(srt_target)}
        if output_path is not None:
            s["output_path"] = str(output_path)
        return s

    return {"state": state, "tmp_path": tmp_path, "workdir": workdir,
            "srt_target": srt_target, "video": video, "calls": calls}


def test_mux_writes_sidecar_matching_srt_vi(mux_env):
    result = mux(mux_env["state"](), Config())
    sidecar = Path(result["srt_sidecar"])
    assert sidecar.exists()
    assert sidecar.read_text(encoding="utf-8") == mux_env["srt_target"].read_text(encoding="utf-8")


def test_mux_sidecar_default_output_keeps_vi_tag(mux_env):
    # Default output is in.vi.mp4 -> sidecar in.vi.srt (the .vi tag survives).
    result = mux(mux_env["state"](), Config())
    assert Path(result["output_path"]).name == "in.vi.mp4"
    assert Path(result["srt_sidecar"]).name == "in.vi.srt"


def test_mux_sidecar_custom_output_preserves_vi_tag(mux_env):
    # An explicit -o bar.mp4 still yields bar.vi.srt, not bar.srt (KTD3).
    out = mux_env["tmp_path"] / "bar.mp4"
    result = mux(mux_env["state"](output_path=out), Config())
    assert Path(result["srt_sidecar"]).name == "bar.vi.srt"
    assert Path(result["srt_sidecar"]).parent == out.parent


def test_mux_fr_target_uses_locale_derived_names_and_metadata(mux_env):
    # R16: an FR run writes <base>.fr.mp4 + <base>.fr.srt and muxes language=fra.
    result = mux(mux_env["state"](), Config(target_lang="fr"))
    assert Path(result["output_path"]).name == "in.fr.mp4"
    assert Path(result["srt_sidecar"]).name == "in.fr.srt"
    args = mux_env["calls"]["args"]
    assert "language=fra" in args  # ISO 639-2 metadata from the profile


def test_mux_es_mx_region_target_naming(mux_env):
    result = mux(mux_env["state"](), Config(target_lang="es-MX"))
    assert Path(result["output_path"]).name == "in.es-mx.mp4"
    assert "language=spa" in mux_env["calls"]["args"]


def test_mux_sidecar_recreated_on_cache_hit_without_rebuild(mux_env):
    state, cfg = mux_env["state"](), Config()
    first = mux(state, cfg)
    assert mux_env["calls"]["count"] == 1          # video encoded once

    sidecar = Path(first["srt_sidecar"])
    sidecar.unlink()
    assert not sidecar.exists()

    second = mux(state, cfg)
    assert mux_env["calls"]["count"] == 1          # cache hit: not rebuilt
    assert sidecar.exists()                        # sidecar rewritten anyway
    assert second["srt_sidecar"] == first["srt_sidecar"]
    assert sidecar.read_text(encoding="utf-8") == mux_env["srt_target"].read_text(encoding="utf-8")


def test_default_branch_is_stream_copy_no_subtitles_filter(mux_env):
    mux(mux_env["state"](), Config())
    args = mux_env["calls"]["args"]
    assert args[args.index("-c:v") + 1] == "copy"
    assert not any("subtitles=" in a for a in args)


# --- U3: opt-in hard burn-in (--burn-subs) ---

@pytest.fixture
def burn_env(mux_env, monkeypatch):
    # A long VI line so the 42-char burn render wraps into multiple cues.
    seg = Segment(index=0, start=0.0, end=10.0, text_src="x",
                  text_target="Đôi khi bạn cần một agent, nhưng lúc khác cần cả đội ngũ phối hợp")
    monkeypatch.setattr(ffmpeg, "probe_video_height", lambda path: 1080)
    state = mux_env["state"]()
    state["segments"] = [seg]
    state["words"] = []
    return {**mux_env, "state": state, "seg": seg}


def test_escape_filter_path_escapes_colons_backslashes_quotes():
    out = ffmpeg.escape_filter_path(r"/tmp/we'ird:dir\sub.srt")
    assert out == r"/tmp/we\'ird\:dir\\sub.srt"


def test_burn_branch_constructs_subtitles_filter_and_keeps_soft_track(burn_env):
    mux(burn_env["state"], Config(subtitle_burn=True))
    args = burn_env["calls"]["args"]
    fc = args[args.index("-filter_complex") + 1]
    assert "subtitles=" in fc                       # libass video branch
    assert "[v]" in args                            # mapped the filtered video
    assert args[args.index("-c:v") + 1] == "libx264"  # re-encode, not copy
    # the soft, toggleable vie track still rides alongside the burn
    assert "mov_text" in args and "language=vie" in args
    assert "-map" in args and "2:s" in args


def test_burn_force_style_has_margin_border_and_scaled_size(burn_env):
    mux(burn_env["state"], Config(subtitle_burn=True))
    fc = burn_env["calls"]["args"][burn_env["calls"]["args"].index("-filter_complex") + 1]
    assert "BorderStyle=1" in fc
    assert "MarginV=" in fc
    assert "FontSize=" in fc
    assert f"FontName={mux_mod.BURN_FONT}" in fc


def test_burn_generates_42char_srt_while_sidecar_keeps_srt_vi(burn_env):
    result = mux(burn_env["state"], Config(subtitle_burn=True))
    burn_srt = burn_env["workdir"] / "transcript.vi.burn.srt"
    assert burn_srt.exists()
    cues = srt.parse_cues(burn_srt.read_text(encoding="utf-8"))
    assert len(cues) > 1                            # wrapped at the 42 limit
    assert all(len(c.text) <= 42 for c in cues)
    # soft track / sidecar keep the wider srt_target unchanged (84-char default)
    sidecar = Path(result["srt_sidecar"])
    assert sidecar.read_text(encoding="utf-8") == burn_env["srt_target"].read_text(encoding="utf-8")


def test_burn_toggle_and_width_change_rebuild_output(burn_env):
    state, calls = burn_env["state"], burn_env["calls"]
    mux(state, Config())                            # burn off
    assert calls["count"] == 1
    mux(state, Config(subtitle_burn=True))          # toggling on -> rebuild (R8)
    assert calls["count"] == 2
    mux(state, Config(subtitle_burn=True))          # same config -> cache hit
    assert calls["count"] == 2
    mux(state, Config(subtitle_burn=True, srt_burn_max_cue_chars=30))  # width -> rebuild
    assert calls["count"] == 3
