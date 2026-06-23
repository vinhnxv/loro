"""Mix the dub with the original audio and mux everything into the output video.

The durable marker `mux.json` lives in the workdir (never next to the user's
video); validity additionally requires the recorded output file to still
exist with matching content — deleting the .vi.mp4 while keeping the workdir
re-runs only this stage (R17)."""

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.state import DubState
from loro.utils import ffmpeg, srt

log = logging.getLogger("loro.mux")

# Fallback burn font when the profile names none. libass resolves a font by name
# via fontconfig; the profile supplies the per-language name + glyph sample, and
# preflight probes actual glyph coverage (R17). A Latin-covering font bundled
# under assets/fonts/ backs tier-1 when the host lacks the profile font (R18).
BURN_FONT = "Arial"
# Repo-bundled fonts, wired into the libass `subtitles` filter via fontsdir=
# (libass otherwise resolves FontName through fontconfig only, ignoring a bundled
# file). Empty/absent -> libass falls back to fontconfig, the prior behavior.
FONTS_DIR = Path(__file__).resolve().parents[3] / "assets" / "fonts"
# Identifies the burn re-encode profile in the mux fingerprint, so tuning the
# encoder rebuilds the output (R8); "copy" is the default fast path.
BURN_ENCODER = "libx264 crf18 preset medium yuv420p"


def _burn_force_style(height: int, font: str) -> str:
    """ASS force_style overrides for burned captions (KTD4): pin the profile's
    target-language font (R17), BorderStyle=1 (outline + drop shadow, legible
    over bright video), a bottom MarginV so captions clear lower-thirds and
    player chrome, and a FontSize scaled to the output height so it reads at any
    resolution. Built here, not a Config field."""
    font_size = max(18, round(height * 0.042))
    margin_v = max(20, round(height * 0.04))
    return (f"FontName={font},FontSize={font_size},"
            f"BorderStyle=1,Outline=2,Shadow=1,MarginV={margin_v}")


def _sidecar_path(output: Path, tag: str) -> Path:
    """The target SRT path beside the shipped video, as <basename>.<tag>.srt
    (KTD3) — the <basename>.<lang>.srt convention players and YouTube expect.
    Keeps the locale tag for both the default foo.<tag>.mp4 (-> foo.<tag>.srt) and
    an explicit -o bar.mp4 (-> bar.<tag>.srt); output.with_suffix('.srt') would
    wrongly drop the tag on a custom -o."""
    return output.parent / (output.stem.removesuffix("." + tag) + f".{tag}.srt")


def _write_sidecar(output: Path, srt_target: str, tag: str) -> Path:
    """Copy the rendered srt_target to the sidecar beside the output (R5). An
    idempotent copy written on every mux (including cache-hit reruns), so
    deleting the sidecar triggers a rewrite without rebuilding the video."""
    sidecar = _sidecar_path(output, tag)
    shutil.copyfile(srt_target, sidecar)
    return sidecar


def mux(state: DubState, cfg: Config) -> DubState:
    video = Path(state["video_path"])
    workdir = Path(state["workdir"])
    profile = cfg.language_profile
    # Locale-derived naming (U10): the VI default keeps .vi.mp4 / .vi.srt /
    # language=vie byte-identical.
    tag = cfg.target_lang.lower()
    output = Path(state.get("output_path") or str(video.with_suffix("")) + f".{tag}.mp4")

    # Burn-in (KTD2/KTD4) is the only path that re-encodes video; the default
    # stays -c:v copy. Resolve force_style up front (it depends on the output
    # height) so it can join the fingerprint — toggling burn, the cue width, the
    # font/margins, or the encoder profile all rebuild the output (R8).
    force_style = (_burn_force_style(ffmpeg.probe_video_height(video), profile.font)
                   if cfg.subtitle_burn else "")

    marker = workdir / "mux.json"
    inputs = {
        "dub_sha": artifacts.cached_file_sha256(state["dub_wav"]),
        "video": artifacts.video_fingerprint(video),
        "mode": cfg.original_audio,
        "duck_volume": cfg.duck_volume,
        "srt_sha": artifacts.file_sha256(state["srt_target"]),
        "output_path": str(output),
        "subtitle_burn": cfg.subtitle_burn,
        "srt_burn_max_cue_chars": cfg.srt_burn_max_cue_chars,
        "burn_force_style": force_style,
        "encoder": BURN_ENCODER if cfg.subtitle_burn else "copy",
    }
    if artifacts.is_valid(marker, inputs):
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if output.exists() and artifacts.file_sha256(output) == recorded["output_sha256"]:
            # Rewrite the sidecar even on a cache hit, so deleting it (or never
            # having produced it on an older run) still lands the .vi.srt (R5).
            sidecar = _write_sidecar(output, state["srt_target"], tag)
            log.info("output reused -> %s", output)
            return {"output_path": str(output), "srt_sidecar": str(sidecar)}

    # srt_target stays input 2, mapped only via -map 2:s for the soft track. The
    # duck audio mix is `[0:a]volume[bg];[1:a][bg]amix[aout]`; replace just
    # passes the dub through. When burning, both audio and the [0:v]->subtitles
    # video branch share one filter_complex (KTD4) so the graph carries both.
    if cfg.original_audio == "duck":
        audio_branch = (f"[0:a]volume={cfg.duck_volume}[bg];"
                        "[1:a][bg]amix=inputs=2:duration=first:normalize=0[aout]")
    else:  # replace (skip slots already carry original audio from fit, R23)
        audio_branch = "[1:a]anull[aout]"

    args = ["-i", str(video), "-i", state["dub_wav"], "-i", state["srt_target"]]
    if cfg.subtitle_burn:
        # Render a burn-specific SRT at the tighter cue width (the soft track and
        # sidecar keep the wider srt_target). The subtitles filter reads it by
        # escaped path, never as a mapped input (KTD4).
        burn_srt = workdir / f"transcript.{tag}.burn.srt"
        burn_srt.write_text(
            srt.to_srt_wrapped(state["segments"], state.get("words") or [], side="target",
                               max_chars=cfg.srt_burn_max_cue_chars, max_dur=cfg.srt_max_cue_dur),
            encoding="utf-8")
        sub = ffmpeg.escape_filter_path(burn_srt)
        # Point libass at the bundled fonts dir so the profile font resolves even
        # when the host fontconfig lacks it (R18); absent dir -> fontconfig only.
        fontsdir = (f":fontsdir={ffmpeg.escape_filter_path(FONTS_DIR)}"
                    if FONTS_DIR.is_dir() else "")
        args += [
            "-filter_complex",
            f"[0:v]subtitles={sub}{fontsdir}:force_style='{force_style}'[v];{audio_branch}",
            "-map", "[v]", "-map", "[aout]",
        ]
        video_codec = ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
                       "-pix_fmt", "yuv420p"]
    elif cfg.original_audio == "duck":
        args += ["-filter_complex", audio_branch, "-map", "0:v", "-map", "[aout]"]
        video_codec = ["-c:v", "copy"]
    else:  # replace, no burn: a bare copy + dub-audio map is enough
        args += ["-map", "0:v", "-map", "1:a"]
        video_codec = ["-c:v", "copy"]

    tmp_out = output.with_name(f".tmp.{uuid.uuid4().hex}.{output.name}")
    args += [
        "-map", "2:s", *video_codec, "-c:a", "aac", "-b:a", "192k",
        "-c:s", "mov_text", "-metadata:s:s:0", f"language={profile.iso639_2}",
        "-shortest", str(tmp_out),
    ]
    try:
        ffmpeg.ffmpeg(*args)
        os.replace(tmp_out, output)
    finally:
        tmp_out.unlink(missing_ok=True)

    marker.unlink(missing_ok=True)
    artifacts.produce_json(
        marker, inputs, "mux",
        lambda: {"output_path": str(output),
                 "output_sha256": artifacts.file_sha256(output)},
    )
    sidecar = _write_sidecar(output, state["srt_target"], tag)
    log.info("output -> %s (+ sidecar %s)", output, sidecar.name)
    return {"output_path": str(output), "srt_sidecar": str(sidecar)}
