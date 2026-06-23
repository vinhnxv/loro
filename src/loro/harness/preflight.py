"""Capability checks before the graph runs: collect every problem, fail once.

Checks capability, not just liveness — e.g. the configured Gemma model must
actually be served, and when cross-check is enabled a short real-speech clip
from the video proves the audio model actually transcribes audio before we burn
40 minutes of pipeline (R4).

Per-engine ASR/TTS prerequisites (key presence, liveness/auth probes, voice-name
validation, interpreter existence) live on each engine's provider (U6); this
module validates the active ASR and TTS provider and keeps the shared checks:
ffmpeg presence, the LLM model-serving probe, the audio-input probe, override
validation, and the burn-in glyph checks.
"""

import base64
import json
import logging
import re
import shutil
import subprocess
from pathlib import Path

from loro import providers
from loro.config import Config
from loro.nodes.mux import FONTS_DIR
from loro.profiles import is_profiled, registered_tags
from loro.providers.base import PROBE_TIMEOUT
from loro.services import llm
from loro.utils import ffmpeg

log = logging.getLogger("loro.preflight")

_SEG_KEY_RE = re.compile(r"seg_(\d+)")


def out_of_range_override_keys(overrides: dict[str, str], num_segments: int) -> list[str]:
    """Override keys that map to no current segment — either not a `seg_NNNN`
    id, or an index at/after the current segment count. Preflight validates
    overrides.json *shape* before the run, but the segment count only exists
    after sentence_seg re-segments; a `seg_NNNN` key that survived an upgrade
    may now point past the new count. Surfacing these (translate skips-and-logs
    them) keeps a user fix from being silently dropped or applied to the wrong
    line (U5)."""
    bad = []
    for key in overrides:
        m = _SEG_KEY_RE.fullmatch(key)
        if not m or int(m.group(1)) >= num_segments:
            bad.append(key)
    return sorted(bad)


class PreflightError(RuntimeError):
    pass


def _extract_probe_clip(video: str | Path, workdir: str | Path) -> Path:
    """A short real-speech clip from the video for the audio probe. Silence is
    useless here: a llama.cpp model returns empty content for silence whether or
    not it can hear, so silence cannot tell a working audio model from a
    text-only one. Real speech transcribes to non-empty text on a model that
    processes audio and stays empty on one that ignores it. Sampled ~10% in to
    skip a title/silent intro, capped at 12s to keep the request small."""
    out = Path(workdir) / ".audio_probe.wav"
    dur = ffmpeg.probe_duration(video)
    start = min(10.0, dur * 0.1) if dur > 2.0 else 0.0
    window = min(12.0, max(1.0, dur - start))
    ffmpeg.ffmpeg("-ss", f"{start:.3f}", "-i", str(video), "-t", f"{window:.3f}",
                  "-ac", "1", "-ar", "16000", str(out))
    return out


def _probe_audio_input(cfg: Config, video: str | Path, workdir: str | Path) -> None:
    # Probe the audio model specifically: on the split profile llm_model is
    # the vision-only 26B, so the probe must target llm_model_audio (the 12B that
    # actually hears audio) or it would test the wrong capability. The clip is
    # real speech (not silence) so an empty reply means the model did not
    # process the audio — llm.chat surfaces that as an empty_response error.
    clip = _extract_probe_clip(video, workdir)
    b64 = base64.b64encode(clip.read_bytes()).decode("ascii")
    content = [
        {"type": "text", "text": "Transcribe this audio. Reply with the text only."},
        {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
    ]
    llm.chat(cfg, [{"role": "user", "content": content}], max_tokens=64,
              temperature=0.0, role=cfg.llm_role("audio"), enable_thinking=False)


def _has_subtitles_filter() -> bool:
    """True when the local ffmpeg was built with the libass `subtitles` filter
    that --burn-subs needs (R9). Matches the filter-name column of
    `ffmpeg -filters`, not a stray mention in some other filter's description."""
    try:
        out = ffmpeg.run(["ffmpeg", "-hide_banner", "-filters"])
    except Exception:
        return False
    return re.search(r"(?m)^\s*\S+\s+subtitles\b", out) is not None


def _renders_glyphs(workdir: str | Path, font: str, sample: str) -> bool:
    """Trial libass render of the profile's representative glyph sample with the
    profile's burn font (R17). Burns the sample (white) over a black frame and
    reads back the average luma via signalstats: a font that truly renders the
    glyphs draws bright pixels (YAVG > 0); a font that resolves but lacks the
    glyphs draws nothing, so the frame stays black (YAVG ~ 0). Probes glyph
    *coverage*, not mere font presence — a presence-only check passes on a Latin
    substitute that then burns tofu. The resolution order is profile font ->
    bundled fallback -> fail (R18): fontsdir exposes the bundled font to libass,
    so a host missing the profile font can still satisfy coverage from the bundle."""
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    probe_srt = wd / ".glyph_probe.srt"
    probe_srt.write_text(
        f"1\n00:00:00,000 --> 00:00:01,000\n{sample}\n", encoding="utf-8")
    sub = ffmpeg.escape_filter_path(probe_srt)
    fontsdir = (f":fontsdir={ffmpeg.escape_filter_path(FONTS_DIR)}"
                if FONTS_DIR.is_dir() else "")
    style = f"FontName={font},FontSize=72"
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "info", "-f", "lavfi",
         "-i", "color=c=black:s=480x160:d=1:r=1",
         "-vf", f"subtitles={sub}{fontsdir}:force_style='{style}',signalstats,metadata=print",
         "-frames:v", "1", "-an", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return False
    m = re.search(r"lavfi\.signalstats\.YAVG=([\d.]+)", proc.stderr)
    return bool(m) and float(m.group(1)) > 0.5


def preflight(cfg: Config, video: str | Path, workdir: str | Path) -> None:
    problems: list[str] = []

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            problems.append(f"`{tool}` not found on PATH")

    # Language profile gate (R4): an unprofiled --target-lang must not silently
    # produce wrong-CPS, tofu-subtitle output after paying for cloud TTS. Reject
    # it unless --allow-fallback opts into the best-effort generic profile, which
    # we then warn about loudly. Profiled tags (and their region variants) pass.
    if not is_profiled(cfg.target_lang):
        if cfg.allow_fallback:
            log.warning("target language %r is unprofiled — running best-effort on the "
                        "generic profile (wrong-CPS/missing-glyph output is possible); "
                        "tier-1 languages: %s", cfg.target_lang,
                        ", ".join(registered_tags()))
        else:
            problems.append(
                f"unprofiled target language {cfg.target_lang!r}; pass --allow-fallback to "
                f"run best-effort on the generic profile, or pick a profiled language "
                f"({', '.join(registered_tags())})")

    # Validate only the selected ASR engine's prerequisites (R9), through its
    # provider (U6) — a cloud engine needs a key + a reachable/authenticating
    # endpoint; the local engine needs the NeMo interpreter (and Granite when
    # cross-check runs). The vision/translate/seg model-server checks below run on
    # every engine.
    asr_provider = providers.asr(cfg.asr_engine)
    problems.extend(asr_provider.preflight(cfg))

    # source_lang="auto" needs an engine that can identify the spoken language;
    # the local Nemotron path cannot, so it requires an explicit --source-lang
    # (U7/R12), gated on the capability flag rather than an engine-name check.
    if cfg.source_lang == "auto" and not asr_provider.detects_language:
        problems.append(
            f"--source-lang auto needs an ASR engine that detects language, but "
            f"`{cfg.asr_engine}` cannot — pass an explicit --source-lang (e.g. en)")
    # The crosscheck ensemble is English-tuned (R3 scope): a non-EN configured
    # source whose run includes the crosscheck may be mis-calibrated — warn, not
    # fail. (For `auto` the detected source is unknown until ASR runs; the runtime
    # surfaces low-confidence/mixed detection instead.)
    if (cfg.source_lang not in ("en", "auto") and asr_provider.wants_crosscheck
            and cfg.enable_cross_check):
        log.warning("source language %r is not English while the cross-check ensemble "
                    "is English-tuned — transcript cross-check calibration may be off "
                    "(R3); consider --no-cross-check", cfg.source_lang)

    if not Path(video).exists():
        problems.append(f"video not found: {video}")

    try:
        wd = Path(workdir)
        wd.mkdir(parents=True, exist_ok=True)
        probe = wd / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        problems.append(f"workdir not writable ({workdir}): {exc}")

    overrides_file = Path(workdir) / "overrides.json"
    if overrides_file.exists():
        try:
            data = json.loads(overrides_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in data.items()
            ):
                raise ValueError("must be a JSON object like {\"seg_0012\": \"<translation>\"}")
        except (json.JSONDecodeError, ValueError) as exc:
            problems.append(f"overrides.json is invalid ({overrides_file}): {exc}")

    # R21: a work dir is single-target by convention — overrides.json maps
    # seg_NNNN -> target text in ONE target language. If this work dir already
    # holds a target subtitle for a DIFFERENT language, switching targets reuses
    # stale per-target overrides/artifacts; warn (the run proceeds — the user owns
    # the work dir). Compare each transcript.<tag>.srt tag against this run's
    # source/target; anything else is a prior target.
    wd = Path(workdir)
    if wd.is_dir():
        known = {cfg.target_lang.lower(), cfg.source_lang.lower()}
        prior = sorted({p.name[len("transcript."):-len(".srt")]
                        for p in wd.glob("transcript.*.srt")
                        if ".burn." not in p.name} - known - {"auto"})
        if prior:
            log.warning("work dir %s already holds target subtitles for %s but this run "
                        "targets %r — work dirs are single-target by convention; "
                        "overrides.json and cached artifacts may be stale for the new "
                        "target (R21). Use a fresh work dir per target language.",
                        workdir, prior, cfg.target_lang)

    # Group every per-role endpoint by (host, api_key) so each distinct host is
    # contacted once and all the models it must serve are checked together. On
    # the single-host profile this is one call; when a role is split onto its own
    # host (e.g. audio on oMLX) that host is checked independently (R1/R2/R38).
    # Source each role's (host, key, model) from the single accessor (KTD7) so
    # the grouping below and the call sites share one contract.
    roles = [
        (cfg.llm_role("vision"), "vision/text", "LLM_MODEL_VISION"),
        (cfg.llm_role("translate"), "translate", "LLM_MODEL_TRANSLATE"),
        (cfg.llm_role("seg"), "segmentation", "LLM_MODEL_SEG"),
    ]
    # The audio endpoint serves only the cross-check re-listen (and the audio
    # probe below) on an engine that wants the crosscheck (the local engine,
    # KTD8), so it is checked only then.
    if providers.asr(cfg.asr_engine).wants_crosscheck and cfg.enable_cross_check:
        roles.append((cfg.llm_role("audio"), "audio", "LLM_MODEL_AUDIO"))

    by_endpoint: dict[tuple[str, str], dict[str, tuple[str, str]]] = {}
    for r, label, env in roles:
        by_endpoint.setdefault((r.host, r.api_key), {}).setdefault(r.model, (label, env))

    alive_endpoints: set[tuple[str, str]] = set()
    for (host, key), models_needed in by_endpoint.items():
        try:
            served = llm.list_models(cfg, timeout=PROBE_TIMEOUT, host=host, api_key=key)
            alive_endpoints.add((host, key))
            for model, (role, env) in models_needed.items():
                if model not in served:
                    problems.append(
                        f"Model server ({host}) is alive but does not serve the {role} model "
                        f"`{model}` (set via {env}; available: {', '.join(served) or 'no models'})"
                    )
        except Exception as exc:
            problems.append(f"Model server unreachable ({host}): {exc}")

    # Validate only the selected TTS engine (R10), through its provider (U6) — the
    # preset cloud engines need a key + a reachable endpoint + known voice names;
    # Higgs needs its server; the vieneu venv is checked lazily in
    # VieNeuClient.__enter__, so a vieneu run adds no TTS check here.
    tts_provider = providers.tts(cfg.tts_engine)
    problems.extend(tts_provider.preflight(cfg))

    # Language-aware cloning gate (R14/KTD7): a cloning-only engine that cannot
    # clone in the target language (VieNeu for any non-VI target) would otherwise
    # produce wrong-language audio that the silence/duration QA gate passes — catch
    # it before any billing. (A preset engine has clones=False, so it never trips.)
    if tts_provider.clones and not tts_provider.clones_in(cfg.target_lang):
        problems.append(
            f"TTS engine `{cfg.tts_engine}` cannot clone a voice in the target "
            f"language {cfg.target_lang!r} (it has no preset voices to fall back "
            f"on) — pick a preset engine (--tts-engine soniox|gemini) or a target "
            f"the engine supports")

    audio_role = cfg.llm_role("audio")
    if (providers.asr(cfg.asr_engine).wants_crosscheck and cfg.enable_cross_check
            and (audio_role.host, audio_role.api_key) in alive_endpoints):
        try:
            _probe_audio_input(cfg, video, workdir)
        except Exception as exc:
            problems.append(
                f"audio model `{cfg.llm_model_audio}` does not accept audio input (needed for "
                f"cross-check; point LLM_MODEL_AUDIO at an audio-capable model, or disable with "
                f"--no-cross-check): {exc}"
            )

    # Burn-in (--burn-subs) gates on a libass-capable ffmpeg and a font that can
    # actually render the target language's glyphs (R17/R18). Inert for the default
    # soft/sidecar delivery, so a non-burn run never probes fonts. Resolution
    # order: profile font -> bundled fallback (via fontsdir) -> fail.
    if cfg.subtitle_burn:
        profile = cfg.language_profile
        if not _has_subtitles_filter():
            problems.append(
                "ffmpeg has no `subtitles` filter (libass) — needed for --burn-subs "
                "to render subtitles into the picture; install an ffmpeg build with libass "
                "or drop --burn-subs"
            )
        elif not _renders_glyphs(workdir, profile.font, profile.glyph_sample):
            problems.append(
                f"no font with adequate glyph coverage for {profile.locale!r} when "
                f"burning subtitles: neither the profile font `{profile.font}` nor a "
                f"bundled fallback rendered the sample {profile.glyph_sample!r} (the "
                f"test render came out empty/tofu) — install a font covering the target "
                f"glyphs (or add one under assets/fonts/) or drop --burn-subs"
            )

    if problems:
        listing = "\n".join(f"  - {p}" for p in problems)
        raise PreflightError(f"Preflight found {len(problems)} problem(s):\n{listing}")
