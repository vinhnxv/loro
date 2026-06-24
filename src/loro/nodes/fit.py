"""Fit each TTS clip into its subtitle slot and assemble the dub track.

A clip may spill into the silence before the next segment; only when it would
overlap the next line is it sped up (capped at cfg.max_tempo — beyond that the
overflow is accepted rather than producing chipmunk audio).

The timeline sample rate is pinned by config (clips at other rates are
resampled); every placed clip gets a short fade at both ends against clicks;
skipped segments leave their slot empty in duck mode (the ducked original
shows through) or carry the original audio at full level in replace mode
(R23). The assembled `fit/dub.vi.wav` is an artifact fingerprinted over the
ordered (clip hash | skip status) sequence, so a previously skipped segment
gaining a clip invalidates the dub and the muxed output (R17)."""

import logging
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

from loro.config import Config
from loro.harness import artifacts
from loro.harness.ledger import SkipLedger
from loro.state import DubState, Segment, segment_id
from loro.utils import ffmpeg

log = logging.getLogger("loro.fit")

# Placement-policy version, folded into the dub fingerprint (U2). Bump this
# whenever the timeline-placement logic changes shape, so an existing
# dub.<lang>.wav cached under the old policy is invalidated and rebuilt once.
# 1 = pre-U2 (over-cap clips kept onset and summed over the next segment);
# 2 = U2 (interior over-cap spill trimmed at the next segment's onset).
PLACEMENT_POLICY = 2

# Trailing silence padded onto the timeline past video_duration. The LAST clip is
# never trimmed at an onset, so it may spill into this headroom (mux preserves it,
# U3/R2); beyond it _place clamps the tail at the timeline end and audio is
# dropped. Shared by build (timeline sizing) and _record_overruns (the last-clip
# overrun gate) so the two never disagree on how much spill is recoverable.
TAIL_HEADROOM_SEC = 1.0


def _slot_end(segments: list[Segment], i: int, video_duration: float) -> float:
    if i + 1 < len(segments):
        return segments[i + 1].start
    return video_duration


def _clip_sha(path: str) -> str:
    return artifacts.cached_file_sha256(path)


def _clip_duration(path: str | Path) -> float:
    """Wav clip duration from the header only — no ffprobe subprocess — for the
    per-call overrun reprobe (U4) on a CACHE HIT, where build did not run and so
    left no probed duration to reuse. The reprobe runs for every clip on such a
    fit() call, so an ffprobe fork per segment would be a real cost on long
    videos; the soundfile header read is microseconds. TTS clips are always wav,
    and frames/samplerate agrees with build's ffmpeg.probe_duration to far within
    the fit_overflow_tolerance band, so the overrun decision is unchanged."""
    info = sf.info(str(path))
    return info.frames / info.samplerate


def _read_mono_at(path: str | Path, sr: int, scratch: Path) -> np.ndarray:
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != sr:
        tmp = scratch / f".resample.{uuid.uuid4().hex}.wav"
        ffmpeg.ffmpeg("-i", str(path), "-ar", str(sr), "-ac", "1", str(tmp))
        try:
            audio, _ = sf.read(str(tmp), dtype="float32", always_2d=False)
        finally:
            tmp.unlink(missing_ok=True)
    return audio


def _fade(audio: np.ndarray, sr: int, fade_ms: float) -> np.ndarray:
    n = min(int(sr * fade_ms / 1000), len(audio) // 2)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype="float32")
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def _placement(start: float, slot_end: float, clip_dur: float, cfg: Config) -> float:
    """Where to place a clip that fits its slot (KTD3). "center" nudges it
    forward by min(fit_max_center_offset, slack/2) so a short clip doesn't
    finish early, clamped so onset never precedes `start` nor overruns the slot;
    "start" left-aligns (the old behavior)."""
    if cfg.fit_alignment != "center":
        return start
    slack = (slot_end - start) - clip_dur
    if slack <= 0:
        return start
    offset = min(cfg.fit_max_center_offset, slack / 2)
    return max(start, min(start + offset, slot_end - clip_dur))


def _place(timeline: np.ndarray, audio: np.ndarray, at: float, sr: int) -> None:
    offset = int(at * sr)
    end = min(offset + len(audio), len(timeline))
    if end > offset:
        timeline[offset:end] += audio[: end - offset]


def _record_overruns(segments: list[Segment], video_duration: float,
                     ledger: SkipLedger, cfg: Config,
                     built_durations: dict[str, float] | None = None) -> None:
    """Reconcile the run's placement-layer `fit_overflow` set in ONE write
    (U4/R3/KTD7).

    Called OUTSIDE the cached `build` (artifacts.produce skips build on a cache
    hit — exactly the resumed run where a VI clip overran and the run still
    exited 0), so the overrun decision is re-derived on EVERY fit() call from the
    segment geometry, not gated on a rebuild. It is computable without the built
    timeline (so it works identically on a cache hit and under both --original-
    audio duck and replace). `built_durations` carries the clip durations build
    already probed on a rebuild, so a fresh run does not re-read every clip header;
    a cache hit (build skipped) falls back to the soundfile header read.

    Two geometries drop audio:
    - INTERIOR clip: U2 trims its spilled tail at the next segment's onset,
      dropping (capped - slot). Flagged when capped > slot * fit_overflow_tolerance
      (a material post-cap overrun; the band keeps a normal few-percent overrun off
      exit 2, KTD7).
    - LAST clip: never trimmed at an onset — it spills into the timeline's trailing
      headroom and is clamped at its end, so audio is dropped only when the capped
      clip exceeds slot + TAIL_HEADROOM_SEC (the headroom mux can still preserve).

    A CPS segment already carrying a best-effort length_overflow stays exit-0 and
    is never promoted here (KTD2); the reconcile drops stale fit_overflows whose
    clip now fits."""
    overflow_ids: set[str] = set()
    for i, seg in enumerate(segments):
        if not seg.tts_wav or seg.skipped:
            continue
        slot = _slot_end(segments, i, video_duration) - seg.start
        if slot <= 0:
            continue
        sid = segment_id(seg)
        if built_durations is not None and sid in built_durations:
            clip_dur = built_durations[sid]
        else:
            clip_dur = _clip_duration(seg.tts_wav)
        if clip_dur <= slot:
            continue
        capped = clip_dur / min(clip_dur / slot, cfg.max_tempo)
        if i + 1 < len(segments):
            overran = capped > slot * cfg.fit_overflow_tolerance
        else:
            overran = capped > slot + TAIL_HEADROOM_SEC
        if overran:
            overflow_ids.add(sid)
    ledger.reconcile_fit_overflows(overflow_ids)


def fit(state: DubState, cfg: Config) -> DubState:
    segments = state["segments"]
    workdir = Path(state["workdir"])
    fit_dir = workdir / "fit"
    fit_dir.mkdir(parents=True, exist_ok=True)
    sr = cfg.timeline_sr
    video_duration = state["video_duration"]

    entries = []
    any_skip = False
    for i, seg in enumerate(segments):
        slot_end = _slot_end(segments, i, video_duration)
        if seg.skipped or not seg.tts_wav:
            any_skip = True
            entries.append([seg.index, "skip", seg.skip_reason,
                            round(seg.start, 3), round(slot_end, 3)])
        else:
            entries.append([seg.index, "clip", _clip_sha(seg.tts_wav),
                            round(seg.start, 3), round(slot_end, 3)])

    inputs = {
        "entries": entries,
        "video_duration": round(video_duration, 3),
        "max_tempo": cfg.max_tempo,
        "sr": sr,
        "fade_ms": cfg.fade_ms,
        "mode": cfg.original_audio,
        # Placement shapes the timeline, so it is part of the dub's identity:
        # changing alignment/offset rebuilds dub.vi.wav and the mux (R17).
        "fit_alignment": cfg.fit_alignment,
        "fit_max_center_offset": cfg.fit_max_center_offset,
        # U2: the over-cap spill-trim changes the placed samples of over-cap
        # clips without changing placed_at, so it would be a silent cache HIT on
        # exactly the resumed runs that hit the overlap bug. The policy version
        # AND the tolerance value (U4 reprobes the fit_overflow decision against
        # the current tolerance on every call) are in the fingerprint so changing
        # either invalidates a stale over-cap dub and forces one rebuild.
        "placement_policy": PLACEMENT_POLICY,
        "fit_overflow_tolerance": cfg.fit_overflow_tolerance,
    }
    if cfg.original_audio == "replace" and any_skip:
        # Skip slots carry original audio: its content shapes the dub (R23)
        inputs["orig_sha"] = artifacts.cached_file_sha256(state["audio_orig"])

    # Clip durations build probes on a rebuild, reused by _record_overruns so a
    # fresh run does not re-read every clip header (a cache hit leaves this empty
    # and the reprobe falls back to the soundfile header read).
    built_durations: dict[str, float] = {}

    def build(tmp: Path) -> None:
        # Whole timeline in RAM: ~346 MB for 60 min at 24 kHz float32; scales
        # linearly with duration * timeline_sr. The +TAIL_HEADROOM_SEC pad gives
        # the last clip room to spill past video_duration (U3/R2).
        timeline = np.zeros(int((video_duration + TAIL_HEADROOM_SEC) * sr), dtype="float32")
        for i, seg in enumerate(segments):
            slot_end = _slot_end(segments, i, video_duration)
            if seg.skipped or not seg.tts_wav:
                if cfg.original_audio == "replace" and slot_end > seg.start:
                    span = fit_dir / f".orig_span.{uuid.uuid4().hex}.wav"
                    ffmpeg.ffmpeg("-i", state["audio_orig"],
                                  "-ss", f"{seg.start:.3f}", "-to", f"{slot_end:.3f}",
                                  "-ar", str(sr), "-ac", "1", str(span))
                    try:
                        audio = _read_mono_at(span, sr, fit_dir)
                    finally:
                        span.unlink(missing_ok=True)
                    _place(timeline, _fade(audio, sr, cfg.fade_ms), seg.start, sr)
                continue

            clip_dur = ffmpeg.probe_duration(seg.tts_wav)
            built_durations[segment_id(seg)] = clip_dur
            slot = slot_end - seg.start
            overflow = clip_dur > slot and slot > 0
            if not overflow:
                seg.fitted_wav = seg.tts_wav
                # Short clip: center it (capped) so it doesn't finish early.
                seg.placed_at = _placement(seg.start, slot_end, clip_dur, cfg)
            else:
                factor = min(clip_dur / slot, cfg.max_tempo)
                out = fit_dir / f"seg_{seg.index:04d}.wav"
                ffmpeg.atempo(seg.tts_wav, str(out), factor)
                seg.fitted_wav = str(out)
                log.info("segment %d: %.2fs into %.2fs slot, tempo x%.2f",
                         seg.index, clip_dur, slot, factor)
                seg.placed_at = seg.start  # overflow: keep onset, let it spill
            audio = _read_mono_at(seg.fitted_wav, sr, fit_dir)
            # A clip that still overruns its slot after the tempo cap must not sum
            # on top of the next segment's clip region (B2/R1): trim the spilled
            # tail at the next segment's onset so overlapping samples are never
            # summed. Geometry, not occupancy — the interior overrun is trimmed
            # even when the next slot is empty (duck-skipped), so placement is
            # mode-independent and stable on a cache-hit reprobe (KTD7). The LAST
            # segment is never trimmed: its tail spills into genuine trailing
            # silence (the +1.0s headroom) and mux preserves it (U3/R2).
            if overflow and i + 1 < len(segments):
                writable = max(0, int(round(slot * sr)))
                audio = audio[:writable]
            _place(timeline, _fade(audio, sr, cfg.fade_ms), seg.placed_at, sr)

        np.clip(timeline, -1.0, 1.0, out=timeline)
        sf.write(str(tmp), timeline, sr)

    # Locale-derived dub-track name (U10); the VI default keeps dub.vi.wav, and
    # the path is not part of the artifact fingerprint, so an existing VI work dir
    # still hits the cache.
    dub_wav = fit_dir / f"dub.{cfg.target_lang.lower()}.wav"
    cached = artifacts.produce(dub_wav, inputs, "fit", build)
    log.info("dub track %s -> %s", "reused" if cached else "built", dub_wav)

    # Surface placement-layer length overruns in the ledger on EVERY call,
    # independent of the cache hit above, so a resumed run still reports them and
    # raises the exit code (U4/R3). built_durations is populated only on a rebuild
    # (empty on a cache hit -> header reprobe).
    _record_overruns(segments, video_duration,
                     SkipLedger.from_cfg(workdir, cfg), cfg, built_durations)
    return {"segments": segments, "dub_wav": str(dub_wav)}
