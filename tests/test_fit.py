import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from loro.config import Config
from loro.harness import artifacts
from loro.harness.ledger import SkipLedger
from loro.nodes import fit as fit_mod
from loro.nodes.fit import fit
from loro.nodes.mux import mux
from loro.state import Segment
from loro.utils import ffmpeg


def _tone(path, seconds, sr=24000, freq=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(path, (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), sr)


def test_fit_assembles_timeline_and_speeds_up_overflow(tmp_path):
    # Segment 0 fits its slot; segment 1 (3s clip) overflows a 1.5s slot.
    short = tmp_path / "s0.wav"
    long = tmp_path / "s1.wav"
    _tone(short, 1.0)
    _tone(long, 3.0)

    segments = [
        Segment(index=0, start=0.0, end=1.5, text_src="a", text_target="a", tts_wav=str(short)),
        Segment(index=1, start=2.0, end=3.0, text_src="b", text_target="b", tts_wav=str(long)),
    ]
    state = {
        "segments": segments,
        "workdir": str(tmp_path),
        "video_duration": 3.5,  # slot for seg 1 = 3.5 - 2.0 = 1.5s
    }
    result = fit(state, Config(max_tempo=1.35))

    # Overflowing clip got time-stretched (capped at max_tempo)
    assert segments[1].fitted_wav != segments[1].tts_wav
    stretched, sr = sf.read(segments[1].fitted_wav)
    assert len(stretched) / sr == pytest.approx(3.0 / 1.35, rel=0.05)

    # Timeline has audio at both placements and silence in the gap
    dub, sr = sf.read(result["dub_wav"])
    assert np.abs(dub[: int(0.5 * sr)]).max() > 0.1            # seg 0 audible
    assert np.abs(dub[int(1.6 * sr) : int(1.9 * sr)]).max() < 1e-4  # gap silent
    assert np.abs(dub[int(2.1 * sr) : int(2.5 * sr)]).max() > 0.1   # seg 1 audible


def test_fit_keeps_clip_that_fits(tmp_path):
    clip = tmp_path / "s0.wav"
    _tone(clip, 1.0)
    segments = [Segment(index=0, start=0.0, end=2.0, text_src="a", text_target="a", tts_wav=str(clip))]
    state = {"segments": segments, "workdir": str(tmp_path), "video_duration": 2.0}
    fit(state, Config())
    assert segments[0].fitted_wav == str(clip)


def test_fit_survives_skipped_segment_zero(tmp_path):
    # The old implementation read segments[0].fitted_wav unconditionally
    clip = tmp_path / "s1.wav"
    _tone(clip, 1.0)
    segments = [
        Segment(index=0, start=0.0, end=1.0, text_src="a", skipped=True,
                skip_reason="translate_failed"),
        Segment(index=1, start=2.0, end=3.0, text_src="b", text_target="b", tts_wav=str(clip)),
    ]
    state = {"segments": segments, "workdir": str(tmp_path), "video_duration": 3.5}
    result = fit(state, Config())  # duck mode: skipped slot stays empty
    dub, sr = sf.read(result["dub_wav"])
    assert np.abs(dub[: int(1.0 * sr)]).max() < 1e-4   # skipped slot silent in dub
    assert np.abs(dub[int(2.1 * sr) : int(2.6 * sr)]).max() > 0.1


def test_fade_brings_clip_edges_to_zero(tmp_path):
    clip = tmp_path / "s0.wav"
    _tone(clip, 1.0)
    segments = [Segment(index=0, start=0.5, end=1.5, text_src="a", text_target="a", tts_wav=str(clip))]
    state = {"segments": segments, "workdir": str(tmp_path), "video_duration": 2.5}
    # fit_alignment="start" keeps the clip left-aligned at seg.start so this
    # fade test reasons about a known placement (U3 centering is exercised
    # separately below).
    result = fit(state, Config(fade_ms=30.0, fit_alignment="start"))
    dub, sr = sf.read(result["dub_wav"])
    start = int(0.5 * sr)
    end = start + sr  # 1s clip
    # First and last couple of samples of the placed clip are faded toward 0
    assert np.abs(dub[start : start + 8]).max() < 0.01
    assert np.abs(dub[end - 8 : end]).max() < 0.01
    # Mid-clip is untouched
    assert np.abs(dub[start + sr // 2 - 100 : start + sr // 2 + 100]).max() > 0.25


def test_resamples_clip_with_other_rate(tmp_path):
    clip = tmp_path / "s0.wav"
    _tone(clip, 1.0, sr=44100)  # differs from timeline_sr
    segments = [Segment(index=0, start=0.0, end=1.5, text_src="a", text_target="a", tts_wav=str(clip))]
    state = {"segments": segments, "workdir": str(tmp_path), "video_duration": 2.0}
    result = fit(state, Config(timeline_sr=24000))
    dub, sr = sf.read(result["dub_wav"])
    assert sr == 24000
    assert np.abs(dub[: int(0.5 * sr)]).max() > 0.1


# --- U3: capped centering of short clips (KTD3) ---

def test_placement_math():
    cfg = Config(fit_max_center_offset=0.2)
    # small slack (< 2*offset): centered by half the slack
    assert fit_mod._placement(0.0, 1.3, 1.0, cfg) == pytest.approx(0.15)
    # large slack: offset capped at fit_max_center_offset
    assert fit_mod._placement(0.0, 10.0, 1.0, cfg) == pytest.approx(0.2)
    # clip equal to slot: no slack, place at start
    assert fit_mod._placement(0.0, 1.0, 1.0, cfg) == 0.0
    # non-positive slot: place at start
    assert fit_mod._placement(2.0, 1.0, 1.0, cfg) == 2.0
    # never overruns the slot even with an absurd cap (clamp binds)
    placed = fit_mod._placement(0.0, 1.2, 1.0, Config(fit_max_center_offset=5.0))
    assert placed + 1.0 <= 1.2 + 1e-9
    # "start" opt-out left-aligns regardless of slack
    assert fit_mod._placement(0.5, 10.0, 1.0, Config(fit_alignment="start")) == 0.5


def test_center_nudges_short_clip_forward(tmp_path):
    clip = tmp_path / "s0.wav"
    _tone(clip, 1.0)
    seg = Segment(index=0, start=0.0, end=5.0, text_src="a", text_target="a", tts_wav=str(clip))
    state = {"segments": [seg], "workdir": str(tmp_path), "video_duration": 10.0}
    fit(state, Config(fit_alignment="center", fit_max_center_offset=0.2))
    assert seg.placed_at == pytest.approx(0.2, abs=1e-3)  # capped offset, not 0.0


def test_center_last_segment_never_overruns_timeline(tmp_path):
    clip = tmp_path / "s0.wav"
    _tone(clip, 1.0)
    seg = Segment(index=0, start=0.0, end=5.0, text_src="a", text_target="a", tts_wav=str(clip))
    state = {"segments": [seg], "workdir": str(tmp_path), "video_duration": 1.2}
    fit(state, Config(fit_max_center_offset=5.0))  # cap larger than the slack
    assert seg.placed_at == pytest.approx(0.1, abs=1e-3)   # bounded by slack/2
    assert seg.placed_at + 1.0 <= 1.2 + 1e-3              # clip ends within video


def test_overflow_clip_keeps_onset_and_caps_tempo(tmp_path):
    clip = tmp_path / "s0.wav"
    _tone(clip, 3.0)
    seg = Segment(index=0, start=0.0, end=5.0, text_src="a", text_target="a", tts_wav=str(clip))
    state = {"segments": [seg], "workdir": str(tmp_path), "video_duration": 1.5}  # slot 1.5
    fit(state, Config(max_tempo=1.35))
    assert seg.placed_at == pytest.approx(0.0)            # overflow keeps onset
    stretched, sr = sf.read(seg.fitted_wav)
    assert len(stretched) / sr == pytest.approx(3.0 / 1.35, rel=0.05)  # capped at max_tempo


def test_alignment_change_rebuilds_dub(tmp_path):
    clip = tmp_path / "s0.wav"
    _tone(clip, 1.0)

    def make():
        return {"segments": [Segment(index=0, start=0.0, end=5.0, text_src="a",
                                     text_target="a", tts_wav=str(clip))],
                "workdir": str(tmp_path), "video_duration": 10.0}

    r1 = fit(make(), Config(fit_alignment="center", fit_max_center_offset=0.2))
    dub_center, sr = sf.read(r1["dub_wav"])
    fit(make(), Config(fit_alignment="center", fit_max_center_offset=0.2))  # cached rerun
    r2 = fit(make(), Config(fit_alignment="start"))  # config change -> rebuild
    dub_start, _ = sf.read(r2["dub_wav"])
    # center placed the clip at 0.2s, start at 0.0s: the opening differs, which
    # is only possible if the artifact rebuilt on the alignment change (R17).
    assert not np.array_equal(dub_center[: int(0.5 * sr)], dub_start[: int(0.5 * sr)])


# --- U2: over-cap spill must not overlap the next segment (B2/R1) ---

def test_overflow_clip_trimmed_at_next_segment_onset(tmp_path):
    # Clip 0 overruns its 1.0s slot even at max_tempo and would spill into
    # segment 1's region; the spilled tail is trimmed at seg 1's onset so the two
    # clips never sum. Proven by isolating each clip's footprint (the other slot
    # skipped): no sample is non-zero in both, and clip 0 is silent past its slot.
    long0 = tmp_path / "c0.wav"
    clip1 = tmp_path / "c1.wav"
    _tone(long0, 3.0, freq=440.0)
    _tone(clip1, 0.8, freq=880.0)

    def make(workdir, skip0=False, skip1=False):
        s0 = (Segment(index=0, start=0.0, end=1.0, text_src="a", skipped=True, skip_reason="qa")
              if skip0 else
              Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(long0)))
        s1 = (Segment(index=1, start=1.0, end=2.0, text_src="b", skipped=True, skip_reason="qa")
              if skip1 else
              Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="b", tts_wav=str(clip1)))
        wd = tmp_path / workdir
        wd.mkdir()
        return {"segments": [s0, s1], "workdir": str(wd), "video_duration": 2.0}

    cfg = Config(max_tempo=1.35)
    clip0_only, sr = sf.read(fit(make("a", skip1=True), cfg)["dub_wav"])
    clip1_only, _ = sf.read(fit(make("b", skip0=True), cfg)["dub_wav"])
    eps = 1e-4
    both_loud = (np.abs(clip0_only) > eps) & (np.abs(clip1_only) > eps)
    assert not both_loud.any()                              # no summation overlap
    assert np.abs(clip0_only[int(1.05 * sr):]).max() < eps  # trimmed at slot end


def test_trailing_overflow_clip_spills_into_trailing_silence(tmp_path):
    # The LAST clip is never trimmed — its over-cap tail spills past
    # video_duration into the +1.0s headroom (audible), preserved for mux (U3/R2).
    clip = tmp_path / "c.wav"
    _tone(clip, 2.0)
    seg = Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(clip))
    state = {"segments": [seg], "workdir": str(tmp_path), "video_duration": 1.0}
    dub, sr = sf.read(fit(state, Config(max_tempo=1.35))["dub_wav"])
    # capped clip = 2.0/1.35 ≈ 1.48s placed at 0.0 -> audible past video_duration
    assert np.abs(dub[int(1.1 * sr):int(1.4 * sr)]).max() > 0.1


def test_no_overflow_run_is_cache_stable(tmp_path):
    # VI regression: the U2 trim only touches over-cap clips, so a no-overflow run
    # is unchanged and reruns from cache with byte-identical output.
    clip = tmp_path / "c.wav"
    _tone(clip, 1.0)

    def make():
        return {"segments": [Segment(index=0, start=0.0, end=2.0, text_src="a",
                                     text_target="a", tts_wav=str(clip))],
                "workdir": str(tmp_path), "video_duration": 2.0}

    dub1, sr = sf.read(fit(make(), Config())["dub_wav"])
    dub2, _ = sf.read(fit(make(), Config())["dub_wav"])
    assert np.array_equal(dub1, dub2)


def test_fit_overflow_tolerance_change_busts_dub_cache(tmp_path):
    # The tolerance is in the dub fingerprint (U2): editing it must invalidate an
    # existing over-cap dub so a U4 reprobe never claims an overrun the cached
    # bytes don't reflect.
    clip = tmp_path / "c.wav"
    _tone(clip, 1.0)

    def make():
        return {"segments": [Segment(index=0, start=0.0, end=2.0, text_src="a",
                                     text_target="a", tts_wav=str(clip))],
                "workdir": str(tmp_path), "video_duration": 2.0}

    fit(make(), Config(fit_overflow_tolerance=1.5))
    fp1 = artifacts.read_meta(tmp_path / "fit" / "dub.vi.wav")["input_fingerprint"]
    fit(make(), Config(fit_overflow_tolerance=1.5))          # same tol -> stable
    assert artifacts.read_meta(tmp_path / "fit" / "dub.vi.wav")["input_fingerprint"] == fp1
    fit(make(), Config(fit_overflow_tolerance=2.0))          # changed -> rebuild
    assert artifacts.read_meta(tmp_path / "fit" / "dub.vi.wav")["input_fingerprint"] != fp1


# --- U4: placement-layer overrun -> ledger fit_overflow + exit 2 (B2/R3/KTD7) ---

def _fit_overflows(workdir):
    return {k for k, v in SkipLedger(workdir).entries().items()
            if v["status"] == "fit_overflow"}


def _two_seg_state(tmp_path, name, clip0_dur, seg1_skipped=False):
    # seg 0 is INTERIOR (followed by seg 1) with a 1.0s slot; an over-cap clip0
    # has its tail trimmed at seg 1's onset (U2).
    wd = tmp_path / name
    wd.mkdir()
    long0 = wd / "c0.wav"
    _tone(long0, clip0_dur)
    s0 = Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(long0))
    if seg1_skipped:
        s1 = Segment(index=1, start=1.0, end=2.0, text_src="b", skipped=True, skip_reason="qa")
    else:
        clip1 = wd / "c1.wav"
        _tone(clip1, 0.8)
        s1 = Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="b", tts_wav=str(clip1))
    return {"segments": [s0, s1], "workdir": str(wd), "video_duration": 2.0}


def test_over_tolerance_overrun_records_fit_overflow(tmp_path):
    # 3.0s clip in a 1.0s slot: capped at 1.35 -> 2.22x slot > 1.5x tolerance.
    state = _two_seg_state(tmp_path, "a", clip0_dur=3.0)
    fit(state, Config(max_tempo=1.35, fit_overflow_tolerance=1.5))
    assert _fit_overflows(state["workdir"]) == {"seg_0000"}


def test_sub_tolerance_overrun_does_not_record(tmp_path):
    # 1.5s clip in a 1.0s slot: capped -> 1.11x slot. With an EXPLICIT wide 1.5x
    # band the tail is trimmed but it is NOT exit-2 signal (KTD7). (The default
    # band is tighter now — see test_material_drop_recorded_at_default_tolerance.)
    state = _two_seg_state(tmp_path, "a", clip0_dur=1.5)
    fit(state, Config(max_tempo=1.35, fit_overflow_tolerance=1.5))
    assert _fit_overflows(state["workdir"]) == set()


def test_material_drop_recorded_at_default_tolerance(tmp_path):
    # Regression for the 1.5 dead band: a 2.0s clip in a 1.0s slot is capped to
    # ~1.48x slot and the interior trim drops ~32% of the post-cap audio. The old
    # default 1.5 left this BELOW the band (silent exit 0); the 1.10 default flags
    # it. Uses Config() so it pins the shipped default, not an explicit value.
    state = _two_seg_state(tmp_path, "a", clip0_dur=2.0)
    fit(state, Config())  # default max_tempo=1.35, fit_overflow_tolerance=1.10
    assert _fit_overflows(state["workdir"]) == {"seg_0000"}


def test_last_clip_over_headroom_records_fit_overflow(tmp_path):
    # The LAST clip is never trimmed at an onset, but _place clamps its tail at the
    # timeline end (slot + TAIL_HEADROOM_SEC). A 3.0s clip in a 1.0s final slot is
    # capped to ~2.22s > 1.0 + 1.0 headroom, so ~0.22s is dropped past the headroom
    # and mux cannot recover it — that drop must raise exit 2 (previously the last
    # segment was skipped by the recorder and exited 0).
    clip = tmp_path / "c.wav"
    _tone(clip, 3.0)
    seg = Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(clip))
    state = {"segments": [seg], "workdir": str(tmp_path), "video_duration": 1.0}
    dub, sr = sf.read(fit(state, Config(max_tempo=1.35))["dub_wav"])
    assert len(dub) == int((1.0 + fit_mod.TAIL_HEADROOM_SEC) * sr)  # tail was clamped
    assert _fit_overflows(state["workdir"]) == {"seg_0000"}


def test_last_clip_within_headroom_not_recorded(tmp_path):
    # The companion to the spill test: a 2.0s last clip caps to ~1.48s, within the
    # 1.0s slot + 1.0s headroom, so it is fully preserved by mux and is NOT exit 2.
    clip = tmp_path / "c.wav"
    _tone(clip, 2.0)
    seg = Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(clip))
    state = {"segments": [seg], "workdir": str(tmp_path), "video_duration": 1.0}
    fit(state, Config(max_tempo=1.35))
    assert _fit_overflows(state["workdir"]) == set()


def test_overrun_reconcile_is_idempotent_across_reruns(tmp_path):
    # U4 idempotency: a resumed/cache-hit rerun with the SAME overrun must not
    # rewrite skips.json. The previous per-segment record_fit_overflow re-saved and
    # re-pushed a demoted window entry on every call; reconcile is a no-op when
    # nothing changed.
    state = _two_seg_state(tmp_path, "wd", clip0_dur=3.0)
    cfg = Config(max_tempo=1.35, fit_overflow_tolerance=1.5)
    fit(state, cfg)
    skips = Path(state["workdir"]) / "skips.json"
    assert _fit_overflows(state["workdir"]) == {"seg_0000"}
    data1 = skips.read_text(encoding="utf-8")
    fit(state, cfg)  # dub is a cache hit; the overrun is unchanged
    assert skips.read_text(encoding="utf-8") == data1  # byte-identical, no rewrite


def test_record_overruns_reuses_prebuilt_durations(tmp_path):
    # On a rebuild fit hands _record_overruns the durations build already probed, so
    # it does not re-read clip headers; the passed value drives the decision. The
    # on-disk clip fits its slot, but the "built" 3.0s value overruns -> recorded.
    clip = tmp_path / "c.wav"
    _tone(clip, 0.5)  # 0.5s header would fit the 1.0s slot
    segs = [
        Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(clip)),
        Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="b", tts_wav=str(clip)),
    ]
    led = SkipLedger(str(tmp_path))
    fit_mod._record_overruns(segs, 2.0, led,
                             Config(max_tempo=1.35, fit_overflow_tolerance=1.5),
                             {"seg_0000": 3.0})
    assert _fit_overflows(str(tmp_path)) == {"seg_0000"}  # passed 3.0 used, not 0.5


def test_overrun_recorded_on_cache_hit_rerun(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    long0 = wd / "c0.wav"
    _tone(long0, 3.0)
    clip1 = wd / "c1.wav"
    _tone(clip1, 0.8)

    def make():
        return {"segments": [
            Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(long0)),
            Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="b", tts_wav=str(clip1)),
        ], "workdir": str(wd), "video_duration": 2.0}

    cfg = Config(max_tempo=1.35, fit_overflow_tolerance=1.5)
    fit(make(), cfg)
    dub = wd / "fit" / "dub.vi.wav"
    sha = artifacts.file_sha256(dub)
    (wd / "skips.json").unlink()                      # wipe the recorded overrun
    fit(make(), cfg)                                  # dub is a cache hit
    assert artifacts.file_sha256(dub) == sha          # build was NOT re-run
    assert _fit_overflows(str(wd)) == {"seg_0000"}    # yet the overrun re-records


def test_over_cap_clip_with_skipped_neighbor_still_records(tmp_path):
    # KTD7: the gate keys on dropped audio (geometry), not neighbor occupancy —
    # a duck-mode skipped neighbor (empty slot) does not let the overrun off.
    state = _two_seg_state(tmp_path, "a", clip0_dur=3.0, seg1_skipped=True)
    fit(state, Config(max_tempo=1.35, original_audio="duck", fit_overflow_tolerance=1.5))
    assert _fit_overflows(state["workdir"]) == {"seg_0000"}


def test_fit_overflow_is_mode_independent(tmp_path):
    def make(name, mode):
        wd = tmp_path / name
        wd.mkdir()
        long0 = wd / "c0.wav"
        _tone(long0, 3.0)
        clip1 = wd / "c1.wav"
        _tone(clip1, 0.8)
        st = {"segments": [
            Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(long0)),
            Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="b", tts_wav=str(clip1)),
        ], "workdir": str(wd), "video_duration": 2.0}
        if mode == "replace":
            st["audio_orig"] = str(_orig_audio(wd))
        return st, wd

    duck, dwd = make("duck", "duck")
    fit(duck, Config(max_tempo=1.35, original_audio="duck", fit_overflow_tolerance=1.5))
    repl, rwd = make("repl", "replace")
    fit(repl, Config(max_tempo=1.35, original_audio="replace", fit_overflow_tolerance=1.5))
    assert _fit_overflows(str(dwd)) == _fit_overflows(str(rwd)) == {"seg_0000"}


def test_cps_length_overflow_not_promoted_to_fit_overflow(tmp_path):
    # KTD2: a CPS segment already carrying an exit-0 length_overflow must not be
    # back-door-promoted to exit-2 by the placement layer.
    state = _two_seg_state(tmp_path, "a", clip0_dur=3.0)
    SkipLedger(state["workdir"]).record_length_overflow("seg_0000")
    fit(state, Config(max_tempo=1.35, fit_overflow_tolerance=1.5))
    entries = SkipLedger(state["workdir"]).entries()
    assert entries["seg_0000"]["status"] == "length_overflow"  # not promoted
    assert _fit_overflows(state["workdir"]) == set()


def test_recompute_without_overrun_clears_stale_fit_overflow(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    c0 = wd / "c0.wav"
    _tone(c0, 3.0)
    c1 = wd / "c1.wav"
    _tone(c1, 0.8)

    def make(clip0):
        return {"segments": [
            Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a", tts_wav=str(clip0)),
            Segment(index=1, start=1.0, end=2.0, text_src="b", text_target="b", tts_wav=str(c1)),
        ], "workdir": str(wd), "video_duration": 2.0}

    cfg = Config(max_tempo=1.35, fit_overflow_tolerance=1.5)
    fit(make(str(c0)), cfg)
    assert _fit_overflows(str(wd)) == {"seg_0000"}
    short = wd / "short.wav"                            # now fits its slot
    _tone(short, 0.5)
    fit(make(str(short)), cfg)
    assert _fit_overflows(str(wd)) == set()            # stale entry cleared


def _orig_audio(tmp_path, seconds=4.0, sr=44100):
    path = tmp_path / "audio_orig.wav"
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (0.4 * np.sin(2 * np.pi * 220.0 * t)).astype("float32")
    sf.write(path, np.stack([audio, audio], axis=1), sr)
    return path


def test_replace_mode_fills_skip_slot_with_original_audio(tmp_path):
    clip = tmp_path / "s1.wav"
    _tone(clip, 0.8)
    orig = _orig_audio(tmp_path)
    segments = [
        Segment(index=0, start=0.5, end=1.5, text_src="a", skipped=True, skip_reason="qa"),
        Segment(index=1, start=2.0, end=3.0, text_src="b", text_target="b", tts_wav=str(clip)),
    ]
    state = {"segments": segments, "workdir": str(tmp_path),
             "video_duration": 3.5, "audio_orig": str(orig)}

    result = fit(state, Config(original_audio="replace"))
    dub, sr = sf.read(result["dub_wav"])
    # R23: the skipped slot carries original audio energy in replace mode
    assert np.abs(dub[int(0.7 * sr) : int(1.5 * sr)]).max() > 0.2

    # Same layout in duck mode: slot must be silent in the dub track
    state2 = {"segments": [
        Segment(index=0, start=0.5, end=1.5, text_src="a", skipped=True, skip_reason="qa"),
        Segment(index=1, start=2.0, end=3.0, text_src="b", text_target="b", tts_wav=str(clip)),
    ], "workdir": str(tmp_path / "duck"), "video_duration": 3.5, "audio_orig": str(orig)}
    (tmp_path / "duck").mkdir()
    result2 = fit(state2, Config(original_audio="duck"))
    dub2, sr2 = sf.read(result2["dub_wav"])
    assert np.abs(dub2[int(0.7 * sr2) : int(1.4 * sr2)]).max() < 1e-4


def test_filled_skip_invalidates_dub(tmp_path):
    # AE1/AE3 invalidation: a previously skipped segment gaining a clip must
    # rebuild dub.vi.wav
    clip0 = tmp_path / "s0.wav"
    clip1 = tmp_path / "s1.wav"
    _tone(clip0, 0.8)
    _tone(clip1, 0.8)

    def make_state(seg0_skipped):
        seg0 = (Segment(index=0, start=0.0, end=1.0, text_src="a", skipped=True,
                        skip_reason="qa")
                if seg0_skipped else
                Segment(index=0, start=0.0, end=1.0, text_src="a", text_target="a",
                        tts_wav=str(clip0)))
        return {"segments": [seg0,
                             Segment(index=1, start=2.0, end=3.0, text_src="b",
                                     text_target="b", tts_wav=str(clip1))],
                "workdir": str(tmp_path), "video_duration": 3.5}

    result = fit(make_state(seg0_skipped=True), Config())
    dub_before, sr = sf.read(result["dub_wav"])
    assert np.abs(dub_before[: int(0.8 * sr)]).max() < 1e-4

    # Rerun unchanged -> cached (content identical)
    fit(make_state(seg0_skipped=True), Config())

    # Skip filled -> dub rebuilt with the new clip audible
    result = fit(make_state(seg0_skipped=False), Config())
    dub_after, sr = sf.read(result["dub_wav"])
    assert np.abs(dub_after[: int(0.8 * sr)]).max() > 0.1


@pytest.fixture
def tiny_video(tmp_path):
    """A 2s test video with a video and an audio stream."""
    path = tmp_path / "in.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=128x72:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path)],
        check=True,
    )
    return path


class TestMux:
    def _state(self, tmp_path, video):
        workdir = tmp_path / "work"
        workdir.mkdir(exist_ok=True)
        dub = workdir / "dub.vi.wav"
        _tone(dub, 2.0)
        srt_target = workdir / "transcript.vi.srt"
        srt_target.write_text("1\n00:00:00,000 --> 00:00:01,000\nxin chào\n", encoding="utf-8")
        return {
            "video_path": str(video),
            "workdir": str(workdir),
            "dub_wav": str(dub),
            "srt_target": str(srt_target),
            "output_path": str(tmp_path / "out.vi.mp4"),
        }

    def test_mux_writes_output_and_marker(self, tmp_path, tiny_video):
        state = self._state(tmp_path, tiny_video)
        result = mux(state, Config())
        out = tmp_path / "out.vi.mp4"
        assert out.exists()
        marker = json.loads((tmp_path / "work" / "mux.json").read_text())
        assert marker["output_sha256"] == artifacts.file_sha256(out)

    def test_deleted_output_remuxes_without_other_stages(self, tmp_path, tiny_video):
        state = self._state(tmp_path, tiny_video)
        mux(state, Config())
        first_sha = artifacts.file_sha256(tmp_path / "out.vi.mp4")

        # Cached path: nothing recomputed when output intact
        result = mux(state, Config())
        assert artifacts.file_sha256(tmp_path / "out.vi.mp4") == first_sha

        # User deletes the video, keeps workdir -> only mux reruns
        (tmp_path / "out.vi.mp4").unlink()
        result = mux(state, Config())
        assert (tmp_path / "out.vi.mp4").exists()
        assert result["output_path"] == str(tmp_path / "out.vi.mp4")

    def test_dub_longer_than_video_keeps_full_tail(self, tmp_path, tiny_video):
        # U3/R2: a dub that spills past the 2s video must not be truncated to the
        # video length — the output audio runs to the dub tail (~3s).
        state = self._state(tmp_path, tiny_video)
        _tone(state["dub_wav"], 3.0)             # 3s dub over a 2s video
        mux(state, Config())
        out = tmp_path / "out.vi.mp4"
        assert ffmpeg.probe_duration(out) == pytest.approx(3.0, abs=0.2)

    def test_dub_within_video_keeps_video_duration(self, tmp_path, tiny_video):
        # U3 regression: standard case (dub <= video) output duration is the video
        # duration, with no appended +1.0s headroom silence.
        state = self._state(tmp_path, tiny_video)      # 2s dub, 2s video
        mux(state, Config())
        out = tmp_path / "out.vi.mp4"
        assert ffmpeg.probe_duration(out) == pytest.approx(2.0, abs=0.2)

    def test_replace_mode_keeps_dub_tail(self, tmp_path, tiny_video):
        state = self._state(tmp_path, tiny_video)
        _tone(state["dub_wav"], 3.0)
        mux(state, Config(original_audio="replace"))
        out = tmp_path / "out.vi.mp4"
        assert ffmpeg.probe_duration(out) == pytest.approx(3.0, abs=0.2)

    def test_changed_subtitles_remux(self, tmp_path, tiny_video):
        state = self._state(tmp_path, tiny_video)
        mux(state, Config())
        first_sha = artifacts.file_sha256(tmp_path / "out.vi.mp4")

        (tmp_path / "work" / "transcript.vi.srt").write_text(
            "1\n00:00:00,000 --> 00:00:01,000\nbản dịch mới\n", encoding="utf-8")
        mux(state, Config())
        assert artifacts.file_sha256(tmp_path / "out.vi.mp4") != first_sha
