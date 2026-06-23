"""Mechanical QA gate for TTS clips (R7): purely local, offline-testable.

A clip passes when it decodes, its duration is plausible for the target text's
expected length, and it is not all-silence. The expected length is sourced from
the active LanguageProfile's counter + rate (U5): syllables/4.3 for Vietnamese,
characters/CPS for every other language — so the gate is language-agnostic, not
Vietnamese-specific. For non-VI profiles this gate is the truncation/silence
sanity floor; the authoritative length gate is the measured-clip-duration-vs-slot
check in the tts node (U6). Failures raise a qa-class StageError so the caller's
retry/skip policy treats them like any other segment failure. The silence
amplitude threshold shares the `silence_threshold_db` knob with cross-check's
silencedetect split points.
"""

import numpy as np
import soundfile as sf

from loro.config import Config
from loro.harness.retry import QA, StageError
from loro.profiles.base import vi_syllable_count


def syllable_count(text_target: str) -> int:
    """The Vietnamese syllable counter — a VI-ONLY internal heuristic (U5), used by
    this QA gate for the VI expected-length floor. The single implementation lives
    in `loro.profiles.base.vi_syllable_count` (the VI profile's `counter`); this is
    a thin re-export so the QA gate and the translate length budget can never drift
    (#9). Non-VI profiles use a character (CPS) counter (`profiles.base.char_count`)."""
    return vi_syllable_count(text_target)


def check_clip(path, text_target: str, cfg: Config) -> None:
    """Raise StageError('tts', 'qa', code) when the clip fails the gate."""
    try:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:
        raise StageError("tts", QA, "undecodable", str(exc)[:200]) from exc
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sample_rate
    # Expected speech duration in the active profile's length model: VI keeps
    # syllables / 4.3 byte-identically; CPS profiles use chars / the profile CPS
    # (U5). The measured-duration gate (U6) layers on top of this for non-VI.
    profile = cfg.language_profile
    expected = profile.counter(text_target) / profile.rate
    min_duration = max(cfg.qa_min_clip_sec, cfg.qa_min_duration_ratio * expected)
    # Absolute floor on the upper bound: very short lines (1-3 syllables) get
    # an unfairly tight window otherwise, and Higgs pads with leading silence
    max_duration = max(cfg.qa_max_duration_ratio * expected, cfg.qa_max_clip_floor_sec)
    if duration < min_duration:
        raise StageError("tts", QA, "too_short",
                         f"{duration:.2f}s for ~{expected:.2f}s of speech")
    if duration > max_duration:
        raise StageError("tts", QA, "too_long",
                         f"{duration:.2f}s for ~{expected:.2f}s of speech")

    peak = float(np.abs(audio).max()) if len(audio) else 0.0
    silence_amplitude = 10 ** (cfg.silence_threshold_db / 20)
    if peak <= silence_amplitude:
        raise StageError("tts", QA, "silent", f"peak {peak:.5f}")
