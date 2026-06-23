"""Small in-memory audio helpers shared across nodes and services.

Lives in utils (a leaf both `nodes` and `services` may import) so neither layer
has to import the other for a pure array op — the tts node and the Gemini
service both trim clip edges with the same primitive.
"""

import numpy as np


def trim_silence_edges(audio: np.ndarray, threshold_db: float) -> np.ndarray:
    """Drop leading/trailing samples below the silence floor so a clip's
    duration reflects its SPEECH span, not padding or the share of an inter-turn
    pause a cut landed in. Returns the audio unchanged when nothing crosses the
    floor (a fully silent clip is rejected upstream by the QA gate).

    This is per-clip EDGE cleanup using instantaneous amplitude — never a way to
    find interior cut points (speech crosses zero constantly, so instantaneous
    amplitude is not a silence-run detector; use silencedetect for that)."""
    if len(audio) == 0:
        return audio
    floor = 10 ** (threshold_db / 20)
    loud = np.abs(audio) > floor
    if not loud.any():
        return audio
    lo = int(np.argmax(loud))
    hi = len(audio) - int(np.argmax(loud[::-1]))
    return audio[lo:hi]
