"""Provider registry: resolve a configured engine name to its provider (R2/R4).

Adding an engine is one new provider module plus one entry in the family tuple
below — no edits to the node, preflight, voice casting, or config dispatch (AE1).
`asr(name)` / `tts(name)` are the single lookups every node, the graph, preflight,
and the Config capability adapters go through.

Note the deliberate name shadow: the lookup functions `asr`/`tts` defined here
share the `loro.providers` namespace with the `asr/` and `tts/` subpackages, so
after this module loads `loro.providers.tts` (attribute access) resolves to the
FUNCTION, not the subpackage. Both established access patterns work regardless —
`from loro.providers import tts` (the function) and `from loro.providers.tts.X
import Y` (the submodule, via sys.modules). Only bare dotted-attribute access like
`loro.providers.tts.soniox` hits the function and AttributeErrors; import the
submodule explicitly instead.
"""

from loro.providers.asr.assemblyai import AssemblyaiAsrProvider
from loro.providers.asr.local import LocalAsrProvider
from loro.providers.asr.soniox import SonioxAsrProvider
from loro.providers.base import AsrProvider, AsrResult, TtsProvider
from loro.providers.tts.gemini import GeminiTtsProvider
from loro.providers.tts.higgs import HiggsTtsProvider
from loro.providers.tts.soniox import SonioxTtsProvider
from loro.providers.tts.vieneu import VieNeuTtsProvider

__all__ = ["asr", "tts", "AsrProvider", "TtsProvider", "AsrResult",
           "UnknownEngineError"]


class UnknownEngineError(ValueError):
    """Raised for an engine name with no registered provider. A named error (not
    a bare KeyError) so a misconfigured ASR_ENGINE/TTS_ENGINE fails legibly."""


_ASR: dict[str, AsrProvider] = {
    p.name: p for p in (SonioxAsrProvider(), AssemblyaiAsrProvider(), LocalAsrProvider())
}
_TTS: dict[str, TtsProvider] = {
    p.name: p for p in (VieNeuTtsProvider(), HiggsTtsProvider(),
                        SonioxTtsProvider(), GeminiTtsProvider())
}


def asr(name: str) -> AsrProvider:
    try:
        return _ASR[name]
    except KeyError:
        raise UnknownEngineError(
            f"unknown ASR engine {name!r}; known: {', '.join(sorted(_ASR))}"
        ) from None


def tts(name: str) -> TtsProvider:
    try:
        return _TTS[name]
    except KeyError:
        raise UnknownEngineError(
            f"unknown TTS engine {name!r}; known: {', '.join(sorted(_TTS))}"
        ) from None
