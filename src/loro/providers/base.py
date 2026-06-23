"""Provider contracts + capability surface for the ASR and TTS engine families.

A provider co-locates everything engine-specific for one engine (KTD2/KTD6):
client/runner construction, its config knobs, its contribution to the artifact
cache fingerprint, its preflight checks, and its capability flags. The pipeline
nodes own the shared orchestration — artifact caching, the skip ledger, QA-retry,
Gemini batch/split/fallback, local-ASR window/merge/dedup — and invoke providers,
never the reverse (R7).

ASR and TTS are different stages (ASR transcribes a whole file; TTS synthesizes
per segment/batch), so they get SEPARATE contracts that share one pattern — a
contract + a registry + self-contained modules — not identical method signatures
(KTD1, R1). Capability flags are plain attributes read in place of the old
engine-name checks (KTD4, R5).
"""

from typing import NamedTuple, Protocol

from loro.config import Config, PresetVoices
from loro.state import Segment

# Shared liveness/auth-probe timeout for provider preflight probes: fail fast
# rather than waiting out the full pipeline timeout. preflight's own model-server
# probe reuses it too.
PROBE_TIMEOUT = 15.0


class AsrResult(NamedTuple):
    """What an ASR provider hands back to the asr node: the raw segment backbone
    and the word stream. The node owns the shared cross-engine tail — the EN SRT
    write and the {segments, words, srt_src} return — so every engine shares it
    byte-for-byte (KTD3).

    source_lang is the resolved input language (U7): the detected language when a
    provider ran language identification (source_lang="auto"), else the configured
    source. None when the provider did not resolve one (the node falls back to
    cfg.source_lang), so the local engine needs no change."""
    segments: list[Segment]
    words: list[dict]
    source_lang: str | None = None


class AsrProvider(Protocol):
    """One ASR engine, transcribing a whole file (KTD1). Cloud engines own a
    single cached API call + raw->(words, segments) mapping + their own no-speech
    guard; the local engine drives the shared node-side windowing toolkit plus the
    Nemotron worker. Either way it yields the same AsrResult contract (R8)."""

    name: str
    # Local engines want the Granite ensemble crosscheck; cloud engines are their
    # own source of truth and skip it. The graph reads this in place of the old
    # `asr_engine == "local"` check (KTD8, R5/R8).
    wants_crosscheck: bool
    # Can the engine identify the spoken language (source_lang="auto")? The cloud
    # engines run language identification; the local Nemotron path cannot, so it
    # requires an explicit --source-lang. Preflight reads this capability instead
    # of an engine-name check (U7, R12).
    detects_language: bool

    def transcribe(self, state: dict, cfg: Config, asr_dir) -> AsrResult: ...

    def preflight(self, cfg: Config) -> list[str]: ...


class TtsProvider(Protocol):
    """One TTS engine, synthesizing per segment / per batch (KTD1). Exposes a
    context-manager client with synthesize(text, out, voice) (and, for batching
    engines, synthesize_batch), its fingerprint contribution, its chunk budget,
    and — for preset engines — its voice config."""

    name: str
    clones: bool             # vieneu/higgs clone a reference voice
    batches: bool            # gemini batches consecutive segments into one call
    native_long_text: bool   # gemini handles long text without per-segment chunking

    def client(self, cfg: Config, ref_audio=None, ref_text=None): ...

    def clones_in(self, locale: str) -> bool:
        """Can this engine clone a voice in `locale` (a BCP-47 tag)? Defaults to
        `self.clones` (back-compat: a cloning engine clones everywhere, a preset
        engine nowhere); VieNeu overrides it to Vietnamese-only (U9, KTD7). The
        STATIC capability gate preflight uses to reject "clone in a language the
        engine cannot clone" before billing — clone QUALITY across languages is
        uneven and validated at tier-1 calibration (U11), not here."""
        ...

    def engine_inputs(self, cfg: Config) -> dict: ...

    def chunk_budget(self, cfg: Config) -> int: ...

    def preset_voices(self, cfg: Config) -> PresetVoices | None: ...

    def preflight(self, cfg: Config) -> list[str]: ...
