"""Shared LangGraph state for the dubbing pipeline."""

from dataclasses import asdict, dataclass, field, fields
from typing import TypedDict


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text_src: str
    text_target: str = ""
    # Diarization label (AssemblyAI speaker id, e.g. "A"/"B"). Consumed by the
    # preset (soniox) engine's voice_ref to cast each speaker to a preset voice
    # (KTD4); the cloning engines ignore it. "" on the local ASR engine and on
    # single-speaker / diarization-off audio, where it routes to the one default
    # voice (R6). Picked up automatically by to_dict/from_dict via `fields`.
    speaker: str = ""
    tts_wav: str = ""
    fitted_wav: str = ""
    # Start offset actually used when placing on the timeline
    placed_at: float = field(default=-1.0)
    # Set when the segment was skipped (see harness.ledger); the slot falls
    # back to original audio in fit/mux instead of a dub clip.
    skipped: bool = False
    skip_reason: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Segment":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def segment_id(segment: "Segment | int") -> str:
    index = segment if isinstance(segment, int) else segment.index
    return f"seg_{index:04d}"


class DubState(TypedDict, total=False):
    video_path: str
    workdir: str
    video_duration: float
    audio_16k: str       # 16kHz mono wav for ASR / reference extraction
    audio_orig: str      # full-quality wav for final mixing
    subs_path: str       # extracted English subtitle track, "" when none (R34)
    # Deduplicated, time-sorted word timestamps from asr; consumed by
    # sentence_seg (the dub backbone) and the sub-style SRT writers. NEVER put
    # this in any node's artifact `inputs` fingerprint — it would bloat cache
    # keys; hash it explicitly (see sentence_seg._words_sha) instead.
    words: list[dict]
    segments: list[Segment]
    # Resolved input language reaching the translation prompt (U7): the configured
    # source_lang (default "en", byte-identical) or, with source_lang="auto", the
    # language the ASR engine detected. Direct source->target, no English pivot.
    source_lang: str
    srt_src: str
    srt_target: str
    ref_audio: str       # voice-clone reference clip (cloning engines only)
    ref_text: str        # transcript of the reference clip (cloning engines only)
    # speaker id -> preset voice, built by voice_ref for the soniox engine
    # (KTD4). The "" key is the no-speaker sentinel (single-speaker / local-ASR
    # audio) mapping to the default voice (R6). Absent for cloning engines.
    voice_cast: dict[str, str]
    video_context: str   # scene summary from the vision agent
    video_keywords: list[str]  # expected terminology from vision (R31)
    seg_visuals: dict[int, str]  # segment index -> its shot's visual description (R39)
    dub_wav: str         # assembled target-language dub track
    output_path: str
    # Target-language SRT written beside the output video as <basename>.<tag>.srt
    # (locale-derived, U10; R5, KTD3). Output-side delivery, distinct from the
    # input-side `subs_path`.
    srt_sidecar: str
