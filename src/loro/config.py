"""Runtime configuration, sourced from environment variables and CLI flags."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple


class PresetVoices(NamedTuple):
    """The active preset engine's voice config (KTD6): the deterministic
    rotation pool, per-speaker pins, and the single default voice. Both
    preset engines (soniox/gemini) resolve theirs through Config.preset_voices
    so voice casting and the tts default-voice fallback stay engine-agnostic."""
    pool: list
    voice_map: dict
    default: str


class LlmRole(NamedTuple):
    """The resolved (host, api_key, model) triple for one LLM role (KTD7) — the
    single place a call site names a role instead of threading three fields:
    cfg.llm_role("translate" | "vision" | "seg" | "audio"). __post_init__ fills
    each role's fields from the base when its override is empty, so the triple is
    identical to passing the three llm_*_<role> fields explicitly (no behavior or
    fingerprint change, R13)."""
    host: str
    api_key: str
    model: str

DEFAULT_NEMOTRON_PYTHON = Path.home() / ".pyenv/versions/nemo/bin/python"
DEFAULT_GRANITE_PYTHON = Path.home() / ".pyenv/versions/granite/bin/python"
DEFAULT_VIENEU_PYTHON = Path.home() / ".pyenv/versions/vieneu/bin/python"


def _parse_voice_map(raw: str) -> dict:
    """Parse SONIOX_VOICE_MAP ("A=Adrian, B=Maya") into {speaker: voice}.
    Whitespace is tolerated; a malformed entry (no "=", or an empty side) is
    skipped so a typo degrades to pool rotation instead of crashing the run."""
    mapping: dict[str, str] = {}
    for entry in raw.split(","):
        speaker, sep, voice = entry.partition("=")
        speaker, voice = speaker.strip(), voice.strip()
        if sep and speaker and voice:
            mapping[speaker] = voice
    return mapping


@dataclass
class Config:
    # OpenAI-compatible model server(s), namespaced LLM_*. The base
    # LLM_HOST/LLM_API_KEY/LLM_MODEL is the default endpoint for every role.
    llm_host: str = field(default_factory=lambda: os.environ.get("LLM_HOST", "http://127.0.0.1:1234/v1"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "llm"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gemma-4-12B-it-8bit"))

    # Per-role endpoint overrides. Each role resolves a (host, api_key, model);
    # an empty override inherits the base above (set in __post_init__). Roles:
    # vision = vision + seg_visual (image); translate = translate + context
    # (text); seg = sentence_seg (text); audio = crosscheck re-listen +
    # preflight probe (audio). Splitting roles across hosts keeps each host on
    # one resident model, so a one-model-resident router (remote llama.cpp)
    # never pays a load/unload swap mid-pipeline.
    llm_host_vision: str = field(default_factory=lambda: os.environ.get("LLM_HOST_VISION", ""))
    llm_api_key_vision: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY_VISION", ""))
    llm_model_vision: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_VISION", ""))
    llm_host_translate: str = field(default_factory=lambda: os.environ.get("LLM_HOST_TRANSLATE", ""))
    llm_api_key_translate: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY_TRANSLATE", ""))
    llm_host_seg: str = field(default_factory=lambda: os.environ.get("LLM_HOST_SEG", ""))
    llm_api_key_seg: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY_SEG", ""))
    llm_host_audio: str = field(default_factory=lambda: os.environ.get("LLM_HOST_AUDIO", ""))
    llm_api_key_audio: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY_AUDIO", ""))
    # Translation LLM. Defaults to llm_model (see __post_init__) so an empty
    # config behaves byte-identically. Set LLM_MODEL_TRANSLATE to A/B a text-only
    # translator (e.g. Qwen) without dragging vision/crosscheck — which need
    # multimodal — onto a model that can't see images or hear audio (R37, KTD1).
    llm_model_translate: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_TRANSLATE", ""))
    # Audio-understanding LLM (crosscheck re-listen + preflight audio probe).
    # Defaults to llm_model (see __post_init__) so an empty config / the old
    # oMLX profile behaves byte-identically. It needs its own field because the
    # llama.cpp vision role (gemma-4-26B-A4B) has no audio, so the one
    # multimodal field can no longer carry both image and audio — set
    # LLM_MODEL_AUDIO to a model that does hear audio (e.g. gemma-4-12B) while
    # llm_model serves vision/text (R2, KTD1).
    llm_model_audio: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_AUDIO", ""))

    # Language selection (R1). target_lang resolves a LanguageProfile (rate model,
    # translation framing, TTS engine + voice strategy, font, script) through the
    # registry; source_lang is the input language fed to the translation prompt.
    # Defaults vi/en keep the historical EN->VI pipeline byte-identical (R19);
    # source_lang="auto" opts into Soniox language identification (a deliberate
    # ASR re-bill, U7). allow_fallback lets an unprofiled target_lang proceed
    # best-effort on the generic profile instead of failing preflight (R4).
    target_lang: str = field(default_factory=lambda: os.environ.get("TARGET_LANG", "vi"))
    source_lang: str = field(default_factory=lambda: os.environ.get("SOURCE_LANG", "en"))
    allow_fallback: bool = field(
        default_factory=lambda: os.environ.get("ALLOW_FALLBACK", "").lower()
        in ("1", "true", "yes"))

    # TTS engine selector (R1): "soniox" (cloud preset voices, default),
    # "vieneu" (on-device worker), or "higgs" (remote server). Soniox is the
    # engine under trial and the new default; --tts-engine overrides the env
    # (CLI wins). The cloning engines (vieneu/higgs) keep their existing paths
    # byte-for-byte; tts_uses_cloning (below) is the single place that answers
    # "does this engine clone?" for voice_ref, tts, and preflight (KTD2/KTD4).
    tts_engine: str = field(default_factory=lambda: os.environ.get("TTS_ENGINE", "soniox"))

    # Soniox cloud TTS (tts-rt-v1) options, used only when tts_engine ==
    # "soniox". Soniox has 28 preset studio voices and no cloning, so each
    # diarized speaker is cast to a preset voice (voice_ref → cast.json). The
    # key is a credential (R11): keep it in the gitignored .env, never a tracked
    # file, and it rides only in the Authorization header. voice_pool is the
    # deterministic rotation over preset voices; voice_map pins specific
    # speakers (SONIOX_VOICE_MAP="A=Adrian,B=Maya"); default_voice covers
    # single-speaker / no-diarization audio (R6).
    soniox_api_key: str = field(default_factory=lambda: os.environ.get("SONIOX_API_KEY", ""))
    soniox_base_url: str = field(
        default_factory=lambda: os.environ.get("SONIOX_BASE_URL", "https://tts-rt.soniox.com"))
    soniox_model: str = field(default_factory=lambda: os.environ.get("SONIOX_MODEL", "tts-rt-v1"))
    # SONIOX_LANGUAGE removed in the multi-language refactor (U9): the Soniox
    # spoken-language param is now profile-derived from --target-lang via
    # `effective_tts_language`, so a stale SONIOX_LANGUAGE env no longer applies.
    soniox_sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("SONIOX_SAMPLE_RATE", "24000")))
    soniox_audio_format: str = field(
        default_factory=lambda: os.environ.get("SONIOX_AUDIO_FORMAT", "wav"))
    soniox_timeout: float = field(
        default_factory=lambda: float(os.environ.get("SONIOX_TIMEOUT", "120.0")))
    soniox_default_voice: str = field(
        default_factory=lambda: os.environ.get("SONIOX_DEFAULT_VOICE", "Adrian"))
    # Ordered, alternating-gender default pool (a shippable default the smoke run
    # may retune, not a placeholder). Override comma-separated via SONIOX_VOICE_POOL.
    soniox_voice_pool: list = field(default_factory=lambda: (
        [v.strip() for v in os.environ["SONIOX_VOICE_POOL"].split(",") if v.strip()]
        if os.environ.get("SONIOX_VOICE_POOL")
        else ["Adrian", "Maya", "Noah", "Nina", "Jack", "Emma"]))
    # Per-speaker pins, parsed from SONIOX_VOICE_MAP="A=Adrian,B=Maya"; malformed
    # entries (no "=", empty side) are skipped so a typo can't crash the run.
    soniox_voice_map: dict = field(default_factory=lambda: _parse_voice_map(
        os.environ.get("SONIOX_VOICE_MAP", "")))

    # Higgs Audio v3 TTS server (sglang-omni)
    higgs_host: str = field(default_factory=lambda: os.environ.get("HIGGS_HOST", "http://localhost:8000"))
    higgs_model: str = field(default_factory=lambda: os.environ.get("HIGGS_MODEL", "bosonai/higgs-audio-v3-tts-4b"))

    # VieNeu-TTS on-device engine. Runs as a subprocess worker in an isolated
    # venv (heavy onnxruntime/sea-g2p/MOSS/torch footprint kept out of the thin
    # main env), the same isolation pattern as the NeMo/Granite workers.
    # VIENEU_MODEL records the model identity in each clip's fingerprint; the
    # worker loads VieNeu's default v3 Turbo weights (the upstream API exposes no
    # local repo-id override). Pinning to a commit SHA plus a load-time integrity
    # check is deferred (see the plan's Risks / Deferred follow-up).
    vieneu_python: str = field(
        default_factory=lambda: os.environ.get("VIENEU_PYTHON", str(DEFAULT_VIENEU_PYTHON))
    )
    vieneu_model: str = field(
        default_factory=lambda: os.environ.get("VIENEU_MODEL", "pnnbao-ump/VieNeu-TTS-v3-Turbo")
    )
    vieneu_temperature: float = field(
        default_factory=lambda: float(os.environ.get("VIENEU_TEMPERATURE", "0.8"))
    )
    vieneu_emotion: str = field(
        default_factory=lambda: os.environ.get("VIENEU_EMOTION", "natural")
    )
    # Voice cloning defaults to audio-only (R5): the verified transcript is
    # English while VieNeu's sea-g2p front end is Vietnamese-first, so feeding
    # ref_text may mis-phonemize the reference. The ref_text is plumbed all the
    # way to the worker but only sent when this is enabled — flip it on (env
    # VIENEU_REF_TEXT=1) once a smoke run shows it helps.
    vieneu_ref_text: bool = field(
        default_factory=lambda: os.environ.get("VIENEU_REF_TEXT", "").lower()
        in ("1", "true", "yes")
    )
    # Per-clip synthesis budget for the warm worker (model load happens once at
    # spawn, so this bounds a single infer call, not a cold start).
    vieneu_timeout: float = 600.0

    # Gemini cloud TTS (generateContent) options, used only when tts_engine ==
    # "gemini". Like Soniox it is a PRESET engine — Gemini has 30 prebuilt voices
    # and no cloning, so each diarized speaker is cast to a preset voice
    # (voice_ref -> cast.json via the shared preset_voices accessor, KTD6). The
    # key is a credential (R10): keep it in the gitignored .env, never a tracked
    # file, and it rides only in the x-goog-api-key header. The engine batches
    # consecutive segments into one multi-speaker call to minimize API calls
    # against Gemini's daily/RPM limits (R4/KTD2): batch_segments caps a batch's
    # length, batch_max_syllables caps its audio duration ("drifts past a few
    # minutes"), and split_min_gap_ms floors the inter-turn silence the splitter
    # cuts on. style_prompt steers tone and asks for inter-turn pauses (R7).
    gemini_api_key: str = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY", ""))
    gemini_base_url: str = field(default_factory=lambda: os.environ.get(
        "GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"))
    gemini_model: str = field(
        default_factory=lambda: os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-tts-preview"))
    # Fixed by the API (24 kHz mono s16le PCM) but folded into clip identity and
    # used to wrap the raw PCM into a WAV (KTD4).
    gemini_sample_rate: int = field(
        default_factory=lambda: int(os.environ.get("GEMINI_SAMPLE_RATE", "24000")))
    gemini_default_voice: str = field(
        default_factory=lambda: os.environ.get("GEMINI_DEFAULT_VOICE", "Kore"))
    # Ordered default pool (a shippable default the smoke run may retune).
    # Override comma-separated via GEMINI_VOICE_POOL.
    gemini_voice_pool: list = field(default_factory=lambda: (
        [v.strip() for v in os.environ["GEMINI_VOICE_POOL"].split(",") if v.strip()]
        if os.environ.get("GEMINI_VOICE_POOL")
        else ["Kore", "Puck", "Aoede", "Charon", "Leda", "Orus"]))
    # Per-speaker pins, parsed from GEMINI_VOICE_MAP="A=Kore,B=Puck"; malformed
    # entries (no "=", empty side) are skipped so a typo can't crash the run.
    gemini_voice_map: dict = field(default_factory=lambda: _parse_voice_map(
        os.environ.get("GEMINI_VOICE_MAP", "")))
    # Batch a run of consecutive to-do segments into one multi-speaker call
    # (1 = per-segment, no batching). A batch ends at this segment count, the
    # syllable budget below, or a third distinct diarized speaker (multi-speaker
    # is capped at 2 voices per request).
    gemini_batch_segments: int = field(
        default_factory=lambda: int(os.environ.get("GEMINI_BATCH_SEGMENTS", "8")))
    # Running-syllable ceiling on a batch (~360 syllables ≈ 84 s of speech at the
    # VI profile rate), keeping batched audio well under Gemini's "drifts
    # past a few minutes" threshold. Identity-bearing (folded into the per-clip
    # fingerprint as the effective chunk budget for Gemini, KTD5).
    gemini_batch_max_syllables: int = field(
        default_factory=lambda: int(os.environ.get("GEMINI_BATCH_MAX_SYLLABLES", "360")))
    # Minimum inter-turn silence (ms) the splitter treats as a segment boundary;
    # shorter dips inside a turn are ignored.
    gemini_split_min_gap_ms: float = field(
        default_factory=lambda: float(os.environ.get("GEMINI_SPLIT_MIN_GAP_MS", "200")))
    # Optional style/audio-tag directive prepended to the synthesis prompt (R7).
    gemini_style_prompt: str = field(
        default_factory=lambda: os.environ.get("GEMINI_STYLE_PROMPT", ""))
    gemini_timeout: float = field(
        default_factory=lambda: float(os.environ.get("GEMINI_TIMEOUT", "300.0")))

    # ASR engine selector (R1): "soniox" (cloud stt-async-v5, the new default),
    # "assemblyai" (cloud universal-3-pro), or "local" (the existing Nemotron
    # windows + Granite/Gemma ensemble). --asr-engine overrides the env (CLI wins,
    # mirroring --tts-engine). Defaulting to a cloud engine sends ASR audio
    # off-machine and bills per minute (KTD5); --asr-engine local (or
    # ASR_ENGINE=local) restores the fully-offline path. All Nemotron/Granite/
    # audio-role fields below stay in place — the local engine still uses them.
    asr_engine: str = field(default_factory=lambda: os.environ.get("ASR_ENGINE", "soniox"))

    # Soniox cloud ASR options (used only when asr_engine == "soniox"). The
    # async stt-async-v5 model transcribes the whole file in one job: upload →
    # create → poll → retrieve token transcript. The STT path REUSES the existing
    # soniox_api_key (KTD4) — STT and TTS are two Soniox products on one account
    # key, so there is no soniox_stt_api_key field; only the base URL, model, and
    # STT options below are STT-specific. Diarization is on by default (R4) and
    # its string speaker labels ride onto each word (KTD8). Recognition context
    # (R5) biases domain terms/background text; it is sent only when non-empty.
    # language_hints / context_terms parse comma-separated like the assemblyai_*
    # and soniox_voice_pool knobs.
    soniox_stt_base_url: str = field(
        default_factory=lambda: os.environ.get("SONIOX_STT_BASE_URL", "https://api.soniox.com"))
    soniox_stt_model: str = field(
        default_factory=lambda: os.environ.get("SONIOX_STT_MODEL", "stt-async-v5"))
    soniox_stt_language_hints: list = field(default_factory=lambda: (
        [h.strip() for h in os.environ["SONIOX_STT_LANGUAGE_HINTS"].split(",") if h.strip()]
        if os.environ.get("SONIOX_STT_LANGUAGE_HINTS")
        else ["en"]))
    soniox_stt_enable_language_identification: bool = field(
        default_factory=lambda: os.environ.get("SONIOX_STT_ENABLE_LANGUAGE_IDENTIFICATION", "").lower()
        in ("1", "true", "yes"))
    soniox_stt_speaker_diarization: bool = field(
        default_factory=lambda: os.environ.get("SONIOX_STT_SPEAKER_DIARIZATION", "1").lower()
        not in ("0", "false", "no"))
    # Recognition context (R5/KTD6): domain terms + background text. Empty = no
    # biasing object is sent. context_terms parses comma-separated.
    soniox_stt_context_terms: list = field(default_factory=lambda: (
        [t.strip() for t in os.environ["SONIOX_STT_CONTEXT_TERMS"].split(",") if t.strip()]
        if os.environ.get("SONIOX_STT_CONTEXT_TERMS")
        else []))
    soniox_stt_context_text: str = field(
        default_factory=lambda: os.environ.get("SONIOX_STT_CONTEXT_TEXT", ""))
    # Best-effort server-side cleanup after retrieval (R7/KTD7): delete the
    # uploaded file + transcription so the user's audio isn't retained on Soniox.
    # A failed delete is logged at warning level and is non-fatal (the local
    # cache is the source of truth for reruns).
    soniox_stt_cleanup: bool = field(
        default_factory=lambda: os.environ.get("SONIOX_STT_CLEANUP", "1").lower()
        not in ("0", "false", "no"))
    # Poll cadence + budgets, mirroring the assemblyai_* knobs. request_timeout
    # bounds each HTTP call; the poll loop's wall-clock ceiling is
    # poll_timeout_base + poll_timeout_per_sec * audio_duration.
    soniox_stt_poll_interval: float = field(
        default_factory=lambda: float(os.environ.get("SONIOX_STT_POLL_INTERVAL", "3.0")))
    soniox_stt_request_timeout: float = field(
        default_factory=lambda: float(os.environ.get("SONIOX_STT_REQUEST_TIMEOUT", "60.0")))
    soniox_stt_poll_timeout_base: float = field(
        default_factory=lambda: float(os.environ.get("SONIOX_STT_POLL_TIMEOUT_BASE", "120.0")))
    soniox_stt_poll_timeout_per_sec: float = field(
        default_factory=lambda: float(os.environ.get("SONIOX_STT_POLL_TIMEOUT_PER_SEC", "2.0")))

    # AssemblyAI cloud ASR options (used only when asr_engine == "assemblyai").
    # The key is a credential: keep it in the gitignored .env, never a tracked
    # file (R8). speech_models is an array-with-fallback (KTD3): universal-3-pro
    # primary, universal-2 the automatic fallback the API selects if the primary
    # is unavailable for the input; override via comma-separated ASSEMBLYAI_SPEECH_MODELS.
    assemblyai_api_key: str = field(default_factory=lambda: os.environ.get("ASSEMBLYAI_API_KEY", ""))
    assemblyai_base_url: str = field(
        default_factory=lambda: os.environ.get("ASSEMBLYAI_BASE_URL", "https://api.assemblyai.com/v2"))
    assemblyai_speech_models: list = field(default_factory=lambda: (
        [m.strip() for m in os.environ["ASSEMBLYAI_SPEECH_MODELS"].split(",") if m.strip()]
        if os.environ.get("ASSEMBLYAI_SPEECH_MODELS")
        else ["universal-3-pro", "universal-2"]))
    # Diarization on by default (R3); IDs are anonymous A/B/C, captured per word +
    # utterance + sentence but not yet driving voice selection (KTD7).
    assemblyai_speaker_labels: bool = field(
        default_factory=lambda: os.environ.get("ASSEMBLYAI_SPEAKER_LABELS", "1").lower()
        not in ("0", "false", "no"))
    # Language detection on by default; pin ASSEMBLYAI_LANGUAGE_CODE (e.g. "en")
    # to fix the language and disable detection (cheaper/safer on reliably-English
    # inputs, U2/KTD). language_detection is dropped from the request when a code
    # is pinned.
    assemblyai_language_detection: bool = field(
        default_factory=lambda: os.environ.get("ASSEMBLYAI_LANGUAGE_DETECTION", "1").lower()
        not in ("0", "false", "no"))
    assemblyai_language_code: str = field(
        default_factory=lambda: os.environ.get("ASSEMBLYAI_LANGUAGE_CODE", ""))
    # Poll cadence + budgets. request_timeout bounds each HTTP call; the poll
    # loop's wall-clock ceiling is poll_timeout_base + poll_timeout_per_sec *
    # audio_duration (duration-scaled, mirroring asr_timeout_*).
    assemblyai_poll_interval: float = field(
        default_factory=lambda: float(os.environ.get("ASSEMBLYAI_POLL_INTERVAL", "3.0")))
    assemblyai_request_timeout: float = field(
        default_factory=lambda: float(os.environ.get("ASSEMBLYAI_REQUEST_TIMEOUT", "60.0")))
    assemblyai_poll_timeout_base: float = field(
        default_factory=lambda: float(os.environ.get("ASSEMBLYAI_POLL_TIMEOUT_BASE", "120.0")))
    assemblyai_poll_timeout_per_sec: float = field(
        default_factory=lambda: float(os.environ.get("ASSEMBLYAI_POLL_TIMEOUT_PER_SEC", "2.0")))

    # Python interpreter of the virtualenv that has nemo_toolkit[asr] installed.
    # NeMo cannot share an env with transformers>=5.4, hence the subprocess worker.
    nemotron_python: str = field(
        default_factory=lambda: os.environ.get("NEMOTRON_PYTHON", str(DEFAULT_NEMOTRON_PYTHON))
    )

    # Python interpreter of the virtualenv that has torch + transformers for
    # granite-speech (isolated for the same reason as NeMo: its transformers
    # pin must not constrain the main env).
    granite_python: str = field(
        default_factory=lambda: os.environ.get("GRANITE_PYTHON", str(DEFAULT_GRANITE_PYTHON))
    )
    granite_model_id: str = field(
        default_factory=lambda: os.environ.get("GRANITE_MODEL_ID", "ibm-granite/granite-speech-4.1-2b")
    )

    # Retry/timeout policy for all external calls (oMLX, Higgs, ASR subprocess)
    retry_attempts: int = 3
    retry_base_delay: float = 1.0
    # Env-overridable (LLM_TIMEOUT) so a remote llama.cpp cold-load + model
    # swap, which can approach the default 180s, is calibrated without a code
    # change (R5, KTD5). A non-numeric value raises here, like vieneu_temperature.
    llm_timeout: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT", "180.0")))
    # Safety net on the per-image payload sent to the vision model. Frames
    # normally arrive as 640px JPEGs (~tens of KB) via ffmpeg, but image_part
    # itself has no guard, so a caller bypassing extract_frames — or a botched
    # scale — could ship a multi-MB original and blow past the server's vision
    # context. Cap the raw bytes per image; an oversized frame degrades that
    # shot/context (R19) instead of crashing. <=0 disables. Env-overridable
    # (LLM_IMAGE_MAX_BYTES).
    llm_image_max_bytes: int = field(
        default_factory=lambda: int(os.environ.get("LLM_IMAGE_MAX_BYTES", str(8 * 1024 * 1024))))
    higgs_timeout: float = 600.0
    # ASR subprocess timeout: base covers model load, per-second scales with audio
    asr_timeout_base: float = 900.0
    asr_timeout_per_sec: float = 2.0
    # Granite verify subprocess: same shape (base = model load, per-sec = audio)
    granite_timeout_base: float = 600.0
    granite_timeout_per_sec: float = 2.0

    # ASR windowing: long audio is transcribed in overlapping windows (R10)
    asr_window: float = 600.0
    asr_overlap: float = 10.0
    # Segmentation budget shared by sentence_seg (KTD2/KTD5). The dub backbone
    # is whole sentences, but even a sentence may run long; one over
    # max_segment_duration is pause-split at inter-word silences so each piece
    # re-anchors to the video and drift cannot accumulate. A pause-split point
    # prefers a silence gap of at least segment_split_min_pause between words.
    max_segment_duration: float = 18.0
    segment_split_min_pause: float = 0.4

    # Sentence segmentation (sentence_seg, KTD2). The dub backbone is whole
    # sentences, not Nemotron's acoustic units. Boundaries come from an LLM pass
    # (Gemma via oMLX) over any span that is both longer than
    # sentence_seg_max_unpunct_dur AND below sentence_seg_min_punct_density
    # (fraction of words ending a sentence) — i.e. the under-punctuated
    # monologue the naive punctuation split fails on. Already-punctuated spans
    # are split on their own punctuation with no LLM call. llm_model_seg
    # defaults to llm_model (see __post_init__); sentence_seg_word_window
    # bounds the per-call word count so a very long span is windowed.
    llm_model_seg: str = field(default_factory=lambda: os.environ.get("LLM_MODEL_SEG", ""))
    sentence_seg_max_unpunct_dur: float = 30.0
    sentence_seg_min_punct_density: float = 0.04
    sentence_seg_word_window: int = 1000

    # Subtitle rendering (U2/KTD4): the dub backbone is whole sentences, but
    # subtitles stay short/readable — each sentence is split into sub-style cues
    # of at most srt_max_cue_chars characters and srt_max_cue_dur seconds. EN
    # cues break at real word timestamps; VI cues tile the span proportionally.
    srt_max_cue_chars: int = 84
    srt_max_cue_dur: float = 6.0
    # Burned-in VI captions (U3/KTD2/KTD4): opt-in via --burn-subs. When on, the
    # video is re-encoded through a libass `subtitles` filter; burned cues wrap
    # tighter than the soft-track/sidecar default so on-screen lines stay
    # readable. The burned force_style (font, outline, margin, height-scaled
    # size) is built in mux, not configured here.
    subtitle_burn: bool = False
    srt_burn_max_cue_chars: int = 42

    # Pipeline behaviour
    enable_vision: bool = True
    vision_frames: int = 6
    # Per-shot visual grounding (R39): describe each shot once (sampled at its
    # scene-cut window) and attach it to the segments it covers. scene_threshold
    # tunes scene-cut sensitivity (~0.3-0.4: lower splits talking-heads too
    # finely, higher merges fast slide changes — calibrate on real clips).
    enable_seg_visual: bool = True
    seg_visual_frames: int = 3
    scene_threshold: float = 0.35
    # Cost floor for per-shot description: scene cuts closer together than this
    # are merged, so a cut-heavy/slideshow video can't explode into hundreds of
    # micro-shots (one Gemma call each). Bounds shots to ~duration/min_shot_dur
    # regardless of scene_threshold. Raise to spend fewer Gemma calls.
    min_shot_duration: float = 15.0
    # Embedded/sidecar subtitles (R34/R35): when a segment is covered by a
    # sub that also aligns with Nemotron, the sub text wins outright and no
    # verify engine is called. Floors guard against bad auto-subs / wrong
    # language / desync — calibrate at U6.
    enable_embedded_subs: bool = True
    sub_coverage_floor: float = 0.8   # fraction of segment time a cue must cover
    sub_align_floor: float = 0.6      # token alignment of sub vs Nemotron
    # Cross-check: Gemma re-listens to each segment to verify Nemotron's text
    enable_cross_check: bool = True
    crosscheck_max_clip: float = 30.0       # Gemma's audio length limit per call
    # Lead-in padding (seconds) prepended to each verify clip. Both verify
    # engines systematically dropped the leading word/ordinal on clips cut
    # exactly at the Nemotron boundary ("One What is…"→"What is…",
    # "Unsupervised"→"supervised"); a small lead-in lets them hear the onset.
    # Verification only — segment timing (Nemotron) is untouched (U6 follow-up).
    crosscheck_clip_pad: float = 0.25
    crosscheck_wer_threshold: float = 0.2   # above this, Gemma's reading wins
    crosscheck_align_floor: float = 0.25    # below this, suspect the verify engine
    crosscheck_min_length_ratio: float = 0.3  # verify reading this short -> suspect
    # Ensemble vote weights (R28). Calibrated at U6 against the ai-interview
    # regression set (docs/notes/2026-06-13-ensemble-findings.md): the plan's
    # granite-lead weights (0.2/0.5/0.3, still the reference table in
    # diff.vote3) let lone Granite replace Nemotron, which on real video was
    # net-harmful — leading-word drops and keyword-bias hallucinations. Raising
    # Nemotron to parity with Granite requires *both* verify engines to agree
    # before replacing (lone Granite ties Nemotron -> keep), which cut the
    # replace rate 29% -> 18% and fixed every known-wrong segment. Lower
    # nemotron back toward 0.2 to restore granite-lead once the clip-onset pad
    # (deferred) recovers the dropped leading words.
    crosscheck_weights: dict = field(default_factory=lambda: {
        "nemotron": 0.5, "granite": 0.5, "gemma": 0.3})
    # Max keywords fed into Granite's biased-ASR prompt (R31) to avoid
    # over-biasing the model into hearing terms that were never spoken.
    crosscheck_keyword_cap: int = 32
    # Shared silence knob: cross-check split points and the TTS QA gate
    silence_threshold_db: float = -40.0
    silence_min_duration: float = 0.3

    # Long passages crash autoregressive TTS (truncation/looping/silence): a
    # ~3500-char segment came back as 81s of audio for ~186s of text. So a
    # segment's Vietnamese text is split into chunks of at most this many
    # syllables, synthesized per chunk, QA'd per chunk, and concatenated.
    # ~60 syllables ≈ 14s of speech at the VI profile rate — inside Higgs's
    # reliable zone. Short segments stay one chunk, so their behavior is
    # unchanged. Raise to send longer requests; lower if degradation persists.
    tts_max_chunk_syllables: int = 60
    # Silence inserted between concatenated chunk clips for natural phrasing
    # (Higgs already pads each clip; chunk edges are trimmed before this gap).
    # tts_chunk_gap_ms applies at sentence/clause joins; tts_hardwrap_gap_ms
    # applies where an over-budget clause was cut mid-phrase — that cut has no
    # natural pause, so it defaults to 0 to avoid silence inside a sentence (U4).
    tts_chunk_gap_ms: float = 120.0
    tts_hardwrap_gap_ms: float = 0.0

    # TTS QA gate (R7) — calibrate after the first real run
    qa_min_duration_ratio: float = 0.3   # clip vs expected speech duration
    qa_max_duration_ratio: float = 3.0
    qa_min_clip_sec: float = 0.3
    qa_max_clip_floor_sec: float = 2.5   # upper-bound floor for very short lines

    # Measured-TTS-duration length gate (U6, R6-R8). For non-VI (CPS) profiles the
    # rendered clip duration vs. its slot is the authoritative length gate; a clip
    # longer than slot * slot_overflow_tolerance is over budget. The band is
    # deliberately WIDE until U11 calibration (the loop must not thrash on the
    # first real runs); tighten on measured data. VI keeps the syllable QA gate
    # (measured_duration_active is False for VI), so VI stays byte-identical.
    slot_overflow_tolerance: float = 1.5
    # Re-translation escalation: when a clip is over budget, shrink the text and
    # re-synthesize, bounded by budget_retry_max. OFF by default until U11
    # calibrates the tolerance band — uncalibrated, the loop can re-bill multiple
    # TTS calls per hard segment. The baseline measured gate (length_overflow
    # recording) is always on for non-VI; only this escalation is flag-gated.
    enable_budget_retry: bool = field(
        default_factory=lambda: os.environ.get("ENABLE_BUDGET_RETRY", "").lower()
        in ("1", "true", "yes"))
    budget_retry_max: int = 2

    # Skip ledger / abort window (R5a) — calibrate after the first real run
    abort_window: int = 20
    abort_threshold: int = 5
    # "duck": keep original audio at low volume under the dub; "replace": dub only
    original_audio: str = "duck"
    duck_volume: float = 0.15
    # Max speed-up applied to TTS audio that overflows its subtitle slot
    max_tempo: float = 1.35
    # Placement of a clip that is SHORTER than its slot (U3/KTD3). "center"
    # nudges the clip forward by min(fit_max_center_offset, slack/2) so a short
    # VI clip no longer finishes early and reads as "ahead" of the picture;
    # capping the offset stops a long slot from pushing onset seconds late
    # (which would convert "finishes early" into "starts late" across shot
    # cuts). "start" is the opt-out (left-align at seg.start, the old behavior);
    # "stretch" is deferred. Onset never drifts more than fit_max_center_offset
    # past seg.start and the clip never overruns the slot.
    fit_alignment: str = "center"
    fit_max_center_offset: float = 0.2
    # Dub timeline sample rate; clips at other rates are resampled by `fit`.
    # VieNeu emits 48 kHz (Higgs 24 kHz); 24000 keeps output modest and
    # downsamples VieNeu's clips (KTD5). Raise to 48000 for full VieNeu fidelity.
    timeline_sr: int = 24000
    # Anti-click fade applied to both ends of every placed clip
    fade_ms: float = 30.0
    # Segments per translation request (translated as one batch for coherence)
    translate_batch: int = 12
    # Layered translation context (R40): how many EN neighbor sentences on each
    # side of a batch are folded into its context. Calibrate on real clips.
    context_neighbors: int = 2
    # Max distinct shot descriptions folded into one batch's context prompt. A
    # batch spanning many shots would otherwise concatenate them all and bloat
    # the prompt; the translator only needs roughly what is on screen.
    context_shot_cap: int = 3
    # Sequential running summary layer (R41). One translate-model call per batch
    # building summary_i = f(summary_{i-1}, EN_i); byte-level fingerprint, so an
    # early edit re-summarizes + re-translates the tail (accepted cost, KTD5).
    enable_summary: bool = True

    # Voice cloning reference. When unset, the longest clean segment of the
    # original speaker is extracted automatically.
    ref_audio: str | None = None
    ref_text: str | None = None

    workdir: Path = Path("work")

    def __post_init__(self) -> None:
        # Normalize the language tags once, up front: lowercase + strip whitespace
        # so every downstream consumer (profile resolution, the translate
        # fingerprint guard, the `auto` source-detection check, locale-derived
        # filenames) treats "VI"/"vi", "EN"/"en", and "AUTO"/"auto" identically.
        # BCP-47 is case-insensitive and `resolve()` already lowercases, so this
        # only removes the inconsistency that let a non-canonical spelling silently
        # bust the byte-identical vi/en cache and re-bill (R19). Region subtags are
        # preserved (resolve() collapses them) so a future region profile is not
        # pre-empted; the translate fingerprint collapses region separately.
        self.target_lang = self.target_lang.strip().lower()
        self.source_lang = self.source_lang.strip().lower()
        # Per-role LLM endpoints inherit the base when their override is empty.
        # default_factory is a 0-arg callable that can't read sibling fields, so
        # the fallback lives here; it runs after every field is initialized, so
        # it honors kwargs (e.g. Config(llm_model="X")), not just env (KTD1).
        # An empty config / the old single-host oMLX profile keeps every role on
        # the same host+model — byte-identical behavior (R3). Set LLM_*_<ROLE>
        # to split a role onto its own host/model (e.g. audio on a separate
        # always-hot host so a one-model-resident router never swaps).
        for role in ("vision", "translate", "seg", "audio"):
            if not getattr(self, f"llm_host_{role}"):
                setattr(self, f"llm_host_{role}", self.llm_host)
            if not getattr(self, f"llm_api_key_{role}"):
                setattr(self, f"llm_api_key_{role}", self.llm_api_key)
            if not getattr(self, f"llm_model_{role}"):
                setattr(self, f"llm_model_{role}", self.llm_model)

    @property
    def tts_uses_cloning(self) -> bool:
        """Does the selected TTS engine clone the original speaker's voice IN THE
        TARGET LANGUAGE? A thin adapter over the provider's `clones_in(locale)`
        capability (KTD4/KTD7) — the single source of truth for the
        cloning-vs-preset branch in voice_ref, tts, and the clip fingerprint, so
        all three agree. Language-aware (U9): VieNeu clones only for vi, so a
        non-VI target resolves to preset (or preflight rejects an engine that can
        only clone). Byte-identical at the VI default. The registry is imported
        lazily because the provider modules import Config (the cycle, KTD4)."""
        from loro.providers import tts
        return tts(self.tts_engine).clones_in(self.target_lang)

    @property
    def effective_tts_language(self) -> str:
        """The spoken-language code for the preset cloud engines (Soniox). The
        profile's `tts_language_code` when it has one; for the generic fallback
        profile (empty code, reached only via --allow-fallback) the target tag's
        base subtag is the best-effort spoken-language hint, so a fallback run
        never sends an empty `language` to the engine (#3). Byte-identical for any
        profiled target — the profile code is non-empty, so the fallback is inert
        and the clip fingerprint is unchanged."""
        return self.language_profile.tts_language_code or self.target_lang.split("-")[0]

    @property
    def preset_voices(self) -> PresetVoices | None:
        """The active preset engine's (pool, voice_map, default) voice config. A
        thin adapter delegating to the provider (KTD4/KTD6), so _voice_cast and the
        tts node's default-voice fallback need no per-engine branch. Only the
        preset engines (soniox/gemini) reach this; the cloning engines are gated
        out by tts_uses_cloning (their provider returns None). Lazy import for the
        same config<->providers cycle reason as tts_uses_cloning."""
        from loro.providers import tts
        return tts(self.tts_engine).preset_voices(self)

    @property
    def language_profile(self):
        """The resolved LanguageProfile for `target_lang` (R2). A thin lazy
        accessor over the registry — the single source every node reads instead of
        branching on language. The import is local to keep `config` import-light
        and to mirror the provider adapters above (the profiles module is pure, so
        there is no cycle; this just defers the cost to first use). An unprofiled
        tag resolves to the generic fallback here; preflight is where it is gated
        behind --allow-fallback (R4)."""
        from loro.profiles import resolve
        return resolve(self.target_lang)

    @property
    def tts_chunk_budget(self) -> int:
        """TTS sub-chunk budget in the active profile's length unit (U5). The
        syllable model (VI) keeps the `tts_max_chunk_syllables` knob, so the VI
        clip fingerprint and chunking stay byte-identical (R19); CPS profiles use
        the profile's char-sized `chunk_budget` so a non-VI segment is not
        fragmented ~4x by feeding a character counter a syllable-valued budget.
        Both `_seg_inputs` (clip fingerprint) and `_synthesize_clip` (actual
        chunking) read this one value so they can never diverge."""
        p = self.language_profile
        return self.tts_max_chunk_syllables if p.length_model == "syllable" else p.chunk_budget

    @property
    def measured_duration_active(self) -> bool:
        """Is the measured-clip-duration-vs-slot length gate active (U6)? On for
        the CPS profiles (non-VI), off for the VI syllable model — so the whole
        measured-duration mechanism (length_overflow recording + the re-translation
        loop) is inert for VI and the VI path stays byte-identical (R19)."""
        return self.language_profile.length_model == "cps"

    def llm_role(self, name: str) -> LlmRole:
        """Resolve role `name` (vision | translate | seg | audio) to its
        (host, api_key, model). The per-role fields are filled from the base in
        __post_init__, so this is the explicit accessor for the role contract that
        call sites and the preflight model-serving check read (KTD7, R13)."""
        return LlmRole(
            getattr(self, f"llm_host_{name}"),
            getattr(self, f"llm_api_key_{name}"),
            getattr(self, f"llm_model_{name}"),
        )
