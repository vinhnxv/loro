"""U1: Golden fingerprint-parity snapshots for every ASR/TTS/voice-cast engine.

Characterization-first GATE (KTD5; R10/R11; AE2). These capture the canonical
cache-fingerprint *inputs dict* of each engine as a golden literal, computed from
the CURRENT production code, before the provider refactor (U2-U8) moves where
each dict is built. `artifacts.fingerprint` hashes a key-sorted JSON dump, so
dict-equality is hash-equality, and that hash is the artifact cache key gating
real per-minute cloud billing — any drift here means a previously-cached
transcript/clip re-uploads, re-synthesizes, and re-bills.

The TTS dicts come straight from `_seg_inputs` (callable, asserted by full
dict-equality). The ASR and voice-cast dicts are built inline inside their nodes,
so we run the node against mocked services and assert the persisted sidecar
`input_fingerprint` equals `fingerprint(<golden literal dict>)` — exercising the
real construction path while keeping the golden literal legible.

These goldens encode the implicit `target_lang="vi"` / `source_lang="en"` default
(those Config knobs do not exist yet — U4 adds them). The multi-language refactor
(U2 rename, U5 CPS budget, U7 source detection, U8 profile prompt) must keep the
VI/EN default byte-identical: the frozen translate-budget hashes below are the
absolute gate for that, robust to however the nodes are later refactored.
"""

import json
import sys
import textwrap
from pathlib import Path

import pytest

from loro.config import Config
from loro.harness import artifacts
from loro.harness.artifacts import fingerprint
from loro.nodes import asr as asr_mod
from loro.nodes import translate as tr_mod
from loro.nodes.asr import MERGE_EPS
from loro.nodes.tts import _seg_inputs
from loro.nodes.voice import _voice_cast
from loro.providers.asr import local as local_provider
from loro.services import assemblyai, soniox_stt
from loro.state import Segment
from loro.utils import ffmpeg
from loro.workers.nemotron_worker import MODEL_ID


# Every env var that feeds an identity-bearing field, cleared so Config() yields
# the documented code defaults the golden literals below encode. A default change
# is itself a fingerprint change (and must update these goldens deliberately).
_IDENTITY_ENV = (
    "TTS_ENGINE", "ASR_ENGINE",
    "VIENEU_MODEL", "VIENEU_TEMPERATURE", "VIENEU_EMOTION",
    "HIGGS_MODEL",
    "SONIOX_MODEL", "SONIOX_SAMPLE_RATE", "SONIOX_AUDIO_FORMAT",
    "GEMINI_MODEL", "GEMINI_SAMPLE_RATE", "GEMINI_STYLE_PROMPT",
    "GEMINI_BATCH_MAX_SYLLABLES",
    "SONIOX_STT_MODEL", "SONIOX_STT_LANGUAGE_HINTS",
    "SONIOX_STT_ENABLE_LANGUAGE_IDENTIFICATION", "SONIOX_STT_SPEAKER_DIARIZATION",
    "SONIOX_STT_CONTEXT_TERMS", "SONIOX_STT_CONTEXT_TEXT",
    "ASSEMBLYAI_SPEECH_MODELS", "ASSEMBLYAI_SPEAKER_LABELS",
    "ASSEMBLYAI_LANGUAGE_DETECTION", "ASSEMBLYAI_LANGUAGE_CODE",
    "SONIOX_VOICE_POOL", "SONIOX_VOICE_MAP", "SONIOX_DEFAULT_VOICE",
    "GEMINI_VOICE_POOL", "GEMINI_VOICE_MAP", "GEMINI_DEFAULT_VOICE",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in _IDENTITY_ENV:
        monkeypatch.delenv(name, raising=False)


# --------------------------------------------------------------------------- #
# TTS — full _seg_inputs dict per engine (callable, so assert dict-equality).
# --------------------------------------------------------------------------- #

_SEG = Segment(index=0, start=0.0, end=1.0, text_src="hello",
               text_target="Xin chào", speaker="A")
_REF_SHA = "ref-sha-fixed"
_REF_TEXT = "the reference transcript"

# Cloning engines fold the reference (ref_sha/ref_text) + the chunking knobs.
_VIENEU_GOLDEN = {
    "text_vi": "Xin chào",
    "engine": "vieneu",
    "model": "pnnbao-ump/VieNeu-TTS-v3-Turbo",
    "temperature": 0.8,
    "emotion": "natural",
    "max_chunk_syllables": 60,
    "chunk_gap_ms": 120.0,
    "hardwrap_gap_ms": 0.0,
    "ref_sha": _REF_SHA,
    "ref_text": _REF_TEXT,
}
_HIGGS_GOLDEN = {
    "text_vi": "Xin chào",
    "engine": "higgs",
    "model": "bosonai/higgs-audio-v3-tts-4b",
    "max_chunk_syllables": 60,
    "chunk_gap_ms": 120.0,
    "hardwrap_gap_ms": 0.0,
    "ref_sha": _REF_SHA,
    "ref_text": _REF_TEXT,
}
# Preset engines fold the per-segment cast voice instead of a reference.
_SONIOX_GOLDEN = {
    "text_vi": "Xin chào",
    "engine": "soniox",
    "model": "tts-rt-v1",
    "language": "vi",
    "sample_rate": 24000,
    "audio_format": "wav",
    "max_chunk_syllables": 60,
    "chunk_gap_ms": 120.0,
    "hardwrap_gap_ms": 0.0,
    "voice": "Adrian",
}
# Gemini omits the chunking knobs (it does not chunk normal segments, KTD5).
_GEMINI_GOLDEN = {
    "text_vi": "Xin chào",
    "engine": "gemini",
    "model": "gemini-3.1-flash-tts-preview",
    "sample_rate": 24000,
    "style_prompt": "",
    "batch_max_syllables": 360,
    "voice": "Kore",
}


def test_vieneu_seg_inputs_golden():
    inputs, _ = _seg_inputs(_SEG, Config(tts_engine="vieneu"), True,
                            _REF_SHA, _REF_TEXT, {})
    assert inputs == _VIENEU_GOLDEN


def test_higgs_seg_inputs_golden():
    inputs, _ = _seg_inputs(_SEG, Config(tts_engine="higgs"), True,
                            _REF_SHA, _REF_TEXT, {})
    assert inputs == _HIGGS_GOLDEN


def test_soniox_seg_inputs_golden():
    inputs, voice = _seg_inputs(_SEG, Config(tts_engine="soniox"), False,
                                None, None, {"A": "Adrian"})
    assert inputs == _SONIOX_GOLDEN
    assert voice == "Adrian"


def test_gemini_seg_inputs_golden():
    inputs, voice = _seg_inputs(_SEG, Config(tts_engine="gemini"), False,
                                None, None, {"A": "Kore"})
    assert inputs == _GEMINI_GOLDEN
    assert voice == "Kore"


# --------------------------------------------------------------------------- #
# translate — budget model + batch inputs dict (the translate cache key).
#
# The syllable budget (the VI profile rate = 4.3) feeds `_seg_hash` (the
# per-segment translate key) and every batch's `[index, text, budget]` line; the
# Vietnamese SYSTEM prompt + model + temperature feed the batch key. U5 (CPS
# budget) demotes syllables to a VI-only model and U8 sources the prompt from the
# profile — both MUST keep the VI/EN default byte-identical, so these are frozen
# absolute hashes, not reconstructions that would drift with the source.
# --------------------------------------------------------------------------- #

# A 1.0s segment -> max(3, int(1.0 * 4.3)) = 4 syllables (the legacy VI model).
_TRANSLATE_BUDGET_VI = 4
_SEG_HASH_GOLDEN = "53cfa70c6405d4eb51fcd2c389084481254bc74caafcb57afc7d22001d9e400e"
# Batch of three 1.5s lines -> int(1.5 * 4.3) = 6 syllables each, no layered
# context (video_context=""). The node-produced sidecar must equal this hash.
_TRANSLATE_BATCH_INPUTS_FP = "921c9235a1f04381469227a1ab59ff7d833cbd2d88c3d8250979e0a0e96c2e15"


def test_translate_budget_model_golden():
    # The VI syllable budget for a fixed segment must not drift: U5 keeps VI on
    # the legacy `syllable_count` + 4.3 model verbatim.
    assert tr_mod._budget(Config(), _SEG) == _TRANSLATE_BUDGET_VI


def test_translate_seg_hash_golden():
    # The per-segment translate cache key folds {"en": text, "budget": int}; both
    # the dict-literal and the live `_seg_hash` path hash to the frozen value.
    assert fingerprint({"en": "hello", "budget": _TRANSLATE_BUDGET_VI}) == _SEG_HASH_GOLDEN
    assert tr_mod._seg_hash(_SEG, _TRANSLATE_BUDGET_VI) == _SEG_HASH_GOLDEN


def test_translate_batch_inputs_fingerprint_golden(tmp_path, monkeypatch):
    # Full batch translate cache key (SYSTEM prompt + model + temperature +
    # per-line [index, text, budget]) for the VI/EN default, no layered context.
    # Run the real node against a mocked LLM and assert the persisted sidecar
    # fingerprint equals the frozen hash — robust to U5/U8 internal refactors.
    def fake_chat(cfg, messages, **kw):
        user = messages[-1]["content"]
        lines = json.loads(user[user.rindex("[{"):])
        return json.dumps([{"i": l["i"], "vi": f"bản dịch {l['i']}"} for l in lines],
                          ensure_ascii=False)

    monkeypatch.setattr(tr_mod.llm, "chat", fake_chat)
    segs = [Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.5,
                    text_src=f"english line {i}") for i in range(3)]
    state = {"workdir": str(tmp_path), "segments": segs, "video_context": ""}
    tr_mod.translate(state, Config(translate_batch=3))

    meta = artifacts.read_meta(tmp_path / "translate" / "batch_0000.json")
    assert meta["input_fingerprint"] == _TRANSLATE_BATCH_INPUTS_FP


@pytest.mark.parametrize("target_lang,source_lang", [
    ("VI", "EN"),        # uppercase canonical spelling
    ("vi-VN", "en-US"),  # region-subtagged spelling of the same languages
    ("Vi", "eN"),        # mixed case
])
def test_translate_fingerprint_is_spelling_invariant(tmp_path, monkeypatch,
                                                     target_lang, source_lang):
    # R19/#1: a non-canonical but equivalent spelling of the vi/en default must
    # resolve to the SAME translate fingerprint as the canonical lowercase tags —
    # else it silently misses the existing cache and re-bills. Config normalizes
    # case; the fingerprint guard collapses region subtags to the base tag.
    def fake_chat(cfg, messages, **kw):
        user = messages[-1]["content"]
        lines = json.loads(user[user.rindex("[{"):])
        return json.dumps([{"i": l["i"], "vi": f"bản dịch {l['i']}"} for l in lines],
                          ensure_ascii=False)

    monkeypatch.setattr(tr_mod.llm, "chat", fake_chat)
    segs = [Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.5,
                    text_src=f"english line {i}") for i in range(3)]
    state = {"workdir": str(tmp_path), "segments": segs, "video_context": ""}
    tr_mod.translate(state, Config(translate_batch=3, target_lang=target_lang,
                                   source_lang=source_lang))

    meta = artifacts.read_meta(tmp_path / "translate" / "batch_0000.json")
    assert meta["input_fingerprint"] == _TRANSLATE_BATCH_INPUTS_FP


# --------------------------------------------------------------------------- #
# ASR — inline inputs dicts, pinned via the persisted sidecar fingerprint.
# --------------------------------------------------------------------------- #

_SONIOX_TOKENS = {
    "tokens": [
        {"text": "Hello", "start_ms": 100, "end_ms": 480, "speaker": "1"},
        {"text": " world.", "start_ms": 500, "end_ms": 900, "speaker": "1"},
    ],
}
_ASSEMBLYAI_RESPONSE = {
    "status": "completed",
    "text": "Hello world.",
    "words": [
        {"start": 100, "end": 480, "text": "Hello", "speaker": "A"},
        {"start": 500, "end": 900, "text": "world.", "speaker": "A"},
    ],
    "utterances": [{"start": 100, "end": 900, "text": "Hello world.", "speaker": "A"}],
}


def _asr_state(tmp_path, body=b"RIFFfake-wav"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    audio = tmp_path / "audio_16k.wav"
    audio.write_bytes(body)
    return {"workdir": str(tmp_path), "audio_16k": str(audio)}, audio


def test_soniox_asr_inputs_golden(tmp_path, monkeypatch):
    state, audio = _asr_state(tmp_path)
    monkeypatch.setattr(soniox_stt, "transcribe", lambda cfg, a, **kw: _SONIOX_TOKENS)
    asr_mod.asr(state, Config(asr_engine="soniox", soniox_api_key="k"))

    golden = {
        "audio_sha": artifacts.file_sha256(audio),
        "engine": "soniox",
        "model": "stt-async-v5",
        "language_hints": ["en"],
        "enable_language_identification": False,
        "enable_speaker_diarization": True,
        "context_terms": [],
        "context_text": "",
    }
    meta = artifacts.read_meta(tmp_path / "asr" / "soniox.json")
    assert meta["input_fingerprint"] == fingerprint(golden)


def test_assemblyai_asr_inputs_golden(tmp_path, monkeypatch):
    state, audio = _asr_state(tmp_path)
    monkeypatch.setattr(assemblyai, "transcribe", lambda cfg, a: _ASSEMBLYAI_RESPONSE)
    asr_mod.asr(state, Config(asr_engine="assemblyai", assemblyai_api_key="k"))

    golden = {
        "audio_sha": artifacts.file_sha256(audio),
        "engine": "assemblyai",
        "speech_models": ["universal-3-pro", "universal-2"],
        "speaker_labels": True,
        "language_detection": True,
        "language_code": "",
    }
    # Both the raw-transcript cache and the utterances.json artifact fold this dict.
    raw_meta = artifacts.read_meta(tmp_path / "asr" / "assemblyai.json")
    utt_meta = artifacts.read_meta(tmp_path / "asr" / "utterances.json")
    assert raw_meta["input_fingerprint"] == fingerprint(golden)
    assert utt_meta["input_fingerprint"] == fingerprint(golden)


# A stub worker that echoes the canned transcription of each input "wav" (a JSON
# file). Same shape as test_asr_merge's stub; pure stdlib so the subprocess is
# fast. Its file content is fixed, so worker_sha is deterministic run to run.
_STUB_WORKER = textwrap.dedent('''
    import json, sys
    for path in sys.argv[1:]:
        with open(path) as f:
            payload = json.load(f)
        print(json.dumps({"path": path, "text": payload["text"],
                          "segments": payload["segments"],
                          "words": payload.get("words")}), flush=True)
''')


def test_assemblyai_pinned_code_fingerprint_invariant_to_detection(tmp_path, monkeypatch):
    # B4/R7/KTD5: a pinned language_code drops language_detection from the
    # fingerprint (mirroring the request), so the two toggles hash identically.
    monkeypatch.setattr(assemblyai, "transcribe", lambda cfg, a: _ASSEMBLYAI_RESPONSE)

    def fp(detection):
        state, _ = _asr_state(tmp_path / f"d{int(detection)}")
        asr_mod.asr(state, Config(asr_engine="assemblyai", assemblyai_api_key="k",
                                  assemblyai_language_code="en",
                                  assemblyai_language_detection=detection))
        return artifacts.read_meta(
            tmp_path / f"d{int(detection)}" / "asr" / "assemblyai.json")["input_fingerprint"]

    assert fp(True) == fp(False)


def test_local_asr_window_and_merge_inputs_golden(tmp_path, monkeypatch):
    worker = tmp_path / "stub_worker.py"
    worker.write_text(_STUB_WORKER)
    monkeypatch.setattr(local_provider, "WORKER", worker)
    # Single window (duration <= asr_window): the audio file is itself the "wav"
    # the stub reads, so it must be the JSON payload the stub echoes.
    monkeypatch.setattr(ffmpeg, "probe_duration", lambda p: 100.0)
    audio = tmp_path / "audio_16k.wav"
    audio.write_text(json.dumps({
        "text": "hello world",
        "segments": [{"start": 1.0, "end": 5.0, "text": "hello world"}],
        "words": [{"start": 1.0, "end": 2.0, "word": "hello"},
                  {"start": 2.0, "end": 5.0, "word": "world"}],
    }))
    state = {"workdir": str(tmp_path), "audio_16k": str(audio)}
    cfg = Config(asr_engine="local", nemotron_python=sys.executable,
                 retry_base_delay=0.0)
    asr_mod.asr(state, cfg)

    win_golden = {
        "audio_sha": artifacts.file_sha256(audio),
        "start": 0.0,
        "length": 100.0,
        "overlap": 10.0,
        "worker_sha": artifacts.file_sha256(worker),
        "model_id": MODEL_ID,
    }
    win = tmp_path / "asr" / "win_0000.json"
    assert artifacts.read_meta(win)["input_fingerprint"] == fingerprint(win_golden)

    merge_golden = {
        "window_hashes": [artifacts.cached_file_sha256(win)],
        "eps": MERGE_EPS,
    }
    merge_meta = artifacts.read_meta(tmp_path / "asr" / "segments.json")
    assert merge_meta["input_fingerprint"] == fingerprint(merge_golden)


def test_local_asr_multi_window_inputs_golden(tmp_path, monkeypatch):
    # Multi-window parity (3 windows): pins each per-window inputs dict at non-zero
    # offsets AND the merge dict's window_hashes LIST — a single-window test can't
    # cover the multi-element hash list whose drift would re-transcribe every
    # cached window on rerun.
    worker = tmp_path / "stub_worker.py"
    worker.write_text(_STUB_WORKER)
    monkeypatch.setattr(local_provider, "WORKER", worker)
    monkeypatch.setattr(ffmpeg, "probe_duration", lambda p: 1500.0)

    def fake_cut(src, out, start, end):
        # Each window's cut "wav" is the JSON payload the stub echoes.
        with open(out, "w") as f:
            json.dump({"text": f"win {start:.0f}",
                       "segments": [{"start": 1.0, "end": 5.0, "text": f"w{start:.0f}"}],
                       "words": [{"start": 1.0, "end": 2.0, "word": "w"}]}, f)

    monkeypatch.setattr(ffmpeg, "cut_audio", fake_cut)
    audio = tmp_path / "audio_16k.wav"
    audio.write_bytes(b"fake-audio-bytes")
    state = {"workdir": str(tmp_path), "audio_16k": str(audio)}
    cfg = Config(asr_engine="local", nemotron_python=sys.executable,
                 retry_base_delay=0.0)
    asr_mod.asr(state, cfg)

    audio_sha = artifacts.file_sha256(audio)
    worker_sha = artifacts.file_sha256(worker)
    # window_bounds(1500, 600, 10) = [(0, 600), (590, 1190), (1180, 1500)]
    expected = [(0.0, 600.0), (590.0, 600.0), (1180.0, 320.0)]
    for i, (start, length) in enumerate(expected):
        win = tmp_path / "asr" / f"win_{i:04d}.json"
        golden = {"audio_sha": audio_sha, "start": start, "length": length,
                  "overlap": 10.0, "worker_sha": worker_sha, "model_id": MODEL_ID}
        assert artifacts.read_meta(win)["input_fingerprint"] == fingerprint(golden)

    merge_golden = {
        "window_hashes": [artifacts.cached_file_sha256(tmp_path / "asr" / f"win_{i:04d}.json")
                          for i in range(3)],
        "eps": MERGE_EPS,
    }
    merge_meta = artifacts.read_meta(tmp_path / "asr" / "segments.json")
    assert merge_meta["input_fingerprint"] == fingerprint(merge_golden)


# --------------------------------------------------------------------------- #
# voice-cast — inline inputs dict, pinned via the persisted cast.json sidecar.
# --------------------------------------------------------------------------- #

_VOICE_POOL = ["Adrian", "Maya", "Noah", "Nina", "Jack", "Emma"]


def _spk_seg(index, speaker):
    return Segment(index=index, start=float(index), end=float(index) + 1.0,
                   text_src=f"s{index}", speaker=speaker)


def test_voice_cast_two_speaker_inputs_golden(tmp_path):
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, "A"), _spk_seg(1, "B"), _spk_seg(2, "A")]}
    _voice_cast(state, Config(tts_engine="soniox"))
    golden = {
        "speakers": ["A", "B"],
        "pool": _VOICE_POOL,
        "map": {},
        "default": "Adrian",
    }
    meta = artifacts.read_meta(tmp_path / "voice" / "cast.json")
    assert meta["input_fingerprint"] == fingerprint(golden)


def test_voice_cast_single_empty_speaker_inputs_golden(tmp_path):
    state = {"workdir": str(tmp_path), "segments": [_spk_seg(0, "")]}
    _voice_cast(state, Config(tts_engine="soniox"))
    golden = {
        "speakers": [""],
        "pool": _VOICE_POOL,
        "map": {},
        "default": "Adrian",
    }
    meta = artifacts.read_meta(tmp_path / "voice" / "cast.json")
    assert meta["input_fingerprint"] == fingerprint(golden)


# The gemini preset engine casts from its own pool/default (resolved through the
# provider, U4), so its voice-cast fingerprint is distinct from soniox's and must
# be pinned separately — a drift re-bills every cached Gemini clip.
_GEMINI_VOICE_POOL = ["Kore", "Puck", "Aoede", "Charon", "Leda", "Orus"]


def test_voice_cast_gemini_two_speaker_inputs_golden(tmp_path):
    state = {"workdir": str(tmp_path),
             "segments": [_spk_seg(0, "A"), _spk_seg(1, "B"), _spk_seg(2, "A")]}
    _voice_cast(state, Config(tts_engine="gemini"))
    golden = {
        "speakers": ["A", "B"],
        "pool": _GEMINI_VOICE_POOL,
        "map": {},
        "default": "Kore",
    }
    meta = artifacts.read_meta(tmp_path / "voice" / "cast.json")
    assert meta["input_fingerprint"] == fingerprint(golden)


def test_voice_cast_gemini_single_empty_speaker_inputs_golden(tmp_path):
    state = {"workdir": str(tmp_path), "segments": [_spk_seg(0, "")]}
    _voice_cast(state, Config(tts_engine="gemini"))
    golden = {
        "speakers": [""],
        "pool": _GEMINI_VOICE_POOL,
        "map": {},
        "default": "Kore",
    }
    meta = artifacts.read_meta(tmp_path / "voice" / "cast.json")
    assert meta["input_fingerprint"] == fingerprint(golden)


# --------------------------------------------------------------------------- #
# Edge: an identity-bearing knob shifts the fingerprint; a non-identity knob
# (e.g. a poll interval / request timeout) does not.
# --------------------------------------------------------------------------- #

def test_identity_knob_changes_tts_fingerprint():
    base, _ = _seg_inputs(_SEG, Config(tts_engine="gemini"), False, None, None,
                          {"A": "Kore"})
    styled, _ = _seg_inputs(_SEG, Config(tts_engine="gemini", gemini_style_prompt="warm"),
                            False, None, None, {"A": "Kore"})
    assert fingerprint(base) != fingerprint(styled)


def test_nonidentity_knob_does_not_change_tts_fingerprint():
    # soniox_timeout shapes nothing in the emitted clip, so it is absent from the
    # clip fingerprint — tuning it must not resynthesize.
    a, _ = _seg_inputs(_SEG, Config(tts_engine="soniox", soniox_timeout=60.0),
                       False, None, None, {"A": "Adrian"})
    b, _ = _seg_inputs(_SEG, Config(tts_engine="soniox", soniox_timeout=999.0),
                       False, None, None, {"A": "Adrian"})
    assert fingerprint(a) == fingerprint(b)


def _soniox_asr_fingerprint(tmp_path, monkeypatch, **cfg_kw):
    state, _ = _asr_state(tmp_path)
    monkeypatch.setattr(soniox_stt, "transcribe", lambda cfg, a, **kw: _SONIOX_TOKENS)
    asr_mod.asr(state, Config(asr_engine="soniox", soniox_api_key="k", **cfg_kw))
    return artifacts.read_meta(tmp_path / "asr" / "soniox.json")["input_fingerprint"]


def test_identity_knob_changes_asr_fingerprint(tmp_path, monkeypatch):
    base = _soniox_asr_fingerprint(tmp_path / "a", monkeypatch)
    ctx = _soniox_asr_fingerprint(tmp_path / "b", monkeypatch,
                                  soniox_stt_context_text="LangGraph dubbing")
    assert base != ctx


def test_nonidentity_knob_does_not_change_asr_fingerprint(tmp_path, monkeypatch):
    # The poll interval is run-cadence, not transcript identity, so it is absent
    # from the inputs dict — changing it leaves the cache key untouched.
    base = _soniox_asr_fingerprint(tmp_path / "a", monkeypatch,
                                   soniox_stt_poll_interval=3.0)
    slow = _soniox_asr_fingerprint(tmp_path / "b", monkeypatch,
                                   soniox_stt_poll_interval=30.0)
    assert base == slow


def test_default_source_lang_keeps_asr_fingerprint_but_auto_busts_it(tmp_path, monkeypatch):
    # R20: the default source_lang=en leaves the Soniox ASR fingerprint identical
    # to the golden (LID off); source_lang=auto is a DELIBERATE bust (LID on,
    # widened hints) — a documented re-bill, not silent drift.
    default = _soniox_asr_fingerprint(tmp_path / "a", monkeypatch)
    explicit_en = _soniox_asr_fingerprint(tmp_path / "b", monkeypatch, source_lang="en")
    auto = _soniox_asr_fingerprint(tmp_path / "c", monkeypatch, source_lang="auto")
    assert default == explicit_en  # en is the byte-identical default
    assert default != auto         # auto flips LID + widens hints (re-bill)
