"""Synthesize target-language speech per segment via the active LanguageProfile.

Each clip is a durable artifact fingerprinted by content — effective text_target,
and either the cloning reference (audio hash + text) or the preset voice, plus
engine + synthesis params — never by index alone, so a changed translation, a
re-pinned speaker, or an engine switch resynthesizes exactly its own clip (R1a,
R4, R8, AE2). Every freshly synthesized clip passes the mechanical QA gate
before the artifact is finalized; a clip that keeps failing is skipped through
the ledger (TTS sampling is nondeterministic, so retrying genuinely helps). The
engine is selected at runtime by cfg.tts_engine through the provider registry
(providers.tts()); per-engine behavior lives in src/loro/providers/tts/. Capability
flags on the provider govern the cloning-vs-preset path (clones) and the batch
path (batches/native_long_text), read in place of engine-name checks (KTD1/KTD4).
"""

import json
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from loro import providers
from loro.config import Config
from loro.harness import artifacts, qa
from loro.harness.ledger import SkipLedger
from loro.harness.retry import StageError, with_retry
from loro.profiles import resolve
from loro.state import DubState, Segment, segment_id
from loro.utils import ffmpeg, srt
from loro.utils.audio import trim_silence_edges
from loro.utils.textchunk import HARDWRAP, chunk_for_tts_typed

log = logging.getLogger("loro.tts")

# FROZEN clip-fingerprint key (KTD6/R19): the per-clip text key is the legacy
# Vietnamese-era name "text_vi" for EVERY target language. Renaming it busts every
# cached TTS clip (re-bill) and must update the fingerprint-parity goldens — it is
# NOT the target language, it is a frozen key (#8).
_K_TEXT = "text_vi"


def _tts_client(cfg: Config, ref_audio: Path | None = None, ref_text: str | None = None):
    """Return the TTS client for the configured engine via its provider (KTD1).
    All clients honor the same context-manager + synthesize(text, out, voice)
    surface, so the rest of the node is engine-agnostic. The preset engines ignore
    ref_audio/ref_text; the cloning engines ignore the per-call voice instead."""
    return providers.tts(cfg.tts_engine).client(cfg, ref_audio, ref_text)


def _engine_inputs(cfg: Config) -> dict:
    """Engine identity + its config-level synthesis params, folded into every
    clip's fingerprint (KTD4/KTD6, R4/R8), sourced from the active provider:
    switching engine/model/params re-synthesizes exactly the affected clips
    instead of reusing a clip another engine produced. Each engine contributes
    only its own keys, so an unrelated engine's knob can't invalidate this one.
    The per-segment preset voice is NOT here (it varies per clip) — _seg_inputs
    folds it into the fingerprint alongside this (R8)."""
    return providers.tts(cfg.tts_engine).engine_inputs(cfg)


def _chunk_budget(cfg: Config) -> int:
    """Per-engine sub-chunk syllable budget for _synthesize_clip, sourced from the
    active provider. The chunking engines (higgs/vieneu/soniox) truncate/loop on
    long paragraphs, so they use the tight tts_max_chunk_syllables. Gemini handles
    long text natively, so a normal segment is one call; its budget is raised to
    gemini_batch_max_syllables so only a pathologically long SINGLE segment is
    split — guarding the per-segment fallback from emitting a truncated/drifted
    clip QA's duration window happens to accept (A4)."""
    return providers.tts(cfg.tts_engine).chunk_budget(cfg)


def _synthesize_clip(client, text_target: str, out: Path, cfg: Config,
                     voice: str | None = None) -> None:
    """Synthesize `text_target` to `out` as one clip in the given preset `voice`
    (ignored by the cloning clients). Long text is split into syllable-bounded
    chunks (autoregressive TTS truncates/loops on whole paragraphs); each chunk
    is synthesized and QA'd on its own so a flaky chunk retries locally instead
    of redoing the segment, then the chunks are concatenated with a short gap.
    Short text takes the unchanged single call."""
    chunks, break_types = chunk_for_tts_typed(text_target, _chunk_budget(cfg),
                                              cfg.language_profile.counter)
    if len(chunks) <= 1:
        client.synthesize(text_target, out, voice)
        return

    parts: list[np.ndarray] = []
    sr: int | None = None
    for n, chunk in enumerate(chunks):
        sub = out.with_name(f".chunk.{out.name}.{n:03d}.wav")

        def synth(sub=sub, chunk=chunk) -> None:
            client.synthesize(chunk, sub, voice)   # infra retry lives inside the client
            qa.check_clip(sub, chunk, cfg)         # qa StageError on a bad chunk

        try:
            with_retry("tts", synth, attempts=cfg.retry_attempts,
                       base_delay=0.0, retry_classes=("qa",))
            audio, file_sr = sf.read(str(sub), dtype="float32", always_2d=False)
        finally:
            sub.unlink(missing_ok=True)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sr = sr or int(file_sr)
        parts.append(trim_silence_edges(audio, cfg.silence_threshold_db))

    # A sentence/clause join gets the natural pause; a mid-clause hard-wrap cut
    # gets tts_hardwrap_gap_ms (0 by default) so a single long sentence reads as
    # continuous audio instead of stuttering at an arbitrary word boundary (U4).
    boundary_gap = np.zeros(int(sr * cfg.tts_chunk_gap_ms / 1000), dtype="float32")
    hardwrap_gap = np.zeros(int(sr * cfg.tts_hardwrap_gap_ms / 1000), dtype="float32")
    pieces: list[np.ndarray] = []
    for n, part in enumerate(parts):
        if n:
            pieces.append(hardwrap_gap if break_types[n - 1] == HARDWRAP else boundary_gap)
        pieces.append(part)
    sf.write(str(out), np.concatenate(pieces), sr)


def _seg_inputs(seg: Segment, cfg: Config, cloning: bool, ref_sha: str | None,
                ref_text: str | None, voice_cast: dict) -> tuple[dict, str | None]:
    """The clip's fingerprint inputs dict + its resolved preset voice. Cloning
    folds the reference into the key; the preset engines fold the per-segment
    cast voice instead (KTD6, R8). The chunking-engine knobs are part of clip
    identity for higgs/vieneu/soniox, but are OMITTED for Gemini — it doesn't
    chunk normal segments (its budget is gemini_batch_max_syllables, already in
    _engine_inputs), so tuning a Higgs/VieNeu chunk knob must not invalidate a
    cached Gemini clip (KTD5)."""
    inputs: dict = {
        _K_TEXT: seg.text_target,
        # Engine identity + its synth params (R4/R8): see _engine_inputs.
        **_engine_inputs(cfg),
    }
    if not providers.tts(cfg.tts_engine).native_long_text:
        # Chunking shapes the synthesized audio, so it is part of the clip's
        # identity: changing the budget or either gap (re)synthesizes. The
        # native-long-text engine (gemini) omits these — it doesn't chunk normal
        # segments (KTD5), read as a capability flag rather than an engine name.
        inputs["max_chunk_syllables"] = cfg.tts_chunk_budget
        inputs["chunk_gap_ms"] = cfg.tts_chunk_gap_ms
        inputs["hardwrap_gap_ms"] = cfg.tts_hardwrap_gap_ms
    voice = None
    if cloning:
        inputs["ref_sha"] = ref_sha
        inputs["ref_text"] = ref_text
    else:
        voice = voice_cast.get(seg.speaker) or cfg.preset_voices.default
        inputs["voice"] = voice
    return inputs, voice


def _resolve_segment(seg: Segment, art: Path, inputs: dict, input_hash: str,
                     ledger: SkipLedger) -> bool:
    """Resolve a segment's cache/ledger state. Returns True when it still needs
    synthesis (to-do). An already-valid clip is reused (tts_wav set) and an
    accepted-skip is marked skipped — both return False."""
    if artifacts.is_valid(art, inputs):
        seg.tts_wav = str(art)
        return False
    if not ledger.should_attempt(segment_id(seg), input_hash):
        entry = ledger.entries()[segment_id(seg)]
        seg.skipped, seg.skip_reason = True, entry["reason"]
        log.info("segment %d accepted-skip (%s)", seg.index, entry["reason"])
        return False
    return True


def _produce_clip(seg: Segment, art: Path, inputs: dict, input_hash: str,
                  cfg: Config, ledger: SkipLedger, build, total: int) -> None:
    """Produce one clip's artifact with QA-retry, recording ledger state. The
    common tail of both the per-segment loop and the batch fallback."""
    log.info("TTS segment %d/%d: %s", seg.index + 1, total, seg.text_target[:60])
    try:
        # Infra retries live inside each client's synthesize(); this layer
        # retries QA failures (resampling may simply produce a good clip the
        # second time — no backoff, the failure is sampling variance not load).
        with_retry(
            "tts",
            lambda: artifacts.produce(art, inputs, "tts", build),
            attempts=cfg.retry_attempts, base_delay=0.0, retry_classes=("qa",),
        )
    except StageError as exc:
        seg.skipped, seg.skip_reason = True, exc.code
        log.warning("segment %d failed TTS (%s) — skipped", seg.index, exc)
        ledger.record_failure(segment_id(seg), input_hash, exc.signature,
                              reason=exc.code)  # may raise AbortRun (R5a)
        return
    seg.tts_wav = str(art)
    ledger.record_ok(segment_id(seg), stage="tts")


def _per_segment_clip(client, seg: Segment, art: Path, inputs: dict,
                      input_hash: str, voice: str | None, cfg: Config,
                      ledger: SkipLedger, total: int) -> None:
    """Synthesize one segment with a single (per-segment) call. The Gemini batch
    fallback and the GEMINI_BATCH_SEGMENTS=1 path both route here."""
    def build(tmp: Path) -> None:
        _synthesize_clip(client, seg.text_target, tmp, cfg, voice)
        qa.check_clip(tmp, seg.text_target, cfg)

    _produce_clip(seg, art, inputs, input_hash, cfg, ledger, build, total)


def _group_batches(todo: list[tuple], cfg: Config) -> list[list[tuple]]:
    """Group the ordered to-do items into batches bounded by gemini_batch_segments,
    a running syllable budget (gemini_batch_max_syllables, keeping batch audio
    under the drift threshold), and <= 2 distinct diarized speakers (the
    multi-speaker voice cap). Order is preserved so the split maps back 1:1."""
    batches: list[list[tuple]] = []
    cur: list[tuple] = []
    cur_units = 0
    cur_speakers: set[str] = set()
    count = cfg.language_profile.counter
    # gemini_batch_max_syllables is calibrated in VI syllables; convert it to the
    # active profile's counter unit (characters for CPS profiles) by the rate ratio
    # so a char count is compared against a duration-equivalent ceiling, not a raw
    # syllable number (#15). VI's ratio is exactly 1.0, so VI batching is unchanged.
    max_units = cfg.gemini_batch_max_syllables * (cfg.language_profile.rate / resolve("vi").rate)
    for item in todo:
        seg = item[0]
        units = count(seg.text_target)
        if cur and (
            len(cur) >= cfg.gemini_batch_segments
            or cur_units + units > max_units
            or len(cur_speakers | {seg.speaker}) > 2
        ):
            batches.append(cur)
            cur, cur_units, cur_speakers = [], 0, set()
        cur.append(item)
        cur_units += units
        cur_speakers.add(seg.speaker)
    if cur:
        batches.append(cur)
    return batches


def _finalize_batch(batch: list[tuple], pieces: list[np.ndarray], sr: int,
                    cfg: Config, ledger: SkipLedger) -> None:
    """Finalize a successfully-split batch. QA EVERY clip first (the atomic gate,
    R5): a single failure raises a qa StageError so the caller falls back without
    half-finalizing the batch. Only when all clips pass are the artifacts written
    + recorded; an already-finalized clip is a cache hit on any later fallback."""
    for (seg, *_rest), piece in zip(batch, pieces):
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        try:
            sf.write(tmp, piece, sr)
            qa.check_clip(tmp, seg.text_target, cfg)   # qa StageError -> caller falls back
        finally:
            Path(tmp).unlink(missing_ok=True)
    for (seg, art, inputs, _input_hash, _voice), piece in zip(batch, pieces):
        artifacts.produce(art, inputs, "tts",
                          lambda tmp, piece=piece: sf.write(str(tmp), piece, sr))
        seg.tts_wav = str(art)
        ledger.record_ok(segment_id(seg), stage="tts")


def _process_batch(client, batch: list[tuple], cfg: Config, ledger: SkipLedger,
                   total: int) -> None:
    """Synthesize one batch in a single multi-speaker call, splitting the result
    into per-segment clips. On a split-count mismatch, any clip failing QA, or a
    batch-call failure, fall back to per-segment synthesis — splitting is a cost
    optimization, never a correctness dependency (R5/KTD2). A batch of one always
    takes the per-segment path (no batch call to save)."""
    if len(batch) > 1:
        turns = [(seg.speaker, seg.text_target, voice)
                 for (seg, _art, _inputs, _input_hash, voice) in batch]
        try:
            pieces, sr = client.synthesize_batch(turns)
            _finalize_batch(batch, pieces, sr, cfg, ledger)
            return
        except StageError as exc:
            # SplitError, a clip's qa failure, or the batch call failing all land
            # here (SplitError is a StageError); the per-segment fallback below
            # re-synthesizes each segment with its own retry + ledger handling.
            log.info("Gemini batch %s -> per-segment fallback (%s)",
                     [b[0].index for b in batch], exc)
    for (seg, art, inputs, input_hash, voice) in batch:
        _per_segment_clip(client, seg, art, inputs, input_hash, voice, cfg, ledger, total)


def _tts_gemini_batched(client, segments: list[Segment], cfg: Config,
                        out_dir: Path, ledger: SkipLedger, voice_cast: dict) -> None:
    """Batch the to-do segments into call-minimizing multi-speaker requests
    (R4/R11). Cache/skip resolution is identical to the per-segment loop — only
    already-invalid, attemptable segments are batched, so a partial rerun re-pays
    only for what changed."""
    todo: list[tuple] = []
    for seg in segments:
        if seg.skipped or not seg.text_target:
            continue  # upstream skip (e.g. translate_failed)
        art = out_dir / f"seg_{seg.index:04d}.wav"
        inputs, voice = _seg_inputs(seg, cfg, False, None, None, voice_cast)
        input_hash = artifacts.fingerprint(inputs)
        if not _resolve_segment(seg, art, inputs, input_hash, ledger):
            continue
        todo.append((seg, art, inputs, input_hash, voice))

    for batch in _group_batches(todo, cfg):
        _process_batch(client, batch, cfg, ledger, len(segments))


def _converged_inputs(seg: Segment, cfg: Config) -> dict:
    """Fingerprint key for a segment's converged target text (U6 determinism): the
    ORIGINAL translated text + slot + tolerance/cap. On a re-run the original text
    is unchanged (translate cache), so a converged-text cache hit short-circuits
    the loop BEFORE any non-deterministic re-synthesis or measurement, and the
    re-run reuses the same clip instead of re-billing."""
    return {"text0": seg.text_target, "slot": round(seg.duration, 3),
            "tol": cfg.slot_overflow_tolerance, "cap": cfg.budget_retry_max}


def _record_length_overflow(seg: Segment, cfg: Config, ledger: SkipLedger,
                            clip_dur: float | None = None) -> None:
    """Record (not escalate) a best-effort length_overflow when the final clip
    can't fit even at max atempo (R7/R8) — the clip is KEPT; `fit` speeds it to
    max_tempo and lets the residual spill, mux still produces a timeline. Shared
    by the per-segment measured gate and the Gemini batch path so length surfacing
    is engine-independent (#4). A pre-measured `clip_dur` avoids a redundant
    ffprobe (#13)."""
    slot = seg.duration
    if seg.skipped or not seg.tts_wav or slot <= 0:
        return
    if clip_dur is None:
        clip_dur = ffmpeg.probe_duration(seg.tts_wav)
    if clip_dur > slot * cfg.max_tempo:
        ledger.record_length_overflow(segment_id(seg))
        log.info("segment %d length_overflow: kept best-effort (%.2fs clip in "
                 "%.2fs slot)", seg.index, clip_dur, slot)


def _measured_gate(seg: Segment, cfg: Config, ledger: SkipLedger, resynth,
                   context_block: str = "") -> bool:
    """Authoritative measured-duration gate for one non-VI clip (U6/R6-R8).

    Two thresholds: a clip over `slot * slot_overflow_tolerance` triggers
    ESCALATION (re-translate shorter under `context_block`, re-synthesize,
    re-measure — bounded by budget_retry_max, only when enabled); a final clip
    still over `slot * max_tempo` (more than the atempo residual cleanup in `fit`
    can absorb) is recorded as a best-effort length_overflow (kept, excluded from
    the abort window, R7). `resynth(seg)` re-produces the clip with the current
    text. Returns True when the target text changed (the caller regenerates the
    SRT)."""
    slot = seg.duration
    if slot <= 0 or not seg.tts_wav:
        return False
    changed = False
    clip_dur: float | None = None  # last measured duration; None => must (re)probe
    if cfg.enable_budget_retry:
        from loro.nodes import translate as _translate
        for _ in range(cfg.budget_retry_max):
            clip_dur = ffmpeg.probe_duration(seg.tts_wav)
            if clip_dur <= slot * cfg.slot_overflow_tolerance:
                break  # within the escalation tolerance — stop shrinking
            new_budget = max(3, int(_translate._budget(cfg, seg) * slot / clip_dur))
            shorter = _translate.translate_segment(cfg, seg, context_block, new_budget)
            if not shorter or shorter == seg.text_target:
                break  # model can't shrink further — accept best-effort
            seg.text_target = shorter
            changed = True
            resynth(seg)
            clip_dur = None  # the clip changed; the overflow check must re-probe (#13)
            if seg.skipped or not seg.tts_wav:
                return changed
    _record_length_overflow(seg, cfg, ledger, clip_dur)
    return changed


def _rewrite_target_srt(state: DubState, cfg: Config, segments: list[Segment]) -> None:
    """Regenerate the target SRT (and the translate/segments.json manifest body)
    after the measured loop changed a segment's text, so the soft/burned subtitles
    match the synthesized audio (U6/R6). mux reads the SRT by path and the burn SRT
    from state segments, so an in-place rewrite suffices."""
    workdir = Path(state["workdir"])
    words = state.get("words") or []
    srt_path = workdir / f"transcript.{cfg.target_lang.lower()}.srt"
    srt_path.write_text(
        srt.to_srt_wrapped(segments, words, side="target",
                           max_chars=cfg.srt_max_cue_chars, max_dur=cfg.srt_max_cue_dur),
        encoding="utf-8")
    manifest = workdir / "translate" / "segments.json"
    if manifest.exists():
        manifest.write_text(
            json.dumps({"segments": [s.to_dict() for s in segments]},
                       ensure_ascii=False, indent=1), encoding="utf-8")


def tts(state: DubState, cfg: Config) -> DubState:
    segments = state["segments"]
    workdir = Path(state["workdir"])
    out_dir = workdir / "tts"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = SkipLedger.from_cfg(workdir, cfg)

    # The cloning-only reads (reference hash + client construction) run only for
    # the cloning engines; a preset run carries no ref_audio/ref_text in state,
    # so gating them here keeps it from KeyError-ing (KTD1/U4).
    provider = providers.tts(cfg.tts_engine)
    cloning = cfg.tts_uses_cloning
    ref_sha = artifacts.cached_file_sha256(state["ref_audio"]) if cloning else None
    ref_text = state.get("ref_text") if cloning else None
    voice_cast = state.get("voice_cast", {})
    client_cm = (_tts_client(cfg, Path(state["ref_audio"]), state["ref_text"])
                 if cloning else _tts_client(cfg))
    # The measured-loop re-translation reuses the global video context (the domain
    # framing that keeps loanwords/terms consistent); the per-batch layered context
    # is not reloaded into the TTS node (#6).
    video_context = state.get("video_context", "")
    gate_context = (f"{cfg.language_profile.context_labels.video_context}: {video_context}"
                    if video_context else "")

    with client_cm as client:
        # A batching engine (gemini) with batching enabled routes through the
        # batch->split->fallback path (the already-entered client is reused —
        # never a second client). Every other engine, and GEMINI_BATCH_SEGMENTS=1,
        # take the per-segment loop — selected by capability, not name. Escalation
        # (re-translate+resynth) only the per-segment path supports, so a batching
        # engine falls through to it when budget retry is on, so the measured loop
        # can shrink over-slot clips (#4).
        batched = provider.batches and cfg.gemini_batch_segments > 1
        if batched and not cfg.enable_budget_retry:
            _tts_gemini_batched(client, segments, cfg, out_dir, ledger, voice_cast)
            # The batch path skips the per-segment measured gate, so still run the
            # length_overflow recording pass for non-VI targets here (R7/R8) —
            # escalation is N/A without budget retry (#4).
            if cfg.measured_duration_active:
                for seg in segments:
                    _record_length_overflow(seg, cfg, ledger)
            return {"segments": segments}

        srt_changed = False
        for seg in segments:
            if seg.skipped or not seg.text_target:
                continue  # upstream skip (e.g. translate_failed)

            # The measured-duration gate is on for non-VI (CPS) profiles only, so
            # the VI path below is exactly the legacy synthesize-once loop (R19).
            gate_on = cfg.measured_duration_active
            escalate = gate_on and cfg.enable_budget_retry
            conv_art = out_dir / f"converged_{seg.index:04d}.json"
            conv_inputs = _converged_inputs(seg, cfg) if escalate else None
            conv_hit = escalate and artifacts.is_valid(conv_art, conv_inputs)
            if conv_hit:
                # Cache-first determinism (U6): reuse the converged text and skip
                # the non-deterministic measure/re-translate loop entirely.
                seg.text_target = json.loads(conv_art.read_text(encoding="utf-8"))["text"]

            def _produce(s: Segment = seg) -> None:
                art = out_dir / f"seg_{s.index:04d}.wav"
                inputs, voice = _seg_inputs(s, cfg, cloning, ref_sha, ref_text, voice_cast)
                input_hash = artifacts.fingerprint(inputs)
                if not _resolve_segment(s, art, inputs, input_hash, ledger):
                    return  # cache hit / accepted-skip
                _per_segment_clip(client, s, art, inputs, input_hash, voice, cfg,
                                  ledger, len(segments))

            _produce()
            if gate_on and not conv_hit and not seg.skipped and seg.tts_wav:
                if _measured_gate(seg, cfg, ledger, _produce, gate_context):
                    srt_changed = True
            if escalate and not seg.skipped:
                # Persist the converged text so the next run takes the cache-first
                # path above (deterministic, no re-bill).
                artifacts.produce_json(conv_art, conv_inputs, "tts",
                                       lambda t=seg.text_target: {"text": t})

        if srt_changed:
            _rewrite_target_srt(state, cfg, segments)

    return {"segments": segments}
