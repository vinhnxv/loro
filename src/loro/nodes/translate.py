"""Length-aware, profile-driven source->target dubbing translation via the LLM.

Direct source->target (no English pivot, U8): the system prompt, context labels,
output key, and per-language length budget are all sourced from the active
LanguageProfile (`cfg.language_profile`) — Vietnamese resolves the legacy
syllable model + the exact legacy prompt (so the VI/EN default stays
byte-identical), every other language uses a characters-per-second budget. When
source and target resolve to the same language the LLM is skipped entirely
(target text = source text) for a voice-replacement run (R11).

Segments are translated in deterministic index-range batches (the model sees
surrounding dialogue) with one durable artifact per batch. When a batch must
be recomputed, peers whose (effective text, length budget) did not change
keep their previous translation verbatim — TTS cache stability beats batch
coherence at the margins. User fixes live in `overrides.json` (user-owned,
pipeline only reads): they are applied after translation and survive any
retranslation (R16).

A batch artifact is only finalized (sidecar written) once every segment in it
has a translation; batches containing failed segments stay invalid so the
next run retries the failures while reusing the successful peers' text from
the artifact body (R5b semantics, reason `translate_failed`, R18).
"""

import json
import logging
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.harness.ledger import SkipLedger
from loro.harness.preflight import out_of_range_override_keys
from loro.harness.retry import StageError, classify
from loro.services import llm
from loro.state import DubState, Segment, segment_id
from loro.utils import srt

log = logging.getLogger("loro.translate")


def _words_sha(words: list[dict]) -> str:
    """Fingerprint the EN word-timing stream that now anchors VI cue times (U1).
    Mirrors sentence_seg._words_sha so a changed word source invalidates the
    translate manifest and regenerates srt_target with the new anchored timing —
    without it a translated-text cache hit would serve the old uniform SRT."""
    return artifacts.fingerprint(
        {"w": [[round(w["start"], 3), round(w["end"], 3), w["word"]] for w in words]}
    )

# The translation system prompt is now sourced from the active LanguageProfile
# (cfg.language_profile.system_prompt) instead of a hardcoded Vietnamese constant
# (U8, R10). The VI profile carries the exact legacy prompt, so the VI translate
# fingerprint — which folds the system prompt — stays byte-identical (R19).


# One symbol feeds the model call AND both fingerprint dicts, so tuning it
# automatically invalidates the right artifacts
TEMPERATURE = 0.3

# FROZEN artifact/fingerprint key names (KTD6/R19). These dict-literal strings are
# part of the on-disk cache identity and are deliberately the legacy
# Vietnamese-era names — they are NOT the current source/target language. Renaming
# any of them silently busts every existing translate cache (re-bill) and MUST
# update the fingerprint-parity goldens in lockstep. Named here so the freeze is a
# single, searchable contract instead of scattered bare literals (#8).
_K_SRC = "en"         # source text in the artifact body + _seg_hash
_K_TGT = "vi"         # target text in the artifact body
_K_BUDGET = "budget"  # length budget in the artifact body + _seg_hash
_K_PREV = "prev_vi"   # prev-target window in the prompt fingerprint
# The canonical default tags the fingerprint guard treats as the byte-identical
# baseline: a run targeting these is NOT folded into the hash (R19).
_DEFAULT_SRC = "en"
_DEFAULT_TGT = "vi"


def _base_tag(tag: str) -> str:
    """The base (primary) subtag, lowercased: 'EN'/'en-US' -> 'en'. Equivalent
    spellings of one language collapse here, so the cache key and the
    source==target check treat them identically (R19/R11, #1)."""
    return tag.split("-")[0].strip().lower()


def _same_language(a: str, b: str) -> bool:
    """True when two BCP-47 tags share a base language (en == en-US), so the
    source==target voice-replacement path triggers regardless of region (R11)."""
    return _base_tag(a) == _base_tag(b)


def _budget(cfg: Config, seg: Segment) -> int:
    # Per-language length budget: duration x the profile's rate, in the profile's
    # own unit (VI: syllables at 4.3/s, byte-identical to the legacy model;
    # CPS profiles: characters at the profile's CPS). Feeds the translate
    # fingerprint, so the VI/EN default stays frozen (U5, R5/R19).
    return max(3, int(seg.duration * cfg.language_profile.rate))


def _load_layers(ctx_art: Path) -> dict | None:
    """Read a batch's context artifact (U4/U5). None when absent or torn — the
    caller falls back to the bare global video_context, exactly as before (R43)."""
    if not ctx_art.exists():
        return None
    try:
        return json.loads(ctx_art.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _context_block(profile, layers: dict, prev_target: list[str]) -> str:
    """Assemble the layered context (R40-42) into the translation prompt: global
    video context + the shot's visual + the running summary + the source neighbour
    lines + the already-translated target text of the previous batches (a
    consistency reference, not to be retranslated). Labels come from the profile
    (R10), so the node owns assembly and the profile owns the language framing."""
    lbl = profile.context_labels
    parts: list[str] = []
    if layers.get("video_context"):
        parts.append(f"{lbl.video_context}: {layers['video_context']}")
    shots = layers.get("shot_visuals") or []
    if shots:
        parts.append(f"{lbl.shot_visuals}: " + " ".join(shots))
    if layers.get("summary"):
        parts.append(f"{lbl.summary}: {layers['summary']}")
    before = layers.get("neighbors_before") or []
    if before:
        parts.append(f"{lbl.neighbors_before}: " + " ".join(before))
    after = layers.get("neighbors_after") or []
    if after:
        parts.append(f"{lbl.neighbors_after}: " + " ".join(after))
    if prev_target:
        parts.append(f"{lbl.prev_translations}: " + " ".join(prev_target))
    return "\n".join(parts)


def _translate_lines(cfg: Config, profile, lines: list[dict],
                     context_block: str) -> dict[int, str]:
    # context_block is the fully assembled layered context (already labelled);
    # an empty block is the degraded path (R43). The system prompt, the per-line
    # directive, and the output key are all profile-sourced (R10).
    user = (
        (context_block + "\n\n" if context_block else "")
        + profile.context_labels.instruction + "\n\n"
        + json.dumps(lines, ensure_ascii=False)
    )
    reply = llm.chat(
        cfg,
        [{"role": "system", "content": profile.system_prompt},
         {"role": "user", "content": user}],
        temperature=TEMPERATURE,
        # 8192 (not 4096): a thinking model served via Ollama Cloud (qwen3.5)
        # burns hundreds-thousands of invisible completion tokens per call on
        # top of the actual JSON; 4096 truncated the 12-segment batch (finish=
        # length -> empty_response -> abort window). 8192 covers a full batch's
        # hidden burn + answer. A non-thinking model (Gemma) stops early and
        # never uses the headroom, so this is free locally.
        max_tokens=8192,
        stage="translate",
        role=cfg.llm_role("translate"),
        enable_thinking=False,
    )
    items = llm.extract_json(reply)
    key = profile.output_key
    # extract_json can return a list OR a dict (a model that wraps the array in an
    # object), and a list item can be the wrong type or mis-keyed. That is a model
    # OUTPUT-shape failure (CONTENT), not a programmer error — classify it as a
    # content StageError so the batch/per-segment handlers record it as a skip,
    # rather than the U8 narrowing letting a malformed-but-parseable reply crash
    # the run. A genuine bug in OUR code stays outside this guard and propagates.
    if not isinstance(items, list):
        raise StageError("translate", "content", "bad_shape",
                         f"expected a JSON array, got {type(items).__name__}")
    try:
        result = {int(item["i"]): str(item[key]).strip() for item in items}
    except (KeyError, TypeError, ValueError) as exc:
        raise StageError("translate", "content", "bad_shape",
                         f"malformed translation item: {exc}") from exc
    # A truncated reply (e.g. a leaked <think> ate max_tokens, KTD6) silently
    # drops indices. Surface it as a content error so the missing segments are
    # retried/strike-counted instead of falling quietly to translate_failed (U6).
    missing = [line["i"] for line in lines if line["i"] not in result]
    if missing:
        raise StageError("translate", "content", "incomplete_array",
                         f"model omitted indices {missing}")
    return result


def _line(seg: Segment, budget: int) -> dict:
    """One model-facing translate line. Neutral keys ("src"/"budget") the
    profile's directive explains; the artifact body + fingerprint keep their
    frozen "en"/"budget" keys separately (KTD6)."""
    return {"i": seg.index, "src": seg.text_src, "budget": budget}


def load_overrides(workdir: Path) -> dict[str, str]:
    """`overrides.json` maps segment ids ("seg_0012") to user-fixed text_target.
    Preflight validates the shape; empty-string values are ignored here so a
    blank entry can never silently mute a segment."""
    path = workdir / "overrides.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    overrides = {}
    for key, value in data.items():
        if str(value).strip():
            overrides[str(key)] = str(value)
        else:
            log.warning("empty override for %s ignored", key)
    return overrides


def _seg_hash(seg: Segment, budget: int) -> str:
    return artifacts.fingerprint({_K_SRC: seg.text_src, _K_BUDGET: budget})


def _compute_batch(cfg: Config, profile, art: Path, batch: list[Segment],
                   context_block: str, ledger: SkipLedger, prompt_sha: str) -> dict:
    """Translate one batch with peer reuse from the previous artifact body.
    Reuse only applies while the prompt-level inputs (system, layered context,
    model, prev-target — all folded into prompt_sha) are unchanged — a new prompt
    must retranslate everything. The artifact body keeps the frozen _K_SRC/_K_TGT/
    _K_BUDGET ("en"/"vi"/"budget") keys (KTD6) regardless of target language."""
    try:
        prev = json.loads(art.read_text(encoding="utf-8"))
        prev_items = prev.get("items", {}) if prev.get("prompt_sha") == prompt_sha else {}
    except (OSError, json.JSONDecodeError):
        prev_items = {}

    items: dict[str, dict] = {}
    need: list[Segment] = []
    for seg in batch:
        budget = _budget(cfg, seg)
        prev = prev_items.get(str(seg.index))
        if prev and prev.get(_K_TGT) and prev[_K_SRC] == seg.text_src and prev[_K_BUDGET] == budget:
            items[str(seg.index)] = {_K_SRC: seg.text_src, _K_BUDGET: budget, _K_TGT: prev[_K_TGT]}
        elif ledger.should_attempt(segment_id(seg), _seg_hash(seg, budget)):
            need.append(seg)
            items[str(seg.index)] = {_K_SRC: seg.text_src, _K_BUDGET: budget, _K_TGT: None}
        else:  # accepted skip: don't retry until its input changes
            items[str(seg.index)] = {_K_SRC: seg.text_src, _K_BUDGET: budget, _K_TGT: None}

    translated: dict[int, str] = {}
    infra_signature = None
    if need:
        lines = [_line(s, _budget(cfg, s)) for s in need]
        try:
            translated = _translate_lines(cfg, profile, lines, context_block)
        except StageError as exc:
            if exc.error_class == "infra":
                # Server down after its own retries: per-segment fallback
                # would just repeat the timeout once per segment. Skip them
                # directly — the strikes drive the abort window (R5a).
                infra_signature = exc.signature
                log.warning("batch translation failed with infra error (%s) — "
                            "skipping per-segment fallback", exc)
            else:
                log.exception("batch translation failed, retrying segment by segment")
        except ValueError:
            # A malformed / parse-failed model reply (extract_json) is a CONTENT
            # failure: fall through to the per-segment retry. A programmer error
            # (KeyError/TypeError/…) is deliberately NOT caught here — it propagates
            # with its stack instead of falling silently through to the per-segment
            # retry and being downgraded to a skip (B5/R8).
            log.exception("batch translation failed, retrying segment by segment")

    for seg in need:
        budget = _budget(cfg, seg)
        text_target = translated.get(seg.index, "")
        if not text_target and infra_signature is not None:
            ledger.record_failure(segment_id(seg), _seg_hash(seg, budget),
                                  infra_signature, reason="translate_failed")
            continue
        if not text_target:
            try:
                single = _translate_lines(cfg, profile, [_line(seg, budget)],
                                          context_block)
                text_target = single.get(seg.index, "")
                if not text_target:
                    raise ValueError(f"no translation returned for segment {seg.index}")
            except (StageError, ValueError) as exc:
                # Only the EXPECTED content failures become a per-segment skip: an
                # LLM StageError, or the explicit "no translation returned" / parse
                # ValueError. A programmer error (KeyError, TypeError, …) propagates
                # with its stack instead of vanishing as a translate_failed skip
                # (B5/R8).
                signature = exc.signature if isinstance(exc, StageError) \
                    else ("translate", *classify(exc))
                log.warning("segment %d translate failed (%s) — skipped", seg.index, exc)
                ledger.record_failure(segment_id(seg), _seg_hash(seg, budget),
                                      signature, reason="translate_failed")
                continue
        items[str(seg.index)][_K_TGT] = text_target
        ledger.record_ok(segment_id(seg), stage="translate")
    return {"range": [batch[0].index, batch[-1].index], "prompt_sha": prompt_sha,
            "items": items}


def translate_segment(cfg: Config, seg: Segment, context_block: str,
                      budget: int) -> str:
    """Re-translate ONE segment under a (tighter) budget, with the caller-supplied
    `context_block` (the global video context during the U6 measured loop, #6).
    The per-segment entrypoint the measured-duration loop calls to shrink an
    over-slot segment; returns the new target text, or "" on ANY failure — so an
    LLM/network error never propagates out to abort the (partially billed) TTS run;
    the caller keeps its best clip and marks length_overflow (#2)."""
    try:
        out = _translate_lines(cfg, cfg.language_profile, [_line(seg, budget)], context_block)
    except Exception as exc:
        log.warning("re-translation of segment %d failed (%s) — keeping current text",
                    seg.index, exc)
        return ""
    return out.get(seg.index, "")


def translate(state: DubState, cfg: Config) -> DubState:
    segments = state["segments"]
    video_context = state.get("video_context", "")
    profile = cfg.language_profile
    source_lang = state.get("source_lang", cfg.source_lang)
    workdir = Path(state["workdir"])
    tdir = workdir / "translate"
    cdir = workdir / "context"
    overrides = load_overrides(workdir)
    # Segments are sentences now (sentence_seg); a seg_NNNN override that
    # survived re-segmentation may point past the new count. Drop+log such keys
    # rather than silently ignoring a user fix or applying it to a stale index
    # (U5). The "different sentence at the same index" case is undetectable and
    # is called out in the README upgrade note.
    for key in out_of_range_override_keys(overrides, len(segments)):
        log.warning("override %s is outside the current segment range (0..%d) — ignored; "
                    "recheck overrides.json after re-segmentation",
                    key, len(segments) - 1)
        overrides.pop(key, None)
    ledger = SkipLedger.from_cfg(workdir, cfg)
    k = cfg.context_neighbors

    # R11: when source and target are the same language, skip the LLM entirely —
    # the target text is the source text — and let TTS/fit/mux still run (voice
    # replacement). Overrides still apply below.
    same_language = _same_language(source_lang, cfg.target_lang)
    if same_language:
        log.warning("source and target language both resolve to %r — skipping LLM "
                    "translation (target text = source text); running voice "
                    "replacement only (R11)", source_lang)

    batch_files = []
    all_items: dict[str, dict] = {}
    prev_target_all: list[str] = []  # target text of prior batches, in order (prev-target source)
    for bi, offset in enumerate(
            [] if same_language else range(0, len(segments), cfg.translate_batch)):
        batch = segments[offset : offset + cfg.translate_batch]
        art = tdir / f"batch_{bi:04d}.json"
        ctx_art = cdir / f"batch_{bi:04d}.json"
        layers = _load_layers(ctx_art)

        # prompt_fp is the prompt-level fingerprint (the peer-reuse gate);
        # inputs is that plus the per-segment lines. Deriving inputs from
        # prompt_fp keeps the layered-context keys in lockstep across both.
        # The "model" key keeps its name; only its value source is
        # llm_model_translate (defaults to llm_model) so the bare-fallback
        # fingerprint stays byte-identical and never busts existing cache —
        # setting LLM_MODEL_TRANSLATE invalidates only the translation (R37, KTD1).
        # The system prompt is profile-sourced (U8): VI's is the legacy string, so
        # this key is byte-identical for VI. source_lang/target_lang are folded in
        # only when their BASE tag differs from the en/vi default — folding the
        # base tag (not the raw string) means equivalent spellings (VI, vi-VN,
        # en-US) share the byte-identical VI/EN cache instead of silently busting
        # it and re-billing (R19, #1); a genuinely different source/target still
        # busts the cache deliberately.
        prompt_fp = {"context": video_context, "system": profile.system_prompt,
                     "model": cfg.llm_model_translate, "temperature": TEMPERATURE}
        src_base, tgt_base = _base_tag(source_lang), _base_tag(cfg.target_lang)
        if src_base != _DEFAULT_SRC:
            prompt_fp["source_lang"] = src_base
        if tgt_base != _DEFAULT_TGT:
            prompt_fp["target_lang"] = tgt_base
        if layers is not None:
            prev_target = prev_target_all[-k:]
            context_block = _context_block(profile, layers, prev_target)
            # The layered context + prev-target must enter the fingerprint AND the
            # peer-reuse gate, or a peer with unchanged (en, budget) would keep its
            # text translated under stale context/summary/prev-target, and a resumed
            # run would diverge from a fresh one (R1/R2/U6).
            prompt_fp["context_sha"] = artifacts.cached_file_sha256(ctx_art)
            prompt_fp[_K_PREV] = prev_target
        else:
            # R43: no context artifact -> bare global video_context, as before
            context_block = (f"{profile.context_labels.video_context}: {video_context}"
                             if video_context else "")
        prompt_sha = artifacts.fingerprint(prompt_fp)
        inputs = {**prompt_fp,
                  "lines": [[s.index, s.text_src, _budget(cfg, s)] for s in batch]}

        if artifacts.is_valid(art, inputs):
            data = json.loads(art.read_text(encoding="utf-8"))
        else:
            log.info("translating batch %d (segments %d-%d)", bi,
                     batch[0].index, batch[-1].index)
            data = _compute_batch(cfg, profile, art, batch, context_block, ledger, prompt_sha)
            complete = all(item[_K_TGT] for item in data["items"].values())
            if complete:
                artifacts.produce(
                    art, inputs, "translate",
                    lambda tmp, data=data: tmp.write_text(
                        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"),
                )
            else:
                # Keep the body for peer reuse, but never finalize a batch
                # with failed segments: the next run must retry them
                artifacts.write_unfinalized(
                    art, json.dumps(data, ensure_ascii=False, indent=1).encode("utf-8"))
        all_items.update(data["items"])
        batch_files.append(art)
        # Accumulate this batch's target text for the next batch's prev-target window.
        for seg in batch:
            txt = data["items"].get(str(seg.index), {}).get(_K_TGT)
            if txt:
                prev_target_all.append(txt)

    for seg in segments:
        item = all_items.get(str(seg.index), {})
        override = overrides.get(segment_id(seg))
        if override is not None:
            text_target = override
        elif same_language:
            text_target = seg.text_src          # R11: identity "translation"
        else:
            text_target = item.get(_K_TGT) or ""
        seg.text_target = text_target
        if not text_target:
            seg.skipped = True
            seg.skip_reason = "translate_failed"

    words = state.get("words") or []
    manifest_inputs = {
        "batch_hashes": [artifacts.cached_file_sha256(f) for f in batch_files],
        "overrides": overrides,
        # VI cue times now anchor to the EN word curve (U1), so the word source
        # is part of this manifest: changing it must regenerate srt_target.
        "words_sha": _words_sha(words),
    }
    artifacts.produce_json(
        tdir / "segments.json", manifest_inputs, "translate",
        lambda: {"segments": [s.to_dict() for s in segments]},
    )

    # Voice replacement (R11): the target text IS the source text, and the target
    # SRT path collides with — and would clobber — the word-timed source SRT
    # (transcript.<tag>.srt, written by asr/crosscheck). Reuse that source SRT as
    # the target sidecar instead of overwriting it (#7).
    if same_language:
        src_srt = workdir / f"transcript.{source_lang.lower()}.srt"
        if src_srt.exists():
            log.info("source==target: reusing source SRT %s as target sidecar (R11)", src_srt)
            return {"segments": segments, "srt_target": str(src_srt)}

    # Target SRT filename derives from the target locale (U10); the VI default
    # keeps transcript.vi.srt byte-identical.
    tgt_tag = cfg.target_lang.lower()
    srt_target = workdir / f"transcript.{tgt_tag}.srt"
    srt_target.write_text(
        srt.to_srt_wrapped(segments, words, side="target",
                           max_chars=cfg.srt_target_max_cue_chars, max_dur=cfg.srt_max_cue_dur),
        encoding="utf-8")
    log.info("target SRT -> %s", srt_target)
    return {"segments": segments, "srt_target": str(srt_target)}
