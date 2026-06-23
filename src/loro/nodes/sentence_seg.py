"""Turn the ASR word stream into natural reading units — the dub backbone.

Nemotron's acoustic segments cut sentences in half or mash a 405-second
monologue into one blob; the dub inherits those units and drifts. This node
replaces that backbone with *clause-level reading units* (a whole sentence when
short, or a coherent clause of a long run-on): it pre-splits on existing
punctuation, sends any long under-punctuated span to Gemma (via `llm.chat`) for
boundaries, aligns the returned units back to the word timestamps, and falls
back to a pause split when Gemma is unavailable or its output doesn't reproduce
the word stream (KTD2/KTD5). Conversational speech is mostly run-on with almost
no sentence punctuation, so asking Gemma for whole "sentences" yields minutes-long
units that the pause-split then chops mid-clause — hence the prompt targets
breath-sized clauses (~5-12s) broken at conjunctions/phrase boundaries. The pure
logic lives in `loro.utils.sentences`; this node wires Gemma in and persists the
artifact.

The artifact `sentence_seg/segments.json` is fingerprinted over the word-stream
hash, raw fallback segments, model, prompt and thresholds, so re-segmentation
cascades correctly into the index-keyed caches downstream. A run that had to
degrade to a pause split is written *unfinalized* (durable body, no valid
sidecar) so the next run retries the LLM — the same R5b/R32 pattern translate
and crosscheck use for transient model outages.
"""

import json
import logging
from collections import Counter
from pathlib import Path

from loro.config import Config
from loro.harness import artifacts
from loro.services import llm
from loro.state import DubState, Segment
from loro.utils import sentences

log = logging.getLogger("loro.sentence_seg")

PROMPT = (
    "Segment the following lightly-punctuated English transcript (from automatic "
    "speech recognition) into natural reading units for dubbing and subtitles. "
    "Each unit should read as one coherent clause or sentence and last roughly 5 "
    "to 12 seconds of speech. Keep a complete sentence whole when it is short "
    "enough; split a long run-on sentence at a major clause boundary — before a "
    "conjunction such as and, so, but, then, or because, or at a clear phrase "
    "break. Do not emit tiny fragments: keep a short clause joined to a neighbour. "
    "Preserve every word exactly and in order — do not add, remove, paraphrase, "
    "correct, or re-spell any word. Reply with ONLY a JSON array of strings, one "
    "unit per element, and nothing else."
)


def _words_sha(words: list[dict]) -> str:
    return artifacts.fingerprint(
        {"w": [[round(w["start"], 3), round(w["end"], 3), w["word"]] for w in words]}
    )


def _segment_speaker(words: list[dict], start: float, end: float) -> str:
    """Majority diarization label among the words in [start, end) (KTD7); ties
    broken by the earliest such word. Returns '' when no word carries a speaker
    (the local engine and single-speaker / diarization-off audio)."""
    inside = [w.get("speaker") for w in words
              if start - 1e-6 <= w["start"] < end and w.get("speaker")]
    if not inside:
        return ""
    counts = Counter(inside)
    top = max(counts.values())
    for spk in inside:  # inside is in word order, so the earliest wins ties
        if counts[spk] == top:
            return spk
    return inside[0]


def _make_llm_fn(cfg: Config):
    """Wrap Gemma as the boundary oracle. Returns a list of sentence strings;
    raises (caught upstream into the pause fallback) on any model failure."""
    def llm_fn(text: str) -> list[str]:
        # Size the budget to the window so a long span is not truncated; the
        # spike used ~6k tokens for ~1051 words (KTD2 Open Question).
        max_tokens = min(8192, max(1024, len(text.split()) * 8))
        reply = llm.chat(
            cfg,
            [{"role": "user", "content": f"{PROMPT}\n\n{text}"}],
            temperature=0.0,
            max_tokens=max_tokens,
            stage="sentence_seg",
            role=cfg.llm_role("seg"),
            enable_thinking=False,
        )
        items = llm.extract_json(reply)
        if not isinstance(items, list):
            raise ValueError("sentence_seg: model did not return a JSON array")
        return [str(s) for s in items if str(s).strip()]

    return llm_fn


def sentence_seg(state: DubState, cfg: Config) -> DubState:
    workdir = Path(state["workdir"])
    sdir = workdir / "sentence_seg"
    sdir.mkdir(parents=True, exist_ok=True)

    words = state.get("words") or []
    raw_segments = state["segments"]  # raw acoustic units from asr (the fallback)
    raw_dicts = [{"start": s.start, "end": s.end, "text": s.text_src} for s in raw_segments]

    inputs = {
        "words_sha": _words_sha(words),
        "raw_segments": [[round(s.start, 3), round(s.end, 3), s.text_src] for s in raw_segments],
        "model": cfg.llm_model_seg,
        "prompt": PROMPT,
        "max_segment_duration": cfg.max_segment_duration,
        "max_unpunct_dur": cfg.sentence_seg_max_unpunct_dur,
        "min_punct_density": cfg.sentence_seg_min_punct_density,
        "min_pause": cfg.segment_split_min_pause,
        "word_window": cfg.sentence_seg_word_window,
    }
    art = sdir / "segments.json"

    if artifacts.is_valid(art, inputs):
        manifest = json.loads(art.read_text(encoding="utf-8"))
    else:
        seg_dicts, degraded = sentences.segment_into_sentences(
            words,
            raw_segments=raw_dicts,
            llm_fn=_make_llm_fn(cfg),
            max_dur=cfg.max_segment_duration,
            min_pause=cfg.segment_split_min_pause,
            max_unpunct_dur=cfg.sentence_seg_max_unpunct_dur,
            min_punct_density=cfg.sentence_seg_min_punct_density,
            word_window=cfg.sentence_seg_word_window,
        )
        # Carry the per-sentence speaker onto each unit (R3/KTD7). speaker is
        # derived from the word stream, not the inputs fingerprint, so a clean
        # local manifest stays valid and reconstructs with "".
        for d in seg_dicts:
            d["speaker"] = _segment_speaker(words, d["start"], d["end"])
        manifest = {"segments": seg_dicts, "degraded": degraded}
        body = json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8")
        if degraded:
            # LLM down or misaligned: keep the pause-split body but leave the
            # artifact invalid so the next run retries Gemma (R5b/R32).
            log.warning("sentence_seg degraded to pause-split (LLM unavailable or "
                        "misaligned) — next run will retry the LLM")
            artifacts.write_unfinalized(art, body)
        else:
            artifacts.produce(art, inputs, "sentence_seg",
                              lambda tmp: tmp.write_bytes(body))

    segments = [
        Segment(index=i, start=s["start"], end=s["end"], text_src=str(s["text"]).strip(),
                speaker=str(s.get("speaker", "")))
        for i, s in enumerate(d for d in manifest["segments"] if str(d["text"]).strip())
    ]
    if not segments:
        raise RuntimeError("sentence segmentation produced no segments — is there speech "
                           "in the video?")
    log.info("%d sentence segment(s)%s", len(segments),
             " (degraded: pause-split)" if manifest.get("degraded") else "")
    return {"segments": segments}
