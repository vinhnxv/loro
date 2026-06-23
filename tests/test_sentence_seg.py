"""Node-level tests for sentence_seg (the dub backbone).

test_sentences.py pins the pure segmentation logic; these pin the *node*: the
Gemma wiring, the durable artifact, the degraded -> write_unfinalized -> retry
loop, and the no-words fallback — paths the full-graph integration test cannot
reach because its stub ASR worker emits no word timestamps.
"""

import json

import pytest

from loro.config import Config
from loro.harness import artifacts
from loro.harness.retry import StageError
from loro.nodes import sentence_seg as ss
from loro.state import Segment


def _unpunctuated_words(n=100, dur=0.3, gap=0.1, start=0.0):
    """A long span with no sentence punctuation, over sentence_seg_max_unpunct_dur
    so the node routes it through the LLM."""
    words, t = [], start
    for i in range(n):
        words.append({"start": round(t, 3), "end": round(t + dur, 3), "word": f"w{i}"})
        t += dur + gap
    return words


def _raw_segment(words):
    return Segment(index=0, start=words[0]["start"], end=words[-1]["end"],
                   text_src=" ".join(w["word"] for w in words))


def _regroup_reply(cfg, messages, **kw):
    """Stub Gemma: re-segment the prompt's text into 8-word 'sentences',
    preserving every word, returned as a JSON array (what extract_json parses).
    Signature matches llm.chat(cfg, messages, **kw)."""
    text = messages[-1]["content"].split("\n\n", 1)[1]
    toks = text.split()
    sents = [" ".join(toks[i:i + 8]) for i in range(0, len(toks), 8)]
    return json.dumps(sents, ensure_ascii=False)


def test_gemma_path_segments_finalizes_and_caches(tmp_path, monkeypatch):
    calls = []

    def chat(cfg, messages, **kw):
        calls.append(kw.get("stage"))
        return _regroup_reply(cfg, messages, **kw)

    monkeypatch.setattr(ss.llm, "chat", chat)
    words = _unpunctuated_words()
    state = {"workdir": str(tmp_path), "words": words, "segments": [_raw_segment(words)]}

    result = ss.sentence_seg(state, Config())
    segs = result["segments"]
    assert len(segs) > 1                       # the long span became many sentences
    assert all(s.duration <= 18.0 + 1e-9 for s in segs)
    assert " ".join(s.text_src for s in segs).split() == [w["word"] for w in words]
    art = tmp_path / "sentence_seg" / "segments.json"
    manifest = json.loads(art.read_text())
    assert manifest["degraded"] is False
    assert artifacts.read_meta(art) is not None  # finalized (valid sidecar)
    n_calls = len(calls)

    # Rerun with identical inputs -> cached, no further model calls.
    ss.sentence_seg(state, Config())
    assert len(calls) == n_calls


def test_gemma_down_degrades_then_retries_next_run(tmp_path, monkeypatch):
    words = _unpunctuated_words()
    state = {"workdir": str(tmp_path), "words": words, "segments": [_raw_segment(words)]}

    def down(cfg, messages, **kw):
        raise StageError("sentence_seg", "infra", "down", "oMLX unreachable")

    monkeypatch.setattr(ss.llm, "chat", down)
    result = ss.sentence_seg(state, Config())  # must NOT raise
    segs = result["segments"]
    assert len(segs) > 1                                  # pause-split fallback ran
    assert all(s.duration <= 18.0 + 1e-9 for s in segs)
    art = tmp_path / "sentence_seg" / "segments.json"
    assert json.loads(art.read_text())["degraded"] is True
    assert artifacts.read_meta(art) is None              # unfinalized: no sidecar

    # Next run with Gemma back: the unfinalized artifact is recomputed (retried).
    monkeypatch.setattr(ss.llm, "chat", _regroup_reply)
    ss.sentence_seg(state, Config())
    assert json.loads(art.read_text())["degraded"] is False
    assert artifacts.read_meta(art) is not None          # now finalized


def test_no_words_falls_back_to_raw_segments_and_caches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ss.llm, "chat",
                        lambda *a, **k: calls.append(1) or "[]")
    raw = [Segment(index=0, start=0.0, end=4.0, text_src="câu một."),
           Segment(index=1, start=4.0, end=8.0, text_src="câu hai.")]
    state = {"workdir": str(tmp_path), "words": [], "segments": raw}

    result = ss.sentence_seg(state, Config())
    segs = result["segments"]
    assert [s.text_src for s in segs] == ["câu một.", "câu hai."]
    assert calls == []                                   # no LLM call without words
    assert json.loads((tmp_path / "sentence_seg" / "segments.json").read_text())["degraded"] is False

    ss.sentence_seg(state, Config())                     # cached, still no LLM call
    assert calls == []


def test_empty_stream_raises_no_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(ss.llm, "chat", lambda *a, **k: "[]")
    state = {"workdir": str(tmp_path), "words": [], "segments": []}
    with pytest.raises(RuntimeError):
        ss.sentence_seg(state, Config())


def test_sentence_seg_model_defaults_to_gemma_kwarg():
    # Mirrors the llm_model_translate fallback: an empty config uses Gemma, but the
    # llm_model kwarg (not only LLM_MODEL_SEG) is honored.
    assert Config(llm_model="X").llm_model_seg == "X"
    assert Config(llm_model="X", llm_model_seg="Y").llm_model_seg == "Y"


# --- speaker capture onto segments (U4/R3) ---


def _spk_words(specs):
    """specs: [(word, speaker)]. Auto-timestamped at 0.5s steps so a short,
    punctuated stream never routes through the LLM (so these stay offline)."""
    words, t = [], 0.0
    for word, speaker in specs:
        w = {"start": round(t, 3), "end": round(t + 0.4, 3), "word": word}
        if speaker is not None:
            w["speaker"] = speaker
        words.append(w)
        t += 0.5
    return words


def _state(tmp_path, words):
    return {"workdir": str(tmp_path), "words": words,
            "segments": [Segment(index=0, start=words[0]["start"], end=words[-1]["end"],
                                 text_src=" ".join(w["word"] for w in words))]}


def test_segment_speaker_matches_dominant_speaker(tmp_path, monkeypatch):
    monkeypatch.setattr(ss.llm, "chat", lambda *a, **k: "[]")
    words = _spk_words([("Hello", "A"), ("world.", "A"),
                        ("How", "B"), ("are", "B"), ("you?", "B")])
    segs = ss.sentence_seg(_state(tmp_path, words), Config())["segments"]
    assert [s.speaker for s in segs] == ["A", "B"]
    # speaker is also persisted into the manifest, not only on the live objects.
    manifest = json.loads((tmp_path / "sentence_seg" / "segments.json").read_text())
    assert [s["speaker"] for s in manifest["segments"]] == ["A", "B"]


def test_no_speaker_keys_yields_empty_speaker(tmp_path, monkeypatch):
    # Local-path parity: no word carries a speaker -> Segment.speaker == "".
    monkeypatch.setattr(ss.llm, "chat", lambda *a, **k: "[]")
    words = _spk_words([("Hello", None), ("world.", None)])
    segs = ss.sentence_seg(_state(tmp_path, words), Config())["segments"]
    assert all(s.speaker == "" for s in segs)


def test_straddling_sentence_takes_majority_not_crash(tmp_path, monkeypatch):
    monkeypatch.setattr(ss.llm, "chat", lambda *a, **k: "[]")
    words = _spk_words([("The", "A"), ("quick", "A"), ("brown", "B"), ("fox.", "A")])
    segs = ss.sentence_seg(_state(tmp_path, words), Config())["segments"]
    assert len(segs) == 1
    assert segs[0].speaker == "A"  # 3x A vs 1x B


def test_segment_speaker_round_trips_through_dict():
    seg = Segment(index=0, start=0.0, end=1.0, text_src="x", speaker="B")
    assert seg.to_dict()["speaker"] == "B"
    assert Segment.from_dict(seg.to_dict()).speaker == "B"
