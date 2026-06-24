import json

import pytest

from loro.config import Config
from loro.harness import artifacts
from loro.harness.ledger import SkipLedger
from loro.nodes import translate as tr
from loro.state import Segment


def _segments(n=6):
    return [
        Segment(index=i, start=i * 2.0, end=i * 2.0 + 1.5, text_src=f"english line {i}")
        for i in range(n)
    ]


@pytest.fixture
def env(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    sent = []  # every payload sent to the model

    def fake_chat(cfg, messages, **kw):
        sent.append(messages)
        user = messages[-1]["content"]
        lines = json.loads(user[user.rindex("[{"):])
        return json.dumps([{"i": l["i"], "vi": f"bản dịch {l['i']}"} for l in lines],
                          ensure_ascii=False)

    monkeypatch.setattr(tr.llm, "chat", fake_chat)
    state = {"workdir": str(workdir), "segments": _segments(), "video_context": "video về ML"}
    return {"state": state, "workdir": workdir, "sent": sent}


def _run(env, **cfg_kw):
    cfg = Config(translate_batch=3, **cfg_kw)
    return tr.translate(env["state"], cfg)


class TestBatchCache:
    def test_full_run_then_cached(self, env):
        result = _run(env)
        assert result["segments"][0].text_target == "bản dịch 0"
        assert len(env["sent"]) == 2  # 6 segments / batch 3

        env["state"]["segments"] = _segments()
        _run(env)
        assert len(env["sent"]) == 2  # all batches cached

    def test_one_changed_segment_recalls_only_its_batch_and_keeps_peers(self, env):
        # AE2 (first half)
        _run(env)
        segs = _segments()
        segs[4].text_src = "a totally new english line"
        env["state"]["segments"] = segs
        _run(env)
        assert len(env["sent"]) == 3  # only batch 1 re-sent

        # The re-sent request contains only the changed segment: peers reused
        last_user = env["sent"][-1][-1]["content"]
        lines = json.loads(last_user[last_user.rindex("[{"):])
        assert [l["i"] for l in lines] == [4]
        # Peers keep their old text verbatim
        assert env["state"]["segments"][3].text_target == "bản dịch 3"
        assert env["state"]["segments"][5].text_target == "bản dịch 5"

    def test_system_prompt_change_invalidates_all_batches(self, env, monkeypatch):
        # The system prompt is now profile-sourced (U8): a changed profile prompt
        # busts every batch. Swap the resolved profile for one with a new prompt.
        import dataclasses
        from loro.config import Config
        _run(env)
        changed = dataclasses.replace(Config().language_profile,
                                      system_prompt="một system prompt hoàn toàn khác")
        monkeypatch.setattr(type(Config()), "language_profile",
                            property(lambda self: changed))
        env["state"]["segments"] = _segments()
        _run(env)
        assert len(env["sent"]) == 4  # both batches recomputed


class TestPrompt:
    def test_loanword_instruction_and_context_present(self, env):
        # AE5 at prompt level
        _run(env)
        system = env["sent"][0][0]["content"]
        assert "từ mượn" in system or "agent" in system
        user = env["sent"][0][-1]["content"]
        assert "video về ML" in user

    def test_degraded_empty_context_still_translates(self, env):
        env["state"]["video_context"] = ""
        result = _run(env)
        assert all(s.text_target for s in result["segments"])


class TestOverrides:
    def test_override_wins_and_survives_recompute(self, env):
        (env["workdir"] / "overrides.json").write_text(
            json.dumps({"seg_0002": "bản sửa tay"}, ensure_ascii=False))
        result = _run(env)
        assert result["segments"][2].text_target == "bản sửa tay"

        # Recompute the batch (change a peer) - override still wins
        segs = _segments()
        segs[0].text_src = "changed"
        env["state"]["segments"] = segs
        _run(env)
        assert env["state"]["segments"][2].text_target == "bản sửa tay"
        # The model's own translation is preserved in the batch artifact
        batch0 = json.loads((env["workdir"] / "translate" / "batch_0000.json").read_text())
        assert batch0["items"]["2"]["vi"] == "bản dịch 2"

    def test_editing_override_changes_manifest(self, env):
        _run(env)
        manifest_file = env["workdir"] / "translate" / "segments.json"
        first = json.loads(manifest_file.read_text())
        (env["workdir"] / "overrides.json").write_text(
            json.dumps({"seg_0001": "sửa mới"}, ensure_ascii=False))
        env["state"]["segments"] = _segments()
        _run(env)
        second = json.loads(manifest_file.read_text())
        assert first != second
        assert second["segments"][1]["text_target"] == "sửa mới"

    def test_out_of_range_override_is_skipped_and_logged_not_misapplied(self, env, caplog):
        # U5: after re-segmentation a seg_NNNN key may point past the new count
        # (here only 6 segments exist). The stale key must be dropped + logged,
        # never applied to a different segment; the in-range override still wins.
        import logging
        (env["workdir"] / "overrides.json").write_text(
            json.dumps({"seg_0002": "bản đúng", "seg_0099": "ngoài phạm vi"},
                       ensure_ascii=False))
        with caplog.at_level(logging.WARNING):
            result = _run(env)
        assert result["segments"][2].text_target == "bản đúng"          # in-range applied
        # No segment carries the stale text, and the model's own VI fills the rest
        assert all(s.text_target != "ngoài phạm vi" for s in result["segments"])
        assert any("seg_0099" in r.message for r in caplog.records)  # surfaced loudly


class TestBudgetProfile:
    def test_vi_budget_is_legacy_syllable_model(self):
        # R19: VI keeps duration x 4.3 syllables (byte-identical).
        seg = Segment(index=0, start=0.0, end=10.0, text_src="x")
        assert tr._budget(Config(), seg) == int(10.0 * 4.3)

    def test_fr_budget_is_cps_not_syllables(self):
        # R5: an FR segment budgets in characters at the FR profile CPS, not a
        # syllable count — a 10s slot at 17 CPS allows ~170 chars, far more than
        # the ~43 the VI syllable model would allow for the same duration.
        seg = Segment(index=0, start=0.0, end=10.0, text_src="x")
        fr = tr._budget(Config(target_lang="fr"), seg)
        vi = tr._budget(Config(), seg)
        assert fr == int(10.0 * 17.0)
        assert fr > vi


class TestProfileFraming:
    def test_fr_uses_profile_prompt_labels_and_output_key(self, env, monkeypatch):
        # R10: an FR run uses the FR system prompt + English context labels and
        # parses the FR profile's "text" output key.
        def fr_fake(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            return json.dumps([{"i": l["i"], "text": f"FR {l['i']}"} for l in lines],
                              ensure_ascii=False)
        monkeypatch.setattr(tr.llm, "chat", fr_fake)
        result = _run(env, target_lang="fr")
        assert result["segments"][0].text_target == "FR 0"     # parsed via "text" key
        system = env["sent"][0][0]["content"]
        assert "traducteur professionnel de doublage" in system  # FR system prompt
        user = env["sent"][0][-1]["content"]
        assert "Video context" in user                          # English labels, not Vietnamese
        assert "Bối cảnh video" not in user

    def test_fr_target_busts_vi_cache(self, env, monkeypatch):
        # R19/R10: target_lang=fr folds target_lang into the fingerprint, so it
        # does not collide with a cached VI batch.
        _run(env)  # VI default
        def fr_fake(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            return json.dumps([{"i": l["i"], "text": f"FR {l['i']}"} for l in lines],
                              ensure_ascii=False)
        monkeypatch.setattr(tr.llm, "chat", fr_fake)
        env["state"]["segments"] = _segments()
        _run(env, target_lang="fr")
        assert len(env["sent"]) == 4  # VI batches not reused for FR


class TestSameLanguage:
    def test_source_equals_target_skips_llm_and_copies_source(self, env, monkeypatch, caplog):
        # R11/AE5: en source + en target -> no LLM call, target text = source text,
        # a warning, and segments stay translatable (not skipped) for TTS.
        import logging
        env["state"]["source_lang"] = "en"
        with caplog.at_level(logging.WARNING):
            result = _run(env, target_lang="en")
        assert env["sent"] == []  # the LLM was never called
        for seg in result["segments"]:
            assert seg.text_target == seg.text_src
            assert not seg.skipped
        assert any("skipping LLM translation" in r.message for r in caplog.records)

    def test_same_language_region_variant(self, env):
        # en source vs en-US target still counts as same language (base tag).
        env["state"]["source_lang"] = "en"
        result = _run(env, target_lang="en-US")
        assert env["sent"] == []
        assert result["segments"][0].text_target == result["segments"][0].text_src


class TestTranslateModel:
    def test_post_init_defaults_to_gemma_kwarg(self):
        # KTD1: the fallback honors the llm_model kwarg, not only the env var
        # (case test_crosscheck.py:272 constructs Config(llm_model=...)).
        assert Config(llm_model="X").llm_model_translate == "X"

    def test_explicit_translate_model_wins(self):
        assert Config(llm_model="X", llm_model_translate="Y").llm_model_translate == "Y"

    def test_changing_translate_model_leaves_gemma_model(self):
        # vision/crosscheck artifacts key on llm_model; A/B-ing the
        # translator must not disturb their fingerprint source.
        assert Config(llm_model_translate="qwen3-14b-4bit").llm_model == Config().llm_model

    def test_default_translate_model_is_byte_identical_fingerprint(self, env):
        # R37: with LLM_MODEL_TRANSLATE unset, llm_model_translate == llm_model, so
        # the batch fingerprint must equal exactly what the pre-split code
        # produced (the "model" key sourced from llm_model). Upgrading the
        # code must NOT bust an existing translate cache.
        cfg = Config(translate_batch=3)
        batch = _segments()[:3]
        old_inputs = {  # the exact inputs dict the pre-split code built
            "lines": [[s.index, s.text_src, tr._budget(cfg, s)] for s in batch],
            "context": "video về ML",
            "system": cfg.language_profile.system_prompt,  # profile-sourced (U8)
            "model": cfg.llm_model,  # old value source
            "temperature": tr.TEMPERATURE,
        }
        old_prompt_sha = artifacts.fingerprint({
            "context": "video về ML", "system": cfg.language_profile.system_prompt,
            "model": cfg.llm_model, "temperature": tr.TEMPERATURE,
        })
        _run(env)
        meta = json.loads(
            (env["workdir"] / "translate" / "batch_0000.json.meta.json").read_text())
        assert meta["input_fingerprint"] == artifacts.fingerprint(old_inputs)
        body = json.loads(
            (env["workdir"] / "translate" / "batch_0000.json").read_text())
        assert body["prompt_sha"] == old_prompt_sha

    def test_different_translate_model_retranslates_all(self, env):
        _run(env)
        env["state"]["segments"] = _segments()
        _run(env, llm_model_translate="qwen3-14b-4bit")
        assert len(env["sent"]) == 4  # both batches retranslated, none cached


def _write_ctx(workdir, bi, layers):
    cdir = workdir / "context"
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"batch_{bi:04d}.json").write_text(
        json.dumps(layers, ensure_ascii=False), encoding="utf-8")


class TestLayeredContext:
    def test_prompt_includes_layers_and_prev_vi(self, env):
        # R42: the batch prompt carries the assembled layers, and the next
        # batch's prompt carries the previous batch's VI as a consistency ref.
        wd = env["workdir"]
        _write_ctx(wd, 0, {"video_context": "vc-global", "shot_visuals": ["a lab"],
                           "summary": "intro to ml", "neighbors_after": ["en 3"]})
        _write_ctx(wd, 1, {"video_context": "vc-global", "shot_visuals": ["a stage"],
                           "summary": "deeper", "neighbors_before": ["en 2"]})
        _run(env)
        u0 = env["sent"][0][-1]["content"]
        assert "vc-global" in u0 and "a lab" in u0 and "intro to ml" in u0
        assert "KHÔNG dịch lại" not in u0  # first batch has no prev-VI section
        u1 = env["sent"][1][-1]["content"]
        assert "a stage" in u1
        assert "KHÔNG dịch lại" in u1     # prev-VI section present
        assert "bản dịch 2" in u1         # prev-VI from the end of batch 0

    def test_context_change_retranslates_only_that_batch(self, env):
        wd = env["workdir"]
        _write_ctx(wd, 0, {"video_context": "vc1"})
        _write_ctx(wd, 1, {"video_context": "vc1"})
        _run(env)
        assert len(env["sent"]) == 2
        _write_ctx(wd, 0, {"video_context": "vc1-CHANGED"})  # only batch 0's context
        env["state"]["segments"] = _segments()
        _run(env)
        assert len(env["sent"]) == 3  # batch 0 re-translated, batch 1 still cached

    def test_upstream_translation_change_cascades_prev_vi(self, env, monkeypatch):
        # R42: an upstream edit changes its VI -> the downstream batch's prev-VI
        # changes -> it retranslates; peers do NOT keep stale vi (prompt_sha
        # carries prev-VI).
        wd = env["workdir"]

        def en_fake(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            return json.dumps([{"i": l["i"], "vi": f"VI[{l['src']}]"} for l in lines],
                              ensure_ascii=False)

        monkeypatch.setattr(tr.llm, "chat", en_fake)
        _write_ctx(wd, 0, {"video_context": "vc"})
        _write_ctx(wd, 1, {"video_context": "vc"})
        _run(env)
        n_first = len(env["sent"])
        segs = _segments()
        segs[2].text_src = "a brand new upstream line"  # last seg of batch 0
        env["state"]["segments"] = segs
        _run(env)
        # batch 0 retranslates (its line changed) AND batch 1 retranslates
        # (its prev-VI window changed)
        assert len(env["sent"]) == n_first + 2

    def test_fresh_and_resume_after_crash_match(self, env):
        # Determinism (R1/R2): prev-VI lives in the fingerprint, so a resumed
        # run (batch 0 cached, batch 1 recomputed) reproduces the fresh result.
        wd = env["workdir"]
        _write_ctx(wd, 0, {"video_context": "vc"})
        _write_ctx(wd, 1, {"video_context": "vc"})
        r1 = _run(env)
        vi_fresh = [s.text_target for s in r1["segments"]]
        (wd / "translate" / "batch_0001.json").unlink()
        (wd / "translate" / "batch_0001.json.meta.json").unlink()
        env["state"]["segments"] = _segments()
        r2 = _run(env)
        assert [s.text_target for s in r2["segments"]] == vi_fresh

    def test_peer_reuse_when_context_and_prev_vi_unchanged(self, env):
        # R5b: identical context + prev-VI across runs -> everything cached.
        wd = env["workdir"]
        _write_ctx(wd, 0, {"video_context": "vc"})
        _write_ctx(wd, 1, {"video_context": "vc"})
        _run(env)
        assert len(env["sent"]) == 2
        env["state"]["segments"] = _segments()
        _run(env)
        assert len(env["sent"]) == 2

    def test_override_wins_with_context_present(self, env):
        # R16: overrides still applied after translation, unaffected by context.
        wd = env["workdir"]
        _write_ctx(wd, 0, {"video_context": "vc"})
        _write_ctx(wd, 1, {"video_context": "vc"})
        (wd / "overrides.json").write_text(
            json.dumps({"seg_0002": "bản sửa tay"}, ensure_ascii=False))
        result = _run(env)
        assert result["segments"][2].text_target == "bản sửa tay"

    def test_truncated_reply_is_content_error_not_silent_skip(self, env, monkeypatch):
        # U6: a reply that omits an index is a content error; the segment is
        # recorded (with a content signature), not silently dropped.
        wd = env["workdir"]
        _write_ctx(wd, 0, {"video_context": "vc"})
        _write_ctx(wd, 1, {"video_context": "vc"})

        def truncating(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            return json.dumps([{"i": l["i"], "vi": f"bản dịch {l['i']}"}
                               for l in lines if l["i"] != 2], ensure_ascii=False)

        monkeypatch.setattr(tr.llm, "chat", truncating)
        result = _run(env)
        seg2 = result["segments"][2]
        assert seg2.skipped and seg2.skip_reason == "translate_failed"
        entry = SkipLedger(wd).entries()["seg_0002"]
        assert entry["signature"][1] == "content"  # not a silent generic drop
        assert result["segments"][0].text_target == "bản dịch 0"  # peers fine

    def test_missing_context_falls_back_to_bare(self, env):
        # R43: no context artifacts -> bare video_context, no raise.
        result = _run(env)
        assert all(s.text_target for s in result["segments"])
        assert "video về ML" in env["sent"][0][-1]["content"]


class TestFailurePolicy:
    def test_failed_segment_skipped_others_translated(self, env, monkeypatch):
        def chat_failing_seg2(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            if any(l["i"] == 2 for l in lines):
                raise ValueError("model returned garbage")
            return json.dumps([{"i": l["i"], "vi": f"bản dịch {l['i']}"} for l in lines],
                              ensure_ascii=False)

        monkeypatch.setattr(tr.llm, "chat", chat_failing_seg2)
        result = _run(env)

        seg2 = result["segments"][2]
        assert seg2.skipped is True
        assert seg2.skip_reason == "translate_failed"
        assert result["segments"][1].text_target == "bản dịch 1"
        assert result["segments"][3].text_target == "bản dịch 3"

        entry = SkipLedger(env["workdir"]).entries()["seg_0002"]
        assert entry["reason"] == "translate_failed"

        # The incomplete batch is not finalized: next run retries seg 2
        assert not (env["workdir"] / "translate" / "batch_0000.json.meta.json").exists()
        monkeypatch.setattr(tr.llm, "chat", lambda cfg, messages, **kw: (
            env["sent"].append(messages),
            json.dumps([{"i": 2, "vi": "bản dịch 2 sửa"}], ensure_ascii=False))[-1])
        env["state"]["segments"] = _segments()
        result = _run(env)
        assert result["segments"][2].text_target == "bản dịch 2 sửa"
        assert result["segments"][2].skipped is False


class TestProgrammerErrorPropagation:
    # U8/B5/R8: a genuine bug in the translation path must surface with a stack,
    # not be silently swallowed as a per-segment content skip.

    def test_batch_level_programmer_error_propagates_not_per_segment(self, env, monkeypatch):
        def boom(cfg, messages, **kw):
            env["sent"].append(messages)
            raise KeyError("unexpected key")
        monkeypatch.setattr(tr.llm, "chat", boom)
        with pytest.raises(KeyError):
            _run(env)
        # propagated from the batch call; the per-segment fallback was never entered
        assert len(env["sent"]) == 1

    def test_per_segment_programmer_error_propagates(self, env, monkeypatch):
        def boom(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            if len(lines) > 1:
                raise ValueError("batch parse fail")       # content -> per-segment retry
            raise KeyError("unexpected per-segment key")   # programmer error -> propagate
        monkeypatch.setattr(tr.llm, "chat", boom)
        with pytest.raises(KeyError):
            _run(env)

    def test_infra_batch_error_skips_per_segment_with_infra_signature(self, env, monkeypatch):
        # Regression: an infra StageError from the batch still short-circuits the
        # per-segment fallback and records each segment with the infra signature.
        from loro.harness.retry import StageError
        def infra(cfg, messages, **kw):
            env["sent"].append(messages)
            raise StageError("translate", "infra", "timeout", "server down")
        monkeypatch.setattr(tr.llm, "chat", infra)
        result = _run(env, abort_threshold=99)   # keep the abort window out of the way
        assert result["segments"][0].skipped
        entry = SkipLedger(env["workdir"]).entries()["seg_0000"]
        assert entry["signature"] == ["translate", "infra", "timeout"]
        # one infra call per batch (2 batches), no per-segment retry storm
        assert len(env["sent"]) == 2

    def test_malformed_object_reply_is_content_skip_not_crash(self, env, monkeypatch):
        # A reply that wraps the array in an OBJECT (or mis-keys items) is a model
        # output-shape (content) failure, not a programmer bug: _translate_lines
        # classifies it as content, so it degrades to a per-segment skip instead of
        # crashing the run with a TypeError.
        def wrapper(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            return json.dumps({"translations": [{"i": l["i"], "vi": f"x{l['i']}"}
                                                 for l in lines]})
        monkeypatch.setattr(tr.llm, "chat", wrapper)
        result = _run(env, abort_threshold=99)        # must NOT raise
        assert result["segments"][0].skipped
        entry = SkipLedger(env["workdir"]).entries()["seg_0000"]
        assert entry["signature"][1] == "content"     # not a propagated crash

    def test_empty_string_translation_records_content_skip(self, env, monkeypatch):
        # Regression: a valid-shape reply with empty translations raises the
        # explicit "no translation returned" ValueError -> recorded as content.
        def empty(cfg, messages, **kw):
            env["sent"].append(messages)
            user = messages[-1]["content"]
            lines = json.loads(user[user.rindex("[{"):])
            return json.dumps([{"i": l["i"], "vi": ""} for l in lines], ensure_ascii=False)
        monkeypatch.setattr(tr.llm, "chat", empty)
        result = _run(env, abort_threshold=99)      # keep the abort window out of the way
        assert result["segments"][0].skipped
        assert result["segments"][0].skip_reason == "translate_failed"
        entry = SkipLedger(env["workdir"]).entries()["seg_0000"]
        assert entry["signature"][1] == "content"   # ValueError -> content class


def test_source_equals_target_reuses_source_srt_without_clobber(tmp_path, monkeypatch):
    # R11/#7: a voice-replacement run (source==target) skips the LLM, sets the
    # target text to the source text, and must NOT clobber the word-timed source
    # SRT (transcript.<tag>.srt). It reuses that file as the target sidecar.
    workdir = tmp_path / "work"
    workdir.mkdir()
    src_srt = workdir / "transcript.en.srt"
    sentinel = "1\n00:00:00,000 --> 00:00:01,000\nhello\n"
    src_srt.write_text(sentinel, encoding="utf-8")

    # No LLM should be called on the same-language path.
    def boom(cfg, messages, **kw):
        raise AssertionError("LLM must not be called when source == target")
    monkeypatch.setattr(tr.llm, "chat", boom)

    segs = [Segment(index=0, start=0.0, end=1.0, text_src="hello")]
    state = {"workdir": str(workdir), "segments": segs, "video_context": "", "words": []}
    out = tr.translate(state, Config(target_lang="en", source_lang="en"))

    assert out["srt_target"] == str(src_srt)            # reused, not a fresh write
    assert src_srt.read_text(encoding="utf-8") == sentinel  # source SRT untouched
    assert segs[0].text_target == "hello"               # identity "translation"
