import json
from pathlib import Path

import pytest

from loro.config import Config
from loro.harness import artifacts, diff
from loro.harness.ledger import AbortRun, SkipLedger
from loro.harness.retry import StageError
from loro.nodes import crosscheck as xck
from loro.state import Segment


class TestDiff:
    def test_single_misheard_term_triggers_replace(self):
        # AE4: one content-word substitution in an otherwise matching segment
        result = diff.compare(
            "we use cooper netties to orchestrate the containers",
            "we use Kubernetes to orchestrate the containers",
        )
        assert result["decision"] == "replace"
        assert result["content_substitution"] is True

    def test_identical_after_normalization_keeps(self):
        result = diff.compare("Hello, World!", "hello world")
        assert result["decision"] == "keep"

    def test_empty_gemma_keeps_low_confidence(self):
        assert diff.compare("a full sentence here", "")["decision"] == "keep_low_confidence"

    def test_gibberish_gemma_keeps_low_confidence(self):
        result = diff.compare(
            "the deployment pipeline builds the artifacts and ships them",
            "lorem ipsum dolor sit amet consectetur adipiscing elit quad",
        )
        assert result["decision"] == "keep_low_confidence"

    def test_stopword_only_difference_keeps(self):
        result = diff.compare("this is the plan", "this is a plan")
        assert result["decision"] == "keep"
        assert result["content_substitution"] is False

    def test_too_short_gemma_low_confidence(self):
        result = diff.compare("one two three four five six seven eight nine ten", "one")
        assert result["decision"] == "keep_low_confidence"

    def test_wer_only_replace_without_substitution(self):
        # Insertions/deletions of content words accumulate WER past the
        # threshold without any single replace opcode
        result = diff.compare(
            "we deploy the model today",
            "we deploy the new trained model today honestly",
        )
        assert result["decision"] == "replace"
        assert result["content_substitution"] is False


class TestSplitBounds:
    def test_short_segment_single_part(self):
        assert xck.split_bounds(20.0, []) == [(0.0, 20.0)]

    def test_45s_with_silence_at_28_cuts_there(self):
        bounds = xck.split_bounds(45.0, [(27.8, 28.2)], max_len=30.0)
        assert bounds == [(0.0, 28.0), (28.0, 45.0)]

    def test_no_silence_hard_cut_at_30(self):
        bounds = xck.split_bounds(45.0, [], max_len=30.0)
        assert bounds == [(0.0, 30.0), (30.0, 45.0)]

    def test_long_segment_multiple_cuts(self):
        bounds = xck.split_bounds(75.0, [(29.0, 29.4), (55.0, 56.0)], max_len=30.0)
        assert bounds[0] == (0.0, 29.2)
        assert bounds[1][1] == pytest.approx(55.5)
        assert bounds[-1][1] == 75.0
        assert all(e - s <= 30.0 + 1e-9 for s, e in bounds)


class TestGranitePrompt:
    def test_no_keywords_is_plain_prompt(self):
        assert xck.granite_prompt([], 32) == xck.GRANITE_BASE_PROMPT

    def test_keywords_use_model_card_syntax(self):
        prompt = xck.granite_prompt(["Kubernetes", "CI/CD"], 32)
        assert prompt == f"{xck.GRANITE_BASE_PROMPT} Keywords: Kubernetes, CI/CD"

    def test_keyword_cap_limits_count(self):
        prompt = xck.granite_prompt([f"kw{i}" for i in range(50)], 3)
        assert prompt.endswith("Keywords: kw0, kw1, kw2")


@pytest.fixture
def env(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    audio = tmp_path / "audio_16k.wav"
    audio.write_bytes(b"fake-16k-audio")

    # ffmpeg cut just creates a placeholder clip file
    monkeypatch.setattr(xck.ffmpeg, "ffmpeg",
                        lambda *args: Path(args[-1]).write_bytes(b"clip"))
    monkeypatch.setattr(xck.ffmpeg, "detect_silences", lambda *a, **kw: [])

    segments = [
        Segment(index=0, start=0.0, end=4.0, text_src="we use cooper netties here"),
        Segment(index=1, start=5.0, end=9.0, text_src="plain matching text"),
    ]
    state = {"workdir": str(workdir), "audio_16k": str(audio), "segments": segments}
    return {"state": state, "workdir": workdir, "segments": segments}


def _seg_index(artifact: Path) -> int:
    # "seg_0000.granite.json" -> 0
    return int(artifact.stem.split("_")[1].split(".")[0])


def _granite(monkeypatch, readings):
    """Fake the Granite worker: write canned readings (keyed by segment index)
    to each job's artifact. Returns a capture dict (call count, last prompt,
    segments seen)."""
    capture = {"calls": 0, "prompt": None, "segments": []}

    def fake(cfg, xdir, jobs, prompt):
        capture["calls"] += 1
        capture["prompt"] = prompt
        for job in jobs:
            idx = _seg_index(job["artifact"])
            capture["segments"].append(idx)
            text = readings[idx]
            artifacts.produce(
                job["artifact"], job["inputs"], "crosscheck",
                lambda tmp, text=text: tmp.write_text(
                    json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8"),
            )

    monkeypatch.setattr(xck, "_run_granite_worker", fake)
    return capture


def _granite_fails(monkeypatch, error_class="infra", code="connection"):
    def fake(cfg, xdir, jobs, prompt):
        raise StageError("crosscheck", error_class, code)

    monkeypatch.setattr(xck, "_run_granite_worker", fake)


def _gemma(monkeypatch, replies):
    """Mock llm.chat (Gemma). Replies are popped in call order; a default is
    returned once the list is exhausted."""
    calls = []

    def fake_chat(cfg, messages, **kw):
        calls.append(messages)
        return replies.pop(0) if replies else "plain matching text"

    monkeypatch.setattr(xck.llm, "chat", fake_chat)
    return calls


def _read_verdict(workdir, index):
    return json.loads((workdir / "crosscheck" / f"seg_{index:04d}.json").read_text())


# retry_attempts=1 keeps the granite-failure path from sleeping through backoff
NOABORT = dict(abort_threshold=99, retry_attempts=1)
# Plan's granite-lead weights (lone Granite can outvote Nemotron). The shipped
# default was calibrated at U6 to require corroboration, so tests that exercise
# the lone-Granite regime pass these explicitly.
GRANITE_LEAD = {"nemotron": 0.2, "granite": 0.5, "gemma": 0.3}


class TestEnsembleVoting:
    def test_agreement_skips_gemma(self, env):
        # R29: N == G, so Gemma's 0.3 can't flip 0.7 — don't call it
        def setup(monkeypatch):
            _granite(monkeypatch, {0: "we use cooper netties here",
                                   1: "plain matching text"})
            return _gemma(monkeypatch, [])

        with pytest.MonkeyPatch.context() as mp:
            gemma_calls = setup(mp)
            result = xck.crosscheck(env["state"], Config(**NOABORT))
        assert gemma_calls == []  # lazy: never consulted
        assert result["segments"][0].text_src == "we use cooper netties here"
        assert _read_verdict(env["workdir"], 0)["decision"] == "keep"

    def test_granite_and_gemma_agree_replaces(self, env):
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            gemma_calls = _gemma(mp, ["we use Kubernetes here"])
            result = xck.crosscheck(env["state"], Config(**NOABORT))
        assert result["segments"][0].text_src == "we use Kubernetes here"
        assert len(gemma_calls) == 1  # only the contested segment
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["decision"] == "replace"
        assert set(verdict["winner"].split("+")) == {"granite", "gemma"}
        assert verdict["text_granite"] == "we use Kubernetes here"

    def test_nemotron_gemma_tie_keeps_and_marks_contested(self, env):
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            # Gemma sides with Nemotron -> under granite-lead weights N+M=0.5
            # ties G=0.5 -> keep Nemotron, flagged contested
            _gemma(mp, ["we use cooper netties here"])
            result = xck.crosscheck(
                env["state"], Config(crosscheck_weights=GRANITE_LEAD, **NOABORT))
        assert result["segments"][0].text_src == "we use cooper netties here"
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["decision"] == "keep"
        assert verdict["contested"] is True

    def test_prompt_never_contains_nemotron_or_granite_text(self, env):
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            calls = _gemma(mp, ["we use Kubernetes here"])
            xck.crosscheck(env["state"], Config(**NOABORT))
        for messages in calls:
            payload = json.dumps(messages)
            assert "cooper netties" not in payload  # Nemotron's reading
            assert "Kubernetes" not in payload      # Granite's reading


class TestEnsembleResume:
    def test_both_tiers_cached_on_rerun(self, env):
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            gemma_calls = _gemma(mp, ["we use Kubernetes here"])
            xck.crosscheck(env["state"], Config(**NOABORT))
            assert cap["calls"] == 1 and len(gemma_calls) == 1

            # Fresh segment objects; artifacts must carry resume, not state
            env["state"]["segments"] = [
                Segment(index=0, start=0.0, end=4.0, text_src="we use cooper netties here"),
                Segment(index=1, start=5.0, end=9.0, text_src="plain matching text"),
            ]
            xck.crosscheck(env["state"], Config(**NOABORT))
        assert cap["calls"] == 1          # tier 1: granite worker not re-run
        assert len(gemma_calls) == 1      # tier 2: verdict cached, gemma not re-called

    def test_keyword_change_reruns_granite_and_verdict(self, env):
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {0: "we use cooper netties here",
                                1: "plain matching text"})
            _gemma(mp, [])
            xck.crosscheck(env["state"], Config(**NOABORT))
            assert cap["calls"] == 1
            assert "Keywords:" not in cap["prompt"]

            # Vision now supplies keywords -> granite prompt changes -> the
            # granite reading invalidates -> verdict recomputes (cascade)
            env["state"]["segments"] = [
                Segment(index=0, start=0.0, end=4.0, text_src="we use cooper netties here"),
                Segment(index=1, start=5.0, end=9.0, text_src="plain matching text"),
            ]
            env["state"]["video_keywords"] = ["Kubernetes"]
            xck.crosscheck(env["state"], Config(**NOABORT))
        assert cap["calls"] == 2
        assert "Keywords: Kubernetes" in cap["prompt"]

    def test_gemma_model_change_does_not_rerun_granite(self, env):
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            _gemma(mp, ["we use Kubernetes here", "we use Kubernetes here"])
            xck.crosscheck(env["state"], Config(**NOABORT))
            assert cap["calls"] == 1

            env["state"]["segments"] = [
                Segment(index=0, start=0.0, end=4.0, text_src="we use cooper netties here"),
                Segment(index=1, start=5.0, end=9.0, text_src="plain matching text"),
            ]
            # Different Gemma model: verdict fingerprint changes, granite's
            # does not -> granite reading is not re-listened
            xck.crosscheck(env["state"], Config(llm_model="other-model", **NOABORT))
        assert cap["calls"] == 1


class TestAudioModelRouting:
    """U2/KTD4: the re-listen arbiter targets llm_model_audio (the 12B that hears
    audio), and only that model's identity gates verdict recompute."""

    def _fresh_segments(self, env):
        # Artifacts must carry resume across runs, so hand each run new objects.
        env["state"]["segments"] = [
            Segment(index=0, start=0.0, end=4.0, text_src="we use cooper netties here"),
            Segment(index=1, start=5.0, end=9.0, text_src="plain matching text"),
        ]

    def test_arbiter_calls_audio_model_not_gemma(self, env):
        # The contested segment's re-listen passes model=cfg.llm_model_audio; the
        # agreeing segment never reaches the arbiter (lazy).
        captured = []

        def fake_chat(cfg, messages, **kw):
            # The re-listen now names the audio role (U7); read its resolved model.
            captured.append(kw["role"].model if "role" in kw else kw.get("model"))
            return "we use Kubernetes here"

        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            mp.setattr(xck.llm, "chat", fake_chat)
            xck.crosscheck(env["state"], Config(llm_model_audio="gemma-4-12B", **NOABORT))
        assert captured == ["gemma-4-12B"]

    def test_audio_model_change_recomputes_verdict(self, env):
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            calls = _gemma(mp, ["we use Kubernetes here", "we use Kubernetes here"])
            xck.crosscheck(env["state"], Config(llm_model_audio="m1", **NOABORT))
            assert len(calls) == 1
            self._fresh_segments(env)
            # llm_model_audio changed -> the re-listen model the verdict depends on
            # changed -> verdict recomputes (arbiter consulted again).
            xck.crosscheck(env["state"], Config(llm_model_audio="m2", **NOABORT))
        assert len(calls) == 2

    def test_vision_model_change_keeps_verdict_when_audio_pinned(self, env):
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            calls = _gemma(mp, ["we use Kubernetes here"])
            xck.crosscheck(env["state"], Config(llm_model_audio="pinned", **NOABORT))
            assert len(calls) == 1
            self._fresh_segments(env)
            # llm_model (vision) changes but llm_model_audio is pinned -> the
            # verdict fingerprint is unchanged, so it is not recomputed (KTD4).
            xck.crosscheck(env["state"], Config(llm_model_audio="pinned",
                                                llm_model="other-vision", **NOABORT))
        assert len(calls) == 1

    def test_default_profile_verdict_not_busted_on_rerun(self, env):
        # R3: llm_model_audio defaults to llm_model, so an unchanged default
        # config (the old oMLX profile) keeps prior verdicts valid — no spurious
        # recompute that would re-listen every segment of a pre-split run.
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            calls = _gemma(mp, ["we use Kubernetes here"])
            xck.crosscheck(env["state"], Config(**NOABORT))
            assert len(calls) == 1
            self._fresh_segments(env)
            xck.crosscheck(env["state"], Config(**NOABORT))
        assert len(calls) == 1


class TestDegradationMatrix:
    def test_granite_batch_failure_falls_back_to_two_way(self, env):
        with pytest.MonkeyPatch.context() as mp:
            _granite_fails(mp)
            # Fallback path: Nemotron x Gemma like the old behaviour
            gemma_calls = _gemma(mp, ["we use Kubernetes here", "plain matching text"])
            result = xck.crosscheck(env["state"], Config(**NOABORT))
        assert result["segments"][0].text_src == "we use Kubernetes here"
        assert len(gemma_calls) == 2  # every segment falls back to Gemma
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["source"] == "granite_fallback"
        assert verdict["decision"] == "replace"
        assert verdict["retryable"] is True  # retry granite next run (R20)

    def test_granite_failure_retried_next_run(self, env):
        with pytest.MonkeyPatch.context() as mp:
            _granite_fails(mp)
            _gemma(mp, ["we use Kubernetes here", "plain matching text"])
            xck.crosscheck(env["state"], Config(**NOABORT))

            # Granite recovers: rerun retries it (verdict was retryable) and
            # the segment now goes through the three-way vote
            env["state"]["segments"] = [
                Segment(index=0, start=0.0, end=4.0, text_src="we use cooper netties here"),
                Segment(index=1, start=5.0, end=9.0, text_src="plain matching text"),
            ]
            cap = _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})
            _gemma(mp, ["we use Kubernetes here"])
            xck.crosscheck(env["state"], Config(**NOABORT))
        assert cap["calls"] == 1
        assert _read_verdict(env["workdir"], 0)["source"] == "vote"

    def test_gemma_fail_on_contested_then_accepts(self, env):
        # Granite-lead weights so the {N,G} fallback vote lets Granite win (R32)
        cfg = Config(crosscheck_weights=GRANITE_LEAD, **NOABORT)
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we use Kubernetes here", 1: "plain matching text"})

            def failing_chat(cfg, messages, **kw):
                raise StageError("crosscheck", "infra", "connection")

            mp.setattr(xck.llm, "chat", failing_chat)
            result = xck.crosscheck(env["state"], cfg)
            # R32: Gemma vote missing -> vote {N, G}, Granite (0.5) wins
            assert result["segments"][0].text_src == "we use Kubernetes here"
            verdict = _read_verdict(env["workdir"], 0)
            assert verdict["decision"] == "replace"
            assert verdict["retryable"] is True
            assert verdict["attempts"] == 1

            # Second failure -> accepted (R20)
            env["state"]["segments"][0].text_src = "we use cooper netties here"
            xck.crosscheck(env["state"], cfg)
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["retryable"] is False
        assert verdict["attempts"] == 2

    def test_both_engines_fail_keeps_low_confidence(self, env):
        cfg = Config(**NOABORT)
        with pytest.MonkeyPatch.context() as mp:
            _granite_fails(mp)

            def failing_chat(cfg, messages, **kw):
                raise StageError("crosscheck", "infra", "connection")

            mp.setattr(xck.llm, "chat", failing_chat)
            result = xck.crosscheck(env["state"], cfg)
        assert result["segments"][0].text_src == "we use cooper netties here"
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["decision"] == "keep"
        assert verdict["low_confidence"] is True
        assert verdict["retryable"] is True

    def test_mass_granite_gibberish_aborts(self, env):
        # R30 across the whole video: Granite degraded everywhere must abort,
        # not bake low-confidence markers for every segment
        env["state"]["segments"] = [
            Segment(index=i, start=float(i), end=i + 0.9,
                    text_src=f"a long enough sentence number {i} here")
            for i in range(6)
        ]
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {i: "zzz qqq xxx unrelated garbage" for i in range(6)})
            _gemma(mp, [])
            with pytest.raises(AbortRun) as exc_info:
                xck.crosscheck(env["state"], Config(abort_threshold=3))
        assert exc_info.value.signature == ("crosscheck", "content", "extreme_mismatch")

    def test_repeated_infra_failures_abort(self, env):
        env["state"]["segments"] = [
            Segment(index=i, start=float(i), end=i + 0.9, text_src=f"text {i}")
            for i in range(6)
        ]
        with pytest.MonkeyPatch.context() as mp:
            _granite_fails(mp)

            def failing_chat(cfg, messages, **kw):
                raise StageError("crosscheck", "infra", "connection")

            mp.setattr(xck.llm, "chat", failing_chat)
            with pytest.raises(AbortRun):
                xck.crosscheck(env["state"], Config(abort_threshold=3, retry_attempts=1))

    def test_abort_does_not_consume_marker_retry_budget(self, env):
        env["state"]["segments"] = [
            Segment(index=i, start=float(i), end=i + 0.9, text_src=f"text {i}")
            for i in range(3)
        ]
        with pytest.MonkeyPatch.context() as mp:
            _granite_fails(mp)

            def failing_chat(cfg, messages, **kw):
                raise StageError("crosscheck", "infra", "connection")

            mp.setattr(xck.llm, "chat", failing_chat)
            with pytest.raises(AbortRun):
                xck.crosscheck(env["state"], Config(abort_threshold=3, retry_attempts=1))
        for i in range(3):
            data = _read_verdict(env["workdir"], i)
            assert data["retryable"] is True, f"seg {i} lost its retry budget"


class TestClipCutting:
    def test_lead_in_pad_applied_to_clip(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(xck.ffmpeg, "ffmpeg",
                            lambda *a: (calls.append(a), Path(a[-1]).write_bytes(b"clip"))[1])
        seg = Segment(index=3, start=5.0, end=9.0, text_src="x")
        xck._cut_clips(Config(crosscheck_clip_pad=0.25), env["state"], seg,
                       env["workdir"], {})
        ss = calls[0][calls[0].index("-ss") + 1]
        assert ss == "4.750"  # 5.0 - 0.25 lead-in

    def test_lead_in_clamps_at_zero(self, env, monkeypatch):
        calls = []
        monkeypatch.setattr(xck.ffmpeg, "ffmpeg",
                            lambda *a: (calls.append(a), Path(a[-1]).write_bytes(b"clip"))[1])
        seg = Segment(index=0, start=0.1, end=4.0, text_src="x")
        xck._cut_clips(Config(crosscheck_clip_pad=0.5), env["state"], seg,
                       env["workdir"], {})
        ss = calls[0][calls[0].index("-ss") + 1]
        assert ss == "0.000"  # never seeks before the start of the audio

    def test_pad_changes_granite_fingerprint(self, env):
        seg = env["segments"][0]
        a = xck._granite_inputs(Config(crosscheck_clip_pad=0.0), seg, "sha", "p", [], "w")
        b = xck._granite_inputs(Config(crosscheck_clip_pad=0.25), seg, "sha", "p", [], "w")
        assert a != b  # changing the pad re-reads Granite

    def test_long_segment_split_feeds_both_engines(self, env):
        # R8a: a 45s segment is split at silence into <=30s parts; both
        # Granite (phase 1) and Gemma (phase 2) consume the same parts
        env["state"]["segments"] = [
            Segment(index=0, start=0.0, end=45.0, text_src="we use cooper netties here"),
        ]
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {0: "we use Kubernetes here"})
            gemma_calls = _gemma(mp, ["we use", "Kubernetes here"])
            result = xck.crosscheck(env["state"], Config(**NOABORT))
        # No silence -> hard cut at 30 -> 2 parts -> 2 Gemma calls
        assert len(gemma_calls) == 2
        # Original segment boundaries are preserved
        assert result["segments"][0].start == 0.0
        assert result["segments"][0].end == 45.0


def _with_subs(env, srt_text):
    path = env["workdir"] / "subs.en.srt"
    path.write_text(srt_text, encoding="utf-8")
    env["state"]["subs_path"] = str(path)
    return path


class TestSubtitleFirst:
    def test_covered_segment_uses_subtitle_skips_engines(self, env):
        env["state"]["segments"] = [
            Segment(index=0, start=0.0, end=4.0,
                    text_src="we deploy the model to production today"),
            Segment(index=1, start=5.0, end=9.0, text_src="plain matching text"),
        ]
        _with_subs(env, "1\n00:00:00,000 --> 00:00:04,000\n"
                        "we deploy the model to production today\n")
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {1: "plain matching text"})
            gemma_calls = _gemma(mp, [])
            result = xck.crosscheck(env["state"], Config(**NOABORT))
        assert result["segments"][0].text_src == "we deploy the model to production today"
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["decision"] == "subtitle"
        assert verdict["winner"] == "subtitle"
        assert 0 not in cap["segments"]  # Granite never asked about the covered seg
        assert gemma_calls == []

    def test_gap_coverage_falls_back_to_ensemble(self, env):
        env["state"]["segments"] = [
            Segment(index=0, start=0.0, end=8.0,
                    text_src="we deploy the model to production today right now"),
        ]
        _with_subs(env, "1\n00:00:00,000 --> 00:00:02,000\nwe deploy\n")  # covers 2s/8s
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {0: "we deploy the model to production today right now"})
            _gemma(mp, [])
            xck.crosscheck(env["state"], Config(**NOABORT))
        assert 0 in cap["segments"]  # ensemble engaged
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["decision"] == "keep"
        assert verdict["sub_rejected"]["reason"] == "low_coverage"

    def test_wrong_language_sub_rejected_below_align(self, env):
        env["state"]["segments"] = [
            Segment(index=0, start=0.0, end=4.0,
                    text_src="we deploy the model to production"),
        ]
        _with_subs(env, "1\n00:00:00,000 --> 00:00:04,000\n"
                        "bonjour le monde ceci est completement autre\n")
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {0: "we deploy the model to production"})
            _gemma(mp, [])
            xck.crosscheck(env["state"], Config(**NOABORT))
        verdict = _read_verdict(env["workdir"], 0)
        assert verdict["sub_rejected"]["reason"] == "below_align"
        assert verdict["decision"] == "keep"

    def test_subtitle_keywords_feed_granite_prompt(self, env):
        # The sub covers no segment but supplies terminology for Granite (R36)
        env["state"]["segments"] = [
            Segment(index=0, start=5.0, end=9.0, text_src="plain matching text"),
        ]
        _with_subs(env, "1\n00:00:00,000 --> 00:00:04,000\n"
                        "Kubernetes orchestrates pods. Kubernetes scales well.\n")
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {0: "plain matching text"})
            _gemma(mp, [])
            xck.crosscheck(env["state"], Config(**NOABORT))
        assert "Kubernetes" in cap["prompt"]

    def test_subtitle_edit_recomputes_covered_segment(self, env):
        env["state"]["segments"] = [
            Segment(index=0, start=0.0, end=4.0,
                    text_src="we deploy the model to production today"),
        ]
        path = _with_subs(env, "1\n00:00:00,000 --> 00:00:04,000\n"
                               "we deploy the model to production today\n")
        with pytest.MonkeyPatch.context() as mp:
            _granite(mp, {})
            _gemma(mp, [])
            r1 = xck.crosscheck(env["state"], Config(**NOABORT))
            assert r1["segments"][0].text_src == "we deploy the model to production today"

            # Edit the sidecar-derived cue: the covered segment must re-derive
            path.write_text("1\n00:00:00,000 --> 00:00:04,000\n"
                            "we deploy the model to staging today\n", encoding="utf-8")
            env["state"]["segments"] = [
                Segment(index=0, start=0.0, end=4.0,
                        text_src="we deploy the model to production today"),
            ]
            r2 = xck.crosscheck(env["state"], Config(**NOABORT))
        assert r2["segments"][0].text_src == "we deploy the model to staging today"


class TestPassthrough:
    def test_flag_off_is_passthrough(self, env):
        with pytest.MonkeyPatch.context() as mp:
            cap = _granite(mp, {})
            calls = _gemma(mp, [])
            result = xck.crosscheck(env["state"], Config(enable_cross_check=False))
        assert cap["calls"] == 0 and calls == []
        assert result["segments"][0].text_src == "we use cooper netties here"
        manifest = json.loads((env["workdir"] / "crosscheck" / "segments.json").read_text())
        assert manifest["passthrough"] is True
