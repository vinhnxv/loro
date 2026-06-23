import numpy as np
import pytest
import soundfile as sf

from loro.config import Config
from loro.harness import qa
from loro.harness.ledger import AbortRun, SkipLedger
from loro.harness.retry import StageError
from loro.nodes import tts as tts_mod
from loro.state import Segment


def _write_wav(path, seconds, sr=24000, amplitude=0.3, freq=440.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    sf.write(str(path), (amplitude * np.sin(2 * np.pi * freq * t)).astype("float32"), sr)


# "sáu âm tiết cho sáu giây" — 6 syllables ≈ 1.4s expected at 4.3 syl/s;
# scale up so the numbers match AE7's 6-second sentence
LONG_TEXT = " ".join(["âm"] * 26)  # ~6s of speech expected


class TestCheckClip:
    def test_short_clip_for_long_sentence_fails(self, tmp_path):
        # AE7: 0.3s wav for a ~6s sentence
        clip = tmp_path / "c.wav"
        _write_wav(clip, 0.3)
        with pytest.raises(StageError) as exc_info:
            qa.check_clip(clip, LONG_TEXT, Config())
        assert exc_info.value.signature == ("tts", "qa", "too_short")

    def test_absurdly_long_clip_fails(self, tmp_path):
        clip = tmp_path / "c.wav"
        _write_wav(clip, 30.0)
        with pytest.raises(StageError) as exc_info:
            qa.check_clip(clip, "ba âm tiết", Config())
        assert exc_info.value.code == "too_long"

    def test_all_silence_fails(self, tmp_path):
        clip = tmp_path / "c.wav"
        _write_wav(clip, 1.5, amplitude=0.0)
        with pytest.raises(StageError) as exc_info:
            qa.check_clip(clip, "xin chào các bạn nhé", Config())
        assert exc_info.value.code == "silent"

    def test_undecodable_fails(self, tmp_path):
        clip = tmp_path / "c.wav"
        clip.write_bytes(b"this is not a wav file at all")
        with pytest.raises(StageError) as exc_info:
            qa.check_clip(clip, "xin chào", Config())
        assert exc_info.value.code == "undecodable"

    def test_valid_clip_passes(self, tmp_path):
        clip = tmp_path / "c.wav"
        _write_wav(clip, 1.4)
        qa.check_clip(clip, "xin chào các bạn hôm nay", Config())  # no raise

    def test_digit_groups_count_spoken_syllables(self):
        # "22" is read "hai mươi hai" — 3 syllables, not 1
        assert qa.syllable_count("22. Regularization là gì?") >= 6
        assert qa.syllable_count("xin chào") == 2

    def test_short_numbered_question_passes_gate(self, tmp_path):
        # The smoke run's seg_0108: a 2.84s clip for "22. Regularization là gì?"
        # was rejected when "22." counted as one syllable
        clip = tmp_path / "c.wav"
        _write_wav(clip, 2.84)
        qa.check_clip(clip, "22. Regularization là gì?", Config())  # no raise


class TestCheckClipCps:
    # U5/R8: for a non-VI (CPS) profile the gate budgets in characters / the
    # profile CPS, not syllables; a silent/truncated clip still hits the floor.
    FR_TEXT = "Bonjour tout le monde, comment allez-vous aujourd'hui ?"  # ~55 chars

    def test_fr_over_slot_clip_rejected_by_cps_gate(self, tmp_path):
        # ~55 chars / 17 CPS ≈ 3.2s expected; a 30s clip is far too long.
        clip = tmp_path / "c.wav"
        _write_wav(clip, 30.0)
        with pytest.raises(StageError) as exc_info:
            qa.check_clip(clip, self.FR_TEXT, Config(target_lang="fr"))
        assert exc_info.value.code == "too_long"

    def test_fr_in_slot_clip_passes(self, tmp_path):
        clip = tmp_path / "c.wav"
        _write_wav(clip, 3.2)  # ≈ expected for ~55 chars at 17 CPS
        qa.check_clip(clip, self.FR_TEXT, Config(target_lang="fr"))  # no raise

    def test_fr_silent_clip_still_caught_by_floor(self, tmp_path):
        clip = tmp_path / "c.wav"
        _write_wav(clip, 3.2, amplitude=0.0)
        with pytest.raises(StageError) as exc_info:
            qa.check_clip(clip, self.FR_TEXT, Config(target_lang="fr"))
        assert exc_info.value.code == "silent"

    def test_fr_budget_differs_from_vi_syllable_budget(self):
        # The same text budgets very differently under CPS vs the VI syllable
        # model — proving the gate is profile-sourced, not syllable-hardcoded.
        fr_expected = (Config(target_lang="fr").language_profile.counter(self.FR_TEXT)
                       / Config(target_lang="fr").language_profile.rate)
        vi_expected = (Config().language_profile.counter(self.FR_TEXT)
                       / Config().language_profile.rate)
        assert abs(fr_expected - vi_expected) > 1.0


class _FakeHiggs:
    """Stands in for HiggsClient: synthesize() delegates to a test callable."""

    instances = []

    def __init__(self, cfg, ref_audio, ref_text):
        self.calls = []
        _FakeHiggs.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def synthesize(self, text, output, voice=None):
        self.calls.append(text)
        self.behavior(text, output)


@pytest.fixture
def env(tmp_path, monkeypatch):
    _FakeHiggs.instances = []
    # Client construction now lives on the higgs provider (U3), so patch the
    # client on the provider module, not the node module.
    from loro.providers.tts import higgs as higgs_provider
    monkeypatch.setattr(higgs_provider, "HiggsClient", _FakeHiggs)
    ref = tmp_path / "ref.wav"
    _write_wav(ref, 5.0)
    workdir = tmp_path / "work"
    workdir.mkdir()

    def make_state(texts):
        return {
            "workdir": str(workdir),
            "ref_audio": str(ref),
            "ref_text": "reference transcript",
            "segments": [
                Segment(index=i, start=i * 3.0, end=i * 3.0 + 2.0,
                        text_src=f"line {i}", text_target=t)
                for i, t in enumerate(texts)
            ],
        }

    # These tests exercise the engine-agnostic tts orchestration through the
    # patched Higgs client, so pin the engine to higgs (the default is vieneu).
    return {"make_state": make_state, "workdir": workdir, "ref": ref,
            "cfg": lambda **kw: Config(tts_engine="higgs", **kw)}


def _good_synth(text, output):
    _write_wav(output, len(text.split()) / 4.3)


class TestTtsNode:
    def test_content_hash_resynthesizes_only_changed_text(self, env):
        # AE2 (second half)
        _FakeHiggs.behavior = staticmethod(_good_synth)
        texts = ["xin chào các bạn", "hẹn gặp lại nhé", "cảm ơn nhiều lắm"]
        state = env["make_state"](texts)
        tts_mod.tts(state, env["cfg"]())
        assert len(_FakeHiggs.instances[-1].calls) == 3

        # Unchanged texts -> no Higgs calls at all
        state = env["make_state"](texts)
        tts_mod.tts(state, env["cfg"]())
        assert len(_FakeHiggs.instances[-1].calls) == 0

        # One text changes -> exactly that clip resynthesized
        texts[1] = "tạm biệt và hẹn gặp lại"
        state = env["make_state"](texts)
        tts_mod.tts(state, env["cfg"]())
        assert _FakeHiggs.instances[-1].calls == ["tạm biệt và hẹn gặp lại"]

    def test_garbage_clip_retried_then_skipped(self, env):
        # AE7 end-to-end at node level: always-too-short clips
        _FakeHiggs.behavior = staticmethod(lambda text, output: _write_wav(output, 0.05))
        state = env["make_state"]([LONG_TEXT])
        tts_mod.tts(state, env["cfg"](retry_attempts=2, abort_threshold=99))

        seg = state["segments"][0]
        assert seg.skipped is True
        assert seg.skip_reason == "too_short"
        assert len(_FakeHiggs.instances[-1].calls) == 2  # retried
        # No valid artifact recorded for the garbage clip
        art = env["workdir"] / "tts" / "seg_0000.wav"
        assert not art.exists() or not tts_mod.artifacts.is_valid(
            art, {"text_vi": LONG_TEXT})
        entry = SkipLedger(env["workdir"]).entries()["seg_0000"]
        assert entry["status"] == "skipped"

    def test_upstream_skipped_segment_not_synthesized(self, env):
        _FakeHiggs.behavior = staticmethod(_good_synth)
        state = env["make_state"](["xin chào các bạn"])
        state["segments"][0].skipped = True
        state["segments"][0].skip_reason = "translate_failed"
        tts_mod.tts(state, env["cfg"]())
        assert _FakeHiggs.instances[-1].calls == []

    def test_repeated_infra_errors_abort_run(self, env):
        # R5a end-to-end at node level: persistent 5xx -> AbortRun from ledger
        def synth_5xx(text, output):
            raise StageError("tts", "infra", "http_503")

        _FakeHiggs.behavior = staticmethod(synth_5xx)
        texts = [f"câu số {i} dài hơn" for i in range(5)]
        state = env["make_state"](texts)
        with pytest.raises(AbortRun):
            tts_mod.tts(state, env["cfg"](abort_threshold=3, retry_attempts=1))
