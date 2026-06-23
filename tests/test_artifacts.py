import json
import os

import pytest

from loro.config import Config
from loro.harness import artifacts
from loro.harness.artifacts import LockError, WorkdirLock
from loro.harness.retry import StageError
from loro.nodes import ingest as ingest_mod
from loro.nodes import vision as vision_mod
from loro.nodes import voice as voice_mod
from loro.state import Segment


def _build_writer(payload: bytes, calls: list):
    def build(tmp_path):
        calls.append(tmp_path)
        tmp_path.write_bytes(payload)
    return build


class TestFingerprint:
    def test_stable_under_key_order(self):
        a = artifacts.fingerprint({"x": 1, "y": [1, 2], "z": {"k": "v"}})
        b = artifacts.fingerprint({"z": {"k": "v"}, "y": [1, 2], "x": 1})
        assert a == b

    def test_any_value_change_changes_it(self):
        base = {"x": 1, "y": [1, 2], "z": {"k": "v"}}
        ref = artifacts.fingerprint(base)
        assert artifacts.fingerprint({**base, "x": 2}) != ref
        assert artifacts.fingerprint({**base, "y": [1, 3]}) != ref
        assert artifacts.fingerprint({**base, "z": {"k": "w"}}) != ref

    def test_unicode_stable(self):
        assert artifacts.fingerprint({"t": "tiếng Việt"}) == artifacts.fingerprint({"t": "tiếng Việt"})


class TestProduce:
    def test_computes_then_reuses(self, tmp_path):
        art = tmp_path / "stage" / "out.json"
        inputs = {"a": 1}
        calls = []
        loaded = artifacts.produce(art, inputs, "stage", _build_writer(b'{"v": 1}', calls))
        assert loaded is False
        assert len(calls) == 1
        assert json.loads(art.read_text()) == {"v": 1}

        # AE1 (mechanism level): valid artifact -> compute must not be called
        loaded = artifacts.produce(art, inputs, "stage", _build_writer(b'{"v": 2}', calls))
        assert loaded is True
        assert len(calls) == 1
        assert json.loads(art.read_text()) == {"v": 1}

    def test_missing_sidecar_recomputes(self, tmp_path):
        art = tmp_path / "out.bin"
        calls = []
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"x", calls))
        artifacts.meta_path(art).unlink()
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"x", calls))
        assert len(calls) == 2

    def test_fingerprint_mismatch_recomputes(self, tmp_path):
        art = tmp_path / "out.bin"
        calls = []
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"x", calls))
        artifacts.produce(art, {"a": 2}, "s", _build_writer(b"y", calls))
        assert len(calls) == 2
        assert art.read_bytes() == b"y"

    def test_tampered_output_recomputes(self, tmp_path):
        # Hand-edited / torn artifact: output hash no longer matches sidecar (R16)
        art = tmp_path / "out.bin"
        calls = []
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"x", calls))
        art.write_bytes(b"tampered")
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"x", calls))
        assert len(calls) == 2
        assert art.read_bytes() == b"x"

    def test_artifact_without_sidecar_is_invalid(self, tmp_path):
        # Simulated crash between rename and sidecar write
        art = tmp_path / "out.bin"
        art.write_bytes(b"partial")
        assert not artifacts.is_valid(art, {"a": 1})

    def test_leftover_tmp_file_never_loaded(self, tmp_path):
        art = tmp_path / "out.bin"
        calls = []

        def crashing_build(tmp):
            tmp.write_bytes(b"half-written")
            raise RuntimeError("crash mid-compute")

        with pytest.raises(RuntimeError):
            artifacts.produce(art, {"a": 1}, "s", crashing_build)
        assert not art.exists()
        assert not artifacts.is_valid(art, {"a": 1})

        # Recovery run computes cleanly despite leftover tmp debris
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"good", calls))
        assert art.read_bytes() == b"good"

    def test_sidecar_contents(self, tmp_path):
        art = tmp_path / "out.bin"
        artifacts.produce(art, {"a": 1}, "mystage", _build_writer(b"x", []))
        meta = json.loads(artifacts.meta_path(art).read_text())
        assert meta["stage"] == "mystage"
        assert meta["input_fingerprint"] == artifacts.fingerprint({"a": 1})
        assert meta["output_sha256"] == artifacts.file_sha256(art)
        assert "written_at" in meta


class TestCachedSha:
    def test_served_from_sidecar_without_reading_content(self, tmp_path, monkeypatch):
        art = tmp_path / "big.wav"
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"audio-bytes", []))
        expected = artifacts.file_sha256(art)

        def no_read(path):
            raise AssertionError("content should not be re-read")

        monkeypatch.setattr(artifacts, "file_sha256", no_read)
        assert artifacts.cached_file_sha256(art) == expected

    def test_falls_back_to_hashing_for_plain_files(self, tmp_path):
        plain = tmp_path / "plain.txt"
        plain.write_bytes(b"no sidecar here")
        assert artifacts.cached_file_sha256(plain) == artifacts.file_sha256(plain)

    def test_modified_artifact_rehashed(self, tmp_path):
        art = tmp_path / "f.bin"
        artifacts.produce(art, {"a": 1}, "s", _build_writer(b"v1", []))
        art.write_bytes(b"v2-different")
        assert artifacts.cached_file_sha256(art) == artifacts.file_sha256(art)


class TestProduceJson:
    def test_roundtrip_and_cache(self, tmp_path):
        art = tmp_path / "seg.json"
        calls = []

        def compute():
            calls.append(1)
            return {"text": "xin chào", "n": 3}

        data = artifacts.produce_json(art, {"k": 1}, "s", compute)
        assert data == {"text": "xin chào", "n": 3}
        data = artifacts.produce_json(art, {"k": 1}, "s", compute)
        assert data == {"text": "xin chào", "n": 3}
        assert len(calls) == 1


class TestLock:
    def test_acquire_and_release(self, tmp_path):
        lock = WorkdirLock(tmp_path)
        with lock:
            assert (tmp_path / "run.lock").exists()
        assert not (tmp_path / "run.lock").exists()

    def test_second_acquire_live_pid_fails(self, tmp_path):
        with WorkdirLock(tmp_path):
            with pytest.raises(LockError):
                WorkdirLock(tmp_path).acquire()

    def test_stale_lock_of_dead_pid_is_broken(self, tmp_path):
        # Find a pid that is certainly dead: spawn-and-reap is overkill; use a
        # huge pid unlikely to exist, verified dead via os.kill probe.
        dead_pid = 99999999
        with pytest.raises((ProcessLookupError, OverflowError)):
            os.kill(dead_pid, 0)
        (tmp_path / "run.lock").write_text(json.dumps({"pid": dead_pid}))
        lock = WorkdirLock(tmp_path)
        lock.acquire()  # must break the stale lock and succeed
        assert json.loads((tmp_path / "run.lock").read_text())["pid"] == os.getpid()
        lock.release()

    def test_corrupt_lockfile_is_broken(self, tmp_path):
        (tmp_path / "run.lock").write_text("garbage")
        lock = WorkdirLock(tmp_path)
        lock.acquire()
        lock.release()


class TestIngestArtifacts:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00" * 256)
        calls = []

        def fake_extract(src, out, rate=16000, channels=1):
            calls.append((rate, channels))
            from pathlib import Path
            Path(out).write_bytes(f"wav-{rate}-{channels}".encode())

        monkeypatch.setattr(ingest_mod.ffmpeg, "extract_audio", fake_extract)
        monkeypatch.setattr(ingest_mod.ffmpeg, "probe_duration", lambda p: 12.5)
        state = {"video_path": str(video), "workdir": str(tmp_path / "work")}
        return {"video": video, "calls": calls, "state": state}

    def test_rerun_does_not_recall_ffmpeg(self, env):
        result = ingest_mod.ingest(env["state"], Config())
        assert result["video_duration"] == 12.5
        assert len(env["calls"]) == 2  # 16k mono + 44.1k stereo

        ingest_mod.ingest(env["state"], Config())
        assert len(env["calls"]) == 2  # cached, no new extraction

    def test_video_mtime_change_recomputes(self, env):
        ingest_mod.ingest(env["state"], Config())
        os.utime(env["video"], (1, 1))  # touch mtime
        ingest_mod.ingest(env["state"], Config())
        assert len(env["calls"]) == 4


class TestVisionArtifacts:
    @pytest.fixture
    def env(self, tmp_path, monkeypatch):
        video = tmp_path / "in.mp4"
        video.write_bytes(b"\x00" * 256)
        frame = tmp_path / "frame_001.jpg"
        frame.write_bytes(b"jpeg")
        monkeypatch.setattr(vision_mod.ffmpeg, "extract_frames",
                            lambda *a, **kw: [frame])
        monkeypatch.setattr(vision_mod.llm, "image_part", lambda p, *a, **k: {"type": "image_url"})
        state = {"video_path": str(video), "workdir": str(tmp_path / "work"),
                 "video_duration": 12.5}
        return {"state": state, "workdir": tmp_path / "work"}

    def test_degraded_artifact_durable_and_reported(self, env, monkeypatch):
        chat_calls = []

        def failing_chat(cfg, messages, **kw):
            chat_calls.append(1)
            raise StageError("vision", "infra", "connection")

        monkeypatch.setattr(vision_mod.llm, "chat", failing_chat)
        result = vision_mod.vision(env["state"], Config())
        assert result["video_context"] == ""
        data = json.loads((env["workdir"] / "vision" / "context.json").read_text())
        assert data["degraded"] is True
        assert data["reason"] == "infra/connection"

        # Rerun: degraded artifact is valid -> oMLX not called again (R19)
        vision_mod.vision(env["state"], Config())
        assert len(chat_calls) == 1

        # Deleting the artifact retries
        (env["workdir"] / "vision" / "context.json").unlink()
        vision_mod.vision(env["state"], Config())
        assert len(chat_calls) == 2

    def test_success_cached(self, env, monkeypatch):
        chat_calls = []

        def ok_chat(cfg, messages, **kw):
            chat_calls.append(1)
            return "a tech talk"

        monkeypatch.setattr(vision_mod.llm, "chat", ok_chat)
        assert vision_mod.vision(env["state"], Config())["video_context"] == "a tech talk"
        assert vision_mod.vision(env["state"], Config())["video_context"] == "a tech talk"
        assert len(chat_calls) == 1

    def test_keywords_line_parsed_into_state(self, env, monkeypatch):
        reply = ("A talk about container orchestration.\n"
                 "KEYWORDS: Kubernetes; CI/CD; transfer learning")
        monkeypatch.setattr(vision_mod.llm, "chat", lambda *a, **kw: reply)
        result = vision_mod.vision(env["state"], Config())
        assert result["video_keywords"] == ["Kubernetes", "CI/CD", "transfer learning"]
        assert result["video_context"] == "A talk about container orchestration."
        data = json.loads((env["workdir"] / "vision" / "context.json").read_text())
        assert data["keywords"] == ["Kubernetes", "CI/CD", "transfer learning"]
        assert data["degraded"] is False

    def test_missing_keywords_line_is_not_degraded(self, env, monkeypatch):
        monkeypatch.setattr(vision_mod.llm, "chat",
                            lambda *a, **kw: "Just a description, no list.")
        result = vision_mod.vision(env["state"], Config())
        assert result["video_keywords"] == []
        assert result["video_context"] == "Just a description, no list."
        data = json.loads((env["workdir"] / "vision" / "context.json").read_text())
        assert data["degraded"] is False

    def test_degraded_vision_yields_empty_keywords(self, env, monkeypatch):
        def failing_chat(cfg, messages, **kw):
            raise StageError("vision", "infra", "connection")

        monkeypatch.setattr(vision_mod.llm, "chat", failing_chat)
        result = vision_mod.vision(env["state"], Config())
        assert result["video_keywords"] == []


class TestVoiceRefArtifacts:
    def test_preset_content_change_invalidates(self, tmp_path, monkeypatch):
        preset = tmp_path / "ref.wav"
        preset.write_bytes(b"voice-v1")
        state = {"workdir": str(tmp_path / "work")}
        # Cloning engine: voice_ref extracts/persists a reference clip (the
        # default engine is now soniox, which casts preset voices instead).
        cfg = Config(tts_engine="vieneu", ref_audio=str(preset), ref_text="hello")

        builds = []
        original_produce_json = artifacts.produce_json

        def counting_produce_json(art, inputs, stage, compute):
            def counted():
                builds.append(1)
                return compute()
            return original_produce_json(art, inputs, stage, counted)

        monkeypatch.setattr(voice_mod.artifacts, "produce_json", counting_produce_json)
        voice_mod.voice_ref(state, cfg)
        voice_mod.voice_ref(state, cfg)
        assert len(builds) == 1  # cached on identical content

        preset.write_bytes(b"voice-v2-different")  # same path, new content
        voice_mod.voice_ref(state, cfg)
        assert len(builds) == 2

    def test_auto_pick_requires_usable_segment(self, tmp_path):
        state = {
            "workdir": str(tmp_path),
            "audio_16k": str(tmp_path / "a.wav"),
            "segments": [Segment(index=0, start=0.0, end=0.5, text_src="hi")],
        }
        with pytest.raises(RuntimeError, match="ref-audio"):
            voice_mod.voice_ref(state, Config(tts_engine="vieneu"))
