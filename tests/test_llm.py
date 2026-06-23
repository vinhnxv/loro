"""llm.chat hardening: a null/empty model response is a clean content-class
StageError, not a None.strip() crash (seen with the qat-4bit Gemma)."""

import pytest

from loro.config import Config
from loro.harness.retry import StageError
from loro.services import llm


def _fake_client(content):
    msg = type("Msg", (), {"content": content})()
    choice = type("Choice", (), {"message": msg})()
    completions = type("Completions", (), {
        "create": lambda self, **kw: type("Resp", (), {"choices": [choice]})()})()
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def test_null_content_raises_clean_error(monkeypatch):
    monkeypatch.setattr(llm, "client", lambda cfg, **kw: _fake_client(None))
    with pytest.raises(StageError) as exc_info:
        llm.chat(Config(), [{"role": "user", "content": "x"}], stage="crosscheck")
    assert exc_info.value.signature == ("crosscheck", "content", "empty_response")


def test_empty_content_raises_clean_error(monkeypatch):
    monkeypatch.setattr(llm, "client", lambda cfg, **kw: _fake_client("   "))
    with pytest.raises(StageError) as exc_info:
        llm.chat(Config(), [{"role": "user", "content": "x"}])
    assert exc_info.value.code == "empty_response"


def test_normal_content_returned_stripped(monkeypatch):
    monkeypatch.setattr(llm, "client", lambda cfg, **kw: _fake_client("  hello  "))
    assert llm.chat(Config(), [{"role": "user", "content": "x"}]) == "hello"


def _capturing_client(calls):
    """Records the kwargs of every create() call so tests can assert `model`."""
    msg = type("Msg", (), {"content": "ok"})()
    choice = type("Choice", (), {"message": msg})()

    def create(self, **kw):
        calls.append(kw)
        return type("Resp", (), {"choices": [choice]})()

    completions = type("Completions", (), {"create": create})()
    chat = type("Chat", (), {"completions": completions})()
    return type("Client", (), {"chat": chat})()


def test_chat_uses_passed_model(monkeypatch):
    # translate passes its own model so vision/crosscheck stay on Gemma (R37)
    calls = []
    monkeypatch.setattr(llm, "client", lambda cfg, **kw: _capturing_client(calls))
    llm.chat(Config(), [{"role": "user", "content": "x"}], model="qwen3-14b-4bit")
    assert calls[0]["model"] == "qwen3-14b-4bit"


def test_chat_defaults_model_to_gemma(monkeypatch):
    # Omitting model keeps every existing caller on llm_model (back-compat)
    calls = []
    monkeypatch.setattr(llm, "client", lambda cfg, **kw: _capturing_client(calls))
    cfg = Config()
    llm.chat(cfg, [{"role": "user", "content": "x"}])
    assert calls[0]["model"] == cfg.llm_model


def test_chat_disables_thinking_via_extra_body(monkeypatch):
    # KTD6: summary/translate calls turn off Qwen thinking via the chat template
    calls = []
    monkeypatch.setattr(llm, "client", lambda cfg, **kw: _capturing_client(calls))
    llm.chat(Config(), [{"role": "user", "content": "x"}], enable_thinking=False)
    assert calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_chat_thinking_on_by_default_sends_no_extra_body(monkeypatch):
    # Default path (Gemma vision/crosscheck) is untouched — no extra_body
    calls = []
    monkeypatch.setattr(llm, "client", lambda cfg, **kw: _capturing_client(calls))
    llm.chat(Config(), [{"role": "user", "content": "x"}])
    assert "extra_body" not in calls[0]


class TestImagePartSizeGuard:
    """image_part caps the per-image payload so an oversized frame degrades the
    shot/context as a content-class StageError instead of choking the server."""

    def test_oversized_image_raises_content_error(self, tmp_path):
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"x" * 100)
        with pytest.raises(StageError) as exc_info:
            llm.image_part(frame, max_bytes=10, stage="vision")
        assert exc_info.value.signature == ("vision", "content", "image_too_large")

    def test_within_limit_encodes(self, tmp_path):
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"jpeg-bytes")
        part = llm.image_part(frame, max_bytes=1024, stage="vision")
        assert part["type"] == "image_url"
        assert part["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_zero_disables_cap(self, tmp_path):
        # Back-compat default: no max_bytes means no guard, large file still sent.
        frame = tmp_path / "frame.jpg"
        frame.write_bytes(b"x" * 5000)
        part = llm.image_part(frame)
        assert part["type"] == "image_url"


class TestImageMaxBytesConfig:
    """LLM_IMAGE_MAX_BYTES is env-overridable; default is 8 MiB."""

    def test_defaults_to_8_mib(self, monkeypatch):
        monkeypatch.delenv("LLM_IMAGE_MAX_BYTES", raising=False)
        assert Config().llm_image_max_bytes == 8 * 1024 * 1024

    def test_reads_env(self, monkeypatch):
        monkeypatch.setenv("LLM_IMAGE_MAX_BYTES", "0")
        assert Config().llm_image_max_bytes == 0


class TestAudioModelConfig:
    """U1: the third Gemma role (audio) is its own field that defaults to
    llm_model, so the old single-multimodal-model profile is unchanged (R3)."""

    def _clean_env(self, monkeypatch):
        for name in ("LLM_MODEL_AUDIO", "LLM_MODEL", "LLM_TIMEOUT"):
            monkeypatch.delenv(name, raising=False)

    def test_audio_model_defaults_to_gemma(self, monkeypatch):
        # Empty config: audio routes onto the same model as before (R3).
        self._clean_env(monkeypatch)
        cfg = Config()
        assert cfg.llm_model_audio == cfg.llm_model

    def test_audio_model_fallback_honors_gemma_kwarg(self, monkeypatch):
        # Fallback lives in __post_init__, so a llm_model kwarg (not just the
        # LLM_MODEL env) is honored — audio follows it when LLM_MODEL_AUDIO unset.
        self._clean_env(monkeypatch)
        cfg = Config(llm_model="vision-26B")
        assert cfg.llm_model_audio == "vision-26B"

    def test_audio_model_env_independent_of_gemma(self, monkeypatch):
        # The split profile: LLM_MODEL_AUDIO (12B) is decoupled from llm_model.
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LLM_MODEL_AUDIO", "gemma-4-12B")
        cfg = Config(llm_model="gemma-4-26B-A4B")
        assert cfg.llm_model_audio == "gemma-4-12B"
        assert cfg.llm_model == "gemma-4-26B-A4B"


class TestPerRoleEndpoints:
    """Each role (vision/translate/seg/audio) resolves a (host, api_key, model);
    an empty override inherits the base LLM_HOST/LLM_API_KEY/LLM_MODEL so the
    single-host profile is unchanged, and a set override splits that role onto
    its own host (so a one-model-resident router never swaps)."""

    def _clean_env(self, monkeypatch):
        for name in ("LLM_HOST", "LLM_API_KEY", "LLM_MODEL"):
            monkeypatch.delenv(name, raising=False)
        for role in ("VISION", "TRANSLATE", "SEG", "AUDIO"):
            for kind in ("HOST", "API_KEY", "MODEL"):
                monkeypatch.delenv(f"LLM_{kind}_{role}", raising=False)

    def test_roles_default_to_base(self, monkeypatch):
        self._clean_env(monkeypatch)
        cfg = Config(llm_host="http://base/v1", llm_api_key="k", llm_model="m")
        for role in ("vision", "translate", "seg", "audio"):
            assert getattr(cfg, f"llm_host_{role}") == "http://base/v1"
            assert getattr(cfg, f"llm_api_key_{role}") == "k"
            assert getattr(cfg, f"llm_model_{role}") == "m"

    def test_audio_host_split_is_independent(self, monkeypatch):
        # The motivating case: audio on its own always-hot host, the rest on the
        # remote router — so the router only ever serves the vision model.
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LLM_HOST", "http://remote:8080/v1")
        monkeypatch.setenv("LLM_HOST_AUDIO", "http://127.0.0.1:1234/v1")
        monkeypatch.setenv("LLM_MODEL_AUDIO", "gemma-4-12B-it-8bit")
        cfg = Config(llm_model="gemma-4-26B-A4B")
        assert cfg.llm_host_audio == "http://127.0.0.1:1234/v1"
        assert cfg.llm_model_audio == "gemma-4-12B-it-8bit"
        # Vision/translate/seg stay on the remote with the base model.
        assert cfg.llm_host_vision == "http://remote:8080/v1"
        assert cfg.llm_host_translate == "http://remote:8080/v1"
        assert cfg.llm_model_vision == "gemma-4-26B-A4B"

    def test_per_role_api_key_override(self, monkeypatch):
        self._clean_env(monkeypatch)
        monkeypatch.setenv("LLM_API_KEY", "base-key")
        monkeypatch.setenv("LLM_API_KEY_AUDIO", "audio-key")
        cfg = Config()
        assert cfg.llm_api_key_audio == "audio-key"
        assert cfg.llm_api_key_vision == "base-key"


class TestLlmRole:
    """U7: cfg.llm_role(name) resolves a role to its (host, api_key, model), and
    llm.chat/list_models accept a `role` descriptor that supplies all three in one
    argument — identical to passing the kwargs explicitly, so no behavior change."""

    def _clean_env(self, monkeypatch):
        for name in ("LLM_HOST", "LLM_API_KEY", "LLM_MODEL"):
            monkeypatch.delenv(name, raising=False)
        for role in ("VISION", "TRANSLATE", "SEG", "AUDIO"):
            for kind in ("HOST", "API_KEY", "MODEL"):
                monkeypatch.delenv(f"LLM_{kind}_{role}", raising=False)

    def test_role_resolves_overridden_triple(self, monkeypatch):
        self._clean_env(monkeypatch)
        cfg = Config(llm_host="http://base/v1", llm_api_key="k", llm_model="m",
                     llm_model_translate="qwen")
        r = cfg.llm_role("translate")
        assert (r.host, r.api_key, r.model) == ("http://base/v1", "k", "qwen")

    def test_role_falls_back_to_base_when_unset(self, monkeypatch):
        # Parity with __post_init__: an empty override inherits the base triple.
        self._clean_env(monkeypatch)
        cfg = Config(llm_host="http://base/v1", llm_api_key="k", llm_model="m")
        for name in ("vision", "translate", "seg", "audio"):
            assert tuple(cfg.llm_role(name)) == ("http://base/v1", "k", "m")

    def test_chat_role_supplies_model_host_and_key(self, monkeypatch):
        self._clean_env(monkeypatch)
        seen, calls = {}, []

        def fake_client(cfg, **kw):
            seen.update(kw)
            return _capturing_client(calls)

        monkeypatch.setattr(llm, "client", fake_client)
        cfg = Config(llm_host_audio="http://audio/v1", llm_api_key_audio="ak",
                     llm_model_audio="gemma-12b")
        llm.chat(cfg, [{"role": "user", "content": "x"}], role=cfg.llm_role("audio"))
        assert calls[0]["model"] == "gemma-12b"
        assert seen["host"] == "http://audio/v1"
        assert seen["api_key"] == "ak"

    def test_list_models_role_supplies_host_and_key(self, monkeypatch):
        self._clean_env(monkeypatch)
        seen = {}

        def fake_client(cfg, **kw):
            seen.update(kw)
            models = type("Models", (), {"list": lambda self: type("R", (), {"data": []})()})()
            return type("Client", (), {"models": models})()

        monkeypatch.setattr(llm, "client", fake_client)
        cfg = Config(llm_host_vision="http://vis/v1", llm_api_key_vision="vk")
        llm.list_models(cfg, role=cfg.llm_role("vision"))
        assert seen["host"] == "http://vis/v1"
        assert seen["api_key"] == "vk"

    def test_role_overrides_explicit_kwargs(self, monkeypatch):
        # When both role= and explicit model=/host=/api_key= are passed, role wins
        # (KTD7) — the descriptor is the single source for the triple.
        self._clean_env(monkeypatch)
        seen, calls = {}, []

        def fake_client(cfg, **kw):
            seen.update(kw)
            return _capturing_client(calls)

        monkeypatch.setattr(llm, "client", fake_client)
        cfg = Config(llm_host_audio="http://audio/v1", llm_api_key_audio="ak",
                     llm_model_audio="audio-model")
        llm.chat(cfg, [{"role": "user", "content": "x"}],
                 model="explicit-model", host="http://explicit/v1", api_key="ek",
                 role=cfg.llm_role("audio"))
        assert calls[0]["model"] == "audio-model"
        assert seen["host"] == "http://audio/v1"
        assert seen["api_key"] == "ak"


class TestOmlxTimeoutConfig:
    """KTD5: llm_timeout is env-overridable for remote cold-load calibration."""

    def test_timeout_defaults_to_180(self, monkeypatch):
        monkeypatch.delenv("LLM_TIMEOUT", raising=False)
        assert Config().llm_timeout == 180.0

    def test_timeout_reads_env(self, monkeypatch):
        monkeypatch.setenv("LLM_TIMEOUT", "420")
        assert Config().llm_timeout == 420.0

    def test_timeout_garbage_raises(self, monkeypatch):
        # A non-numeric value is a clear, immediate error (like vieneu_temperature),
        # not a silent fallback that masks a typo in .env.
        monkeypatch.setenv("LLM_TIMEOUT", "soon")
        with pytest.raises(ValueError):
            Config()
