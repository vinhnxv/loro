"""Client for the OpenAI-compatible model server (oMLX or llama.cpp).

All calls go through the harness retry policy (R3); the SDK's own retries are
disabled so backoff is governed in exactly one place.
"""

import base64
import json
import re
from pathlib import Path

from openai import OpenAI

from loro.config import Config, LlmRole
from loro.harness.retry import CONTENT, StageError, with_retry


def client(cfg: Config, timeout: float | None = None,
           host: str | None = None, api_key: str | None = None) -> OpenAI:
    # host/api_key override the base endpoint so a caller can target a per-role
    # host (LLM_HOST_<ROLE>); None falls back to the base (LLM_HOST).
    return OpenAI(
        base_url=host or cfg.llm_host,
        api_key=api_key or cfg.llm_api_key,
        timeout=timeout if timeout is not None else cfg.llm_timeout,
        max_retries=0,
    )


def chat(
    cfg: Config,
    messages: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 2048,
    stage: str = "llm",
    model: str | None = None,
    host: str | None = None,
    api_key: str | None = None,
    enable_thinking: bool = True,
    role: LlmRole | None = None,
) -> str:
    # `model`/`host`/`api_key` default to the base endpoint; each role passes its
    # own (e.g. translate -> cfg.llm_model_translate on cfg.llm_host_translate,
    # audio -> cfg.llm_model_audio on cfg.llm_host_audio) so roles can live on
    # separate hosts (R37, KTD1). A `role` descriptor supplies all three in one
    # argument (KTD7); the explicit kwargs remain for back-compat.
    if role is not None:
        model, host, api_key = role.model, role.host, role.api_key

    def call() -> str:
        extra: dict = {}
        target = host or cfg.llm_host
        is_ollama = "ollama.com" in (target or "").lower()
        if not enable_thinking:
            # Qwen-style models leak <think> blocks that burn max_tokens and can
            # truncate / confuse extract_json. Disable thinking via the chat
            # template (KTD6). The exact mechanism is build-dependent and is
            # verified against the live oMLX at impl time (/no_think in the
            # prompt is the documented fallback); Gemma ignores the unknown
            # template kwarg, so this is safe on the default path.
            # Ollama Cloud does NOT honor chat_template_kwargs (an oMLX/llama.cpp
            # mechanism; an unknown chat_template_kwargs there makes the request
            # hang). Its native toggle extra_body={"think": False} is accepted
            # but, empirically on qwen3.5:397b, BACKFIRES — it consumes MORE
            # completion tokens (1305 vs 270 with no toggle) while still
            # emitting clean content (the think block is not leaked into content
            # on this serving, it is burned invisibly). So for ollama.com we send
            # NO extra_body (fewest hidden tokens, clean content) and instead
            # floor max_tokens below to absorb the hidden burn. Local oMLX/
            # llama.cpp keeps the chat_template_kwargs path. Generalize by
            # adding another host here if a different cloud provider needs its
            # own toggle.
            if not is_ollama:
                extra["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        eff_max_tokens = max_tokens
        if is_ollama:
            # qwen3.5:397b on Ollama Cloud burns 270-1300+ invisible completion
            # tokens per call (the think toggle cannot disable it). Floor the
            # budget so small-node calls (context=160, seg_visual=256,
            # vision=512) do not hit finish=length with empty content. The
            # actual answer is small; the floor only absorbs the hidden burn.
            eff_max_tokens = max(max_tokens, 2048)
        response = client(cfg, host=host, api_key=api_key).chat.completions.create(
            model=model or cfg.llm_model,
            messages=messages,
            temperature=temperature,
            max_tokens=eff_max_tokens,
            **extra,
        )
        # Some quantized models occasionally return null content; surface it
        # as a clean content-class error rather than crashing on None.strip().
        content = response.choices[0].message.content
        if not content or not content.strip():
            raise StageError(stage, "content", "empty_response",
                             "model returned no content")
        return content.strip()

    # Retry empty_response in addition to infra: a nondeterministic LLM
    # (e.g. qwen3.5 on Ollama Cloud) intermittently returns empty content for a
    # structured call, and retrying the same prompt commonly yields a non-empty
    # reply. The StageError keeps class=content for ledger/abort semantics
    # (ACCEPTABLE_CLASSES), so this only adds retry chances, it does not
    # reclassify the failure as infra.
    return with_retry(stage, call, attempts=cfg.retry_attempts,
                      base_delay=cfg.retry_base_delay,
                      retry_codes=("empty_response",))


def list_models(cfg: Config, timeout: float | None = None,
                host: str | None = None, api_key: str | None = None,
                role: LlmRole | None = None) -> list[str]:
    # A `role` descriptor supplies host/api_key in one argument (KTD7), mirroring
    # chat(); the model field is unused here (list_models enumerates the host).
    if role is not None:
        host, api_key = role.host, role.api_key
    return [m.id for m in client(cfg, timeout=timeout, host=host,
                                 api_key=api_key).models.list().data]


def image_part(path: Path, max_bytes: int = 0, stage: str = "llm") -> dict:
    # max_bytes <= 0 disables the cap (back-compat default). An oversized frame
    # is a content-class StageError so the caller degrades that shot/context
    # rather than letting an outsized request reach — and choke — the server.
    raw = path.read_bytes()
    if max_bytes > 0 and len(raw) > max_bytes:
        raise StageError(stage, CONTENT, "image_too_large",
                         f"{path.name}: {len(raw)} bytes > {max_bytes}")
    data = base64.b64encode(raw).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{data}"}}


def audio_part(path: Path) -> dict:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}


def extract_json(text: str):
    """Parse the first JSON array/object in a model reply, tolerating code fences."""
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    start = min((i for i in (text.find("["), text.find("{")) if i >= 0), default=-1)
    if start < 0:
        raise ValueError(f"no JSON found in model reply: {text[:200]!r}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj
