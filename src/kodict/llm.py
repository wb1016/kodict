"""OpenRouter chat-completions streaming client (SSE).

See api-docs/api.md: POST /api/v1/chat/completions with ``stream: true``,
provider routing via the ``provider`` object (here: fp8-only), usage reported
once in the final chunk before ``data: [DONE]``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import httpx

from kodict.config import ModelConfig

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# OpenRouter app attribution headers (optional but nice)
HEADERS = {
    "HTTP-Referer": "https://github.com/wb1016/kodict",
    "X-Title": "kodict",
}

# status codes worth retrying with backoff
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
# status codes that indicate a broken setup, not a transient problem
_FATAL_STATUS = {400, 401, 402, 403, 404}


class LLMError(Exception):
    def __init__(self, message: str, status: int | None = None,
                 retryable: bool = False, fatal: bool = False,
                 detail: dict | None = None):
        super().__init__(message)
        self.status = status
        self.retryable = retryable
        self.fatal = fatal
        self.detail = detail or {}  # stream diagnostics for the meta sidecar


@dataclass
class StreamResult:
    text: str = ""
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    provider: str | None = None
    events: int = 0  # number of content deltas (for the blinker stats)
    reasoning_text: str = ""
    salvaged_from_reasoning: bool = False  # content was empty, reasoning used
    errors: list[str] = field(default_factory=list)

    @property
    def reasoning_chars(self) -> int:
        return len(self.reasoning_text)

    def detail(self) -> dict:
        """Stream diagnostics, attached to errors and meta sidecars."""
        return {
            "finish_reason": self.finish_reason,
            "native_finish_reason": self.native_finish_reason,
            "provider": self.provider,
            "events": self.events,
            "reasoning_chars": self.reasoning_chars,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
        }


async def stream_chat(
    client: httpx.AsyncClient,
    model: ModelConfig,
    messages: list[dict],
    api_key: str,
    on_token=None,
    on_activity=None,
) -> StreamResult:
    """Stream one chat completion.

    ``on_token(text)`` fires per content delta (blinker + parser);
    ``on_activity(text)`` fires per reasoning delta (blinker only, so
    thinking phases visibly blink too without feeding the parser).
    """
    payload = {
        "model": model.name,
        "messages": messages,
        "stream": True,
        "temperature": model.temperature,
        "max_tokens": model.max_tokens,
        "provider": {
            "quantizations": list(model.quantizations),
            "allow_fallbacks": model.allow_fallbacks,
            # cheapest fp8 endpoints first; fallbacks still walk the list
            **({"sort": model.sort} if model.sort else {}),
        },
        # thinking must stay off unless explicitly requested: some fp8
        # providers answer entirely inside `reasoning`, leaving `content` empty
        "reasoning": {"enabled": model.reasoning_enabled},
    }
    if model.frequency_penalty:
        payload["frequency_penalty"] = model.frequency_penalty
    if model.repetition_penalty != 1.0:
        payload["repetition_penalty"] = model.repetition_penalty
    headers = {**HEADERS, "Authorization": f"Bearer {api_key}",
               "Content-Type": "application/json"}
    result = StreamResult()

    try:
        async with client.stream(
            "POST", API_URL, json=payload, headers=headers,
            timeout=httpx.Timeout(60.0, connect=15.0),
        ) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode("utf-8", "replace")[:500]
                raise LLMError(
                    f"HTTP {resp.status_code}: {body}",
                    status=resp.status_code,
                    retryable=resp.status_code in _RETRYABLE_STATUS,
                    fatal=resp.status_code in _FATAL_STATUS,
                )
            async for line in resp.aiter_lines():
                if not line or line.startswith(":"):
                    continue  # heartbeat comment
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if isinstance(chunk, dict) and chunk.get("error"):
                    err = chunk["error"]
                    message = err.get("message", str(err)) if isinstance(err, dict) else str(err)
                    raise LLMError(f"stream error: {message}", retryable=True)
                _absorb(chunk, result, on_token, on_activity)
    except httpx.HTTPError as e:
        raise LLMError(f"network error: {e}", retryable=True) from e

    # A stream that ends without usable content is a provider problem, not a
    # data problem: surface it so the retry/backoff logic engages and the real
    # reason lands in the checkpoint instead of looking like "no entries".
    if result.errors:
        raise LLMError(
            f"provider error: {'; '.join(result.errors)} "
            f"(finish={result.finish_reason}, provider={result.provider})",
            retryable=True, detail=result.detail(),
        )
    if result.finish_reason in ("error", "content_filter"):
        raise LLMError(
            f"stream finished with {result.finish_reason} "
            f"(native={result.native_finish_reason}, provider={result.provider})",
            retryable=True, detail=result.detail(),
        )
    if not result.text.strip():
        if result.reasoning_text.strip():
            # Provider routed the whole answer into `reasoning` (thinking on
            # despite the flag). Salvage it: our parser only pulls complete
            # objects/words out, thinking prose yields nothing.
            result.text = result.reasoning_text
            result.reasoning_text = ""
            result.salvaged_from_reasoning = True
        else:
            raise LLMError(
                f"empty stream (finish={result.finish_reason}, "
                f"provider={result.provider}, reasoning_chars=0)",
                retryable=True, detail=result.detail(),
            )
    return result


def _absorb(chunk: dict, result: StreamResult, on_token, on_activity=None) -> None:
    for choice in chunk.get("choices") or ():
        delta = choice.get("delta") or {}
        content = delta.get("content")
        if content:
            result.text += content
            result.events += 1
            if on_token is not None:
                on_token(content)
        reasoning = delta.get("reasoning")
        if reasoning:
            result.reasoning_text += reasoning
            if on_activity is not None:
                on_activity(reasoning)  # thinking still blinks the job
        finish = choice.get("finish_reason")
        if finish:
            result.finish_reason = finish
        native = choice.get("native_finish_reason")
        if native:
            result.native_finish_reason = native
        error = choice.get("error")
        if error:
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            result.errors.append(message)
    usage = chunk.get("usage") or {}
    if usage:
        result.prompt_tokens = usage.get("prompt_tokens", result.prompt_tokens)
        result.completion_tokens = usage.get("completion_tokens", result.completion_tokens)
        result.cost = usage.get("cost", result.cost) or result.cost
    if chunk.get("provider"):
        result.provider = chunk["provider"]
