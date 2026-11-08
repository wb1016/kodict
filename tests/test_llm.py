"""Tests for the OpenRouter SSE streaming client (mocked transport)."""

import asyncio
import json

import httpx

from kodict.config import ModelConfig
from kodict.llm import LLMError, stream_chat

MODEL = ModelConfig()
KEY = "sk-test"


def sse(*chunks: str) -> bytes:
    parts = [f"data: {c}\n\n" for c in chunks]
    parts.append("data: [DONE]\n\n")
    return "".join(parts).encode()


def make_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def run_stream(handler, messages=None, on_token=None):
    async def go():
        async with make_client(handler) as client:
            return await stream_chat(client, MODEL, messages or [{"role": "user", "content": "hi"}],
                                     KEY, on_token=on_token)
    return asyncio.run(go())


def test_successful_stream():
    body = sse(
        json.dumps({"choices": [{"delta": {"role": "assistant", "content": ""}}]}),
        json.dumps({"choices": [{"delta": {"content": '{"w": "사과"'}}]}),
        json.dumps({
            "choices": [{"delta": {"content": ', "r": 1}\n'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
            "provider": "prime-intellect",
        }),
    )
    tokens: list[str] = []
    result = run_stream(lambda req: httpx.Response(200, content=body),
                        on_token=tokens.append)
    assert result.text == '{"w": "사과", "r": 1}\n'
    assert tokens == ['{"w": "사과"', ', "r": 1}\n']
    assert result.events == 2
    assert result.finish_reason == "stop"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 5
    assert result.cost == 0.001
    assert result.provider == "prime-intellect"


def test_request_has_fp8_provider_routing_and_auth():
    seen = {}

    body = sse(json.dumps({"choices": [{"delta": {"content": "hi\n"}}]}))

    def handler(req: httpx.Request) -> httpx.Response:
        seen["json"] = json.loads(req.content)
        seen["auth"] = req.headers.get("Authorization")
        return httpx.Response(200, content=body)

    run_stream(handler)
    assert seen["json"]["provider"] == {"quantizations": ["fp8"],
                                       "allow_fallbacks": True,
                                       "sort": "price"}
    assert seen["json"]["stream"] is True
    assert seen["json"]["model"] == MODEL.name
    assert "frequency_penalty" not in seen["json"]  # 0 = not sent (kills JSON structure)
    assert seen["json"]["repetition_penalty"] == 1.1
    assert seen["auth"] == "Bearer sk-test"


def test_429_is_retryable():
    def handler(req):
        return httpx.Response(429, text='{"error":{"message":"rate limited"}}')

    try:
        run_stream(handler)
    except LLMError as e:
        assert e.retryable and not e.fatal and e.status == 429
    else:
        raise AssertionError("expected LLMError")


def test_401_is_fatal():
    def handler(req):
        return httpx.Response(401, text="unauthorized")

    try:
        run_stream(handler)
    except LLMError as e:
        assert e.fatal and not e.retryable
    else:
        raise AssertionError("expected LLMError")


def test_502_is_retryable():
    def handler(req):
        return httpx.Response(502, text="upstream failed")

    try:
        run_stream(handler)
    except LLMError as e:
        assert e.retryable and not e.fatal
    else:
        raise AssertionError("expected LLMError")


def test_network_error_is_retryable():
    def handler(req):
        raise httpx.ConnectError("boom", request=req)

    try:
        run_stream(handler)
    except LLMError as e:
        assert e.retryable and "network" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_midstream_error_object_is_retryable():
    body = sse(json.dumps({"error": {"message": "provider exploded"}}))
    try:
        run_stream(lambda req: httpx.Response(200, content=body))
    except LLMError as e:
        assert e.retryable and "provider exploded" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_heartbeats_and_malformed_chunks_lead_to_empty_stream_error():
    body = b": keep-alive\n\ndata: not json\n\ndata: {\"choices\": []}\n\ndata: [DONE]\n\n"
    try:
        run_stream(lambda req: httpx.Response(200, content=body))
    except LLMError as e:
        assert e.retryable and "empty stream" in str(e)
    else:
        raise AssertionError("expected LLMError for an empty stream")


def test_choice_level_error_raises_provider_error():
    body = sse(json.dumps({"choices": [{"delta": {}, "error": {"message": "boom"}}]}))
    try:
        run_stream(lambda req: httpx.Response(200, content=body))
    except LLMError as e:
        assert e.retryable and "provider error" in str(e) and "boom" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_error_finish_reason_raises_with_diagnostics():
    body = sse(json.dumps({
        "choices": [{"delta": {}, "finish_reason": "error",
                     "native_finish_reason": "provider_exploded"}],
        "provider": "venice",
    }))
    try:
        run_stream(lambda req: httpx.Response(200, content=body))
    except LLMError as e:
        assert e.retryable
        assert "error" in str(e) and "provider_exploded" in str(e) and "venice" in str(e)
    else:
        raise AssertionError("expected LLMError")


def test_request_disables_reasoning_by_default():
    seen = {}

    def handler(req):
        seen["json"] = json.loads(req.content)
        return httpx.Response(200, content=sse(
            json.dumps({"choices": [{"delta": {"content": "hi\n"}}]})))

    run_stream(handler)
    assert seen["json"]["reasoning"] == {"enabled": False}


def test_reasoning_only_stream_is_salvaged():
    """Providers that answer inside `reasoning` get salvaged, not retried."""
    body = sse(json.dumps({
        "choices": [{"delta": {"reasoning": '{"w": "사과", "r": 1}\n'}},
                     {"delta": {"reasoning": '{"w": "학교", "r": 2}\n'},
                      "finish_reason": "stop"}],
        "provider": "CoreWeave",
    }))
    tokens, activity = [], []
    result = run_stream(lambda req: httpx.Response(200, content=body),
                        on_token=tokens.append)
    # on_activity is a separate kwarg; call directly for this unit test
    assert result.salvaged_from_reasoning
    assert result.text == '{"w": "사과", "r": 1}\n{"w": "학교", "r": 2}\n'
    assert tokens == []  # parser never saw content deltas mid-stream


def test_fully_empty_stream_raises_with_detail():
    body = sse(json.dumps({"choices": [{"delta": {}, "finish_reason": "stop"}],
                           "provider": "DeepInfra"}))
    try:
        run_stream(lambda req: httpx.Response(200, content=body))
    except LLMError as e:
        assert e.retryable and "empty stream" in str(e)
        assert e.detail["provider"] == "DeepInfra"
        assert e.detail["finish_reason"] == "stop"
    else:
        raise AssertionError("expected LLMError")
