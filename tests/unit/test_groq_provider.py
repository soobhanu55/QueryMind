"""The Groq provider against a mocked HTTP transport: request shape, 429 back-off, token usage, error handling.
(No call to Groq itself is made; the real API has not been exercised.)"""
import asyncio
import json

import httpx
import pytest

from app.config import get_settings
from app.llm import groq_provider
from app.llm import router as r

OK_BODY = {
    "choices": [{"message": {"content": json.dumps({"sql": "SELECT 1", "confidence": 0.9, "explanation": "x"})}}],
    "usage": {"prompt_tokens": 812, "completion_tokens": 41},
}


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0))  # no real back-off in tests
    yield groq_provider.GroqProvider()
    get_settings.cache_clear()


def mock_transport(monkeypatch, handler):
    real = httpx.AsyncClient
    monkeypatch.setattr(groq_provider.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))


async def test_request_shape_and_token_usage(provider, monkeypatch):
    seen = {}

    def handler(request):
        seen["auth"], seen["body"] = request.headers["authorization"], json.loads(request.content)
        return httpx.Response(200, json=OK_BODY)

    mock_transport(monkeypatch, handler)
    out = await provider.generate("how many orders?", "schema")
    assert seen["auth"] == "Bearer test-key" and seen["body"]["temperature"] == 0 and seen["body"]["response_format"] == {"type": "json_object"}
    assert out.sql == "SELECT 1" and out.model == "llama-3.3-70b-versatile" and (out.input_tokens, out.output_tokens) == (812, 41)
    assert r.cost_usd(out.model, out.input_tokens, out.output_tokens) == pytest.approx((812 * 0.59 + 41 * 0.79) / 1e6, abs=1e-6)


async def test_rate_limit_is_retried_with_back_off(provider, monkeypatch):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429) if len(calls) < 3 else httpx.Response(200, json=OK_BODY)

    mock_transport(monkeypatch, handler)
    assert (await provider.generate("q", "s")).sql == "SELECT 1" and len(calls) == 3


async def test_a_server_error_raises_so_the_router_can_fall_back(provider, monkeypatch):
    mock_transport(monkeypatch, lambda request: httpx.Response(503))
    with pytest.raises(httpx.HTTPStatusError):
        await provider.generate("q", "s")


async def test_a_missing_key_fails_at_construction(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(RuntimeError):
        groq_provider.GroqProvider()
    get_settings.cache_clear()
