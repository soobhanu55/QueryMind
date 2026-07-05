"""
Cache layer for two hot paths that would otherwise repeat expensive work under load:

1. Schema-relevance lookups (schema_store.retrieve_relevant_tables + render_schema_text)
   -- pure CPU/string work today, but stands in for a future embedding-similarity
   lookup and is on the critical path of every request.
2. Question -> generated SQL results (keyed on a normalized hash of the question).
   This is the "cache repeated question embeddings" requirement: instead of paying
   for a fresh LLM call (or, in production, an embedding-similarity search over past
   questions) every time the same/near-duplicate question is asked, we serve the
   previously-generated SQL straight out of cache. The guardrail check is still
   re-run on every cache hit before execution -- it's cheap (sub-ms) and caching a
   security decision is never worth the risk.

Backed by Redis when available; falls back to an in-process TTL dict so the service
(and its test suite) still works without Redis running.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Optional

import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

_settings = get_settings()

try:
    import redis.asyncio as aioredis
except ImportError:  # pragma: no cover
    aioredis = None


class _InMemoryTTLCache:
    """Simple process-local fallback so caching still works without Redis."""

    def __init__(self):
        self._store: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> Optional[str]:
        item = self._store.get(key)
        if item is None:
            return None
        expires_at, value = item
        if expires_at < time.monotonic():
            self._store.pop(key, None)
            return None
        return value

    async def set(self, key: str, value: str, ttl: int) -> None:
        self._store[key] = (time.monotonic() + ttl, value)

    async def close(self):
        pass


class CacheClient:
    def __init__(self):
        self._backend = None
        self._is_redis = False

    async def connect(self):
        if not _settings.cache_enabled or not _settings.redis_url:
            # No REDIS_URL configured at all -- skip the network attempt entirely
            # rather than trying (and failing) to reach a default localhost Redis
            # that was never meant to exist in this deployment.
            self._backend = _InMemoryTTLCache()
            return
        if aioredis is not None:
            try:
                client = aioredis.from_url(_settings.redis_url, decode_responses=True, socket_connect_timeout=1)
                await client.ping()
                self._backend = client
                self._is_redis = True
                logger.info("cache_backend_selected", backend="redis")
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("redis_unavailable_falling_back_to_memory", error=str(exc))
        self._backend = _InMemoryTTLCache()
        logger.info("cache_backend_selected", backend="in_memory")

    async def close(self):
        if self._backend is not None:
            await self._backend.close()

    async def get_json(self, key: str) -> Optional[Any]:
        raw = await self._backend.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def set_json(self, key: str, value: Any, ttl: int) -> None:
        await self._backend.set(key, json.dumps(value), ttl)


_cache_client: Optional[CacheClient] = None


async def get_cache() -> CacheClient:
    global _cache_client
    if _cache_client is None:
        _cache_client = CacheClient()
        await _cache_client.connect()
    return _cache_client


def normalize_question(question: str) -> str:
    """Collapse whitespace/case/punctuation so near-identical questions share a cache key.

    This is a cheap stand-in for a semantic embedding cache: exact/near-duplicate
    phrasing (the overwhelming majority of repeated traffic in practice -- dashboards
    re-asking the same question on a refresh timer, multiple users asking the same
    canned question) hits the cache without needing an embedding model call.
    """
    q = question.strip().lower()
    q = re.sub(r"[^\w\s]", "", q)
    q = re.sub(r"\s+", " ", q)
    return q


def question_cache_key(question: str) -> str:
    normalized = normalize_question(question)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]
    return f"q2sql:{digest}"


def schema_cache_key(table_names: list[str]) -> str:
    digest = hashlib.sha256(",".join(sorted(table_names)).encode("utf-8")).hexdigest()[:24]
    return f"schema:{digest}"
