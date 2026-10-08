"""Cost- and latency-aware routing: the free rule-based parser answers first, a model only when it is not sure.

    question -> [private?] -> rules tier --(confidence >= threshold)--> answer
                                   \\--(low confidence / error)--> model tier --> answer
                                                                       \\--(down / over budget)--> the rules answer, marked degraded

* Escalation by confidence: on the 74-question benchmark the rules answer 72 of them (see docs/routing_eval.md).
* Private questions (matching `router_private_pattern`, e.g. e-mail addresses) never go to a hosted API.
* Each tier has a timeout and a circuit breaker; a tier that keeps failing is skipped for `reset_after` seconds.
* The model tier has a daily call budget, a guard for free-tier quotas and for runaway loops.
* Every answer says which tier produced it, how many tokens it used and what that costs at list price.
"""
from __future__ import annotations

import asyncio
import re
import time
from collections import Counter
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable

import structlog

from app.config import get_settings
from app.llm.base import NL2SQLProvider, SQLGenerationResult

logger = structlog.get_logger(__name__)

HOSTED = {"anthropic", "gemini", "groq"}
# USD per 1M tokens (input, output) at list price; Groq's free tier costs 0, so this is the cost you would pay beyond it.
# A model missing here is reported as unpriced, never guessed.
PRICES = {"llama-3.3-70b-versatile": (0.59, 0.79), "rules": (0.0, 0.0)}


def cost_usd(model: str | None, tokens_in: int | None, tokens_out: int | None) -> float | None:
    if model == "rules":
        return 0.0
    if model not in PRICES or tokens_in is None or tokens_out is None:
        return None
    p_in, p_out = PRICES[model]
    return round((tokens_in * p_in + tokens_out * p_out) / 1e6, 6)


class Breaker:
    """Opens after `threshold` consecutive failures and stays open for `reset_after` seconds, then lets a probe through."""

    def __init__(self, threshold: int = 3, reset_after: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.reset_after, self.clock = threshold, reset_after, clock
        self.failures, self.opened_at = 0, None

    def allow(self) -> bool:
        return self.opened_at is None or self.clock() - self.opened_at >= self.reset_after

    def ok(self) -> None:
        self.failures, self.opened_at = 0, None

    def fail(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.opened_at = self.clock()

    @property
    def state(self) -> str:
        return "closed" if self.opened_at is None else ("half_open" if self.allow() else "open")


@dataclass
class Tier:
    name: str
    make: Callable[[], NL2SQLProvider]
    breaker: Breaker = field(default_factory=Breaker)
    _provider: NL2SQLProvider | None = None

    def provider(self) -> NL2SQLProvider:
        if self._provider is None:
            self._provider = self.make()  # a missing key or model fails here, inside the tier's error handling
        return self._provider


STATS: Counter = Counter()


def snapshot(router: "RoutedProvider | None" = None) -> dict:
    out = dict(STATS)
    if router is not None:
        out["breakers"] = {t.name: t.breaker.state for t in router.tiers}
        out["llm_calls_today"] = router.calls_today
    return out


class RoutedProvider(NL2SQLProvider):
    def __init__(self, rules: NL2SQLProvider | None = None, llm: NL2SQLProvider | None = None) -> None:
        from app.llm.factory import build

        s = get_settings()
        self.settings = s
        self.rules = Tier("rules", lambda: rules or build("mock"))
        self.llm = Tier(s.router_llm, lambda: llm or build(s.router_llm))
        self.tiers = [self.rules, self.llm]
        self.private_re = re.compile(s.router_private_pattern, re.IGNORECASE)
        self._day, self.calls_today = self._today(), 0

    @staticmethod
    def _today() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _budget_left(self) -> bool:
        if self._today() != self._day:
            self._day, self.calls_today = self._today(), 0
        return self.calls_today < self.settings.router_llm_daily_calls

    async def _run(self, tier: Tier, question: str, schema_text: str) -> SQLGenerationResult | None:
        if not tier.breaker.allow():
            STATS[f"{tier.name}_skipped_breaker_open"] += 1
            return None
        started = time.perf_counter()
        try:
            result = await asyncio.wait_for(tier.provider().generate(question, schema_text), self.settings.router_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - any tier failure is handled the same way
            tier.breaker.fail()
            STATS[f"{tier.name}_failures"] += 1
            logger.warning("llm_tier_failed", tier=tier.name, error=type(exc).__name__, detail=str(exc)[:120])
            return None
        tier.breaker.ok()
        STATS[f"{tier.name}_calls"] += 1
        model = "rules" if tier is self.rules else result.model
        STATS["input_tokens"] += result.input_tokens or 0
        STATS["output_tokens"] += result.output_tokens or 0
        logger.info("llm_tier_answered", tier=tier.name, model=model, ms=round((time.perf_counter() - started) * 1000, 1),
                    confidence=result.confidence, tokens_in=result.input_tokens, tokens_out=result.output_tokens)
        return replace(result, model=model, route=tier.name)

    async def generate(self, question: str, schema_text: str) -> SQLGenerationResult:
        s = self.settings
        private = bool(self.private_re.search(question)) and s.router_llm in HOSTED
        draft = await self._run(self.rules, question, schema_text)
        if draft is not None and (draft.confidence >= s.router_min_confidence or private):
            if private:
                STATS["private_kept_off_hosted_api"] += 1
            return draft  # answered by the free tier

        if private or not self._budget_left():
            STATS["llm_skipped_budget" if not private else "private_without_rules_answer"] += 1
        else:
            self.calls_today += 1
            answer = await self._run(self.llm, question, schema_text)
            if answer is not None:
                STATS["escalated"] += 1
                return replace(answer, escalated=True)
            STATS["llm_fallback_to_rules"] += 1

        if draft is not None:  # the model could not help: a low-confidence rules answer still beats an error
            return replace(draft, route="rules_degraded")
        raise RuntimeError("no tier could generate SQL for this question")
