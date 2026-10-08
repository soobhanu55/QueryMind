"""Routing: cheap tier first, escalation by confidence, fallback, breaker, privacy rule, budget, timeout."""
import asyncio

import pytest

from app.config import get_settings
from app.llm import router as r
from app.llm.base import NL2SQLProvider, SQLGenerationResult


class Fake(NL2SQLProvider):
    def __init__(self, confidence=0.9, fail=None, delay=0.0, model=None, tokens=(1000, 100)):
        self.confidence, self.fail, self.delay, self.model, self.tokens, self.calls = confidence, fail, delay, model, tokens, 0

    async def generate(self, question, schema_text, retry=None):
        self.calls += 1
        self.last_retry = retry
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise self.fail
        return SQLGenerationResult("SELECT 1", self.confidence, "x", model=self.model, input_tokens=self.tokens[0], output_tokens=self.tokens[1])


@pytest.fixture(autouse=True)
def settings(monkeypatch):
    for k, v in {"ROUTER_LLM": "groq", "ROUTER_MIN_CONFIDENCE": "0.55", "ROUTER_TIMEOUT_SECONDS": "0.2", "ROUTER_LLM_DAILY_CALLS": "500"}.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()
    r.STATS.clear()
    yield
    get_settings.cache_clear()


def routed(rules, llm):
    return r.RoutedProvider(rules=rules, llm=llm)


async def ask(p, q="total revenue by region"):
    return await p.generate(q, "schema")


async def test_a_confident_rules_answer_never_calls_the_model():
    rules, llm = Fake(0.9), Fake(model="llama-3.3-70b-versatile")
    out = await ask(routed(rules, llm))
    assert out.route == "rules" and out.model == "rules" and not out.escalated and llm.calls == 0


async def test_a_low_confidence_rules_answer_escalates_to_the_model():
    rules, llm = Fake(0.3), Fake(0.95, model="llama-3.3-70b-versatile")
    out = await ask(routed(rules, llm))
    assert out.route == "groq" and out.escalated and out.model == "llama-3.3-70b-versatile" and llm.calls == 1
    assert r.cost_usd(out.model, out.input_tokens, out.output_tokens) == pytest.approx((1000 * 0.59 + 100 * 0.79) / 1e6)


async def test_rules_failure_also_escalates():
    out = await ask(routed(Fake(fail=ValueError("cannot parse")), Fake(0.9, model="m")))
    assert out.route == "groq" and out.escalated


async def test_when_the_model_is_down_the_rules_answer_is_returned_as_degraded():
    out = await ask(routed(Fake(0.3), Fake(fail=ConnectionError("down"))))
    assert out.route == "rules_degraded" and out.confidence == 0.3 and r.STATS["llm_fallback_to_rules"] == 1


async def test_a_failing_model_trips_the_breaker_and_stops_being_called():
    rules, llm = Fake(0.3), Fake(fail=ConnectionError("down"))
    p = routed(rules, llm)
    for _ in range(3):
        await ask(p)
    assert p.llm.breaker.state == "open" and llm.calls == 3
    await ask(p)
    assert llm.calls == 3 and r.STATS["groq_skipped_breaker_open"] == 1
    assert r.snapshot(p)["breakers"]["groq"] == "open"


async def test_a_slow_model_times_out_and_counts_as_a_failure():
    out = await ask(routed(Fake(0.3), Fake(delay=1.0)))
    assert out.route == "rules_degraded" and r.STATS["groq_failures"] == 1


async def test_if_nothing_can_answer_it_raises():
    with pytest.raises(RuntimeError):
        await ask(routed(Fake(fail=ValueError("x")), Fake(fail=ConnectionError("y"))))


async def test_private_questions_stay_off_a_hosted_api():
    rules, llm = Fake(0.2), Fake(0.9, model="m")
    out = await ask(routed(rules, llm), "show me the email addresses of APAC customers")
    assert llm.calls == 0 and out.route == "rules" and r.STATS["private_kept_off_hosted_api"] == 1


async def test_private_questions_may_use_a_local_model(monkeypatch):
    monkeypatch.setenv("ROUTER_LLM", "local")
    get_settings.cache_clear()
    rules, llm = Fake(0.2), Fake(0.9, model="local-model")
    out = await ask(routed(rules, llm), "show me the email addresses of APAC customers")
    assert llm.calls == 1 and out.escalated


async def test_the_daily_call_budget_caps_model_use(monkeypatch):
    monkeypatch.setenv("ROUTER_LLM_DAILY_CALLS", "2")
    get_settings.cache_clear()
    rules, llm = Fake(0.2), Fake(0.9, model="m")
    p = routed(rules, llm)
    routes = [(await ask(p)).route for _ in range(4)]
    assert routes == ["groq", "groq", "rules_degraded", "rules_degraded"] and llm.calls == 2


def test_cost_is_unpriced_rather_than_guessed():
    assert r.cost_usd("some-new-model", 10, 10) is None
    assert r.cost_usd("rules", None, None) == 0.0
    assert r.cost_usd("llama-3.3-70b-versatile", None, None) is None


def test_breaker_recovers_after_the_reset_period():
    now = [0.0]
    b = r.Breaker(threshold=2, reset_after=10, clock=lambda: now[0])
    b.fail()
    b.fail()
    assert b.state == "open" and not b.allow()
    now[0] = 11
    assert b.state == "half_open" and b.allow()
    b.ok()
    assert b.state == "closed"


async def test_a_retry_skips_the_rules_and_goes_to_the_model():
    rules, llm = Fake(0.9), Fake(0.9, model="m")
    out = await routed(rules, llm).generate("q", "s", retry=("SELECT bad", "column x does not exist"))
    assert rules.calls == 0 and llm.calls == 1 and llm.last_retry == ("SELECT bad", "column x does not exist")
    assert out.route == "groq" and out.escalated and r.STATS["repairs"] == 1


async def test_a_retry_for_a_private_question_is_refused_on_a_hosted_api():
    llm = Fake(0.9, model="m")
    with pytest.raises(RuntimeError):
        await routed(Fake(0.9), llm).generate("send the email addresses", "s", retry=("SELECT 1", "err"))
    assert llm.calls == 0


async def test_a_failing_model_cannot_repair():
    with pytest.raises(RuntimeError):
        await routed(Fake(0.9), Fake(fail=ConnectionError("down"))).generate("q", "s", retry=("SELECT 1", "err"))
