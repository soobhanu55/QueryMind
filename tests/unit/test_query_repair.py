"""/query repairs a model's SQL once when the database rejects it (embedded SQLite, scripted provider, no network)."""
import pytest
from fastapi.testclient import TestClient

from app import cache as cache_module
from app.config import get_settings
from app.llm.base import NL2SQLProvider, SQLGenerationResult
from app.main import app
from app.routers import query as query_module


class Scripted(NL2SQLProvider):
    """First answer is whatever `first` says; a retry returns `fixed` (or raises)."""

    def __init__(self, first, fixed=None, fail_retry=False):
        self.first, self.fixed, self.fail_retry, self.retries = first, fixed, fail_retry, []

    async def generate(self, question, schema_text, retry=None):
        if retry is None:
            return SQLGenerationResult(self.first, 0.8, "first try", model="m", route="llm")
        self.retries.append(retry)
        if self.fail_retry:
            raise ConnectionError("model down")
        return SQLGenerationResult(self.fixed, 0.7, "repaired", model="m", route="llm")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    monkeypatch.setenv("CACHE_ENABLED", "false")
    get_settings.cache_clear()
    cache_module._cache_client = None
    yield TestClient(app)
    cache_module._cache_client = None
    get_settings.cache_clear()


def use(monkeypatch, provider):
    monkeypatch.setattr(query_module, "get_provider", lambda: provider)
    return provider


def ask(client):
    return client.post("/query", json={"question": "how many customers are there?"})


def test_a_failing_query_is_repaired_once_and_flagged(client, monkeypatch):
    p = use(monkeypatch, Scripted("SELECT COUNT(*) FROM customers WHERE nope = 1", "SELECT COUNT(*) FROM customers"))
    r = ask(client)
    assert r.status_code == 200 and r.json()["rows"] == [[500]] and r.json()["repaired"] is True
    assert len(p.retries) == 1 and p.retries[0][0].startswith("SELECT COUNT(*) FROM customers WHERE nope") and "nope" in p.retries[0][1]


def test_a_working_query_is_not_repaired(client, monkeypatch):
    p = use(monkeypatch, Scripted("SELECT COUNT(*) FROM customers"))
    r = ask(client)
    assert r.status_code == 200 and r.json()["repaired"] is False and p.retries == []


def test_when_the_repair_also_fails_the_original_error_is_returned(client, monkeypatch):
    use(monkeypatch, Scripted("SELECT nope FROM customers", "SELECT also_nope FROM customers"))
    r = ask(client)
    assert r.status_code == 400 and "nope" in r.json()["detail"]


def test_when_the_model_is_down_the_original_error_is_returned(client, monkeypatch):
    use(monkeypatch, Scripted("SELECT nope FROM customers", fail_retry=True))
    assert ask(client).status_code == 400


def test_a_repair_that_the_guardrail_rejects_is_not_executed(client, monkeypatch):
    use(monkeypatch, Scripted("SELECT nope FROM customers", "DELETE FROM customers"))
    assert ask(client).status_code == 400
    assert TestClient(app).get("/health").json()["db_ok"] is True


def test_the_rules_provider_is_never_asked_to_repair(client, monkeypatch):
    from app.llm.mock_provider import MockNL2SQLProvider

    class Rules(MockNL2SQLProvider):
        retries = 0

        async def generate(self, question, schema_text, retry=None):
            if retry:
                Rules.retries += 1
            return SQLGenerationResult("SELECT nope FROM customers", 0.9, "x")

    use(monkeypatch, Rules())
    assert ask(client).status_code == 400 and Rules.retries == 0
