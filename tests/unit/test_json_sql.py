import pytest

from app.llm.json_sql import parse_sql_response


def test_parses_a_json_object():
    r = parse_sql_response('{"sql": "SELECT 1;", "confidence": 0.9, "explanation": "one"}')
    assert (r.sql, r.confidence, r.explanation) == ("SELECT 1", 0.9, "one")


def test_strips_markdown_fences_and_clamps_confidence():
    r = parse_sql_response('```json\n{"sql": "SELECT 2", "confidence": 7}\n```')
    assert r.sql == "SELECT 2" and r.confidence == 1.0


def test_falls_back_to_the_first_select_statement():
    r = parse_sql_response("Sure! Here you go:\nSELECT a FROM t WHERE x = 1; -- done")
    assert r.sql == "SELECT a FROM t WHERE x = 1" and r.confidence == 0.3


def test_json_without_sql_falls_back_to_text_search():
    r = parse_sql_response('{"answer": "select * from wallets"}')
    assert r.sql.lower().startswith("select * from wallets")


def test_rejects_output_with_no_sql():
    with pytest.raises(ValueError):
        parse_sql_response("I cannot help with that.")


def test_factory_selects_the_new_providers(monkeypatch):
    from app.config import get_settings
    from app.llm import factory

    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    get_settings.cache_clear()
    factory.get_provider.cache_clear()
    try:
        assert type(factory.get_provider()).__name__ == "GroqProvider"
    finally:
        get_settings.cache_clear()
        factory.get_provider.cache_clear()
