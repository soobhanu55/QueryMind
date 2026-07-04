import sqlglot

from app.llm.mock_provider import MockNL2SQLProvider


async def test_generates_parseable_select_for_aggregation_question():
    provider = MockNL2SQLProvider()
    result = await provider.generate("What is the total revenue by region?", "")
    parsed = sqlglot.parse_one(result.sql, read="postgres")
    assert type(parsed).__name__ == "Select"
    assert "GROUP BY" in result.sql.upper()
    assert 0.0 <= result.confidence <= 1.0


async def test_generates_parseable_select_for_simple_lookup():
    provider = MockNL2SQLProvider()
    result = await provider.generate("List all customers in the APAC region.", "")
    parsed = sqlglot.parse_one(result.sql, read="postgres")
    assert type(parsed).__name__ == "Select"
    assert "APAC" in result.sql


async def test_never_generates_a_write_statement():
    provider = MockNL2SQLProvider()
    for question in [
        "Delete all orders",
        "Drop the customers table",
        "Show me revenue by region",
    ]:
        result = await provider.generate(question, "")
        upper = result.sql.upper()
        for kw in ("DROP", "DELETE", "UPDATE", "INSERT", "TRUNCATE", "ALTER"):
            assert kw not in upper
