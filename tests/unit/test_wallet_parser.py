import pytest
import sqlglot

from app.guardrails.rules import check_sql
from app.llm.mock_provider import MockNL2SQLProvider
from app.schema_store import get_schema_store

CASES = [
    ("How many wallets are in the high risk band?", "SELECT COUNT(*) AS wallet_count FROM wallets WHERE risk_band = 'high'"),
    ("What is the average risk score of wallets?", "SELECT AVG(risk_score) AS avg_risk_score FROM wallets"),
    ("Show the top 5 wallets by total borrowed amount.",
     "SELECT wallet_id, borrow_usd FROM wallets ORDER BY borrow_usd DESC, wallet_id LIMIT 5"),
    ("Average deposits by risk band",
     "SELECT risk_band, AVG(deposit_usd) AS avg_deposit_usd FROM wallets GROUP BY risk_band ORDER BY risk_band"),
    ("How many wallets have never been liquidated?", "SELECT COUNT(*) AS wallet_count FROM wallets WHERE was_liquidated = 0"),
    ("How many wallets have more than 100 transactions?", "SELECT COUNT(*) AS wallet_count FROM wallets WHERE n_tx > 100"),
    ("What is the average number of transactions per risk band?",
     "SELECT risk_band, AVG(n_tx) AS avg_n_tx FROM wallets GROUP BY risk_band ORDER BY risk_band"),
]


@pytest.mark.parametrize("question,sql", CASES)
async def test_wallet_questions_generate_expected_sql(question, sql):
    assert (await MockNL2SQLProvider().generate(question, "")).sql == sql


async def test_wallet_sql_passes_the_guardrail():
    store = get_schema_store()
    for question, _ in CASES:
        sql = (await MockNL2SQLProvider().generate(question, "")).sql
        assert type(sqlglot.parse_one(sql, read="postgres")).__name__ == "Select"
        assert check_sql(sql, allowed_tables=store.table_names(), row_limit_cap=100, question=question).allowed


async def test_sales_questions_are_unaffected():
    assert "wallets" not in (await MockNL2SQLProvider().generate("What is the total revenue by region?", "")).sql
