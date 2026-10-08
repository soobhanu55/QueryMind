"""The embedded SQLite demo database: data loads, Postgres dates are rewritten, it is read-only, and the benchmark runs on it."""
import asyncio
import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

from app import embedded
from app.config import get_settings
from app.executor import QueryExecutionError, execute_query


@pytest.fixture(autouse=True)
def embedded_mode(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///:memory:")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def count(table):
    return embedded.run(f"SELECT COUNT(*) FROM {table}", 10, 5)[1][0][0]


def test_every_table_is_loaded():
    assert {t: count(t) for t in ("customers", "employees", "products", "orders", "order_items", "payments", "wallets")} == {
        "customers": 500, "employees": 25, "products": 40, "orders": 5000, "order_items": 12435, "payments": 4610, "wallets": 1625}


@pytest.mark.parametrize("pg,expected", [
    ("SELECT EXTRACT(YEAR FROM o.order_date) FROM orders o", "CAST(STRFTIME('%Y', o.order_date) AS INTEGER)"),
    ("SELECT date_trunc('month', o.order_date) FROM orders o", "DATE(o.order_date, 'start of month')"),
    ("SELECT 1 FROM orders o WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'", "DATE('now', '-30 day')"),
    ("SELECT 1 FROM orders o WHERE o.order_date >= CURRENT_DATE - INTERVAL '1 year'", "DATE('now', '-1 year')"),
])
def test_postgres_date_constructs_are_rewritten(pg, expected):
    assert expected in embedded.to_sqlite(pg)


def test_rewritten_date_queries_return_sensible_data():
    years = embedded.run("SELECT DISTINCT EXTRACT(YEAR FROM o.order_date) FROM orders o ORDER BY 1", 10, 5)[1]
    assert [y[0] for y in years] == [2023, 2024, 2025, 2026]
    months = embedded.run("SELECT date_trunc('month', o.order_date) AS m, COUNT(*) FROM orders o GROUP BY 1 ORDER BY 1 LIMIT 3", 10, 5)[1]
    assert months[0][0].endswith("-01") and months[0][1] > 0


def test_the_connection_is_read_only():
    with pytest.raises(sqlite3.OperationalError):
        embedded.run("DELETE FROM customers", 10, 5)
    assert count("customers") == 500


def test_a_runaway_query_is_interrupted_by_the_time_limit():
    slow = "SELECT COUNT(*) FROM order_items a, order_items b, order_items c"
    with pytest.raises(sqlite3.OperationalError, match="interrupted"):
        embedded.run(slow, 10, 0.05)


async def test_execute_query_serialises_rows_and_flags_truncation():
    result = await execute_query("SELECT customer_id, name FROM customers ORDER BY customer_id", row_cap=5)
    assert result.row_count == 5 and result.truncated and result.columns == ["customer_id", "name"]


async def test_execute_query_maps_errors_and_timeouts():
    with pytest.raises(QueryExecutionError) as bad:
        await execute_query("SELECT nope FROM customers")
    assert not bad.value.is_timeout
    get_settings.cache_clear()
    import os

    os.environ["QUERY_TIMEOUT_SECONDS"] = "0.05"
    get_settings.cache_clear()
    try:
        with pytest.raises(QueryExecutionError) as slow:
            await execute_query("SELECT COUNT(*) FROM order_items a, order_items b, order_items c")
        assert slow.value.is_timeout
    finally:
        del os.environ["QUERY_TIMEOUT_SECONDS"]


def test_the_accuracy_benchmark_gives_the_same_score_as_on_postgres():
    """Runs the 74-question benchmark end to end on the embedded database: 91.9% (68/74), as measured on Postgres."""
    spec = importlib.util.spec_from_file_location("run_accuracy", Path(__file__).resolve().parents[1] / "accuracy" / "run_accuracy.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_accuracy"] = mod
    spec.loader.exec_module(mod)
    report = asyncio.run(mod.run("mock", False))
    assert (report["overall_correct"], report["overall_total"]) == (68, 74)
