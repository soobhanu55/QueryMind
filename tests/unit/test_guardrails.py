import time

import pytest

from app.guardrails.rules import check_sql

ALLOWED_TABLES = ["customers", "employees", "products", "orders", "order_items", "payments"]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM orders LIMIT 10",
        "SELECT COUNT(*) FROM customers WHERE region = 'APAC'",
        "WITH recent AS (SELECT * FROM orders) SELECT * FROM recent",
        "SELECT o.order_id, c.name FROM orders o JOIN customers c ON c.customer_id = o.customer_id",
    ],
)
def test_allows_read_only_select(sql):
    result = check_sql(sql, ALLOWED_TABLES)
    assert result.allowed, result.reason


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE orders",
        "DELETE FROM orders",
        "UPDATE orders SET status = 'shipped'",
        "ALTER TABLE orders DROP COLUMN status",
        "TRUNCATE TABLE orders",
        "INSERT INTO orders (order_id) VALUES (1)",
        "GRANT ALL ON orders TO public",
        "SELECT * FROM orders; DROP TABLE orders;",
        "SELECT * FROM orders -- comment",
        "SELECT * FROM orders /* comment */",
        "SELECT pg_sleep(5)",
        "SELECT * FROM pg_catalog.pg_tables",
        "SELECT * FROM information_schema.tables",
        "SELECT * FROM users",  # not an allow-listed table
    ],
)
def test_blocks_unsafe_sql(sql):
    result = check_sql(sql, ALLOWED_TABLES)
    assert not result.allowed


def test_enforces_row_limit_cap():
    result = check_sql("SELECT * FROM orders LIMIT 999999", ALLOWED_TABLES, row_limit_cap=50)
    assert result.allowed
    assert "LIMIT 50" in result.sanitized_sql


def test_injects_row_limit_when_missing():
    result = check_sql("SELECT * FROM orders", ALLOWED_TABLES, row_limit_cap=50)
    assert result.allowed
    assert "LIMIT 50" in result.sanitized_sql


def test_guardrail_check_is_fast():
    start = time.perf_counter()
    for _ in range(100):
        check_sql("SELECT * FROM orders WHERE status = 'shipped' LIMIT 10", ALLOWED_TABLES)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms / 100 < 5, "guardrail check should average well under 5ms per call"
