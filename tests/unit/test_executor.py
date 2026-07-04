"""Integration-style tests against the real Postgres instance (docker compose up + seed.py)."""
import pytest

from app.executor import QueryExecutionError, execute_query


async def test_executes_simple_select():
    result = await execute_query("SELECT 1 AS one")
    assert result.columns == ["one"]
    assert result.rows == [[1]]
    assert result.row_count == 1
    assert not result.truncated


async def test_row_cap_is_enforced():
    result = await execute_query("SELECT * FROM generate_series(1, 100) AS g", row_cap=10)
    assert result.row_count == 10
    assert result.truncated


async def test_invalid_sql_raises_execution_error():
    with pytest.raises(QueryExecutionError):
        await execute_query("SELECT * FROM this_table_does_not_exist")
