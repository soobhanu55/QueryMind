"""Executes guardrail-approved, read-only SQL against Postgres with a timeout and row cap."""
from __future__ import annotations

import decimal
import time
from dataclasses import dataclass
from typing import Any, Optional

import structlog
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, SQLAlchemyError

from app.config import get_settings
from app.db import session_scope

logger = structlog.get_logger(__name__)


class QueryExecutionError(Exception):
    def __init__(self, message: str, *, is_timeout: bool = False):
        super().__init__(message)
        self.is_timeout = is_timeout


@dataclass
class ExecutionResult:
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    execution_ms: float


def _serialize(value: Any) -> Any:
    if isinstance(value, decimal.Decimal):
        return float(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


async def execute_query(sql: str, row_cap: Optional[int] = None) -> ExecutionResult:
    """Run SQL that has already passed the guardrail. Never call this on unvalidated input."""
    settings = get_settings()
    row_cap = row_cap or settings.max_result_rows
    timeout_ms = int(settings.query_timeout_seconds * 1000)

    start = time.perf_counter()
    columns: list[str] = []
    fetched: list[Any] = []
    async with session_scope() as session:
        try:
            await session.execute(text(f"SET LOCAL statement_timeout = {timeout_ms}"))
            result = await session.execute(text(sql))
            columns = list(result.keys())
            fetched = result.fetchmany(row_cap + 1)  # +1 lets us detect truncation cheaply
        except DBAPIError as exc:
            orig = str(getattr(exc, "orig", exc))
            is_timeout = "statement timeout" in orig.lower() or "canceling statement" in orig.lower()
            logger.warning("query_execution_failed", error=orig, is_timeout=is_timeout, sql=sql[:2000])
            raise QueryExecutionError(orig, is_timeout=is_timeout) from exc
        except SQLAlchemyError as exc:
            logger.warning("query_execution_failed", error=str(exc), sql=sql[:2000])
            raise QueryExecutionError(str(exc)) from exc
        finally:
            await session.rollback()  # read-only path: never commit, always release cleanly

    truncated = len(fetched) > row_cap
    rows = fetched[:row_cap]
    execution_ms = (time.perf_counter() - start) * 1000
    serialized_rows = [[_serialize(v) for v in row] for row in rows]

    logger.info(
        "query_executed",
        row_count=len(serialized_rows),
        truncated=truncated,
        execution_ms=round(execution_ms, 2),
    )
    return ExecutionResult(
        columns=columns,
        rows=serialized_rows,
        row_count=len(serialized_rows),
        truncated=truncated,
        execution_ms=execution_ms,
    )
