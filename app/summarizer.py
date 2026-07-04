"""Template-based natural-language summary of a query result set.

Deliberately not an LLM call: it's on the same request path as generation and
execution, and a template covers the small number of result shapes (empty, single
scalar, single row, tabular) that analytics queries actually produce.
"""
from __future__ import annotations

from typing import Any

from app.executor import ExecutionResult


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return f"{value:,.0f}"
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def summarize(result: ExecutionResult) -> str:
    if result.row_count == 0:
        return "The query ran successfully but returned no matching rows."

    if result.row_count == 1 and len(result.columns) == 1:
        col, val = result.columns[0], result.rows[0][0]
        return f"{col.replace('_', ' ')}: {_format_value(val)}."

    if result.row_count == 1:
        pairs = ", ".join(f"{c.replace('_', ' ')} = {_format_value(v)}" for c, v in zip(result.columns, result.rows[0]))
        return f"Returned 1 row: {pairs}."

    preview_rows = result.rows[:3]
    preview = "; ".join(
        ", ".join(f"{c}={_format_value(v)}" for c, v in zip(result.columns, row)) for row in preview_rows
    )
    suffix = " (capped at the server row limit; refine the question to narrow results)" if result.truncated else ""
    more = f" and {result.row_count - 3} more" if result.row_count > 3 else ""
    return (
        f"Returned {result.row_count} row{'s' if result.row_count != 1 else ''} "
        f"across columns [{', '.join(result.columns)}]. First rows: {preview}{more}.{suffix}"
    )
