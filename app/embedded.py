"""Embedded demo database: the sample sales schema, its seed data and the wallet table in an in-memory SQLite database.

Used when `DATABASE_URL` starts with `sqlite` (and by the Streamlit demo when its Postgres is unreachable), so the demo needs no
hosted database that can be paused or deleted. The pipeline is unchanged: guardrail checks run on the Postgres-dialect SQL first,
then `to_sqlite` rewrites the few Postgres-only constructs the generators emit (EXTRACT, DATE_TRUNC, CURRENT_DATE - INTERVAL) and
the statement runs on a read-only connection with a time limit. Real deployments still use Postgres; this is for demos and tests.
"""
from __future__ import annotations

import csv
import re
import sqlite3
import threading
import time
from pathlib import Path

import sqlglot
from sqlglot import exp

DB_DIR = Path(__file__).resolve().parent.parent / "db"
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None


def to_sqlite(sql: str) -> str:
    """Postgres-dialect SQL -> SQLite, covering the date constructs sqlglot leaves untouched."""

    def fix(node: exp.Expression) -> exp.Expression:
        if isinstance(node, (exp.TimestampTrunc, exp.DateTrunc)):
            unit = node.args["unit"].name.upper()
            modifier = {"MONTH": "start of month", "YEAR": "start of year", "DAY": "start of day"}.get(unit)
            if modifier:
                return exp.Anonymous(this="DATE", expressions=[node.this, exp.Literal.string(modifier)])
        if isinstance(node, exp.Extract):
            fmt = {"YEAR": "%Y", "MONTH": "%m", "DAY": "%d"}.get(node.this.name.upper())
            if fmt:
                call = exp.Anonymous(this="STRFTIME", expressions=[exp.Literal.string(fmt), node.expression])
                return exp.Cast(this=call, to=exp.DataType.build("int"))
        if isinstance(node, exp.Sub) and isinstance(node.this, exp.CurrentDate) and isinstance(node.expression, exp.Interval):
            interval = node.expression
            unit = interval.args["unit"].name.upper().rstrip("S").lower() if interval.args.get("unit") else "day"
            return exp.Anonymous(this="DATE", expressions=[exp.Literal.string("now"), exp.Literal.string(f"-{int(interval.this.name)} {unit}")])
        return node

    return sqlglot.parse_one(sql, read="postgres").transform(fix).sql(dialect="sqlite")


def _statements(script: str):
    """Split a SQL script into complete statements (sqlite3.complete_statement understands quotes and comments)."""
    buf = ""
    for line in script.splitlines(keepends=True):
        buf += line
        if sqlite3.complete_statement(buf):
            yield buf.strip()
            buf = ""


def _build() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    schema = (DB_DIR / "schema.sql").read_text(encoding="utf-8")
    schema = schema.split("-- Read-only analytics role")[0]  # the rest is Postgres role/grant DDL
    schema = re.sub(r"\s+CASCADE\b", "", schema, flags=re.I)
    schema = re.sub(r"\bSERIAL\s+PRIMARY\s+KEY\b", "INTEGER PRIMARY KEY", schema, flags=re.I)
    conn.executescript(schema)
    for stmt in _statements((DB_DIR / "supabase_seed.sql").read_text(encoding="utf-8")):
        if stmt.upper().startswith("INSERT INTO"):
            conn.execute(stmt)
    with (DB_DIR / "wallets.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    cols = rows[0]
    conn.executemany(f"INSERT INTO wallets ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", rows[1:])
    conn.commit()
    conn.execute("PRAGMA query_only = ON")  # nothing the pipeline runs can change the data
    return conn


def connection() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            _conn = _build()
        return _conn


def run(sql: str, row_cap: int, timeout_seconds: float) -> tuple[list[str], list[tuple], bool]:
    """(columns, rows, truncated). Raises sqlite3.Error; an interrupted statement raises OperationalError('interrupted')."""
    conn = connection()
    deadline = time.monotonic() + timeout_seconds
    with _lock:
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        try:
            cur = conn.execute(to_sqlite(sql))
            columns = [d[0] for d in cur.description] if cur.description else []
            fetched = cur.fetchmany(row_cap + 1)
        finally:
            conn.set_progress_handler(None, 0)
    return columns, fetched[:row_cap], len(fetched) > row_cap
