"""
Fast, deterministic SQL safety guardrail.

Runs entirely in-process (no network/LLM call) so it never becomes the bottleneck
under load: parsing + validating a typical generated SQL string with sqlglot and a
handful of compiled regexes takes well under 1ms. See app/guardrails/config.yml for
the policy this module enforces, and README.md ("Guardrail design decision") for why
this replaces NeMo Guardrails as the execution-time gate.

Defense in depth, in order:
  1. raw substring blocklist (comments / statement-smuggling markers)
  2. reject multiple statements
  3. keyword blocklist on the raw text (catches anything sqlglot might parse leniently)
  4. parse with sqlglot; fail closed on parse errors
  5. AST root must be a SELECT (CTEs included -- sqlglot attaches WITH to the Select)
  6. every referenced table must be in the caller-supplied allow-list; blocked schemas rejected
  7. blocked function calls rejected (pg_sleep, dblink, file/superuser functions, etc.)
  8. LIMIT is enforced/rewritten down to the configured row cap
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Optional

import sqlglot
import structlog
import yaml
from sqlglot import exp

logger = structlog.get_logger("guardrail")

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yml")


@dataclass(frozen=True)
class GuardrailConfig:
    allowed_statement_root_types: frozenset
    blocked_statement_keywords: frozenset
    blocked_functions: frozenset
    blocked_schemas: frozenset
    blocked_raw_substrings: tuple
    max_row_limit: int
    allow_multiple_statements: bool

    @classmethod
    def load(cls, path: str = _CONFIG_PATH) -> "GuardrailConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(
            allowed_statement_root_types=frozenset(raw["allowed_statement_root_types"]),
            blocked_statement_keywords=frozenset(k.upper() for k in raw["blocked_statement_keywords"]),
            blocked_functions=frozenset(f.lower() for f in raw["blocked_functions"]),
            blocked_schemas=frozenset(s.lower() for s in raw["blocked_schemas"]),
            blocked_raw_substrings=tuple(raw["blocked_raw_substrings"]),
            max_row_limit=int(raw["max_row_limit"]),
            allow_multiple_statements=bool(raw["allow_multiple_statements"]),
        )


@lru_cache(maxsize=1)
def get_guardrail_config() -> GuardrailConfig:
    return GuardrailConfig.load()


@lru_cache(maxsize=1)
def _keyword_regex() -> re.Pattern:
    keywords = get_guardrail_config().blocked_statement_keywords
    escaped = [re.escape(k) for k in keywords]
    return re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(escaped) + r")(?![A-Za-z0-9_])", re.IGNORECASE)


@dataclass
class GuardrailResult:
    allowed: bool
    sanitized_sql: Optional[str] = None
    reason: Optional[str] = None
    blocked_category: Optional[str] = None
    check_duration_ms: float = 0.0


def check_sql(
    sql: str,
    allowed_tables: Iterable[str],
    row_limit_cap: Optional[int] = None,
    question: Optional[str] = None,
) -> GuardrailResult:
    """Validate and sanitize a generated SQL string before it is allowed to execute."""
    start = time.perf_counter()
    cfg = get_guardrail_config()
    allowed_tables_set = {t.lower() for t in allowed_tables}
    row_limit_cap = row_limit_cap or cfg.max_row_limit

    def _block(reason: str, category: str) -> GuardrailResult:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.warning(
            "guardrail_blocked",
            reason=reason,
            category=category,
            sql=(sql or "")[:2000],
            question=question,
            duration_ms=round(duration_ms, 4),
        )
        return GuardrailResult(
            allowed=False, reason=reason, blocked_category=category, check_duration_ms=duration_ms
        )

    if not sql or not sql.strip():
        return _block("empty SQL", "empty")

    stripped = sql.strip()
    lowered_raw = stripped.lower()

    # 1. Raw substring blocklist
    for substr in cfg.blocked_raw_substrings:
        if substr.lower() in lowered_raw:
            return _block(f"blocked substring present: {substr!r}", "raw_substring")

    # 2. Reject multiple statements (allow one optional trailing semicolon)
    body = stripped[:-1] if stripped.endswith(";") else stripped
    if ";" in body and not cfg.allow_multiple_statements:
        return _block("multiple statements are not allowed", "multiple_statements")

    # 3. Keyword blocklist as defense-in-depth
    kw_match = _keyword_regex().search(body)
    if kw_match:
        return _block(f"blocked keyword: {kw_match.group(0).upper()}", "blocked_keyword")

    # 4. Parse with sqlglot; fail closed on parse errors
    try:
        parsed = [p for p in sqlglot.parse(body, read="postgres") if p is not None]
    except Exception as exc:  # noqa: BLE001
        return _block(f"SQL failed to parse: {exc}", "parse_error")

    if len(parsed) != 1:
        return _block(f"expected exactly one statement, found {len(parsed)}", "multiple_statements")

    statement = parsed[0]
    root_type = type(statement).__name__
    if root_type not in cfg.allowed_statement_root_types:
        return _block(
            f"statement type '{root_type}' is not allowed (read-only SELECT only)",
            "disallowed_statement_type",
        )

    # 5. Every referenced table must be allow-listed; reject blocked schemas outright.
    # CTE aliases (WITH cte_name AS (...)) are local names, not real tables -- exclude
    # them from the allow-list check (the tables *inside* the CTE body are still checked).
    with_clause = statement.args.get("with")
    cte_names = {cte.alias.lower() for cte in with_clause.expressions} if with_clause else set()
    for table_expr in statement.find_all(exp.Table):
        table_name = (table_expr.name or "").lower()
        schema_name = (table_expr.db or "").lower()
        if schema_name and schema_name in cfg.blocked_schemas:
            return _block(f"reference to blocked schema '{schema_name}'", "blocked_schema")
        if table_name and table_name not in allowed_tables_set and table_name not in cte_names:
            return _block(f"reference to non-allow-listed table '{table_name}'", "unknown_table")

    # 6. Blocked function calls (unrecognized/exotic functions parse as Anonymous)
    for func_expr in statement.find_all(exp.Anonymous):
        fname = (func_expr.this or "").lower()
        if fname in cfg.blocked_functions:
            return _block(f"blocked function call: {fname}()", "blocked_function")

    # 7. Enforce row limit cap (inject if missing, rewrite down if too large)
    existing_limit = statement.args.get("limit")
    needs_cap = True
    if existing_limit is not None:
        try:
            limit_value = int(existing_limit.expression.this)
            needs_cap = limit_value > row_limit_cap
        except (AttributeError, ValueError, TypeError):
            needs_cap = True
    if needs_cap:
        statement.set("limit", exp.Limit(expression=exp.Literal.number(row_limit_cap)))

    sanitized_sql = statement.sql(dialect="postgres")
    duration_ms = (time.perf_counter() - start) * 1000
    logger.debug("guardrail_allowed", duration_ms=round(duration_ms, 4))
    return GuardrailResult(allowed=True, sanitized_sql=sanitized_sql, check_duration_ms=duration_ms)
