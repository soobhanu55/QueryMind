"""The core /query pipeline: schema retrieval -> prompt -> LLM -> guardrail -> execute -> summarize."""
from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, HTTPException

from app.cache import get_cache, question_cache_key, schema_cache_key
from app.config import get_settings
from app.executor import QueryExecutionError, execute_query
from app.guardrails.rules import check_sql
from app.llm import SQLGenerationResult, get_provider
from app.llm.mock_provider import MockNL2SQLProvider
from app.llm.router import cost_usd
from app.models import QueryRequest, QueryResponse
from app.schema_store import get_schema_store
from app.summarizer import summarize

logger = structlog.get_logger(__name__)
router = APIRouter()


async def _repair(provider, question, schema_text, generation, error, allowed_tables, row_cap):
    """One repair attempt after an execution error: the model sees its SQL and the database error, and the new SQL goes through
    the same guardrail and executor. Returns (generation, guardrail_result, result), or None when it cannot be fixed."""
    try:
        fixed = await provider.generate(question, schema_text, retry=(generation.sql, error))
        verdict = check_sql(fixed.sql, allowed_tables=allowed_tables, row_limit_cap=row_cap, question=question)
        if not verdict.allowed:
            return None
        return fixed, verdict, await execute_query(verdict.sanitized_sql, row_cap=row_cap)
    except Exception as exc:  # noqa: BLE001 - a failed repair just means the original error is reported
        logger.info("repair_failed", error=type(exc).__name__)
        return None


@router.post("/query", response_model=QueryResponse)
async def run_query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    store = get_schema_store()
    cache = await get_cache()

    # 1. Schema-aware retrieval (cached: table selection + rendered schema text are
    #    pure functions of the question, so repeated/similar questions skip the work).
    relevant_tables = store.retrieve_relevant_tables(request.question)
    schema_key = schema_cache_key(relevant_tables)
    schema_text = await cache.get_json(schema_key)
    if schema_text is None:
        schema_text = store.render_schema_text(relevant_tables)
        await cache.set_json(schema_key, schema_text, settings.schema_cache_ttl_seconds)

    # 2. SQL generation (cached by normalized question text).
    q_key = question_cache_key(request.question)
    cached_generation = await cache.get_json(q_key)
    generation_start = time.perf_counter()
    cached = False
    if cached_generation is not None:
        generation = SQLGenerationResult(**cached_generation)
        generation.route, cached = "cache", True
    else:
        provider = get_provider()
        generation = await provider.generate(request.question, schema_text)
        await cache.set_json(
            q_key,
            {"sql": generation.sql, "confidence": generation.confidence, "explanation": generation.explanation},
            settings.question_cache_ttl_seconds,
        )
    generation_ms = (time.perf_counter() - generation_start) * 1000

    # 3. Guardrail check -- always re-run, even on a cache hit. Fast (<1ms typical) so
    #    this is never the bottleneck, and a security decision should never be cached.
    row_cap = min(request.max_rows or settings.max_result_rows, settings.max_result_rows)
    guardrail_result = check_sql(
        generation.sql,
        allowed_tables=store.table_names(),
        row_limit_cap=row_cap,
        question=request.question,
    )
    if not guardrail_result.allowed:
        raise HTTPException(
            status_code=400,
            detail={
                "detail": "Generated SQL was blocked by the guardrail layer.",
                "reason": guardrail_result.reason,
                "category": guardrail_result.blocked_category,
            },
        )

    # 4. Execute against the database with timeout + row cap. A query that fails to run (not a timeout) gets one repair attempt
    #    from a model provider; the rules cannot repair themselves.
    repaired = False
    try:
        result = await execute_query(guardrail_result.sanitized_sql, row_cap=row_cap)
    except QueryExecutionError as exc:
        fix = None
        if not exc.is_timeout and not isinstance(get_provider(), MockNL2SQLProvider):
            fix = await _repair(get_provider(), request.question, schema_text, generation, str(exc), store.table_names(), row_cap)
        if fix is None:
            status_code = 504 if exc.is_timeout else 400
            raise HTTPException(status_code=status_code, detail=str(exc)) from exc
        generation, guardrail_result, result = fix
        repaired, cached = True, False
        await cache.set_json(
            q_key, {"sql": generation.sql, "confidence": generation.confidence, "explanation": generation.explanation},
            settings.question_cache_ttl_seconds)

    # 5. Natural-language summary + response assembly.
    summary_text = summarize(result)
    return QueryResponse(
        question=request.question,
        sql=guardrail_result.sanitized_sql,
        confidence=generation.confidence,
        explanation=generation.explanation,
        columns=result.columns,
        rows=result.rows,
        row_count=result.row_count,
        truncated=result.truncated,
        execution_ms=round(result.execution_ms, 3),
        guardrail_check_ms=round(guardrail_result.check_duration_ms, 4),
        generation_ms=round(generation_ms, 3),
        cached=cached,
        summary=summary_text,
        repaired=repaired,
        route=generation.route,
        model=generation.model,
        escalated=generation.escalated,
        input_tokens=generation.input_tokens,
        output_tokens=generation.output_tokens,
        cost_usd=0.0 if cached else cost_usd(generation.model, generation.input_tokens, generation.output_tokens),
    )
