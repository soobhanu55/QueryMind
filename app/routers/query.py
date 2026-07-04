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
from app.models import QueryRequest, QueryResponse
from app.schema_store import get_schema_store
from app.summarizer import summarize

logger = structlog.get_logger(__name__)
router = APIRouter()


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
        cached = True
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

    # 4. Execute against Postgres with timeout + row cap.
    try:
        result = await execute_query(guardrail_result.sanitized_sql, row_cap=row_cap)
    except QueryExecutionError as exc:
        status_code = 504 if exc.is_timeout else 400
        raise HTTPException(status_code=status_code, detail=str(exc)) from exc

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
    )
