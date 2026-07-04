from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Natural-language analytics question.")
    max_rows: Optional[int] = Field(None, ge=1, le=500, description="Optional row cap, capped by server config.")


class SQLGenerationResult(BaseModel):
    sql: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str


class QueryResponse(BaseModel):
    question: str
    sql: str
    confidence: float
    explanation: str
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    execution_ms: float
    guardrail_check_ms: float
    generation_ms: float
    cached: bool
    summary: str


class BlockedQueryError(BaseModel):
    detail: str
    reason: str
    category: str


class HealthResponse(BaseModel):
    status: str
    db_ok: bool
    cache_backend: str
