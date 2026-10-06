"""Shared parsing for providers that answer with free-form text instead of a tool call or response schema."""
from __future__ import annotations

import json
import re

from app.llm.base import SQLGenerationResult

JSON_INSTRUCTION = (
    'Respond with ONLY a JSON object: {"sql": "<one read-only PostgreSQL SELECT>", "confidence": <0..1>, '
    '"explanation": "<one sentence>"}. No markdown fences, no text outside the JSON.'
)

_SELECT_RE = re.compile(r"\bselect\b.*|\bwith\s+\w+\s+as\s*\(.*", re.IGNORECASE | re.DOTALL)


def parse_sql_response(text: str) -> SQLGenerationResult:
    """JSON object preferred; otherwise the first SELECT/WITH statement found in the text (confidence 0.3)."""
    cleaned = re.sub(r"^```(?:json|sql)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data.get("sql"), str) and data["sql"].strip():
                return SQLGenerationResult(
                    sql=data["sql"].strip().rstrip(";"),
                    confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
                    explanation=str(data.get("explanation", "")),
                )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    sql = _SELECT_RE.search(cleaned)
    if not sql:
        raise ValueError(f"no SQL found in model output: {text[:120]!r}")
    return SQLGenerationResult(sql=sql.group(0).split(";")[0].strip(), confidence=0.3,
                               explanation="SQL extracted from unstructured model output.")
