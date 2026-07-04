from __future__ import annotations

import json

import structlog
from google import genai
from google.genai import types

from app.config import get_settings
from app.llm.base import NL2SQLProvider, SQLGenerationResult
from app.prompt import build_prompt

logger = structlog.get_logger(__name__)

_RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "sql": {"type": "STRING", "description": "A single read-only PostgreSQL SELECT statement."},
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence between 0 and 1 that this SQL correctly answers the question.",
        },
        "explanation": {"type": "STRING", "description": "One-sentence explanation of what the query does."},
    },
    "required": ["sql", "confidence", "explanation"],
}


class GeminiProvider(NL2SQLProvider):
    """Generates SQL via Gemini's structured-output (response_schema) mode."""

    def __init__(self):
        settings = get_settings()
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set; cannot use llm_provider=gemini")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    async def generate(self, question: str, schema_text: str) -> SQLGenerationResult:
        system_prompt, user_prompt = build_prompt(question, schema_text)
        response = await self._client.aio.models.generate_content(
            model=self._model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                response_mime_type="application/json",
                response_schema=_RESPONSE_SCHEMA,
                temperature=0,
            ),
        )

        if not response.text:
            raise RuntimeError("Gemini response contained no text output")

        data = json.loads(response.text)
        return SQLGenerationResult(
            sql=data["sql"],
            confidence=float(data["confidence"]),
            explanation=data["explanation"],
        )
