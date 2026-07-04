from __future__ import annotations

import structlog
from anthropic import AsyncAnthropic

from app.config import get_settings
from app.llm.base import NL2SQLProvider, SQLGenerationResult
from app.prompt import GENERATE_SQL_TOOL, build_prompt

logger = structlog.get_logger(__name__)


class AnthropicProvider(NL2SQLProvider):
    """Generates SQL via a forced tool-use call so output is always structured JSON."""

    def __init__(self):
        settings = get_settings()
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set; cannot use llm_provider=anthropic")
        self._client = AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    async def generate(self, question: str, schema_text: str) -> SQLGenerationResult:
        system_prompt, user_prompt = build_prompt(question, schema_text)
        response = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            tools=[GENERATE_SQL_TOOL],
            tool_choice={"type": "tool", "name": "generate_sql"},
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "generate_sql":
                data = block.input
                return SQLGenerationResult(
                    sql=data["sql"],
                    confidence=float(data["confidence"]),
                    explanation=data["explanation"],
                )

        raise RuntimeError("Anthropic response did not contain a generate_sql tool call")
