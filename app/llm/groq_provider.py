from __future__ import annotations

from dataclasses import replace

import httpx

from app.config import get_settings
from app.llm.base import NL2SQLProvider, SQLGenerationResult
from app.llm.json_sql import JSON_INSTRUCTION, parse_sql_response
from app.prompt import build_prompt

URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqProvider(NL2SQLProvider):
    """Llama 3.3 70B on Groq's free tier (OpenAI-compatible API, JSON mode, temperature 0)."""

    def __init__(self):
        settings = get_settings()
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set; cannot use llm_provider=groq")
        self._key, self._model = settings.groq_api_key, settings.groq_model

    async def generate(self, question: str, schema_text: str, retry: tuple[str, str] | None = None) -> SQLGenerationResult:
        system_prompt, user_prompt = build_prompt(question, schema_text, retry)
        body = {
            "model": self._model, "temperature": 0, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system_prompt + "\n" + JSON_INSTRUCTION},
                         {"role": "user", "content": user_prompt}],
        }
        async with httpx.AsyncClient(timeout=60) as client:
            for attempt in range(4):  # the free tier rate-limits (429): back off and retry
                r = await client.post(URL, json=body, headers={"Authorization": f"Bearer {self._key}"})
                if r.status_code != 429:
                    break
                import asyncio
                await asyncio.sleep(2 ** attempt * 2)
            r.raise_for_status()
        data = r.json()
        usage = data.get("usage") or {}
        return replace(parse_sql_response(data["choices"][0]["message"]["content"]), model=self._model,
                       input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"))
