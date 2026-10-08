from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SQLGenerationResult:
    sql: str
    confidence: float
    explanation: str
    # filled in by the providers that report usage and by the router
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    route: str | None = None
    escalated: bool = False


class NL2SQLProvider(ABC):
    @abstractmethod
    async def generate(self, question: str, schema_text: str) -> SQLGenerationResult:
        """Generate a single read-only SQL statement (not yet guardrail-checked) for `question`."""
        raise NotImplementedError
