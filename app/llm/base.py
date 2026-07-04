from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SQLGenerationResult:
    sql: str
    confidence: float
    explanation: str


class NL2SQLProvider(ABC):
    @abstractmethod
    async def generate(self, question: str, schema_text: str) -> SQLGenerationResult:
        """Generate a single read-only SQL statement (not yet guardrail-checked) for `question`."""
        raise NotImplementedError
