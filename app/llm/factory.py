from functools import lru_cache

from app.config import get_settings
from app.llm.base import NL2SQLProvider


@lru_cache(maxsize=1)
def get_provider() -> NL2SQLProvider:
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if settings.llm_provider == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider()
    from app.llm.mock_provider import MockNL2SQLProvider

    return MockNL2SQLProvider()
