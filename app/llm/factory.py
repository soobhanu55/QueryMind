from functools import lru_cache

from app.config import get_settings
from app.llm.base import NL2SQLProvider


def build(name: str) -> NL2SQLProvider:
    """One provider by name (imports are lazy so a missing SDK or key only matters for the provider you use)."""
    if name == "anthropic":
        from app.llm.anthropic_provider import AnthropicProvider

        return AnthropicProvider()
    if name == "gemini":
        from app.llm.gemini_provider import GeminiProvider

        return GeminiProvider()
    if name == "groq":
        from app.llm.groq_provider import GroqProvider

        return GroqProvider()
    if name == "local":
        from app.llm.local_provider import LocalProvider

        return LocalProvider()
    if name == "routed":
        from app.llm.router import RoutedProvider

        return RoutedProvider()
    from app.llm.mock_provider import MockNL2SQLProvider

    return MockNL2SQLProvider()


@lru_cache(maxsize=1)
def get_provider() -> NL2SQLProvider:
    return build(get_settings().llm_provider)
