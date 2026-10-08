from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/analytics"
    db_pool_min_size: int = 5
    db_pool_max_size: int = 20
    db_pool_max_overflow: int = 10
    query_timeout_seconds: float = 5.0
    max_result_rows: int = 500

    # --- cache ---
    # Empty by default -- an unset REDIS_URL means "no Redis in this deployment,
    # use the in-memory fallback", rather than attempting (and failing) to reach
    # a default localhost Redis that isn't guaranteed to exist. Local dev/docker-
    # compose sets this explicitly via .env.
    redis_url: str = ""
    cache_enabled: bool = True
    schema_cache_ttl_seconds: int = 3600
    question_cache_ttl_seconds: int = 600

    # --- llm ---
    llm_provider: Literal["mock", "anthropic", "gemini", "groq", "local", "routed"] = "mock"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    local_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # --- routing (llm_provider="routed"): the rule-based parser answers first, a model only when it is not confident ---
    router_llm: Literal["anthropic", "gemini", "groq", "local"] = "groq"
    router_min_confidence: float = 0.55  # rules answers below this escalate to the model
    router_timeout_seconds: float = 30.0
    router_llm_daily_calls: int = 500  # budget guard: beyond it the model tier is skipped for the rest of the UTC day
    # questions matching this never go to a hosted API (only to the rules, or to a local model)
    router_private_pattern: str = r"\b(e-?mails?|phone numbers?|addresses)\b"

    # --- guardrails ---
    guardrail_row_limit_default: int = 200

    # --- app ---
    log_level: str = "INFO"
    app_env: Literal["dev", "test", "prod"] = "dev"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
