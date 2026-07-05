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
    llm_provider: Literal["mock", "anthropic", "gemini"] = "mock"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # --- guardrails ---
    guardrail_row_limit_default: int = 200

    # --- app ---
    log_level: str = "INFO"
    app_env: Literal["dev", "test", "prod"] = "dev"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
