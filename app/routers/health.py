from fastapi import APIRouter
from sqlalchemy import text

from app.cache import get_cache
from app.db import is_embedded, session_scope
from app.llm import get_provider
from app.llm.router import RoutedProvider, snapshot
from app.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    db_ok = True
    try:
        if is_embedded():
            from app import embedded

            embedded.connection().execute("SELECT 1")
        else:
            async with session_scope() as session:
                await session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_ok = False

    cache = await get_cache()
    backend = "redis" if cache._is_redis else "in_memory"

    return HealthResponse(status="ok" if db_ok else "degraded", db_ok=db_ok, cache_backend=backend)


@router.get("/llm/stats")
async def llm_stats() -> dict:
    """Routing counters since start: calls per tier, escalations, fallbacks, tokens, and each tier's breaker state."""
    provider = get_provider()
    return snapshot(provider if isinstance(provider, RoutedProvider) else None)
