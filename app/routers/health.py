from fastapi import APIRouter
from sqlalchemy import text

from app.cache import get_cache
from app.db import session_scope
from app.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    db_ok = True
    try:
        async with session_scope() as session:
            await session.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001
        db_ok = False

    cache = await get_cache()
    backend = "redis" if cache._is_redis else "in_memory"

    return HealthResponse(status="ok" if db_ok else "degraded", db_ok=db_ok, cache_backend=backend)
