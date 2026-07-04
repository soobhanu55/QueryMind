from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.cache import get_cache
from app.db import dispose_engine, get_engine
from app.logging_config import configure_logging
from app.routers import health, query

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    await get_cache()
    yield
    await dispose_engine()


app = FastAPI(
    title="Enterprise Text-to-SQL Analytics Agent",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(health.router, tags=["health"])
app.include_router(query.router, tags=["query"])


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
