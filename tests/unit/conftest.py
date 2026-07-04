import pytest_asyncio

from app.db import dispose_engine


@pytest_asyncio.fixture(autouse=True)
async def _dispose_db_engine_after_each_test():
    """pytest-asyncio gives each test function its own event loop, but app.db caches a
    module-level engine/pool tied to whichever loop created it. Reusing that pool from
    a later test's (different) loop crashes on teardown. Disposing after every test
    forces a fresh engine bound to the next test's loop."""
    yield
    await dispose_engine()
