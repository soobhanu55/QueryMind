"""A failed dispose must still drop the engine, or the next Streamlit question reuses an engine
bound to a closed event loop ("Event loop is closed")."""
import asyncio

import pytest

from app import db


class _BrokenEngine:
    async def dispose(self):
        raise RuntimeError("connection already dead")


def test_failed_dispose_still_resets_engine():
    db._engine, db._session_factory = _BrokenEngine(), object()
    with pytest.raises(RuntimeError):
        asyncio.run(db.dispose_engine())
    assert db._engine is None and db._session_factory is None
