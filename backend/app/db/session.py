"""
Async database engine and session management.

Purpose
-------
Owns the SQLAlchemy async engine and hands out sessions to the API (via FastAPI
dependency injection) and to scripts (via an explicit context manager).

How it works
------------
One engine per process, created lazily. ``pool_pre_ping`` is on because
connections sit idle between requests and Postgres or an intervening proxy may
drop them; without it the first query after an idle period fails.

Failure scenarios
-----------------
* Postgres not yet accepting connections at container start → ``wait_for_db``
  retries with backoff before the app declares itself ready.
* Connection dropped mid-idle → ``pool_pre_ping`` transparently reconnects.
* Pool exhausted → requests block, then raise ``TimeoutError``. Raise
  ``DB_POOL_SIZE`` if this appears in logs; the default of 5 is sized for a
  single-user workload.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.sql import text

from app.config import get_settings
from app.logging_config import get_logger

log = get_logger(__name__)

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Return the process-wide async engine, creating it on first use."""
    global _engine
    if _engine is None:
        s = get_settings()
        _engine = create_async_engine(
            s.database_url,
            echo=s.db_echo,
            pool_size=s.db_pool_size,
            max_overflow=s.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=1800,
            # Server-side statement cache is disabled because pgbouncer-style
            # poolers break prepared statements. Harmless here, and it keeps the
            # door open for adding a pooler later.
            connect_args={"statement_cache_size": 0},
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory."""
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            bind=get_engine(),
            expire_on_commit=False,
            autoflush=False,
        )
    return _sessionmaker


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """Transactional scope for non-request code (scripts, CLI).

    Commits on clean exit, rolls back on exception, always closes.
    """
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a session.

    Deliberately does *not* auto-commit: route handlers commit explicitly so
    that the transaction boundary is visible in the code that owns it.
    """
    factory = get_sessionmaker()
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise


async def wait_for_db(*, attempts: int = 30, delay: float = 2.0) -> None:
    """Block until Postgres answers ``SELECT 1``, or raise after ``attempts``.

    Called at container start. Compose ``depends_on: service_healthy`` already
    covers the common case; this handles the window where Postgres is accepting
    TCP but has not finished recovery.
    """
    engine = get_engine()
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            log.info("database.ready", attempt=attempt)
            return
        except Exception as exc:  # noqa: BLE001 — any driver error means "not ready yet"
            last_error = exc
            log.warning("database.not_ready", attempt=attempt, max_attempts=attempts)
            await asyncio.sleep(delay)
    raise RuntimeError(f"Database unreachable after {attempts} attempts") from last_error


async def dispose_engine() -> None:
    """Close all pooled connections. Called on graceful shutdown."""
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _sessionmaker = None
