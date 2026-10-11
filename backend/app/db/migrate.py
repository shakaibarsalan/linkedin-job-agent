"""
Forward-only SQL migration runner.

Purpose
-------
Applies ``database/migrations/NNN_*.sql`` in filename order, exactly once each,
recording applied versions in ``schema_migrations``.

How it works
------------
Each file runs inside its own transaction. If a file fails, the transaction
rolls back and the runner aborts — so a partially applied migration can never
be recorded as complete. Already-applied versions are skipped, which makes the
runner safe to invoke on every container start.

Why this bypasses SQLAlchemy and drives asyncpg directly
--------------------------------------------------------
A migration file contains many statements. SQLAlchemy's asyncpg dialect routes
**every** statement — including ``exec_driver_sql()`` — through
``asyncpg.Connection.prepare()``, and PostgreSQL rejects that::

    asyncpg.exceptions.PostgresSyntaxError:
        cannot insert multiple commands into a prepared statement

The only way to execute a multi-statement script is PostgreSQL's *simple query*
protocol, which asyncpg uses when ``Connection.execute()`` is called with **no**
bind arguments. So this module takes the raw asyncpg connection out of the
SQLAlchemy pool and uses it directly.

That also sidesteps a second problem: ``text()`` scans for ``:name`` bind
placeholders, which is simply wrong for an arbitrary SQL file.

The advisory lock is **session**-scoped, not transaction-scoped, so it is held
for the whole run regardless of the per-migration transactions.

Configuration
-------------
``MIGRATIONS_DIR`` (default ``/app/migrations`` in the image).

Failure scenarios
-----------------
* SQL error in a migration → that transaction rolls back, the runner raises,
  the API refuses to start, the container is unhealthy, and the deploy rolls
  back. The previous container keeps serving throughout. See docs/ROLLBACK.md.
* Two containers starting at once → the advisory lock serialises them; the
  loser waits, then finds every version already applied and no-ops.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import sys
from typing import Any

from app.db.session import dispose_engine, get_engine, wait_for_db
from app.logging_config import configure_logging, get_logger

log = get_logger(__name__)

# Arbitrary but fixed: identifies *this* application's migration lock so we can
# never collide with another tenant of the same Postgres instance.
_ADVISORY_LOCK_KEY = 8_140_233_901


def _migrations_dir() -> pathlib.Path:
    return pathlib.Path(os.environ.get("MIGRATIONS_DIR", "/app/migrations"))


def discover_migrations(directory: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Return migration files sorted by filename (which encodes ordering)."""
    directory = directory or _migrations_dir()
    if not directory.is_dir():
        raise FileNotFoundError(f"Migrations directory not found: {directory}")
    return sorted(directory.glob("*.sql"))


async def _applied_versions(pg: Any) -> set[str]:
    """Read already-applied versions, tolerating a virgin database."""
    present = await pg.fetchval("SELECT to_regclass('public.schema_migrations') IS NOT NULL")
    if not present:
        return set()
    rows = await pg.fetch("SELECT version FROM schema_migrations")
    return {row["version"] for row in rows}


async def run_migrations(directory: pathlib.Path | None = None) -> list[str]:
    """Apply all pending migrations. Returns the versions applied this call."""
    files = discover_migrations(directory)
    if not files:
        log.warning("migrations.none_found", directory=str(directory or _migrations_dir()))
        return []

    engine = get_engine()
    applied_now: list[str] = []

    async with engine.connect() as conn:
        # Drop to the raw asyncpg connection. See the module docstring: a
        # multi-statement script can only run over the simple query protocol,
        # which SQLAlchemy never uses.
        raw = await conn.get_raw_connection()
        pg: Any = raw.driver_connection
        if pg is None:  # pragma: no cover — would mean a driver/dialect change
            raise RuntimeError(
                "No raw asyncpg connection available. Migrations require the "
                "asyncpg driver; check that DATABASE_URL uses postgresql+asyncpg://"
            )

        # Serialise concurrent runners. Session-scoped, so it is held across
        # the per-migration transactions below.
        await pg.execute("SELECT pg_advisory_lock($1)", _ADVISORY_LOCK_KEY)

        try:
            already = await _applied_versions(pg)
            log.info("migrations.discovered", total=len(files), already_applied=len(already))

            for path in files:
                version = path.stem
                if version in already:
                    log.debug("migrations.skip", version=version)
                    continue

                sql = path.read_text(encoding="utf-8")
                log.info("migrations.apply", version=version)

                try:
                    # Commits on clean exit, rolls back on exception — so a
                    # half-applied migration can never be recorded as complete.
                    async with pg.transaction():
                        # No bind arguments: this is what selects the simple
                        # query protocol and allows multiple statements.
                        await pg.execute(sql)
                        # Migrations are expected to self-record, but one that
                        # forgets must not be re-applied on the next boot.
                        await pg.execute(
                            "INSERT INTO schema_migrations (version) VALUES ($1) "
                            "ON CONFLICT (version) DO NOTHING",
                            version,
                        )
                except Exception:
                    log.error("migrations.failed", version=version, exc_info=True)
                    raise

                applied_now.append(version)
                log.info("migrations.applied", version=version)

        finally:
            # Best-effort. The lock is released anyway when the connection
            # returns to the pool, so a failure here must not mask the real error.
            try:
                await pg.execute("SELECT pg_advisory_unlock($1)", _ADVISORY_LOCK_KEY)
            except Exception:  # noqa: BLE001 — cleanup must not shadow the real failure
                log.warning("migrations.unlock_failed", exc_info=True)

    log.info("migrations.complete", applied=len(applied_now), versions=applied_now)
    return applied_now


async def _main() -> int:
    from app.config import get_settings

    s = get_settings()
    configure_logging(level=s.log_level, fmt=s.log_format, log_dir=s.log_dir, component="migrate")
    try:
        await wait_for_db()
        await run_migrations()
        return 0
    except Exception:  # noqa: BLE001 — CLI boundary: any failure becomes exit code 1
        log.critical("migrations.aborted", exc_info=True)
        return 1
    finally:
        await dispose_engine()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
