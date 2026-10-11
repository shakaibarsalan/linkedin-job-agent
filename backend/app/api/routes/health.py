"""
Health and readiness endpoints.

Purpose
-------
Backs the Docker healthcheck. If this endpoint is wrong, a broken container
looks healthy.

How it works
------------
``/health`` is a liveness probe: the process is up. Deliberately cheap and
dependency-free so a Postgres blip does not cause Docker to kill a healthy API.

``/health/ready`` is a readiness probe: the database is reachable. An API that
cannot reach its database is not ready, however healthy the process is.

Both are unauthenticated — Docker's healthcheck would otherwise need the API key
baked into the compose file, and neither endpoint discloses anything sensitive.

Failure scenarios
-----------------
* Postgres down → ``/health/ready`` returns 503 with ``checks.database=false``.
* AI provider down → reported but **not** fatal: the system degrades to keyword
  scoring rather than refusing to serve.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app import __version__
from app.api.deps import AppSettings, DbSession
from app.logging_config import get_logger
from app.schemas.api import Health

log = get_logger(__name__)

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Health, summary="Liveness probe")
async def health(settings: AppSettings) -> Health:
    """Report that the process is alive. Never touches a dependency."""
    return Health(
        status="ok",
        version=__version__,
        environment=settings.environment,
        checks={"process": True},
    )


@router.get(
    "/health/ready",
    response_model=Health,
    summary="Readiness probe — verifies dependencies",
    responses={503: {"description": "A required dependency is unavailable"}},
)
async def readiness(
    response: Response,
    settings: AppSettings,
    session: DbSession,
) -> Health:
    """Verify Postgres. Returns 503 if it is unreachable."""
    checks: dict[str, bool] = {"process": True}
    detail: dict[str, str] = {}

    try:
        await session.execute(text("SELECT 1"))
        checks["database"] = True
    except Exception as exc:  # noqa: BLE001 — the point is to report, not raise
        checks["database"] = False
        detail["database"] = str(exc)[:300]
        log.error("health.database_unreachable", error=str(exc)[:300])

    # Required dependencies. The AI provider is intentionally absent from this
    # list — the system degrades to rules-based scoring without it.
    required = ("database",)
    ready = all(checks.get(name, False) for name in required)

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return Health(
        status="ok" if ready else "error",
        version=__version__,
        environment=settings.environment,
        checks=checks,
        detail=detail,
    )
