"""
FastAPI application entrypoint.

Purpose
-------
Assembles the API: logging, middleware, routers, lifespan, and error handling.

How it works
------------
On startup the app waits for Postgres, applies pending migrations, and logs the
effective review policy, so the container logs always show what rules this
deployment is actually running under. On shutdown it closes the database pool.

Migrations run here rather than in a separate init container so that a rollback
to a previous image cannot leave the schema ahead of the code in a way nobody
notices.

Security
--------
Binds ``0.0.0.0`` **inside the container**; Docker publishes it as
``127.0.0.1:8088`` on the host, so nothing is reachable from the internet. CORS
is closed by default. All routes except ``/health*`` and the docs require
``X-API-Key``.

Failure scenarios
-----------------
* Migration failure at startup → the process exits non-zero and the container
  reports unhealthy.
* Database unreachable → ``wait_for_db`` retries for ~60 s before giving up.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from app import __version__
from app.api.routes import (
    applications,
    health,
    intake,
    resumes,
    stats,
    system,
)
from app.config import get_settings
from app.db.migrate import run_migrations
from app.db.session import dispose_engine, wait_for_db
from app.logging_config import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Start-up and shut-down sequence."""
    settings = get_settings()

    log.info("api.starting", version=__version__, environment=settings.environment)

    await wait_for_db()
    applied = await run_migrations()
    if applied:
        log.info("api.migrations_applied", versions=applied)

    # Logged every boot: the container's own record of the rules it runs under.
    log.info(
        "api.review_policy",
        review_mode=settings.review_mode,
        min_score_to_apply=settings.min_score_to_apply,
        auto_approve_threshold=settings.auto_apply_score_threshold,
        ai_provider=settings.ai_provider,
        notifications_configured=settings.any_notifier_enabled,
    )

    if not settings.any_notifier_enabled:
        log.warning(
            "api.no_notifications_configured",
            impact="Review requests will only be visible through the API.",
        )

    yield

    log.info("api.shutting_down")
    await dispose_engine()
    log.info("api.stopped")


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()

    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        log_dir=settings.log_dir,
        component="api",
    )

    app = FastAPI(
        title="LinkedIn Job Agent",
        description=(
            "Scores jobs against your CV, drafts cover letters that are checked "
            "against it, and queues everything for you to review. It never applies "
            "for you.\n\n"
            "**Authentication:** every endpoint except `/health*` requires an "
            "`X-API-Key` header.\n\n"
            "**Access:** the API binds to `127.0.0.1` on the host."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS is empty by default. Nothing needs cross-origin access unless you
    # add a UI on another origin.
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["X-API-Key", "Content-Type"],
        )
        log.info("api.cors_enabled", origins=settings.cors_origins)

    @app.middleware("http")
    async def log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Log every request with its duration, excluding health-check noise."""
        started = time.monotonic()
        response = await call_next(request)
        duration_ms = (time.monotonic() - started) * 1000

        if not request.url.path.startswith("/health"):
            log.info(
                "api.request",
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=round(duration_ms, 1),
            )
        response.headers["X-Response-Time-Ms"] = f"{duration_ms:.1f}"
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
        """Return field-level validation detail instead of a bare 422."""
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed",
                "errors": [
                    {
                        "field": ".".join(str(p) for p in err["loc"][1:]),
                        "message": err["msg"],
                    }
                    for err in exc.errors()
                ],
            },
        )

    @app.exception_handler(IntegrityError)
    async def integrity_handler(_request: Request, exc: IntegrityError) -> JSONResponse:
        """Turn a constraint violation into a comprehensible 409.

        The most common one by far is the unique index on
        ``applications.job_id`` — the never-apply-twice guard doing its job.
        """
        log.warning("api.integrity_error", error=str(exc.orig)[:300])
        detail = "This conflicts with existing data"
        if "ux_applications_job" in str(exc.orig):
            detail = "An application for this job already exists — it is never applied to twice"
        return JSONResponse(status_code=409, content={"detail": detail})

    @app.exception_handler(Exception)
    async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
        """Log the full trace, return a message without internals."""
        log.error(
            "api.unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc)[:500],
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "Internal server error. Check the API logs for the trace.",
            },
        )

    app.include_router(health.router)
    for module in (resumes, applications, stats, system, intake):
        app.include_router(module.router, prefix="/api/v1")

    return app


app = create_app()


if __name__ == "__main__":  # pragma: no cover — dev convenience only
    import uvicorn

    s = get_settings()
    uvicorn.run(
        "app.main:app",
        host=s.api_host,
        port=s.api_port,
        reload=s.environment == "development",
        log_config=None,  # structlog owns logging
    )
