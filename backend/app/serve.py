"""
API server entrypoint.

Purpose
-------
Starts uvicorn with logging fully delegated to structlog.

Why this file exists rather than a bare `uvicorn` command
---------------------------------------------------------
uvicorn's ``--log-config`` CLI flag expects a **file** and parses it with
``logging.config.fileConfig``. There is no CLI way to say "do not configure
logging at all" — passing ``/dev/null`` raises ``RuntimeError: /dev/null is an
empty file``, and omitting the flag makes uvicorn install its own handlers,
which then emit plain text alongside our JSON.

The Python API *does* accept ``log_config=None``, which is exactly what we want:
:func:`app.logging_config.configure_logging` has already set up the root logger,
and uvicorn should leave it alone.

Failure scenarios
-----------------
* Port already bound → uvicorn exits non-zero, the container is unhealthy, and
  the deploy rolls back. Check nothing else holds the port inside the container.
"""

from __future__ import annotations

import sys

import uvicorn

from app.config import get_settings
from app.logging_config import configure_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    """Run the API server. Returns a process exit code."""
    settings = get_settings()

    # Configure logging BEFORE uvicorn starts so its own loggers inherit our
    # root handlers rather than installing their own.
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        log_dir=settings.log_dir,
        component="api",
    )

    log.info(
        "serve.starting",
        host=settings.api_host,
        port=settings.api_port,
        environment=settings.environment,
    )

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        # Single worker by design: the API is a control plane for one person,
        # and multiple workers would each try to run migrations at startup.
        # (The advisory lock makes that safe, but pointless.)
        workers=1,
        # The whole reason this module exists.
        log_config=None,
        # Requests are logged by our own middleware with timing and structure.
        access_log=False,
        # Give in-flight requests a chance to finish on SIGTERM.
        timeout_graceful_shutdown=30,
        reload=settings.environment == "development",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
