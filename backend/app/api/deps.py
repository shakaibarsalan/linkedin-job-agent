"""
API dependencies: authentication and shared resources.

Purpose
-------
Provides the `X-API-Key` guard and the shared singletons (settings, database)
routes depend on.

Security note
-------------
The API binds to ``127.0.0.1`` on the host, so the network boundary is the
primary control and the API key is defence in depth. Both matter: anyone with a
shell on the box could otherwise reach an endpoint that reads your resume, your
cover letters, and your application history.

Comparison uses :func:`secrets.compare_digest` — a plain ``==`` on a secret is
timing-attack-prone, and costs nothing to avoid.

Failure scenarios
-----------------
* Missing/incorrect key → 401 with a ``WWW-Authenticate`` header.
* Health and docs endpoints are deliberately unauthenticated so Docker's
  healthcheck works without embedding a secret in the compose file.
"""

from __future__ import annotations

import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.session import get_db
from app.logging_config import get_logger

log = get_logger(__name__)


async def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> None:
    """Reject a request without a valid ``X-API-Key`` header."""
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.api_key):
        log.warning("api.unauthorized", key_present=bool(x_api_key))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="A valid X-API-Key header is required",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def db_session() -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped database session."""
    async for session in get_db():
        yield session


# Reusable annotated types, so route signatures stay readable.
DbSession = Annotated[AsyncSession, Depends(db_session)]
AppSettings = Annotated[Settings, Depends(get_settings)]
Authenticated = Depends(require_api_key)
