"""
System status and audit endpoints.

Purpose
-------
Answers "is this thing configured the way I think it is?" without shelling into
the container, and exposes the audit event stream.

Security note
-------------
:func:`system_status` reports configuration *shape* — which provider, which
model, which review policy — and never a secret value. The API key and AI key
are not exposed by any endpoint.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query
from sqlalchemy import select

from app import __version__
from app.api.deps import AppSettings, Authenticated, DbSession
from app.db.models import Event
from app.logging_config import get_logger
from app.schemas.api import SystemStatus

log = get_logger(__name__)

router = APIRouter(prefix="/system", tags=["system"], dependencies=[Authenticated])


@router.get("/status", response_model=SystemStatus, summary="Configuration snapshot")
async def system_status(settings: AppSettings) -> SystemStatus:
    """Report effective configuration. Never returns a secret value."""
    enabled_channels = [
        name
        for name, enabled in (
            ("telegram", settings.notify_telegram_enabled),
            ("discord", settings.notify_discord_enabled),
            ("slack", settings.notify_slack_enabled),
            ("email", settings.notify_email_enabled),
        )
        if enabled
    ]

    return SystemStatus(
        version=__version__,
        environment=settings.environment,
        review_mode=settings.review_mode,
        min_score_to_apply=settings.min_score_to_apply,
        auto_approve_threshold=settings.auto_apply_score_threshold,
        ai_provider=settings.ai_provider,
        ai_model=settings.ai_model if settings.ai_provider != "rules" else "n/a",
        notifications=enabled_channels,
    )


@router.get("/events", summary="Audit event stream")
async def list_events(
    session: DbSession,
    run_id: int | None = None,
    application_id: int | None = None,
    level: Annotated[str | None, Query(description="debug|info|warning|error|critical")] = None,
    limit: int = 100,
) -> list[dict[str, object]]:
    """Recent audit events, newest first.

    This is the durable record of what the agent did and why — logs rotate,
    this table does not.
    """
    conditions = []
    if run_id is not None:
        conditions.append(Event.run_id == run_id)
    if application_id is not None:
        conditions.append(Event.application_id == application_id)
    if level:
        conditions.append(Event.level == level)

    stmt = select(Event).where(*conditions).order_by(Event.created_at.desc()).limit(min(limit, 500))
    rows = (await session.execute(stmt)).scalars().all()

    return [
        {
            "id": event.id,
            "created_at": event.created_at.isoformat(),
            "level": event.level,
            "event_type": event.event_type,
            "message": event.message,
            "run_id": event.run_id,
            "application_id": event.application_id,
            "job_id": event.job_id,
            "data": event.data,
        }
        for event in rows
    ]
