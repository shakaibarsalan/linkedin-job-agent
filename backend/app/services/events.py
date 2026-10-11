"""
Audit event recording.

Purpose
-------
Writes an append-only trail of what the agent did and why. Logs rotate and get
lost; this table is the durable record that answers "why did it apply to that?"
weeks later, and it is what the dashboard timeline renders.

How it works
------------
Thin wrapper over the ``events`` table. Writes are flushed but not committed —
the caller owns the transaction boundary, so an event and the state change it
describes commit together or not at all.

Failure scenarios
-----------------
* Insert fails → logged at warning, never raised. Losing an audit row must not
  fail the run it is describing.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Event
from app.logging_config import get_logger

log = get_logger(__name__)


async def record(
    session: AsyncSession,
    *,
    event_type: str,
    message: str = "",
    level: str = "info",
    run_id: int | None = None,
    application_id: int | None = None,
    job_id: int | None = None,
    data: dict[str, Any] | None = None,
) -> None:
    """Append one audit event.

    Args:
        session: Open session; the row is flushed, not committed.
        event_type: Stable machine-readable slug, e.g. ``"job.scored"``.
        message: Human-readable summary.
        level: ``debug`` | ``info`` | ``warning`` | ``error`` | ``critical``.
        run_id / application_id / job_id: Optional correlation keys.
        data: Structured detail, stored as JSONB.
    """
    try:
        session.add(
            Event(
                run_id=run_id,
                application_id=application_id,
                job_id=job_id,
                level=level,
                event_type=event_type,
                message=message[:4000],
                data=_sanitise(data or {}),
            )
        )
        await session.flush()
    except Exception:  # noqa: BLE001 — audit must never break the operation
        log.warning("events.write_failed", event_type=event_type, exc_info=True)


# Never persist these into JSONB, even if a caller passes them by mistake.
_FORBIDDEN_KEYS = frozenset(
    {"password", "api_key", "token", "cookie", "session_state", "secret", "authorization"}
)


def _sanitise(data: dict[str, Any]) -> dict[str, Any]:
    """Drop secret-shaped keys and bound the size of the payload."""
    cleaned: dict[str, Any] = {}
    for key, value in list(data.items())[:50]:
        if any(part in key.lower() for part in _FORBIDDEN_KEYS):
            cleaned[key] = "***REDACTED***"
        elif isinstance(value, str):
            cleaned[key] = value[:2000]
        elif isinstance(value, list):
            cleaned[key] = [str(v)[:200] for v in value[:30]]
        else:
            cleaned[key] = value
    return cleaned
