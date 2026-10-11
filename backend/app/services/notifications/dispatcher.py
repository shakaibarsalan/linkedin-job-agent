"""
Notification fan-out.

Purpose
-------
Sends one notification to every configured channel concurrently, records the
outcome per channel, and guarantees the caller never sees an exception.

How it works
------------
Channels are built from settings at construction. :meth:`Dispatcher.send`
gathers all deliveries with ``return_exceptions=True``, so one dead webhook
cannot prevent the others. Every attempt is written to ``notification_log``,
which is what lets you answer "did I actually get told about that?" after
the fact.

Failure scenarios
-----------------
* **No channels configured** → :meth:`send` logs at warning and returns. The
  system still works; you just find out by looking at the dashboard.
* **Channel raises** → captured as a :class:`DeliveryResult` with ``ok=False``.
* **Database unavailable** → the log write is skipped, delivery still happens.
"""

from __future__ import annotations

import asyncio

from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import NotificationLog
from app.logging_config import get_logger
from app.services.notifications.base import (
    DeliveryResult,
    Notification,
    NotificationChannel,
    Priority,
)
from app.services.notifications.channels import (
    DiscordChannel,
    EmailChannel,
    SlackChannel,
    TelegramChannel,
)

log = get_logger(__name__)


def build_channels(settings: Settings | None = None) -> list[NotificationChannel]:
    """Instantiate every channel the configuration enables."""
    s = settings or get_settings()

    candidates: list[tuple[bool, type[NotificationChannel]]] = [
        (s.notify_telegram_enabled, TelegramChannel),
        (s.notify_discord_enabled, DiscordChannel),
        (s.notify_slack_enabled, SlackChannel),
        (s.notify_email_enabled, EmailChannel),
    ]

    channels: list[NotificationChannel] = []
    for enabled, channel_type in candidates:
        if not enabled:
            continue
        channel = channel_type(s)  # type: ignore[call-arg]
        if not channel.configured:
            # Enabled but missing credentials is a config mistake worth shouting
            # about — the user believes they will be notified and will not be.
            log.error(
                "notifications.channel_enabled_but_unconfigured",
                channel=channel.name,
                action="channel skipped; check the corresponding .env values",
            )
            continue
        channels.append(channel)

    if channels:
        log.info("notifications.channels_ready", channels=[c.name for c in channels])
    else:
        log.warning(
            "notifications.none_configured",
            impact="Review requests will not reach you outside the API",
        )
    return channels


class Dispatcher:
    """Fans a notification out to every configured channel."""

    def __init__(
        self,
        channels: list[NotificationChannel] | None = None,
        *,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.channels = channels if channels is not None else build_channels(self.settings)

    @property
    def enabled(self) -> bool:
        return bool(self.channels)

    async def send(
        self, notification: Notification, *, session: AsyncSession | None = None
    ) -> list[DeliveryResult]:
        """Deliver to all channels. Never raises.

        Args:
            notification: What to send.
            session: Optional session for writing ``notification_log`` rows.

        Returns:
            One :class:`DeliveryResult` per channel.
        """
        if not self.channels:
            log.warning(
                "notifications.dropped",
                title=notification.title[:120],
                reason="no channels configured",
            )
            return []

        results = await asyncio.gather(
            *(self._deliver(channel, notification) for channel in self.channels),
            return_exceptions=False,
        )

        if session is not None:
            await self._record(session, notification, results)

        failures = [r for r in results if not r.ok]
        if failures:
            log.warning(
                "notifications.partial_failure",
                failed=[r.channel for r in failures],
                delivered=[r.channel for r in results if r.ok],
            )
        return list(results)

    async def _deliver(
        self, channel: NotificationChannel, notification: Notification
    ) -> DeliveryResult:
        try:
            await channel.send(notification)
            log.info(
                "notifications.sent",
                channel=channel.name,
                event_type=notification.event_type,
                priority=notification.priority.value,
            )
            return DeliveryResult(channel=channel.name, ok=True)
        except Exception as exc:  # noqa: BLE001 — delivery must never propagate
            log.error(
                "notifications.failed",
                channel=channel.name,
                event_type=notification.event_type,
                error=str(exc)[:300],
            )
            return DeliveryResult(channel=channel.name, ok=False, error=str(exc)[:500])

    async def _record(
        self,
        session: AsyncSession,
        notification: Notification,
        results: list[DeliveryResult],
    ) -> None:
        """Write delivery outcomes. Failure here is logged, never raised."""
        try:
            session.add_all(
                [
                    NotificationLog(
                        channel=result.channel,
                        event_type=notification.event_type,
                        subject=notification.title[:500],
                        status="sent" if result.ok else "failed",
                        error=result.error or None,
                    )
                    for result in results
                ]
            )
            await session.flush()
        except Exception:  # noqa: BLE001
            log.warning("notifications.log_write_failed", exc_info=True)

    async def aclose(self) -> None:
        """Close every channel's transport."""
        for channel in self.channels:
            try:
                await channel.aclose()
            except Exception:  # noqa: BLE001
                log.debug("notifications.close_failed", channel=channel.name, exc_info=True)


# ── Prebuilt notifications ────────────────────────────────────────────
# Centralised so wording stays consistent and every call site sets the right
# priority. Anything a human must act on is HIGH or CRITICAL.


def review_required(*, count: int, dashboard_url: str) -> Notification:
    return Notification(
        title=f"{count} job{'s' if count != 1 else ''} waiting for your review",
        body=(
            "Scored, with a cover letter checked against your CV. Nothing is sent "
            "anywhere: you read it, and you apply yourself."
        ),
        priority=Priority.HIGH,
        event_type="review_required",
        url=f"{dashboard_url}/api/v1/applications?status=pending_review",
    )
