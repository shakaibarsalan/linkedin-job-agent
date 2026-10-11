"""Extensible notification layer: Telegram, Discord, Slack, email."""

from app.services.notifications.base import (
    DeliveryResult,
    Notification,
    NotificationChannel,
    Priority,
    build_payload_fields,
)
from app.services.notifications.channels import (
    DiscordChannel,
    EmailChannel,
    SlackChannel,
    TelegramChannel,
)
from app.services.notifications.dispatcher import (
    Dispatcher,
    build_channels,
    review_required,
)

__all__ = [
    "DeliveryResult",
    "DiscordChannel",
    "Dispatcher",
    "EmailChannel",
    "Notification",
    "NotificationChannel",
    "Priority",
    "SlackChannel",
    "TelegramChannel",
    "build_channels",
    "build_payload_fields",
    "review_required",
]
