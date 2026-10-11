"""
Concrete notification channels.

Purpose
-------
Telegram, Discord, Slack, and SMTP email implementations of
:class:`~app.services.notifications.base.NotificationChannel`.

Adding a channel
----------------
Subclass :class:`NotificationChannel`, implement ``configured`` and ``send``,
and add it to :func:`build_channels` in ``dispatcher.py``. Nothing else changes.

Failure scenarios
-----------------
* Bad token/webhook → the provider returns 4xx; the dispatcher logs it and
  records a failed row in ``notification_log``.
* SMTP timeout → email runs in a thread executor so it cannot block the event
  loop; failures are logged, never raised to the caller.
"""

from __future__ import annotations

import asyncio
import smtplib
from email.message import EmailMessage

import httpx

from app.config import Settings
from app.logging_config import get_logger
from app.services.notifications.base import (
    Notification,
    NotificationChannel,
    Priority,
)

log = get_logger(__name__)

_HTTP_TIMEOUT = httpx.Timeout(20.0, connect=10.0)


def _client(settings: Settings) -> httpx.AsyncClient:
    """HTTP client for a notification channel, optionally proxied.

    NOTIFY_PROXY_URL routes ONLY notification traffic. The AI provider always
    uses the direct connection.

    The proxy exists because some ISPs block Telegram (Pakistan among them),
    so a machine can reach everything else it needs but not api.telegram.org.
    """
    kwargs: dict[str, object] = {"timeout": _HTTP_TIMEOUT}
    if settings.notify_proxy_url:
        kwargs["proxy"] = settings.notify_proxy_url
    return httpx.AsyncClient(**kwargs)  # type: ignore[arg-type]


# Colour bar per priority, for the platforms that support it.
_PRIORITY_COLOUR = {
    Priority.LOW: 0x95A5A6,
    Priority.NORMAL: 0x3498DB,
    Priority.HIGH: 0xE67E22,
    Priority.CRITICAL: 0xE74C3C,
}

_PRIORITY_EMOJI = {
    Priority.LOW: "🔹",
    Priority.NORMAL: "✅",
    Priority.HIGH: "⚠️",
    Priority.CRITICAL: "🚨",
}


class TelegramChannel(NotificationChannel):
    """Telegram Bot API."""

    name = "telegram"

    def __init__(self, settings: Settings) -> None:
        self._token = settings.telegram_bot_token
        self._chat_id = settings.telegram_chat_id
        self._client = _client(settings)

    @property
    def configured(self) -> bool:
        return bool(self._token and self._chat_id)

    async def send(self, notification: Notification) -> None:
        emoji = _PRIORITY_EMOJI[notification.priority]
        text = f"{emoji} {notification.as_markdown()}"

        response = await self._client.post(
            f"https://api.telegram.org/bot{self._token}/sendMessage",
            json={
                "chat_id": self._chat_id,
                "text": text[:4096],  # Telegram's hard limit
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            },
        )
        if response.status_code >= 400:
            # Markdown escaping is fiddly; a parse failure should not lose the
            # message, so resend as plain text.
            if "can't parse entities" in response.text.lower():
                log.warning("telegram.markdown_rejected_resending_plain")
                response = await self._client.post(
                    f"https://api.telegram.org/bot{self._token}/sendMessage",
                    json={
                        "chat_id": self._chat_id,
                        "text": f"{emoji} {notification.as_plain_text()}"[:4096],
                        "disable_web_page_preview": True,
                    },
                )
            response.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


class DiscordChannel(NotificationChannel):
    """Discord incoming webhook."""

    name = "discord"

    def __init__(self, settings: Settings) -> None:
        self._webhook_url = settings.discord_webhook_url
        self._client = _client(settings)

    @property
    def configured(self) -> bool:
        return bool(self._webhook_url)

    async def send(self, notification: Notification) -> None:
        embed: dict[str, object] = {
            "title": notification.title[:256],
            "description": notification.body[:4096],
            "color": _PRIORITY_COLOUR[notification.priority],
        }
        if notification.fields:
            embed["fields"] = [
                {"name": key[:256], "value": str(value)[:1024], "inline": True}
                for key, value in list(notification.fields.items())[:25]
            ]
        if notification.url:
            embed["url"] = notification.url

        response = await self._client.post(self._webhook_url, json={"embeds": [embed]})
        response.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


class SlackChannel(NotificationChannel):
    """Slack incoming webhook."""

    name = "slack"

    def __init__(self, settings: Settings) -> None:
        self._webhook_url = settings.slack_webhook_url
        self._client = _client(settings)

    @property
    def configured(self) -> bool:
        return bool(self._webhook_url)

    async def send(self, notification: Notification) -> None:
        emoji = _PRIORITY_EMOJI[notification.priority]
        blocks: list[dict[str, object]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {notification.title}"[:150]},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": notification.body[:3000]},
            },
        ]
        if notification.fields:
            blocks.append(
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*{k}*\n{v}"[:2000]}
                        for k, v in list(notification.fields.items())[:10]
                    ],
                }
            )
        if notification.url:
            blocks.append(
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": f"<{notification.url}|Open>"},
                }
            )

        response = await self._client.post(
            self._webhook_url,
            json={"text": f"{emoji} {notification.title}", "blocks": blocks},
        )
        response.raise_for_status()

    async def aclose(self) -> None:
        await self._client.aclose()


class EmailChannel(NotificationChannel):
    """SMTP email."""

    name = "email"

    def __init__(self, settings: Settings) -> None:
        self._host = settings.smtp_host
        self._port = settings.smtp_port
        self._username = settings.smtp_username
        self._password = settings.smtp_password
        self._from = settings.smtp_from or settings.smtp_username
        self._to = [addr.strip() for addr in settings.smtp_to.split(",") if addr.strip()]
        self._use_tls = settings.smtp_use_tls

    @property
    def configured(self) -> bool:
        return bool(self._host and self._from and self._to)

    async def send(self, notification: Notification) -> None:
        message = EmailMessage()
        message["Subject"] = f"[LinkedIn Agent] {notification.title}"[:200]
        message["From"] = self._from
        message["To"] = ", ".join(self._to)
        message.set_content(notification.as_plain_text())

        # smtplib is blocking; run it off the event loop.
        await asyncio.to_thread(self._send_sync, message)

    def _send_sync(self, message: EmailMessage) -> None:
        if self._port == 465:
            with smtplib.SMTP_SSL(self._host, self._port, timeout=30) as server:
                self._authenticate_and_send(server, message)
        else:
            with smtplib.SMTP(self._host, self._port, timeout=30) as server:
                if self._use_tls:
                    server.starttls()
                self._authenticate_and_send(server, message)

    def _authenticate_and_send(self, server: smtplib.SMTP, message: EmailMessage) -> None:
        if self._username and self._password:
            server.login(self._username, self._password)
        server.send_message(message)
