"""
Notification channel interface.

Purpose
-------
Defines the contract every delivery channel implements, so adding a new one
(Matrix, Pushover, SMS, webhook) means writing one class and registering it —
no changes to any calling code.

Design rule
-----------
**A notification failure must never fail the operation that triggered it.**
Not being told about a submitted application is annoying; failing the run
because Telegram was down would be much worse. Every send is wrapped by the
dispatcher and reduced to a logged result.

Failure scenarios
-----------------
* Channel misconfigured → :meth:`NotificationChannel.configured` returns False
  and the channel is skipped with a startup warning.
* Transport error → caught by the dispatcher, recorded in ``notification_log``.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any


class Priority(str, enum.Enum):
    """How urgently a human needs to see this."""

    LOW = "low"  # routine progress
    NORMAL = "normal"  # run completed, application submitted
    HIGH = "high"  # needs review, run paused
    CRITICAL = "critical"  # session expired, account restricted


@dataclass(slots=True)
class Notification:
    """A message to deliver."""

    title: str
    body: str
    priority: Priority = Priority.NORMAL
    event_type: str = "generic"
    url: str | None = None
    fields: dict[str, str] = field(default_factory=dict)

    def as_plain_text(self) -> str:
        """Render for channels without rich formatting (email, SMS)."""
        lines = [self.title, "", self.body]
        if self.fields:
            lines.append("")
            lines += [f"{key}: {value}" for key, value in self.fields.items()]
        if self.url:
            lines += ["", self.url]
        return "\n".join(lines)

    def as_markdown(self) -> str:
        """Render for channels supporting Markdown (Telegram, Discord, Slack)."""
        lines = [f"*{_escape_markdown(self.title)}*", "", _escape_markdown(self.body)]
        if self.fields:
            lines.append("")
            lines += [
                f"• *{_escape_markdown(k)}:* {_escape_markdown(v)}" for k, v in self.fields.items()
            ]
        if self.url:
            lines += ["", self.url]
        return "\n".join(lines)


def _escape_markdown(text: str) -> str:
    """Escape the characters that break Markdown parsers.

    Job titles routinely contain ``*`` and ``_`` (``Senior C++ Dev *Remote*``),
    which produce an unparseable message and a 400 from Telegram.
    """
    for char in ("\\", "*", "_", "`", "["):
        text = text.replace(char, f"\\{char}")
    return text


@dataclass(slots=True)
class DeliveryResult:
    """Outcome of one delivery attempt."""

    channel: str
    ok: bool
    error: str = ""


class NotificationChannel(abc.ABC):
    """A delivery destination."""

    name: str = "base"

    @property
    @abc.abstractmethod
    def configured(self) -> bool:
        """True when this channel has everything it needs to send."""

    @abc.abstractmethod
    async def send(self, notification: Notification) -> None:
        """Deliver ``notification``. May raise; the dispatcher catches."""

    async def aclose(self) -> None:
        """Release resources. Default is a no-op."""
        return None

    def __repr__(self) -> str:
        return f"<{type(self).__name__} configured={self.configured}>"


def build_payload_fields(**kwargs: Any) -> dict[str, str]:
    """Build a display field map, dropping empty values."""
    return {
        key.replace("_", " ").title(): str(value)
        for key, value in kwargs.items()
        if value not in (None, "", [])
    }
