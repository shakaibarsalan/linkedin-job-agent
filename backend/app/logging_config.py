"""
Structured logging setup.

Purpose
-------
Every component emits machine-parseable JSON
logs with a consistent shape, so that an incident can be reconstructed from
`docker logs` alone.

How it works
------------
`structlog` wraps the stdlib logger. In production we render JSON; locally we
render coloured key-value pairs. A file handler with rotation is added when
``LOG_DIR`` is writable, so logs survive `docker compose down` via the volume.

Secret redaction is applied as a structlog processor, not at each call site —
relying on developers to remember not to log a cookie is not a control.

Failure scenarios
-----------------
* ``LOG_DIR`` not writable → file logging is skipped with a warning; stdout
  logging (which Docker captures) continues. The process never dies over logs.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import re
import sys
from typing import Any

import structlog

# Keys whose values must never reach a log sink, matched case-insensitively as
# substrings so that `linkedin_password` and `AI_API_KEY` are both caught.
_SENSITIVE_KEY_PARTS = (
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "session_state",
    "encryption_key",
    "credential",
)

_REDACTED = "***REDACTED***"

# Catches secrets that appear inside a free-text message rather than a field.
_INLINE_SECRET_RE = re.compile(
    r"(?i)\b(li_at|JSESSIONID|bearer|api[_-]?key)\b\s*[=:]\s*\S+",
)


def _redact(_logger: Any, _method: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Strip secret-looking values from every log record.

    Numbers and booleans are exempt, because a secret is always a string and
    ``cookies=10`` is a count, not a credential. Without that exemption the
    matching is substring-based and over-fires: session diagnostics logged
    ``cookies=***REDACTED***`` and ``has_li_at=***REDACTED***``, which is
    exactly the information you need when a session has gone bad. Failing
    safe is right; failing safe on an integer is just blindness.
    """
    for key in list(event_dict.keys()):
        lowered = key.lower()
        if not any(part in lowered for part in _SENSITIVE_KEY_PARTS):
            continue
        if isinstance(event_dict[key], int | float):  # bool is an int subclass
            continue
        event_dict[key] = _REDACTED

    event = event_dict.get("event")
    if isinstance(event, str):
        event_dict["event"] = _INLINE_SECRET_RE.sub(r"\1=" + _REDACTED, event)
    return event_dict


def configure_logging(
    *,
    level: str = "INFO",
    fmt: str = "json",
    log_dir: str | None = None,
    component: str = "app",
) -> None:
    """Configure stdlib + structlog. Safe to call more than once.

    Args:
        level: Root log level name.
        fmt: ``"json"`` for production, ``"console"`` for human-readable local output.
        log_dir: Directory for rotating file logs. Skipped if not writable.
        component: Bound to every record as ``component`` so lines from
            different processes can be told apart when logs are aggregated.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        _redact,
    ]

    renderer: Any
    if fmt == "console":
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(numeric_level)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    root.addHandler(stream_handler)

    if log_dir:
        try:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, f"{component}.log"),
                maxBytes=10 * 1024 * 1024,  # 10 MB
                backupCount=5,
                encoding="utf-8",
            )
            file_handler.setFormatter(logging.Formatter("%(message)s"))
            root.addHandler(file_handler)
        except OSError as exc:  # pragma: no cover — environment-dependent
            root.warning("file logging disabled: %s", exc)

    # These libraries are chatty at INFO and drown out our own records.
    for noisy in ("httpx", "httpcore", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    structlog.contextvars.bind_contextvars(component=component)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for ``name``."""
    return structlog.stdlib.get_logger(name)
