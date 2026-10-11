"""
Central configuration.

Purpose
-------
Single source of truth for every tunable in the platform. Values come from
environment variables (in production, injected by Docker Compose from `.env`);
`Settings` validates them at import time so a misconfigured deployment fails
loudly at startup rather than silently misbehaving at 3 a.m.

How it works
------------
`pydantic-settings` reads env vars, coerces them to the declared types, and runs
the validators below. `get_settings()` is `lru_cache`d, so the object is built
once per process and can be used as a FastAPI dependency.

Failure scenarios
-----------------
* Missing/short ``API_KEY`` → ``ValidationError`` at startup.
  Recovery: generate one with ``python -c "import secrets; print(secrets.token_urlsafe(32))"``.
* ``AI_PROVIDER`` set to an unknown value → ``ValidationError``.
  Recovery: use one of ``openai_compatible``, ``ollama``, ``rules``.
* An auto-approve threshold below the minimum score → ``ValidationError``.
"""

from __future__ import annotations

import functools
from typing import Annotated, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

AIProviderName = Literal["openai_compatible", "ollama", "rules"]
ReviewMode = Literal["always", "threshold", "never"]
LogFormat = Literal["json", "console"]


class Settings(BaseSettings):
    """Validated application settings. Instantiate via :func:`get_settings`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Application ───────────────────────────────────────────────────
    app_name: str = "linkedin-job-agent"
    environment: Literal["development", "production", "test"] = "production"
    debug: bool = False
    timezone: str = "UTC"

    # ── API ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"  # noqa: S104 — inside the container; host binding is 127.0.0.1
    api_port: int = 8000
    # Shared secret for the X-API-Key header. The API is loopback-only in
    # production, so this is defence in depth rather than the primary control.
    api_key: str = Field(min_length=16)
    # `NoDecode` is load-bearing. Without it, pydantic-settings treats any
    # `list` field as "complex" and runs json.loads() on the raw env value
    # inside the settings *source*, before any field validator can see it.
    # `CORS_ORIGINS=` (the documented default, and what .env ships with) then
    # raises JSONDecodeError at import time and the container crash-loops.
    # NoDecode hands the raw string to `_split_origins` below instead.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # ── Database ──────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://linkedin:linkedin@postgres:5432/linkedin_agent"
    )
    db_pool_size: int = Field(default=5, ge=1, le=50)
    db_max_overflow: int = Field(default=5, ge=0, le=50)
    db_echo: bool = False

    # ── Logging ───────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: LogFormat = "json"
    log_dir: str = "/data/logs"

    # ── Storage paths (all inside the container, backed by volumes) ───
    data_dir: str = "/data"
    resume_dir: str = "/data/resumes"
    profile_path: str = "/data/profile/answers.yaml"

    # Base URL used in notification links. Point it at wherever you reach the API.
    public_url: str = "http://localhost:8088"

    # ── AI provider ───────────────────────────────────────────────────
    ai_provider: AIProviderName = "openai_compatible"
    ai_base_url: str = "https://api.openai.com/v1"
    ai_api_key: str = ""
    ai_model: str = "gpt-4o-mini"
    ai_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    ai_max_tokens: int = Field(default=1500, ge=64, le=32000)
    ai_timeout_seconds: float = Field(default=90.0, ge=5.0, le=600.0)
    ai_max_retries: int = Field(default=3, ge=0, le=10)

    # ── Matching / scoring policy ─────────────────────────────────────
    # Below this score a job is skipped, not shortlisted.
    min_score_to_apply: int = Field(default=70, ge=0, le=100)
    # At or above this score, and with review_mode="threshold", a clean match
    # goes straight to your "approved" shortlist. You still apply by hand.
    auto_apply_score_threshold: int = Field(default=85, ge=0, le=100)
    review_mode: ReviewMode = "always"
    generate_cover_letters: bool = True

    # ── Notifications ─────────────────────────────────────────────────
    # Optional proxy for OUTBOUND NOTIFICATIONS ONLY. Nothing else routes
    # through it. Useful where an ISP blocks Telegram.
    #
    #   e.g. socks5://127.0.0.1:1080
    notify_proxy_url: str = ""

    notify_telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    notify_discord_enabled: bool = False
    discord_webhook_url: str = ""

    notify_slack_enabled: bool = False
    slack_webhook_url: str = ""

    notify_email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_from: str = ""
    smtp_to: str = ""
    smtp_use_tls: bool = True

    # ── Validators ────────────────────────────────────────────────────

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, v: str) -> str:
        try:
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"Unknown timezone {v!r}") from exc
        return v

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Parse the raw env value into a list.

        Accepts a comma-separated string (what `.env` provides), an empty
        string (meaning "no origins"), or an already-parsed list (tests).
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("database_url")
    @classmethod
    def _async_driver(cls, v: str) -> str:
        """Fail fast on a sync driver URL — the whole data layer is async."""
        if not v.startswith("postgresql+asyncpg://"):
            raise ValueError(
                "DATABASE_URL must use the asyncpg driver, "
                "e.g. postgresql+asyncpg://user:pass@host:5432/db"
            )
        return v

    @model_validator(mode="after")
    def _coherent_ranges(self) -> Settings:
        if self.auto_apply_score_threshold < self.min_score_to_apply:
            raise ValueError(
                "AUTO_APPLY_SCORE_THRESHOLD must be >= MIN_SCORE_TO_APPLY, otherwise "
                "auto-approve would fire on jobs the scorer already rejected"
            )
        if self.ai_provider == "openai_compatible" and not self.ai_api_key:
            raise ValueError(
                "AI_API_KEY is required when AI_PROVIDER=openai_compatible. "
                "Use AI_PROVIDER=rules for a no-LLM deployment."
            )
        return self

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def any_notifier_enabled(self) -> bool:
        return any(
            (
                self.notify_telegram_enabled,
                self.notify_discord_enabled,
                self.notify_slack_enabled,
                self.notify_email_enabled,
            )
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
