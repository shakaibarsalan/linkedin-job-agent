"""
Settings parsing from a real environment.

These tests exist because the unit suite constructs ``Settings(**kwargs)``
directly, which bypasses the environment-variable *source* layer entirely — and
that layer is where a whole class of container-only bugs lives.

The specific bug that prompted this file: ``pydantic-settings`` treats any
``list`` field as "complex" and runs ``json.loads()`` on the raw env value
inside the settings source, *before* any ``mode="before"`` field validator can
intervene. ``CORS_ORIGINS=`` — the documented default, and what ``.env.example``
ships — therefore raised ``JSONDecodeError`` at import time and the API
container crash-looped. It passed every local test, because locally the
variable was simply unset.

Everything here goes through ``_env_file=None`` + ``os.environ`` so the source
layer is genuinely exercised.
"""

from __future__ import annotations

import pytest

from app.config import Settings

# A minimal valid environment. Individual tests layer overrides on top.
_BASE_ENV = {
    "API_KEY": "test-api-key-000000000000",
    "AI_PROVIDER": "rules",
    "DATABASE_URL": "postgresql+asyncpg://u:p@postgres:5432/db",
}


def _build(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> Settings:
    """Construct Settings purely from environment variables."""
    for key in (
        "CORS_ORIGINS",
        "REVIEW_MODE",
        "GENERATE_COVER_LETTERS",
        "AI_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)

    for key, value in {**_BASE_ENV, **overrides}.items():
        monkeypatch.setenv(key, value)

    # _env_file=None so a developer's local .env cannot influence the result.
    return Settings(_env_file=None)  # type: ignore[call-arg]


class TestCorsOriginsParsing:
    """Regression: an empty CORS_ORIGINS must not crash the process."""

    def test_empty_string_yields_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`CORS_ORIGINS=` is what .env.example ships. It must parse."""
        settings = _build(monkeypatch, CORS_ORIGINS="")
        assert settings.cors_origins == []

    def test_unset_yields_empty_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _build(monkeypatch)
        assert settings.cors_origins == []

    def test_single_origin(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _build(monkeypatch, CORS_ORIGINS="https://example.com")
        assert settings.cors_origins == ["https://example.com"]

    def test_comma_separated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _build(monkeypatch, CORS_ORIGINS="https://a.com,https://b.com,https://c.com")
        assert settings.cors_origins == ["https://a.com", "https://b.com", "https://c.com"]

    def test_whitespace_is_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _build(monkeypatch, CORS_ORIGINS=" https://a.com , https://b.com ")
        assert settings.cors_origins == ["https://a.com", "https://b.com"]

    def test_trailing_comma_does_not_produce_an_empty_origin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty origin string would be a permissive CORS entry."""
        settings = _build(monkeypatch, CORS_ORIGINS="https://a.com,")
        assert settings.cors_origins == ["https://a.com"]

    def test_json_array_still_works(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Not the documented format, but it should not be a hard failure."""
        settings = _build(monkeypatch, CORS_ORIGINS='["https://a.com"]')
        # Parsed as a single comma-free string; the important part is that it
        # does not raise. The documented format is comma-separated.
        assert settings.cors_origins


class TestFullEnvironmentBoot:
    """Every value from .env.example must produce a usable Settings object."""

    def test_documented_defaults_parse(self, monkeypatch: pytest.MonkeyPatch) -> None:
        settings = _build(
            monkeypatch,
            ENVIRONMENT="production",
            DEBUG="false",
            TIMEZONE="Europe/London",
            CORS_ORIGINS="",
            LOG_LEVEL="INFO",
            LOG_FORMAT="json",
            LOG_DIR="/data/logs",
            MIN_SCORE_TO_APPLY="70",
            AUTO_APPLY_SCORE_THRESHOLD="85",
            REVIEW_MODE="always",
            GENERATE_COVER_LETTERS="true",
            PUBLIC_URL="http://localhost:8088",
            NOTIFY_TELEGRAM_ENABLED="false",
        )
        assert settings.review_mode == "always"
        assert settings.min_score_to_apply == 70
        assert settings.cors_origins == []
        assert settings.tz.key == "Europe/London"

    @pytest.mark.parametrize("value", ["true", "True", "1", "yes", "on"])
    def test_boolean_truthy_forms(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        assert _build(monkeypatch, GENERATE_COVER_LETTERS=value).generate_cover_letters is True

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off"])
    def test_boolean_falsy_forms(self, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        assert _build(monkeypatch, GENERATE_COVER_LETTERS=value).generate_cover_letters is False

    def test_invalid_review_mode_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError):
            _build(monkeypatch, REVIEW_MODE="whenever")

    def test_invalid_timezone_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValueError, match="Unknown timezone"):
            _build(monkeypatch, TIMEZONE="Mars/Olympus_Mons")

    def test_rules_provider_needs_no_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The no-LLM deployment must boot without a provider key."""
        settings = _build(monkeypatch, AI_PROVIDER="rules", AI_API_KEY="")
        assert settings.ai_provider == "rules"

    def test_openai_provider_without_a_key_is_rejected_at_boot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(ValueError, match="AI_API_KEY is required"):
            _build(monkeypatch, AI_PROVIDER="openai_compatible", AI_API_KEY="")


class TestDerivedValues:
    def test_any_notifier_enabled_reflects_configuration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        assert _build(monkeypatch).any_notifier_enabled is False
        assert _build(monkeypatch, NOTIFY_TELEGRAM_ENABLED="true").any_notifier_enabled is True
