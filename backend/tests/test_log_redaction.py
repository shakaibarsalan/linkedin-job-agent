"""The log redactor must hide secrets without hiding diagnostics.

Both directions matter. Leaking a session cookie into a log file is a real
compromise. But over-redacting is not free either: during a live session
failure the logs read ``cookies=***REDACTED***``, which is the count of
cookies, and the operator lost time to a control that was working exactly as
written and still told them nothing.
"""

from __future__ import annotations

import pytest

from app.logging_config import _REDACTED, _redact


def redact(**fields: object) -> dict[str, object]:
    return _redact(None, "info", dict(fields))


class TestSecretsAreHidden:
    @pytest.mark.parametrize(
        "key",
        [
            "password",
            "linkedin_password",
            "AI_API_KEY",
            "api_key",
            "apikey",
            "authorization",
            "session_state",
            "encryption_key",
            "google_credential",
            "access_token",
            "cookie",
            "cookie_value",
        ],
    )
    def test_string_values_on_sensitive_keys_are_redacted(self, key: str) -> None:
        assert redact(**{key: "AQEDATruesecret"})[key] == _REDACTED

    def test_a_cookie_jar_is_redacted_even_though_it_is_not_a_string(self) -> None:
        # A container can carry the secret itself; only scalars are exempt.
        out = redact(cookies=[{"name": "li_at", "value": "AQEDAT..."}])
        assert out["cookies"] == _REDACTED

    def test_a_credential_dict_is_redacted(self) -> None:
        assert redact(credentials={"private_key": "-----BEGIN"})["credentials"] == _REDACTED

    def test_none_is_redacted_rather_than_reported_as_absent(self) -> None:
        # Whether a secret field is set is itself mildly informative; the value
        # slot is not a safe place to say so.
        assert redact(api_key=None)["api_key"] == _REDACTED

    def test_inline_secrets_in_the_message_are_scrubbed(self) -> None:
        out = redact(event="restoring li_at=AQEDATxxxxx from disk")
        assert "AQEDATxxxxx" not in str(out["event"])
        assert _REDACTED in str(out["event"])


class TestDiagnosticsSurvive:
    """Counts and flags are not credentials. This is the regression."""

    def test_a_cookie_count_is_not_a_secret(self) -> None:
        assert redact(cookies=10)["cookies"] == 10

    def test_a_boolean_flag_is_not_a_secret(self) -> None:
        # bool is an int subclass -- assert the value, not just the type.
        assert redact(has_session_cookie=True)["has_session_cookie"] is True
        assert redact(token_present=False)["token_present"] is False

    def test_an_expiry_timestamp_is_not_a_secret(self) -> None:
        assert redact(cookie_expires_in_days=27.5)["cookie_expires_in_days"] == 27.5

    def test_ordinary_fields_are_untouched(self) -> None:
        out = redact(run_id=4, company="Acme", scanned=25)
        assert out == {"run_id": 4, "company": "Acme", "scanned": 25}
