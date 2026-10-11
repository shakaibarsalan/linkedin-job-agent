"""
Tests for the safety policy layer.

These encode the rules that keep a human in charge: score thresholds, and the
three cases that force review whatever the configuration says.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.config import Settings
from app.schemas.ai import JobScoreResult
from app.services.scoring import ScoredMatch, needs_human_review, should_apply


def _settings(**overrides: object) -> Settings:
    """Build settings with the test defaults plus overrides."""
    base: dict[str, object] = {
        "api_key": "test-api-key-000000000000",
        "ai_provider": "rules",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


def _match(score: int, *, verdict: str = "strong_match", degraded: bool = False) -> ScoredMatch:
    return ScoredMatch(
        resume=MagicMock(label="test-cv"),
        result=JobScoreResult(score=score, verdict=verdict),  # type: ignore[arg-type]
        provider="test",
        model="test-model",
        degraded=degraded,
    )


class TestShouldApply:
    def test_score_below_minimum_is_rejected(self) -> None:
        settings = _settings(min_score_to_apply=70)
        ok, reason = should_apply(_match(65), settings=settings)
        assert not ok
        assert "below the minimum" in reason

    def test_score_at_the_minimum_is_accepted(self) -> None:
        settings = _settings(min_score_to_apply=70)
        ok, _ = should_apply(_match(70), settings=settings)
        assert ok

    def test_not_a_match_verdict_overrides_a_passing_score(self) -> None:
        """A numeric score can drift; an explicit "no" from the model wins."""
        settings = _settings(min_score_to_apply=70)
        ok, reason = should_apply(_match(95, verdict="not_a_match"), settings=settings)
        assert not ok
        assert "not_a_match" in reason


class TestHumanReview:
    def test_always_mode_always_requires_review(self) -> None:
        settings = _settings(review_mode="always")
        needed, reason = needs_human_review(_match(99), settings=settings)
        assert needed
        assert reason == "REVIEW_MODE=always"

    def test_never_mode_skips_review_for_a_clean_match(self) -> None:
        settings = _settings(review_mode="never")
        needed, _ = needs_human_review(_match(80), settings=settings)
        assert not needed

    def test_threshold_mode_reviews_below_the_threshold(self) -> None:
        settings = _settings(review_mode="threshold", auto_apply_score_threshold=85)
        assert needs_human_review(_match(80), settings=settings)[0] is True
        assert needs_human_review(_match(90), settings=settings)[0] is False

    # ── The three un-bypassable forcings ──────────────────────────────

    def test_flagged_cover_letter_forces_review_even_in_never_mode(self) -> None:
        """Config must not be able to switch off the honesty backstop."""
        settings = _settings(review_mode="never")
        needed, reason = needs_human_review(
            _match(99), cover_letter_flagged=True, settings=settings
        )
        assert needed
        assert "not supported by the resume" in reason

    def test_unanswered_question_forces_review_even_in_never_mode(self) -> None:
        settings = _settings(review_mode="never")
        needed, reason = needs_human_review(
            _match(99), has_unanswered_questions=True, settings=settings
        )
        assert needed
        assert "could not be answered" in reason

    def test_degraded_scoring_forces_review_even_in_never_mode(self) -> None:
        """A keyword score is not a judgement of fit; a human must confirm."""
        settings = _settings(review_mode="never")
        needed, reason = needs_human_review(_match(99, degraded=True), settings=settings)
        assert needed
        assert "without a language model" in reason


class TestSettingsValidation:
    """Incoherent safety configuration must fail at startup, not at runtime."""

    def test_auto_threshold_below_minimum_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="AUTO_APPLY_SCORE_THRESHOLD"):
            _settings(min_score_to_apply=80, auto_apply_score_threshold=70)

    def test_openai_provider_without_a_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="AI_API_KEY is required"):
            _settings(ai_provider="openai_compatible", ai_api_key="")

    def test_sync_database_driver_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="asyncpg"):
            _settings(database_url="postgresql://user:pass@localhost/db")

    def test_scores_outside_0_to_100_are_rejected(self) -> None:
        with pytest.raises(ValueError):
            _settings(min_score_to_apply=101)
        with pytest.raises(ValueError):
            _settings(auto_apply_score_threshold=-1)
