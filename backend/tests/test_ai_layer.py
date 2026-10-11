"""
Tests for the AI layer: JSON extraction, schema validation, the rules provider,
and graceful degradation.

Model output is untrusted input; these tests cover the shapes models actually
emit, including the malformed ones.
"""

from __future__ import annotations

import pytest

from app.schemas.ai import CoverLetterResult, JobScoreResult, QuestionAnswerResult
from app.services.ai.base import AIResponseError, AIUnavailableError, Message, extract_json
from app.services.ai.factory import ResilientProvider
from app.services.ai.prompts import build_scoring_prompt
from app.services.ai.providers import RulesProvider


class TestExtractJson:
    def test_bare_json_object(self) -> None:
        assert extract_json('{"score": 80}') == {"score": 80}

    def test_fenced_code_block(self) -> None:
        assert extract_json('```json\n{"score": 80}\n```') == {"score": 80}

    def test_unlabelled_fence(self) -> None:
        assert extract_json('```\n{"score": 80}\n```') == {"score": 80}

    def test_prose_preamble(self) -> None:
        text = 'Sure! Here is the assessment:\n\n{"score": 72, "verdict": "possible_match"}'
        assert extract_json(text)["score"] == 72

    def test_nested_objects_survive(self) -> None:
        text = 'Result: {"a": {"b": {"c": [1, 2, 3]}}, "d": "x"}'
        assert extract_json(text)["a"]["b"]["c"] == [1, 2, 3]

    def test_braces_inside_strings_do_not_break_balancing(self) -> None:
        text = '{"reasoning": "Uses {curly} braces and \\"quotes\\"", "score": 5}'
        assert extract_json(text)["score"] == 5

    def test_json_array(self) -> None:
        assert extract_json("[1, 2, 3]") == [1, 2, 3]

    def test_empty_response_raises(self) -> None:
        with pytest.raises(AIResponseError, match="Empty response"):
            extract_json("")

    def test_no_json_raises(self) -> None:
        with pytest.raises(AIResponseError, match="No parseable JSON"):
            extract_json("I'm sorry, I cannot help with that.")


class TestSchemaCoercion:
    """Small models return sloppy shapes; coerce what is safe, reject what is not."""

    def test_comma_separated_string_becomes_a_list(self) -> None:
        result = JobScoreResult.model_validate(
            {"score": 80, "verdict": "strong_match", "matched_skills": "Python, SQL, Docker"}
        )
        assert result.matched_skills == ["Python", "SQL", "Docker"]

    def test_null_list_becomes_empty(self) -> None:
        result = JobScoreResult.model_validate(
            {"score": 50, "verdict": "weak_match", "concerns": None}
        )
        assert result.concerns == []

    def test_out_of_range_score_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            JobScoreResult.model_validate({"score": 150, "verdict": "strong_match"})

    def test_unknown_verdict_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            JobScoreResult.model_validate({"score": 80, "verdict": "amazing_match"})

    def test_runaway_lists_are_capped(self) -> None:
        result = JobScoreResult.model_validate(
            {
                "score": 80,
                "verdict": "strong_match",
                "matched_skills": [f"s{i}" for i in range(500)],
            }
        )
        assert len(result.matched_skills) == 40

    def test_implausibly_short_cover_letter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="implausibly short"):
            CoverLetterResult.model_validate({"cover_letter": "Hi."})

    def test_question_answer_defaults_are_safe(self) -> None:
        """An omitted confidence must default to 0, not to a confident value."""
        result = QuestionAnswerResult.model_validate({"answer": "5"})
        assert result.confidence == 0.0
        assert result.needs_human is False


class TestRulesProvider:
    """The no-LLM provider: honest about what it can and cannot do."""

    async def test_scores_by_keyword_overlap(self, resume_text: str) -> None:
        provider = RulesProvider()

        class FakeJob:
            id = 1
            title = "Senior Backend Engineer"
            company = "Acme"
            location = "London"
            description = "Python, Django, PostgreSQL, Docker, AWS required."
            seniority = None
            employment_type = None

        messages = build_scoring_prompt(
            resume_text=resume_text,
            job_title=FakeJob.title,
            company=FakeJob.company,
            location=FakeJob.location,
            job_description=FakeJob.description,
        )
        result, _ = await provider.complete_json(messages, JobScoreResult)

        assert 0 <= result.score <= 100
        assert result.matched_skills
        # It must be transparent that no model was involved.
        assert "no language model" in result.reasoning.lower()

    async def test_refuses_to_generate_a_cover_letter(self) -> None:
        """There is no honest way to write a letter without a model."""
        provider = RulesProvider()
        with pytest.raises(AIUnavailableError, match="cannot generate free text"):
            await provider.complete([Message(role="user", content="Write a cover letter")])

    async def test_unrelated_job_scores_low(self, resume_text: str) -> None:
        provider = RulesProvider()
        messages = build_scoring_prompt(
            resume_text=resume_text,
            job_title="Dental Hygienist",
            company="Smile Clinic",
            location="Leeds",
            job_description=(
                "Scaling, polishing, periodontal charting, fluoride treatment, "
                "patient dental education, radiography certification required."
            ),
        )
        result, _ = await provider.complete_json(messages, JobScoreResult)
        assert result.score < 40
        assert result.verdict in {"weak_match", "not_a_match"}

    async def test_health_is_always_true(self) -> None:
        assert await RulesProvider().health() is True


class TestResilientProvider:
    """Degradation must be graceful for scoring and refused for generation."""

    class _DeadProvider(RulesProvider):
        name = "dead"

        async def complete(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            raise AIUnavailableError("upstream is down")

    async def test_scoring_falls_back_to_rules(self, resume_text: str) -> None:
        provider = ResilientProvider(primary=self._DeadProvider(), fallback=RulesProvider())

        messages = build_scoring_prompt(
            resume_text=resume_text,
            job_title="Backend Engineer",
            company="Acme",
            location="London",
            job_description="Python and PostgreSQL required.",
        )
        result, _ = await provider.complete_json(messages, JobScoreResult)

        assert 0 <= result.score <= 100
        assert provider.last_call_degraded is True

    async def test_generation_still_raises_when_primary_is_down(self) -> None:
        """Falling back must never mean substituting a guess for a real answer."""
        provider = ResilientProvider(primary=self._DeadProvider(), fallback=RulesProvider())
        with pytest.raises(AIUnavailableError):
            await provider.complete([Message(role="user", content="Write a cover letter")])

    async def test_breaker_opens_after_repeated_failures(self) -> None:
        provider = ResilientProvider(primary=self._DeadProvider(), fallback=RulesProvider())
        assert not provider.breaker_open

        for _ in range(3):
            with pytest.raises(AIUnavailableError):
                await provider.complete([Message(role="user", content="generate")])

        assert provider.breaker_open


class TestPrompts:
    """The honesty rules must be present in every prompt that reaches a model."""

    def test_scoring_prompt_carries_the_honesty_rules(self, resume_text: str) -> None:
        messages = build_scoring_prompt(
            resume_text=resume_text,
            job_title="Engineer",
            company="Acme",
            location="London",
            job_description="Python.",
        )
        combined = " ".join(m.content for m in messages)
        assert "ABSOLUTE RULES" in combined
        assert "only source of truth" in combined
        assert "Never invent" in combined

    def test_sections_the_rules_provider_parses_are_present(self, resume_text: str) -> None:
        """`RulesProvider` extracts these headers; renaming them breaks scoring."""
        messages = build_scoring_prompt(
            resume_text=resume_text,
            job_title="Engineer",
            company="Acme",
            location="London",
            job_description="Python.",
        )
        user = messages[-1].content
        assert "### RESUME" in user
        assert "### JOB DESCRIPTION" in user
        assert "SCORING_TASK" in user

    def test_long_input_is_truncated_with_a_marker(self) -> None:
        messages = build_scoring_prompt(
            resume_text="x" * 50_000,
            job_title="Engineer",
            company="Acme",
            location="London",
            job_description="y" * 50_000,
        )
        user = messages[-1].content
        assert "truncated at" in user
        assert len(user) < 40_000


class TestEmptyResponseDiagnostics:
    """Reasoning models can return HTTP 200 with an empty body.

    Gemini 3.x spends the token budget on hidden thinking before emitting
    anything visible. Exhaust it and the API returns 200, finish_reason "stop",
    and an empty string — with no error field. Measured on the real endpoint:
    a trivial prompt used 6 prompt + 1 visible + ~71 thinking tokens, so
    max_tokens=10 silently produced nothing.

    Surfacing that as "no parseable JSON" three layers later is a bad debugging
    experience, so the provider names the cause at the point it happens.
    """

    async def test_empty_content_raises_with_an_actionable_message(self) -> None:
        from unittest.mock import AsyncMock

        from app.services.ai.providers import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            base_url="https://example.invalid/v1", api_key="k", model="gemini-3.6-flash"
        )
        provider._post_with_retry = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "choices": [{"message": {"content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 6, "completion_tokens": 1, "total_tokens": 78},
            }
        )

        with pytest.raises(AIUnavailableError) as exc_info:
            await provider.complete([Message(role="user", content="score this")])

        message = str(exc_info.value)
        assert "empty response" in message.lower()
        assert "AI_MAX_TOKENS" in message, "must name the setting to change"
        assert "thinking" in message.lower(), "must explain why it was empty"
        await provider.aclose()

    async def test_whitespace_only_is_also_treated_as_empty(self) -> None:
        from unittest.mock import AsyncMock

        from app.services.ai.providers import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            base_url="https://example.invalid/v1", api_key="k", model="m"
        )
        provider._post_with_retry = AsyncMock(  # type: ignore[method-assign]
            return_value={"choices": [{"message": {"content": "   \n  "}}], "usage": {}}
        )

        with pytest.raises(AIUnavailableError):
            await provider.complete([Message(role="user", content="x")])
        await provider.aclose()


class TestMissingContentKey:
    """Gemini omits `content` ENTIRELY when thinking exhausts the budget.

    Observed live on gemini-3.6-flash with max_tokens=8:

        {"choices": [{"finish_reason": "length",
                      "message": {"role": "assistant"}}],
         "usage": {"completion_tokens": 0, "prompt_tokens": 8}}

    Note there is no "content" key at all — not an empty string. Subscripting
    it raised KeyError, which surfaced as a generic "unexpected response shape"
    and hid the real, fixable cause.
    """

    async def test_absent_content_key_gives_the_actionable_error(self) -> None:
        from unittest.mock import AsyncMock

        from app.services.ai.providers import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            base_url="https://example.invalid/v1", api_key="k", model="gemini-3.6-flash"
        )
        provider._post_with_retry = AsyncMock(  # type: ignore[method-assign]
            return_value={
                "choices": [{"finish_reason": "length", "message": {"role": "assistant"}}],
                "usage": {"completion_tokens": 0, "prompt_tokens": 8, "total_tokens": 13},
            }
        )

        with pytest.raises(AIUnavailableError) as exc_info:
            await provider.complete([Message(role="user", content="hi")])

        message = str(exc_info.value)
        assert "AI_MAX_TOKENS" in message, "must name the setting to change"
        assert (
            "unexpected response shape" not in message.lower()
        ), "must not fall back to the generic shape error, which hides the cause"
        await provider.aclose()

    async def test_null_message_is_handled(self) -> None:
        from unittest.mock import AsyncMock

        from app.services.ai.providers import OpenAICompatibleProvider

        provider = OpenAICompatibleProvider(
            base_url="https://example.invalid/v1", api_key="k", model="m"
        )
        provider._post_with_retry = AsyncMock(  # type: ignore[method-assign]
            return_value={"choices": [{"message": None}], "usage": {}}
        )
        with pytest.raises(AIUnavailableError):
            await provider.complete([Message(role="user", content="hi")])
        await provider.aclose()

    def test_health_probe_budget_accommodates_thinking_tokens(self) -> None:
        """A tiny health-check budget reports a healthy provider as down."""
        from app.services.ai.base import AIProvider

        assert AIProvider._HEALTH_MAX_TOKENS >= 256, (
            "reasoning models spend ~70+ tokens thinking before any visible "
            "output; too small a budget makes health() always return False"
        )
