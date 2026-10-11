"""
Screening question answering.

Purpose
-------
Answers the screening questions application forms ask, with a strict
provenance chain so that every answer can be traced to a fact you wrote down —
never to a guess. You paste the questions; you fill in the form.

How it works
------------
Resolution order, most trustworthy first:

1. **Profile** (:mod:`app.services.profile`) — exact, pattern, or skill-years
   match against your YAML facts. Confidence 1.0.
2. **AI**, constrained — the model may only answer from profile facts and the
   resume, and is instructed to return ``needs_human`` otherwise. Its answer is
   then re-verified by the fabrication guard.
3. **Human** — anything unresolved comes back unanswered, with the reason.

There is deliberately no fourth stage. A blank or guessed answer to
"are you legally authorised to work in the UK?" is not an acceptable outcome.

Failure scenarios
-----------------
* AI unavailable → the question is unresolved → human review. Correct behaviour.
* Model returns an answer that fails the fabrication guard → discarded, human
  review.
* Model returns an option not in the select list → discarded, human review.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.schemas.ai import QuestionAnswerResult
from app.services.ai import AIError, AIProvider
from app.services.ai.prompts import build_question_prompt
from app.services.fabrication_guard import verify_answer
from app.services.profile import Profile

log = get_logger(__name__)

# An AI answer below this confidence is treated as unresolved. Screening
# answers are high-stakes; a hesitant model is not good enough.
_MIN_AI_CONFIDENCE = 0.75


@dataclass(slots=True)
class FormQuestion:
    """A question scraped from an application form."""

    question: str
    question_type: str = "text"
    options: list[str] | None = None
    required: bool = True
    field_id: str = ""


@dataclass(slots=True)
class AnsweredQuestion:
    """A resolved (or unresolved) question, with provenance."""

    question: FormQuestion
    answer: str | None
    source: str  # profile | resume | ai | default | unknown
    confidence: float
    requires_review: bool
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.answer is not None and not self.requires_review

    def to_row(self, application_id: int) -> dict[str, object]:
        """Shape for insertion into ``application_answers``."""
        return {
            "application_id": application_id,
            "question": self.question.question[:2000],
            "question_type": self.question.question_type,
            "answer": self.answer,
            "source": self.source,
            "confidence": Decimal(f"{self.confidence:.2f}"),
            "requires_review": self.requires_review,
        }


class QuestionAnswerer:
    """Resolves form questions from profile facts, then a constrained model."""

    def __init__(
        self,
        *,
        profile: Profile,
        provider: AIProvider,
        resume_text: str,
        settings: Settings | None = None,
    ) -> None:
        self.profile = profile
        self.provider = provider
        self.resume_text = resume_text
        self.settings = settings or get_settings()
        self._profile_facts = profile.as_prompt_facts()

    async def answer(self, question: FormQuestion) -> AnsweredQuestion:
        """Resolve a single question."""
        # ── Stage 1: profile ──────────────────────────────────────────
        hit = self.profile.lookup(
            question.question,
            question_type=question.question_type,
            options=question.options,
        )
        if hit is not None:
            log.info(
                "qa.resolved_from_profile",
                question=question.question[:100],
                matched_by=hit.matched_by,
            )
            return AnsweredQuestion(
                question=question,
                answer=hit.value,
                source="profile",
                confidence=hit.confidence,
                requires_review=False,
                reason=f"Matched profile ({hit.matched_by})",
            )

        # ── Stage 2: constrained AI ───────────────────────────────────
        try:
            result = await self._ask_model(question)
        except AIError as exc:
            log.warning("qa.ai_unavailable", question=question.question[:100], error=str(exc)[:200])
            return self._unresolved(question, "AI unavailable and no profile fact matched")

        if result.needs_human:
            return self._unresolved(
                question, result.reasoning or "Model declined to answer from known facts"
            )

        if result.confidence < _MIN_AI_CONFIDENCE:
            return self._unresolved(
                question,
                f"Model confidence {result.confidence:.2f} is below the "
                f"{_MIN_AI_CONFIDENCE} threshold for screening answers",
            )

        if not result.answer.strip():
            return self._unresolved(question, "Model returned an empty answer")

        # A select answer must be verbatim one of the options.
        if question.options and result.answer not in question.options:
            lowered = {o.lower(): o for o in question.options}
            exact = lowered.get(result.answer.lower().strip())
            if exact is None:
                return self._unresolved(
                    question,
                    f"Model answer {result.answer[:60]!r} is not one of the " f"available options",
                )
            result.answer = exact

        # Re-verify: the model was told not to invent, now check that it didn't.
        report = verify_answer(result.answer, self.resume_text, self._profile_facts)
        if not report.ok:
            log.warning(
                "qa.ai_answer_rejected",
                question=question.question[:100],
                flags=report.as_strings(),
            )
            return self._unresolved(
                question,
                "Model answer made claims not supported by the resume: "
                + "; ".join(report.as_strings()),
            )

        log.info(
            "qa.resolved_from_ai",
            question=question.question[:100],
            confidence=result.confidence,
        )
        return AnsweredQuestion(
            question=question,
            answer=result.answer,
            source="ai",
            confidence=result.confidence,
            # An AI-sourced answer always warrants a look, even when accepted.
            requires_review=True,
            reason=result.reasoning or "Answered by model from resume/profile",
        )

    async def answer_all(self, questions: list[FormQuestion]) -> list[AnsweredQuestion]:
        """Resolve every question. Sequential by design — parallel model calls
        against a rate-limited endpoint save little and risk 429s mid-application.
        """
        return [await self.answer(q) for q in questions]

    async def _ask_model(self, question: FormQuestion) -> QuestionAnswerResult:
        messages = build_question_prompt(
            question=question.question,
            question_type=question.question_type,
            options=question.options,
            profile_facts=self._profile_facts,
            resume_text=self.resume_text,
        )
        result, _completion = await self.provider.complete_json(
            messages, QuestionAnswerResult, temperature=0.0, max_retries=1
        )
        return result

    @staticmethod
    def _unresolved(question: FormQuestion, reason: str) -> AnsweredQuestion:
        log.info("qa.unresolved", question=question.question[:120], reason=reason[:200])
        return AnsweredQuestion(
            question=question,
            answer=None,
            source="unknown",
            confidence=0.0,
            requires_review=True,
            reason=reason,
        )


def summarise(answers: list[AnsweredQuestion]) -> tuple[int, int, list[str]]:
    """Return ``(resolved, unresolved, unresolved_question_texts)``."""
    unresolved = [a for a in answers if not a.resolved]
    return (
        len(answers) - len(unresolved),
        len(unresolved),
        [a.question.question[:200] for a in unresolved],
    )
