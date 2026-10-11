"""
Cover letter generation with mandatory honesty verification.

Purpose
-------
Produces a tailored cover letter grounded strictly in the resume, then verifies
it mechanically before it can be attached to an application.

How it works
------------
Generate → verify → decide. :mod:`app.services.fabrication_guard` checks the
letter against the resume. A ``high``-severity flag means the letter is
**discarded**, not merely marked: sending it would be sending a lie. Lower-
severity flags keep the letter but force human review.

Configuration
-------------
``GENERATE_COVER_LETTERS`` — master switch.

Failure scenarios
-----------------
* AI unavailable (including the rules provider, which refuses to generate) →
  returns ``None`` with a reason; the job is queued for you without a letter.
* Letter fails verification at high severity → returns ``None`` with the flags
  attached. One regeneration is attempted with the flags fed back first.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings, get_settings
from app.db.models import Job, Resume
from app.logging_config import get_logger
from app.schemas.ai import CoverLetterResult
from app.services.ai import AIError, AIProvider, Message
from app.services.ai.prompts import build_cover_letter_prompt
from app.services.fabrication_guard import FabricationReport, verify_against_resume

log = get_logger(__name__)


@dataclass(slots=True)
class CoverLetterOutcome:
    """Result of attempting to generate a verified cover letter."""

    text: str | None
    flagged: bool = False
    flags: list[str] = field(default_factory=list)
    warning: str = ""
    skipped_reason: str = ""

    @property
    def usable(self) -> bool:
        """True when there is a letter that may be attached."""
        return self.text is not None


async def generate_cover_letter(
    provider: AIProvider,
    *,
    job: Job,
    resume: Resume,
    matched_skills: list[str] | None = None,
    extra_context: str = "",
    settings: Settings | None = None,
) -> CoverLetterOutcome:
    """Generate and verify a cover letter for ``job`` using ``resume``.

    Returns:
        A :class:`CoverLetterOutcome`. ``text`` is ``None`` when no letter
        could be produced honestly — that is a normal outcome, not an error.
    """
    s = settings or get_settings()

    if not s.generate_cover_letters:
        return CoverLetterOutcome(text=None, skipped_reason="Cover letters disabled by config")

    messages = build_cover_letter_prompt(
        resume_text=resume.content_text,
        job_title=job.title,
        company=job.company,
        job_description=job.description,
        matched_skills=matched_skills,
        extra_context=extra_context,
    )

    try:
        result, _completion = await provider.complete_json(
            messages, CoverLetterResult, temperature=0.6, max_retries=1
        )
    except AIError as exc:
        log.warning("cover_letter.generation_failed", job_id=job.id, error=str(exc)[:300])
        return CoverLetterOutcome(
            text=None, skipped_reason=f"AI unavailable for cover letter generation: {exc}"
        )

    report = _verify(result, job=job, resume=resume)

    if report.ok:
        return _accept(result, report, job=job)

    # One correction attempt: show the model exactly what failed. Regeneration
    # succeeds often enough to be worth a single extra call, and never more.
    log.warning(
        "cover_letter.regenerating",
        job_id=job.id,
        flags=report.as_strings(),
    )
    retry_messages = [
        *messages,
        Message(role="assistant", content=result.model_dump_json()),
        Message(
            role="user",
            content=(
                "That letter contains claims the resume does not support:\n"
                + "\n".join(f"- {f}" for f in report.as_strings())
                + "\n\nRewrite it using ONLY what the resume evidences. Remove every "
                "unsupported claim rather than rephrasing it. A shorter, honest "
                "letter is the correct outcome. Return ONLY the JSON object."
            ),
        ),
    ]

    try:
        result, _completion = await provider.complete_json(
            retry_messages, CoverLetterResult, temperature=0.3, max_retries=1
        )
    except AIError as exc:
        log.warning("cover_letter.regeneration_failed", job_id=job.id, error=str(exc)[:300])
        return CoverLetterOutcome(
            text=None,
            flagged=True,
            flags=report.as_strings(),
            skipped_reason="Letter failed honesty verification and could not be regenerated",
        )

    report = _verify(result, job=job, resume=resume)

    if not report.ok:
        # Two strikes. Do not attach a letter making unsupported claims.
        log.error(
            "cover_letter.rejected",
            job_id=job.id,
            company=job.company[:60],
            flags=report.as_strings(),
        )
        return CoverLetterOutcome(
            text=None,
            flagged=True,
            flags=report.as_strings(),
            skipped_reason=(
                "Letter still made unsupported claims after regeneration; discarded. "
                "Write this one by hand."
            ),
        )

    return _accept(result, report, job=job)


def _verify(result: CoverLetterResult, *, job: Job, resume: Resume) -> FabricationReport:
    return verify_against_resume(
        result.cover_letter,
        resume.content_text,
        job_description=job.description,
        # Without this the employer's own name is flagged as a borrowed
        # technology on every letter -- naming the company you are writing to
        # is the opposite of an invented claim.
        company=job.company or "",
        claims=result.claims_made,
    )


def _accept(
    result: CoverLetterResult, report: FabricationReport, *, job: Job
) -> CoverLetterOutcome:
    """Package an accepted letter, carrying any non-blocking flags forward."""
    log.info(
        "cover_letter.generated",
        job_id=job.id,
        company=job.company[:60],
        words=len(result.cover_letter.split()),
        soft_flags=len(report.flags),
        model_warning=bool(result.warning),
    )
    return CoverLetterOutcome(
        text=result.cover_letter,
        # Medium/low flags do not block, but they do force human review via
        # scoring.needs_human_review().
        flagged=report.requires_review,
        flags=report.as_strings(),
        warning=result.warning,
    )
