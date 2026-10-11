"""
Job compatibility scoring and resume selection.

Purpose
-------
Answers two questions for every discovered job: *how well does this candidate
fit?* and *which CV variant should be submitted?*

How it works
------------
Each active resume is scored against the job description independently, and the
highest-scoring one wins. Scoring every variant (rather than asking a model to
pick) is both cheaper in judgement and more defensible: the score that decided
the resume choice is the same number stored on the application, so the audit
trail is coherent.

Results are cached in ``job_scores`` keyed on ``(job_id, resume_id)``, so
re-running a search never re-pays for scoring the same pair.

Configuration
-------------
``MIN_SCORE_TO_APPLY`` — below this, no application is created.
``AUTO_APPLY_SCORE_THRESHOLD`` — at/above this, ``REVIEW_MODE=threshold`` may
submit without human approval.

Failure scenarios
-----------------
* AI provider down → :class:`ResilientProvider` degrades to keyword scoring and
  the score row records ``provider="rules"``, so a flat run is explainable.
* No active resumes → :class:`NoResumeError`; the run fails fast with an
  actionable message rather than silently scoring nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import Job, JobScore, Resume
from app.logging_config import get_logger
from app.schemas.ai import JobScoreResult
from app.services.ai import AIError, AIProvider
from app.services.ai.prompts import build_scoring_prompt

log = get_logger(__name__)


class NoResumeError(RuntimeError):
    """Raised when scoring is attempted with no active resume configured."""


@dataclass(slots=True)
class ScoredMatch:
    """The winning resume for a job, and why."""

    resume: Resume
    result: JobScoreResult
    provider: str
    model: str
    degraded: bool

    @property
    def score(self) -> int:
        return self.result.score

    @property
    def meets_threshold(self) -> bool:
        return self.result.score >= get_settings().min_score_to_apply


async def load_active_resumes(session: AsyncSession) -> list[Resume]:
    """Return active resumes, default first."""
    stmt = (
        select(Resume)
        .where(Resume.is_active.is_(True))
        .order_by(Resume.is_default.desc(), Resume.id.asc())
    )
    return list((await session.execute(stmt)).scalars().all())


async def _cached_score(session: AsyncSession, job_id: int, resume_id: int) -> JobScore | None:
    stmt = select(JobScore).where(JobScore.job_id == job_id, JobScore.resume_id == resume_id)
    return (await session.execute(stmt)).scalar_one_or_none()


async def _persist_score(
    session: AsyncSession,
    *,
    job_id: int,
    resume_id: int,
    result: JobScoreResult,
    provider: str,
    model: str,
) -> None:
    """Upsert a score row. Concurrent scoring of the same pair is harmless."""
    stmt = (
        pg_insert(JobScore)
        .values(
            job_id=job_id,
            resume_id=resume_id,
            score=result.score,
            verdict=result.verdict,
            reasoning=result.reasoning,
            matched_skills=result.matched_skills,
            missing_skills=result.missing_skills,
            concerns=result.concerns,
            provider=provider,
            model=model,
        )
        .on_conflict_do_update(
            index_elements=[JobScore.job_id, JobScore.resume_id],
            set_={
                "score": result.score,
                "verdict": result.verdict,
                "reasoning": result.reasoning,
                "matched_skills": result.matched_skills,
                "missing_skills": result.missing_skills,
                "concerns": result.concerns,
                "provider": provider,
                "model": model,
            },
        )
    )
    await session.execute(stmt)


async def score_job_against_resume(
    provider: AIProvider,
    *,
    job: Job,
    resume: Resume,
    settings: Settings | None = None,
) -> tuple[JobScoreResult, str, str, bool]:
    """Score one job against one resume.

    Returns:
        ``(result, provider_name, model_name, degraded)`` where ``degraded``
        means the score came from keyword matching rather than a model.
    """
    # `settings` is accepted for interface symmetry with the rest of the
    # service layer, but scoring deliberately reads nothing from it: the
    # temperature is pinned below rather than configurable.
    messages = build_scoring_prompt(
        resume_text=resume.content_text,
        job_title=job.title,
        company=job.company,
        location=job.location,
        job_description=job.description,
        seniority=job.seniority,
        employment_type=job.employment_type,
    )

    result, completion = await provider.complete_json(
        messages,
        JobScoreResult,
        # Pinned to zero, deliberately NOT AI_TEMPERATURE.
        #
        # A score is a judgement that gates a real action, so the same job and
        # the same CV must produce the same answer. At 0.2 they did not:
        # "AI/ML Engineer (Remote)" scored 82 on one run and 48 on a re-run
        # minutes later, with nothing changed. Those sit either side of the
        # 70 threshold, so the same posting was both applied-for and
        # discarded depending on when it happened to be scored.
        #
        # AI_TEMPERATURE still governs the cover letter, where some variation
        # is wanted. Here it only makes the threshold arbitrary.
        temperature=0.0,
        max_retries=1,
    )
    degraded = bool(getattr(provider, "last_call_degraded", False))
    return result, completion.provider, completion.model, degraded


async def score_job(
    session: AsyncSession,
    provider: AIProvider,
    job: Job,
    *,
    resumes: list[Resume] | None = None,
    use_cache: bool = True,
    settings: Settings | None = None,
) -> ScoredMatch:
    """Score ``job`` against every active resume and return the best match.

    Args:
        session: Open database session; score rows are written but not committed.
        provider: AI provider to use.
        job: The job to assess.
        resumes: Override the resume set (used by tests).
        use_cache: Reuse an existing ``job_scores`` row when present.
        settings: Settings override.

    Raises:
        NoResumeError: if there are no active resumes.
        AIError: if every resume failed to score.
    """
    s = settings or get_settings()
    candidates = resumes if resumes is not None else await load_active_resumes(session)

    if not candidates:
        raise NoResumeError(
            "No active resume configured. Upload one via POST /api/v1/resumes "
            "before starting a run."
        )

    best: ScoredMatch | None = None
    errors: list[str] = []

    for resume in candidates:
        if use_cache:
            cached = await _cached_score(session, job.id, resume.id)
            if cached is not None:
                log.debug("scoring.cache_hit", job_id=job.id, resume=resume.label)
                candidate = ScoredMatch(
                    resume=resume,
                    result=JobScoreResult(
                        score=cached.score,
                        verdict=cached.verdict,
                        reasoning=cached.reasoning,
                        matched_skills=list(cached.matched_skills),
                        missing_skills=list(cached.missing_skills),
                        concerns=list(cached.concerns),
                    ),
                    provider=cached.provider,
                    model=cached.model,
                    degraded=cached.provider == "rules",
                )
                if best is None or candidate.score > best.score:
                    best = candidate
                continue

        try:
            result, provider_name, model_name, degraded = await score_job_against_resume(
                provider, job=job, resume=resume, settings=s
            )
        except AIError as exc:
            # One resume failing must not abandon the others.
            errors.append(f"{resume.label}: {exc}")
            log.warning(
                "scoring.resume_failed", job_id=job.id, resume=resume.label, error=str(exc)[:300]
            )
            continue

        await _persist_score(
            session,
            job_id=job.id,
            resume_id=resume.id,
            result=result,
            provider=provider_name,
            model=model_name,
        )

        log.info(
            "scoring.scored",
            job_id=job.id,
            title=job.title[:80],
            company=job.company[:60],
            resume=resume.label,
            score=result.score,
            verdict=result.verdict,
            degraded=degraded,
        )

        candidate = ScoredMatch(
            resume=resume,
            result=result,
            provider=provider_name,
            model=model_name,
            degraded=degraded,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        raise AIError(
            f"Could not score job {job.id} against any resume. Errors: {'; '.join(errors)}"
        )

    log.info(
        "scoring.best_match",
        job_id=job.id,
        resume=best.resume.label,
        score=best.score,
        threshold=s.min_score_to_apply,
        meets_threshold=best.score >= s.min_score_to_apply,
    )
    return best


def should_apply(match: ScoredMatch, *, settings: Settings | None = None) -> tuple[bool, str]:
    """Decide whether an application should be created for this match.

    Returns:
        ``(apply, reason)`` — ``reason`` is stored as ``skip_reason`` when False.
    """
    s = settings or get_settings()

    if match.score < s.min_score_to_apply:
        return False, (
            f"Score {match.score} is below the minimum of {s.min_score_to_apply} "
            f"({match.result.verdict})"
        )
    if match.result.verdict == "not_a_match":
        return False, "Model verdict is 'not_a_match' despite a passing numeric score"
    return True, ""


def needs_human_review(
    match: ScoredMatch,
    *,
    cover_letter_flagged: bool = False,
    has_unanswered_questions: bool = False,
    settings: Settings | None = None,
) -> tuple[bool, str]:
    """Decide whether a human must approve before submission.

    Review is forced — regardless of ``REVIEW_MODE`` — when the honesty guard
    flagged the letter, when a screening question could not be answered from
    known facts, or when the score was produced without a model. Those are
    exactly the cases where automation should not be trusted, so config cannot
    switch the check off.
    """
    s = settings or get_settings()

    if cover_letter_flagged:
        return True, "Cover letter contains claims not supported by the resume"
    if has_unanswered_questions:
        return True, "One or more screening questions could not be answered from known facts"
    if match.degraded:
        return True, "Scored without a language model (AI provider unavailable)"

    if s.review_mode == "always":
        return True, "REVIEW_MODE=always"
    if s.review_mode == "never":
        return False, ""
    # threshold
    if match.score >= s.auto_apply_score_threshold:
        return False, ""
    return True, (
        f"Score {match.score} is below the auto-apply threshold "
        f"of {s.auto_apply_score_threshold}"
    )
