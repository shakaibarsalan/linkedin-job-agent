"""
Browser-free triage: ingest jobs from an external source, score them, write
cover letters, and queue them for human review — without touching LinkedIn.

Why this exists
---------------
The first version of this project read every posting through a logged-in
browser: one LinkedIn page load per job merely considered. That traffic is
what got the account a warning, and almost none of it was necessary. Reading a
posting and deciding whether it is worth applying to needs the job's *text*,
not your session.

So this is the whole pipeline now:

    any job source  ->  ingest  ->  score  ->  cover letter  ->  fact check
                                                                     |
    you apply, by hand  <-  you review  <----------------------------+

Nothing here holds a session, opens a browser, or submits anything.

Failure scenarios
-----------------
* **A job arrives with no description** → rejected at ingest rather than
  scored. Scoring an empty description produces a confident, meaningless
  number, which is worse than skipping it.
* **The AI provider is down** → triage stops with the jobs still stored and
  unscored. Re-running picks up exactly where it left off, because "needs
  triage" is derived from the absence of a score, not from a queue.
* **The same job arrives twice** → ``linkedin_job_id`` is unique; the second
  arrival refreshes the row and is reported as a duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db.models import Application, Job, JobScore
from app.logging_config import get_logger
from app.services import events
from app.services.ai.base import AIProvider
from app.services.cover_letter import generate_cover_letter
from app.services.scoring import NoResumeError, needs_human_review, score_job, should_apply

log = get_logger(__name__)

# A posting shorter than this is not a posting. LinkedIn boilerplate alone runs
# to a few hundred characters, and scoring an empty description yields a
# confident number derived from nothing.
_MIN_DESCRIPTION_CHARS = 200


@dataclass(slots=True)
class IngestResult:
    """What happened to a batch of incoming jobs."""

    received: int = 0
    stored: int = 0
    duplicates: int = 0
    rejected: int = 0
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "received": self.received,
            "stored": self.stored,
            "duplicates": self.duplicates,
            "rejected": self.rejected,
            "reasons": self.reasons[:20],
        }


@dataclass(slots=True)
class TriageResult:
    """What happened when the stored jobs were scored."""

    considered: int = 0
    scored: int = 0
    queued_for_review: int = 0
    auto_approved: int = 0
    skipped: int = 0
    errors: int = 0
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "considered": self.considered,
            "scored": self.scored,
            "queued_for_review": self.queued_for_review,
            "auto_approved": self.auto_approved,
            "skipped": self.skipped,
            "errors": self.errors,
            "failures": self.failures[:20],
        }


def _coerce_posted_at(value: Any) -> datetime | None:
    """Best-effort parse of whatever the source calls a date.

    Returns None rather than guessing. A wrong posted-at silently distorts
    "newest first" ordering, and nothing downstream would ever flag it.
    """
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        # Milliseconds if it is far too large to be seconds.
        seconds = float(value) / 1000.0 if value > 10_000_000_000 else float(value)
        try:
            return datetime.fromtimestamp(seconds, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


async def ingest_jobs(
    session: AsyncSession,
    jobs: list[dict[str, Any]],
    *,
    source: str = "manual",
) -> IngestResult:
    """Store externally-sourced jobs. No session, no browser.

    Args:
        session: Open database session. The caller commits.
        jobs: Raw job dicts. ``linkedin_job_id``, ``title`` and ``description``
            are required; everything else is optional.
        source: Recorded on each row so a job's provenance is visible later.

    Returns:
        Counts of what was stored, refreshed, and refused.
    """
    result = IngestResult(received=len(jobs))

    for raw in jobs:
        job_id = str(raw.get("linkedin_job_id") or "").strip()
        title = str(raw.get("title") or "").strip()
        description = str(raw.get("description") or "").strip()

        if not job_id or not job_id.isdigit():
            result.rejected += 1
            result.reasons.append(f"missing or non-numeric linkedin_job_id: {job_id!r}")
            continue
        if not title:
            result.rejected += 1
            result.reasons.append(f"job {job_id}: no title")
            continue
        if len(description) < _MIN_DESCRIPTION_CHARS:
            # Deliberately refused, not stored-and-skipped: an unscoreable job
            # sitting in the table looks like a job we chose not to apply to.
            result.rejected += 1
            result.reasons.append(
                f"job {job_id}: description is {len(description)} chars, "
                f"below the {_MIN_DESCRIPTION_CHARS} needed to score it honestly"
            )
            continue

        existing = (
            await session.execute(select(Job.id).where(Job.linkedin_job_id == job_id))
        ).scalar_one_or_none()

        payload = dict(raw)
        payload["_source"] = source

        stmt = (
            pg_insert(Job)
            .values(
                linkedin_job_id=job_id,
                url=str(raw.get("url") or f"https://www.linkedin.com/jobs/view/{job_id}/"),
                title=title,
                company=str(raw.get("company") or ""),
                location=str(raw.get("location") or ""),
                workplace_type=raw.get("workplace_type") or None,
                employment_type=raw.get("employment_type") or None,
                seniority=raw.get("seniority") or None,
                description=description,
                # Not asserted from the feed: sources report this unreliably.
                # You see the real form when you open the posting.
                easy_apply=False,
                applicant_count=raw.get("applicant_count"),
                posted_at=_coerce_posted_at(raw.get("posted_at")),
                raw=payload,
            )
            .on_conflict_do_update(
                index_elements=[Job.linkedin_job_id],
                set_={
                    "title": title,
                    "company": str(raw.get("company") or ""),
                    "location": str(raw.get("location") or ""),
                    "description": description,
                    "applicant_count": raw.get("applicant_count"),
                    "raw": payload,
                },
            )
        )
        await session.execute(stmt)

        if existing is None:
            result.stored += 1
        else:
            result.duplicates += 1

    log.info(
        "triage.ingested",
        source=source,
        received=result.received,
        stored=result.stored,
        duplicates=result.duplicates,
        rejected=result.rejected,
    )
    return result


async def _jobs_needing_triage(session: AsyncSession, limit: int) -> list[Job]:
    """Stored jobs with no score and no application yet.

    Derived from the data rather than a queue, so an interrupted triage resumes
    correctly and a job is never scored twice.
    """
    stmt = (
        select(Job)
        .outerjoin(JobScore, JobScore.job_id == Job.id)
        .outerjoin(Application, Application.job_id == Job.id)
        .where(JobScore.id.is_(None), Application.id.is_(None))
        .order_by(Job.posted_at.desc().nullslast(), Job.discovered_at.desc())
        .limit(limit)
    )
    return list((await session.execute(stmt)).scalars().all())


async def triage_jobs(
    session: AsyncSession,
    provider: AIProvider,
    *,
    limit: int = 25,
    settings: Settings | None = None,
) -> TriageResult:
    """Score stored jobs and queue the good ones for review. No browser.

    A clean, strong match may go straight to ``approved`` when ``REVIEW_MODE``
    allows it. A flagged letter or a degraded score always waits for you.

    Args:
        session: Open database session. The caller commits.
        provider: AI provider used for scoring and the cover letter.
        limit: Most jobs to process in one call. Bounds both runtime and spend.
        settings: Settings override.
    """
    s = settings or get_settings()
    result = TriageResult()

    jobs = await _jobs_needing_triage(session, limit)
    result.considered = len(jobs)
    if not jobs:
        log.info("triage.nothing_to_do")
        return result

    log.info("triage.starting", count=len(jobs), limit=limit)

    for job in jobs:
        try:
            match = await score_job(session, provider, job, settings=s)
        except NoResumeError:
            # Nothing can be scored without a resume, so stop rather than
            # burning the rest of the batch on the same failure.
            log.error("triage.no_resume", action="Upload a resume before triaging.")
            raise
        except Exception as exc:  # one job must not kill the batch
            result.errors += 1
            result.failures.append(f"job {job.linkedin_job_id}: scoring failed: {exc}"[:300])
            log.error("triage.score_failed", job_id=job.linkedin_job_id, exc_info=True)
            continue

        result.scored += 1

        apply_ok, skip_reason = should_apply(match, settings=s)
        if apply_ok and match.score < s.min_score_to_apply:
            apply_ok = False
            skip_reason = f"Score {match.score} is below the minimum of {s.min_score_to_apply}"

        if not apply_ok:
            session.add(
                Application(
                    job_id=job.id,
                    resume_id=match.resume.id,
                    score=match.score,
                    status="skipped",
                    skip_reason=skip_reason,
                )
            )
            await events.record(
                session,
                job_id=job.id,
                event_type="triage.job_skipped",
                message=skip_reason,
                data={"score": match.score, "verdict": match.result.verdict},
            )
            result.skipped += 1
            log.info(
                "triage.job_skipped",
                job_id=job.linkedin_job_id,
                score=match.score,
                reason=skip_reason,
            )
            continue

        try:
            letter = await generate_cover_letter(
                provider,
                job=job,
                resume=match.resume,
                matched_skills=match.result.matched_skills,
                settings=s,
            )
        except Exception as exc:  # a letter failure is not fatal
            result.errors += 1
            result.failures.append(f"job {job.linkedin_job_id}: cover letter failed: {exc}"[:300])
            log.error("triage.cover_letter_failed", job_id=job.linkedin_job_id, exc_info=True)
            continue

        review, review_reason = needs_human_review(
            match, cover_letter_flagged=letter.flagged, settings=s
        )

        # Honour REVIEW_MODE rather than forcing review on everything. An
        # "approved" row is a shortlist entry and nothing more: you still open
        # the posting and apply yourself.
        #
        # What stays un-bypassable, and deliberately lives in needs_human_review
        # rather than here: a letter the honesty guard flagged, a screening
        # question that could not be answered from stated facts, or a score
        # produced without a model. Those force review whatever REVIEW_MODE
        # says, because they are exactly the cases where automation should not
        # be trusted. Automating the approval must not automate away the one
        # control that stops a fabricated claim going out under someone's name.
        status = "pending_review" if review else "approved"
        session.add(
            Application(
                job_id=job.id,
                resume_id=match.resume.id,
                score=match.score,
                status=status,
                review_reason=review_reason or None,
                cover_letter=letter.text,
                cover_letter_flags=letter.flags,
            )
        )
        await events.record(
            session,
            job_id=job.id,
            event_type=("triage.queued_for_review" if review else "triage.auto_approved"),
            message=f"{job.title} at {job.company} scored {match.score}",
            data={
                "score": match.score,
                "flagged": letter.flagged,
                "review_reason": review_reason,
            },
        )
        if review:
            result.queued_for_review += 1
        else:
            result.auto_approved += 1
        log.info(
            "triage.queued_for_review" if review else "triage.auto_approved",
            job_id=job.linkedin_job_id,
            score=match.score,
            flagged=letter.flagged,
            reason=review_reason or None,
        )

    log.info(
        "triage.complete",
        considered=result.considered,
        scored=result.scored,
        queued=result.queued_for_review,
        auto_approved=result.auto_approved,
        skipped=result.skipped,
        errors=result.errors,
    )
    return result
