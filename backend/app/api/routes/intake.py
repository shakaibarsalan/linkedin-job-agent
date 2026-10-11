"""
Intake routes — bring jobs in from outside LinkedIn, and score them there.

Purpose
-------
Lets any external source (a job board API, an RSS feed, an n8n workflow, or a
JSON file you put together by hand) hand jobs to the system. Reading a posting
and deciding whether it is worth applying to needs the job's text, not a
logged-in browser.

Two endpoints, deliberately separate:

``POST /jobs/ingest``
    Fast and cheap. Stores what it is given and returns immediately, so the
    caller is not held open while models run.

``POST /jobs/triage``
    Slow and metered. Scores stored jobs, writes cover letters, runs the
    anti-fabrication check, and queues survivors for human review. Bounded by
    ``limit`` so one call cannot run away with the AI budget.

Splitting them means a triage that dies halfway has still banked the ingest,
and re-running resumes rather than restarting.

Nothing here can submit an application. Everything it creates is
``pending_review``, ``approved`` (a shortlist, nothing more) or ``skipped``.
You press Apply yourself.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query
from pydantic import BaseModel, Field

from app.api.deps import AppSettings, Authenticated, DbSession
from app.logging_config import get_logger
from app.services.ai.factory import build_provider
from app.services.notifications import Dispatcher, review_required
from app.services.triage import ingest_jobs, triage_jobs

log = get_logger(__name__)

router = APIRouter(tags=["intake"], dependencies=[Authenticated])


class IncomingJob(BaseModel):
    """One job from an external source.

    Only the three fields needed to score a posting honestly are required.
    Everything else is best-effort: a missing company costs context, a missing
    description means the job cannot be judged at all.
    """

    linkedin_job_id: str = Field(..., description="Numeric LinkedIn job id")
    title: str
    description: str = Field(..., description="Full posting text, not a summary")
    company: str = ""
    location: str = ""
    url: str = ""
    workplace_type: str | None = None
    employment_type: str | None = None
    seniority: str | None = None
    applicant_count: int | None = None
    posted_at: str | None = None

    model_config = {"extra": "allow"}


class IngestRequest(BaseModel):
    jobs: list[IncomingJob]
    source: str = Field("manual", description="Recorded on each row for provenance")


@router.post(
    "/jobs/ingest",
    summary="Store jobs from any source (nothing here logs in anywhere)",
)
async def ingest(
    session: DbSession,
    payload: IngestRequest = Body(...),
) -> dict[str, Any]:
    """Store a batch of externally-sourced jobs.

    Duplicates are refreshed rather than rejected, so re-running a scraper over
    an overlapping window is safe and cheap.

    Jobs without a real description are refused outright. Storing one would
    make it indistinguishable from a job that was considered and passed over,
    and scoring one produces a confident number derived from nothing.
    """
    result = await ingest_jobs(
        session,
        [job.model_dump() for job in payload.jobs],
        source=payload.source,
    )
    await session.commit()
    return result.as_dict()


@router.post(
    "/jobs/triage",
    summary="Score stored jobs, write cover letters, queue for review",
)
async def triage(
    session: DbSession,
    settings: AppSettings,
    limit: int = Query(25, ge=1, le=200, description="Most jobs to process in one call"),
) -> dict[str, Any]:
    """Score everything stored but not yet judged.

    "Not yet judged" is derived from the absence of a score, not from a queue,
    so an interrupted call resumes exactly where it stopped and no job is ever
    scored twice.

    Whatever the score, nothing here applies anywhere. A clean, strong match
    may be shortlisted as ``approved`` when ``REVIEW_MODE`` allows it; anything
    flagged is held for you as ``pending_review``.
    """
    provider = build_provider(settings)
    try:
        result = await triage_jobs(session, provider, limit=limit, settings=settings)
        await session.commit()
        if result.queued_for_review:
            dispatcher = Dispatcher(settings=settings)
            try:
                await dispatcher.send(
                    review_required(
                        count=result.queued_for_review, dashboard_url=settings.public_url
                    ),
                    session=session,
                )
                await session.commit()
            finally:
                await dispatcher.aclose()
        return result.as_dict()
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()
