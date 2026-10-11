"""
Statistics and job-catalogue endpoints.

Purpose
-------
The application funnel, score distribution, the raw job catalogue, and a
flat export for spreadsheets (n8n's Google Sheets node reads it).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, or_, select

from app.api.deps import AppSettings, Authenticated, DbSession
from app.db.models import Application, Job, JobScore, Resume
from app.logging_config import get_logger
from app.schemas.api import DailyCount, JobOut, Page, Statistics

log = get_logger(__name__)

router = APIRouter(tags=["statistics"], dependencies=[Authenticated])


class JobList(Page):
    items: list[JobOut]


@router.get("/statistics", response_model=Statistics, summary="Funnel statistics")
async def statistics(session: DbSession, settings: AppSettings, days: int = 30) -> Statistics:
    """Aggregate counts and recent activity."""
    days = max(1, min(days, 365))

    total_jobs = (await session.execute(select(func.count()).select_from(Job))).scalar_one()

    # One grouped query rather than seven counts.
    status_rows = (
        await session.execute(select(Application.status, func.count()).group_by(Application.status))
    ).all()
    by_status = {row[0]: row[1] for row in status_rows}

    average_score = (await session.execute(select(func.avg(JobScore.score)))).scalar_one_or_none()

    top_companies = [
        {"company": row[0], "applications": row[1]}
        for row in (
            await session.execute(
                select(Job.company, func.count(Application.id))
                .join(Application, Application.job_id == Job.id)
                .where(Job.company != "")
                .group_by(Job.company)
                .order_by(func.count(Application.id).desc())
                .limit(10)
            )
        ).all()
    ]

    since = datetime.now(UTC) - timedelta(days=days)
    applied_by_day = [
        DailyCount(date=row[0].strftime("%Y-%m-%d"), count=row[1])
        for row in (
            await session.execute(
                select(
                    func.date_trunc("day", Application.submitted_at).label("day"),
                    func.count(),
                )
                .where(Application.submitted_at.is_not(None), Application.submitted_at >= since)
                .group_by("day")
                .order_by("day")
            )
        ).all()
    ]

    midnight = datetime.now(settings.tz).replace(hour=0, minute=0, second=0, microsecond=0)
    applied_today = (
        await session.execute(
            select(func.count())
            .select_from(Application)
            .where(Application.submitted_at >= midnight)
        )
    ).scalar_one()

    return Statistics(
        total_jobs_discovered=total_jobs,
        total_applications=sum(by_status.values()),
        pending_review=by_status.get("pending_review", 0),
        approved=by_status.get("approved", 0),
        applied=by_status.get("submitted", 0),
        skipped=by_status.get("skipped", 0),
        applied_today=applied_today,
        average_score=round(float(average_score), 1) if average_score is not None else None,
        top_companies=top_companies,
        applied_by_day=applied_by_day,
    )


@router.get("/jobs", response_model=JobList, summary="Job catalogue")
async def list_jobs(
    session: DbSession,
    company: str | None = None,
    easy_apply: bool | None = None,
    search: str | None = Query(default=None, description="Substring of title or company"),
    limit: int = 50,
    offset: int = 0,
) -> JobList:
    """Every job ever discovered, newest first."""
    # Heterogeneous SQLAlchemy clause types (BinaryExpression, BooleanClauseList)
    # share no useful common supertype, so the list is typed loosely.
    conditions: list[Any] = []
    if company:
        conditions.append(Job.company.ilike(f"%{company}%"))
    if easy_apply is not None:
        conditions.append(Job.easy_apply.is_(easy_apply))
    if search:
        conditions.append(or_(Job.title.ilike(f"%{search}%"), Job.company.ilike(f"%{search}%")))

    total = (
        await session.execute(select(func.count()).select_from(Job).where(*conditions))
    ).scalar_one()
    stmt = (
        select(Job)
        .where(*conditions)
        .order_by(Job.discovered_at.desc())
        .limit(min(limit, 200))
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()

    return JobList(
        total=total, limit=limit, offset=offset, items=[JobOut.model_validate(r) for r in rows]
    )


@router.get("/jobs/{job_id}", response_model=JobOut, summary="Get one job")
async def get_job(job_id: int, session: DbSession) -> JobOut:
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobOut.model_validate(job)


@router.get("/jobs/{job_id}/description", summary="Full job description text")
async def get_job_description(job_id: int, session: DbSession) -> dict[str, str]:
    """Return the stored description — the text every score was based on."""
    job = await session.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return {"job_id": str(job_id), "title": job.title, "description": job.description}


class ExportRow(BaseModel):
    """One flat row, shaped for a spreadsheet.

    Deliberately denormalised. n8n's Google Sheets node maps a flat object to
    columns, and making n8n join applications to jobs would put business logic
    in a place that is hard to test and easy to break.
    """

    application_id: int
    status: str
    score: int | None
    title: str
    company: str
    location: str
    url: str
    resume: str
    cover_letter_flagged: bool
    review_reason: str
    skip_reason: str
    discovered: str
    submitted: str


@router.get(
    "/export/applications",
    response_model=list[ExportRow],
    summary="Flat application rows for a spreadsheet",
)
async def export_applications(session: DbSession, limit: int = 500) -> list[ExportRow]:
    """Every application as a flat row, newest first.

    Backs the Google Sheets sync. `application_id` is the stable key the sheet
    matches on, so re-running the sync updates rows in place rather than
    appending duplicates.
    """
    stmt = (
        select(Application, Job, Resume.label)
        .join(Job, Job.id == Application.job_id)
        .outerjoin(Resume, Resume.id == Application.resume_id)
        .order_by(Application.created_at.desc())
        .limit(min(limit, 2000))
    )
    rows = (await session.execute(stmt)).all()

    return [
        ExportRow(
            application_id=a.id,
            status=a.status,
            score=a.score,
            title=j.title,
            company=j.company,
            location=j.location,
            url=j.url,
            resume=label or "",
            cover_letter_flagged=a.cover_letter_flagged,
            review_reason=a.review_reason or "",
            skip_reason=a.skip_reason or "",
            # ISO dates so Sheets parses them rather than treating them as text.
            discovered=a.created_at.strftime("%Y-%m-%d %H:%M"),
            submitted=a.submitted_at.strftime("%Y-%m-%d %H:%M") if a.submitted_at else "",
        )
        for a, j, label in rows
    ]
