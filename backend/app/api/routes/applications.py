"""
Application and human-review endpoints.

Purpose
-------
The review queue: everything the agent prepared and stopped on, the
approve/reject decision, and a record of what you applied to by hand.

Nothing in this API submits an application anywhere. "Approved" means "on my
shortlist"; you open the posting and apply yourself, then tell the API you did.

How it works
------------
``GET /applications/{id}`` returns everything needed to decide on one screen:
the job, the score with its reasoning, the cover letter with any honesty flags,
and every screening answer with its provenance.

``POST /applications/{id}/review`` records the decision. Approving moves the
application to ``approved`` (your shortlist). The cover letter may be edited in
the same call — correcting a sentence is far more useful than rejecting
outright.

``POST /applications/{id}/answers`` takes the screening questions you see on
the real form and answers them from your profile and CV, or refuses.

``POST /applications/{id}/applied`` records that you applied, by hand.

Failure scenarios
-----------------
* Reviewing an application that is not ``pending_review`` → 409.
* Approving one whose letter is flagged → allowed, but the flags are shown and
  the override is recorded in ``notes`` for the audit trail.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from app.api.deps import AppSettings, Authenticated, DbSession
from app.db.models import Application, ApplicationAnswer, Job, JobScore, Resume
from app.logging_config import get_logger
from app.schemas.api import (
    AnsweredOut,
    AnswerOut,
    ApplicationDetail,
    ApplicationList,
    ApplicationOut,
    AppliedIn,
    JobOut,
    Message,
    QuestionsIn,
    ReviewDecision,
    ScoreOut,
)
from app.services import events
from app.services.ai.factory import build_provider
from app.services.profile import Profile, ProfileError
from app.services.question_answering import FormQuestion, QuestionAnswerer

log = get_logger(__name__)

router = APIRouter(prefix="/applications", tags=["applications"], dependencies=[Authenticated])

VALID_STATUSES = frozenset(
    {
        "pending_review",
        "approved",
        "submitting",
        "submitted",
        "skipped",
        "needs_manual",
        "failed",
    }
)


@router.get("", response_model=ApplicationList, summary="List applications")
async def list_applications(
    session: DbSession,
    status_filter: Annotated[str | None, Query(alias="status")] = None,
    run_id: int | None = None,
    limit: int = 50,
    offset: int = 0,
) -> ApplicationList:
    """List applications, newest first. Filter by status and/or run."""
    if status_filter and status_filter not in VALID_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown status {status_filter!r}. Valid: {sorted(VALID_STATUSES)}",
        )

    conditions = []
    if status_filter:
        conditions.append(Application.status == status_filter)
    if run_id is not None:
        conditions.append(Application.run_id == run_id)

    total = (
        await session.execute(select(func.count()).select_from(Application).where(*conditions))
    ).scalar_one()

    stmt = (
        select(Application)
        .where(*conditions)
        .order_by(Application.created_at.desc())
        .limit(min(limit, 200))
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()

    return ApplicationList(
        total=total,
        limit=limit,
        offset=offset,
        items=[ApplicationOut.model_validate(r) for r in rows],
    )


@router.get(
    "/pending-review",
    response_model=ApplicationList,
    summary="The review queue",
)
async def pending_review(session: DbSession, limit: int = 50) -> ApplicationList:
    """Applications waiting for your decision, best score first."""
    conditions = [Application.status == "pending_review"]
    total = (
        await session.execute(select(func.count()).select_from(Application).where(*conditions))
    ).scalar_one()

    stmt = (
        select(Application)
        .where(*conditions)
        .order_by(Application.score.desc().nullslast(), Application.created_at.asc())
        .limit(min(limit, 200))
    )
    rows = (await session.execute(stmt)).scalars().all()

    return ApplicationList(
        total=total, limit=limit, offset=0, items=[ApplicationOut.model_validate(r) for r in rows]
    )


@router.get(
    "/{application_id}",
    response_model=ApplicationDetail,
    summary="Everything needed to review one application",
)
async def get_application(application_id: int, session: DbSession) -> ApplicationDetail:
    stmt = (
        select(Application)
        .where(Application.id == application_id)
        .options(selectinload(Application.answers))
    )
    application = (await session.execute(stmt)).scalar_one_or_none()
    if application is None:
        raise HTTPException(status_code=404, detail=f"Application {application_id} not found")

    job = await session.get(Job, application.job_id)
    if job is None:  # pragma: no cover — FK guarantees this
        raise HTTPException(status_code=500, detail="Application references a missing job")

    resume_label: str | None = None
    score_detail: ScoreOut | None = None

    if application.resume_id is not None:
        resume = await session.get(Resume, application.resume_id)
        resume_label = resume.label if resume else None

        score_row = (
            await session.execute(
                select(JobScore).where(
                    JobScore.job_id == application.job_id,
                    JobScore.resume_id == application.resume_id,
                )
            )
        ).scalar_one_or_none()
        if score_row is not None:
            score_detail = ScoreOut.model_validate(score_row)

    return ApplicationDetail(
        **ApplicationOut.model_validate(application).model_dump(),
        job=JobOut.model_validate(job),
        resume_label=resume_label,
        score_detail=score_detail,
        answers=[AnswerOut.model_validate(a) for a in application.answers],
    )


@router.post(
    "/{application_id}/review",
    response_model=ApplicationOut,
    summary="Approve or reject an application",
)
async def review_application(
    application_id: int, decision: ReviewDecision, session: DbSession
) -> ApplicationOut:
    """Record a human decision.

    Approving puts the application on your shortlist. Rejecting marks it
    skipped; the job is never re-offered because ``applications.job_id`` is
    unique.
    """
    application = await session.get(Application, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail=f"Application {application_id} not found")

    if application.status not in {"pending_review", "needs_manual"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Application {application_id} is {application.status!r}; only "
                "'pending_review' and 'needs_manual' applications can be reviewed"
            ),
        )

    was_flagged = application.cover_letter_flagged
    application.reviewed_at = datetime.now(UTC)

    if decision.cover_letter is not None:
        # A hand-edited letter is authored by you, so the machine flags no
        # longer apply — clear them rather than leaving a stale warning.
        application.cover_letter = decision.cover_letter
        application.cover_letter_flagged = False
        application.cover_letter_flags = []

    if decision.notes:
        application.notes = decision.notes

    if decision.decision == "approve":
        application.status = "approved"
        application.review_reason = None
        message = "approved"
    else:
        application.status = "skipped"
        application.skip_reason = decision.notes or "Rejected during human review"
        message = "rejected"

    await events.record(
        session,
        application_id=application_id,
        job_id=application.job_id,
        run_id=application.run_id,
        event_type=f"application.{message}",
        message=f"Human {message} this application",
        data={
            "score": application.score,
            "cover_letter_edited": decision.cover_letter is not None,
            # Explicitly record when a human overrode an honesty flag.
            "approved_despite_flags": bool(
                was_flagged and decision.decision == "approve" and decision.cover_letter is None
            ),
        },
    )
    await session.commit()
    await session.refresh(application)

    log.info(
        "api.application_reviewed",
        application_id=application_id,
        decision=decision.decision,
        score=application.score,
        letter_edited=decision.cover_letter is not None,
    )
    return ApplicationOut.model_validate(application)


@router.post(
    "/{application_id}/applied",
    response_model=ApplicationOut,
    summary="Record that you applied by hand",
)
async def mark_applied(
    application_id: int, session: DbSession, payload: AppliedIn | None = None
) -> ApplicationOut:
    """You pressed Apply yourself. Record it, so the job never comes back."""
    application = await session.get(Application, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail=f"Application {application_id} not found")

    if application.status not in {"pending_review", "approved", "needs_manual"}:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Application {application_id} is {application.status!r} "
                "and cannot be marked applied"
            ),
        )

    application.status = "submitted"
    application.submitted_at = datetime.now(UTC)
    if payload is not None and payload.notes:
        application.notes = payload.notes

    await events.record(
        session,
        application_id=application_id,
        job_id=application.job_id,
        event_type="application.applied_by_hand",
        message="You applied to this job yourself",
        data={"score": application.score},
    )
    await session.commit()
    await session.refresh(application)
    log.info("api.application_applied_by_hand", application_id=application_id)
    return ApplicationOut.model_validate(application)


@router.post(
    "/{application_id}/answers",
    response_model=list[AnsweredOut],
    summary="Answer the screening questions on a form you are filling in",
)
async def answer_questions(
    application_id: int,
    payload: QuestionsIn,
    session: DbSession,
    settings: AppSettings,
) -> list[AnsweredOut]:
    """Answer screening questions from your profile and CV, or refuse.

    Paste the questions from the real application form. Each one is answered
    from a fact in your profile first, then by a model that may only use your
    profile and CV, and whose answer is re-checked by the fabrication guard.
    Anything else comes back unanswered, with the reason. Nothing is guessed.
    """
    application = await session.get(Application, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail=f"Application {application_id} not found")
    resume = await session.get(Resume, application.resume_id) if application.resume_id else None
    if resume is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This application has no resume attached, so there is nothing to answer from",
        )
    try:
        profile = Profile.load(settings.profile_path)
    except ProfileError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    provider = build_provider(settings)
    try:
        answerer = QuestionAnswerer(
            profile=profile, provider=provider, resume_text=resume.content_text, settings=settings
        )
        answered = await answerer.answer_all(
            [
                FormQuestion(question=q.question, question_type=q.question_type, options=q.options)
                for q in payload.questions
            ]
        )
    finally:
        aclose = getattr(provider, "aclose", None)
        if aclose is not None:
            await aclose()

    session.add_all([ApplicationAnswer(**a.to_row(application_id)) for a in answered])
    await session.commit()

    return [
        AnsweredOut(
            question=a.question.question,
            question_type=a.question.question_type,
            answer=a.answer,
            source=a.source,
            confidence=a.confidence,
            requires_review=a.requires_review,
            reason=a.reason,
        )
        for a in answered
    ]


@router.get(
    "/{application_id}/answers",
    response_model=list[AnswerOut],
    summary="Screening answers and their provenance",
)
async def get_answers(application_id: int, session: DbSession) -> list[AnswerOut]:
    """Every question asked, its answer, and where the answer came from."""
    stmt = (
        select(ApplicationAnswer)
        .where(ApplicationAnswer.application_id == application_id)
        .order_by(ApplicationAnswer.id.asc())
    )
    rows = (await session.execute(stmt)).scalars().all()
    return [AnswerOut.model_validate(r) for r in rows]


@router.delete("/{application_id}", response_model=Message, summary="Skip an application")
async def skip_application(application_id: int, session: DbSession) -> Message:
    """Mark an application skipped."""
    application = await session.get(Application, application_id)
    if application is None:
        raise HTTPException(status_code=404, detail=f"Application {application_id} not found")

    if application.status == "submitted":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="You already applied to this job; skipping it now would hide that",
        )

    application.status = "skipped"
    application.skip_reason = "Manually skipped"
    await session.commit()

    return Message(message=f"Application {application_id} skipped")
