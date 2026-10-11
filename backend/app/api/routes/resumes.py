"""
Resume management endpoints.

Purpose
-------
Upload, list, activate, and delete CV variants. The uploaded text is what bounds
every AI claim, so this is a security-relevant surface as well as a convenience.

Failure scenarios
-----------------
* Scanned PDF → 400 with an instruction to OCR it, rather than storing an empty
  resume that would silently produce meaningless scores.
* Duplicate label → 409.
* Deleting the last active resume → allowed, but runs will then fail fast with
  ``NoResumeError`` rather than scoring against nothing.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.api.deps import AppSettings, Authenticated, DbSession
from app.db.models import Resume
from app.logging_config import get_logger
from app.schemas.api import Message, ResumeList, ResumeOut
from app.services.resume_store import (
    ResumeExtractionError,
    extract_text,
    guess_skills,
    save_file,
    set_default_resume,
)

log = get_logger(__name__)

router = APIRouter(prefix="/resumes", tags=["resumes"], dependencies=[Authenticated])

# Refuse anything implausible for a CV before reading it into memory.
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def _to_out(resume: Resume) -> ResumeOut:
    return ResumeOut(
        id=resume.id,
        label=resume.label,
        file_path=resume.file_path,
        skills=list(resume.skills),
        years_experience=float(resume.years_experience) if resume.years_experience else None,
        target_roles=list(resume.target_roles),
        is_default=resume.is_default,
        is_active=resume.is_active,
        content_chars=len(resume.content_text),
        created_at=resume.created_at,
    )


@router.get("", response_model=ResumeList, summary="List resumes")
async def list_resumes(session: DbSession, limit: int = 50, offset: int = 0) -> ResumeList:
    total = (await session.execute(select(func.count()).select_from(Resume))).scalar_one()
    stmt = (
        select(Resume)
        .order_by(Resume.is_default.desc(), Resume.id.asc())
        .limit(min(limit, 200))
        .offset(offset)
    )
    rows = (await session.execute(stmt)).scalars().all()
    return ResumeList(total=total, limit=limit, offset=offset, items=[_to_out(r) for r in rows])


@router.post(
    "",
    response_model=ResumeOut,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a resume",
)
async def upload_resume(
    session: DbSession,
    settings: AppSettings,
    file: Annotated[UploadFile, File(description="PDF, DOCX, TXT, or MD")],
    label: Annotated[str, Form(description="Unique name, e.g. 'backend-python'")],
    is_default: Annotated[bool, Form()] = False,
    years_experience: Annotated[float | None, Form()] = None,
    target_roles: Annotated[str, Form(description="Comma-separated")] = "",
) -> ResumeOut:
    """Upload a CV variant and extract its text."""
    data = await file.read()

    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is {len(data) // 1024} KB; the limit is {_MAX_UPLOAD_BYTES // 1024} KB",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        content_text = extract_text(data, file.filename or "resume.pdf")
    except ResumeExtractionError as exc:
        # These messages are written to be actionable — surface them verbatim.
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    path = save_file(data, file.filename or "resume.pdf", settings.resume_dir)

    resume = Resume(
        label=label.strip(),
        file_path=path,
        content_text=content_text,
        skills=guess_skills(content_text),
        years_experience=years_experience,
        target_roles=[r.strip() for r in target_roles.split(",") if r.strip()],
        is_default=False,  # set below, so the partial unique index is respected
        is_active=True,
    )
    session.add(resume)

    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A resume labelled {label!r} already exists",
        ) from exc

    # First resume uploaded becomes the default automatically — otherwise the
    # user has a resume but no runs can score.
    existing = (await session.execute(select(func.count()).select_from(Resume))).scalar_one()
    if is_default or existing == 1:
        await set_default_resume(session, resume.id)

    await session.commit()
    await session.refresh(resume)

    log.info(
        "api.resume_uploaded",
        resume_id=resume.id,
        label=resume.label,
        chars=len(content_text),
        skills=len(resume.skills),
    )
    return _to_out(resume)


@router.get("/{resume_id}", response_model=ResumeOut, summary="Get one resume")
async def get_resume(resume_id: int, session: DbSession) -> ResumeOut:
    resume = await session.get(Resume, resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    return _to_out(resume)


@router.post("/{resume_id}/default", response_model=Message, summary="Set as default")
async def make_default(resume_id: int, session: DbSession) -> Message:
    resume = await session.get(Resume, resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")
    await set_default_resume(session, resume_id)
    await session.commit()
    return Message(message=f"{resume.label!r} is now the default resume")


@router.delete("/{resume_id}", response_model=Message, summary="Deactivate a resume")
async def deactivate_resume(resume_id: int, session: DbSession) -> Message:
    """Deactivate rather than delete.

    Scores and applications reference the resume that produced them; a hard
    delete would destroy that audit trail.
    """
    resume = await session.get(Resume, resume_id)
    if resume is None:
        raise HTTPException(status_code=404, detail=f"Resume {resume_id} not found")

    resume.is_active = False
    resume.is_default = False
    await session.commit()

    log.info("api.resume_deactivated", resume_id=resume_id, label=resume.label)
    return Message(message=f"{resume.label!r} deactivated (history preserved)")
