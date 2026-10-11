"""
API request/response schemas.

Purpose
-------
The typed contract for the dashboard API. Response models are explicit rather
than serialising ORM objects directly, so an added database column can never
silently leak into an HTTP response.

Failure scenarios
-----------------
* Invalid request body → FastAPI returns 422 with per-field detail.
* Out-of-range values → rejected here, before any database write.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ── Common ────────────────────────────────────────────────────────────


class Health(BaseModel):
    """Health probe response."""

    status: Literal["ok", "degraded", "error"]
    version: str
    environment: str
    checks: dict[str, bool]
    detail: dict[str, str] = Field(default_factory=dict)


class Message(BaseModel):
    """Generic acknowledgement."""

    message: str


class Page(BaseModel):
    """Pagination envelope."""

    total: int
    limit: int
    offset: int


# ── Resumes ───────────────────────────────────────────────────────────


class ResumeOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    label: str
    file_path: str
    skills: list[str]
    years_experience: float | None
    target_roles: list[str]
    is_default: bool
    is_active: bool
    content_chars: int = 0
    created_at: datetime


class ResumeList(Page):
    items: list[ResumeOut]


# ── Jobs & applications ───────────────────────────────────────────────


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    linkedin_job_id: str
    url: str
    title: str
    company: str
    location: str
    workplace_type: str | None
    employment_type: str | None
    seniority: str | None
    easy_apply: bool
    applicant_count: int | None
    posted_at: datetime | None
    discovered_at: datetime


class ScoreOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    score: int
    verdict: str
    reasoning: str
    matched_skills: list[str]
    missing_skills: list[str]
    concerns: list[str]
    provider: str
    model: str


class AnswerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    question: str
    question_type: str
    answer: str | None
    source: str
    confidence: float | None
    requires_review: bool


class ApplicationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    job_id: int
    run_id: int | None
    resume_id: int | None
    score: int | None
    status: str
    cover_letter: str | None
    cover_letter_flagged: bool
    cover_letter_flags: list[str]
    review_reason: str | None
    skip_reason: str | None
    notes: str | None
    error: str | None
    attempts: int
    reviewed_at: datetime | None
    submitted_at: datetime | None
    created_at: datetime


class ApplicationDetail(ApplicationOut):
    """Everything needed to make a review decision on one screen."""

    job: JobOut
    resume_label: str | None = None
    score_detail: ScoreOut | None = None
    answers: list[AnswerOut] = Field(default_factory=list)


class ApplicationList(Page):
    items: list[ApplicationOut]


class ReviewDecision(BaseModel):
    """A human's verdict on a pending application."""

    decision: Literal["approve", "reject"]
    # Lets you fix a cover letter before approving rather than rejecting outright.
    cover_letter: str | None = None
    notes: str | None = Field(default=None, max_length=4000)


class AppliedIn(BaseModel):
    """You applied by hand. Recorded so the job is never offered again."""

    notes: str | None = Field(default=None, max_length=4000)


class QuestionIn(BaseModel):
    """One screening question, copied from the form you are filling in."""

    question: str = Field(..., min_length=3, max_length=2000)
    question_type: Literal["text", "numeric", "boolean", "select", "radio", "checkbox"] = "text"
    options: list[str] | None = None


class QuestionsIn(BaseModel):
    questions: list[QuestionIn] = Field(..., min_length=1, max_length=40)


class AnsweredOut(AnswerOut):
    """An answer plus why it was given (or why it was refused)."""

    reason: str


# ── Statistics ────────────────────────────────────────────────────────


class DailyCount(BaseModel):
    date: str
    count: int


class Statistics(BaseModel):
    """Funnel summary."""

    total_jobs_discovered: int
    total_applications: int
    pending_review: int
    approved: int
    applied: int
    skipped: int
    applied_today: int
    average_score: float | None
    top_companies: list[dict[str, object]]
    applied_by_day: list[DailyCount]


# ── System ────────────────────────────────────────────────────────────


class SystemStatus(BaseModel):
    """Configuration snapshot. Never contains a secret value."""

    version: str
    environment: str
    review_mode: str
    min_score_to_apply: int
    auto_approve_threshold: int
    ai_provider: str
    ai_model: str
    notifications: list[str]
