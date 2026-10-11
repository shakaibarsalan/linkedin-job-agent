"""
SQLAlchemy ORM models.

Purpose
-------
Typed Python mirror of ``database/migrations/*.sql``. The SQL files remain the
source of truth for DDL; these classes exist so application code gets type
checking and relationship traversal.

How it works
------------
Declarative mapping against the existing tables. Nothing here creates schema —
``app.db.migrate`` applies the SQL files. ``tests/test_schema_parity.py``
asserts that every column declared here actually exists in the migrated
database, which is what stops the two definitions drifting apart.

Failure scenarios
-----------------
* Column added to SQL but not here → harmless; the ORM ignores it.
* Column added here but not to SQL → ``UndefinedColumnError`` at query time.
  The parity test catches this in CI before it can reach production.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    ARRAY,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base for all models."""


# Reusable column types
_TS = DateTime(timezone=True)


class Resume(Base):
    """A CV variant. ``content_text`` bounds what the AI layer may claim."""

    __tablename__ = "resumes"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    label: Mapped[str] = mapped_column(Text, unique=True)
    file_path: Mapped[str] = mapped_column(Text)
    content_text: Mapped[str] = mapped_column(Text)
    skills: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    years_experience: Mapped[Decimal | None] = mapped_column(Numeric(4, 1), nullable=True)
    target_roles: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())

    scores: Mapped[list[JobScore]] = relationship(back_populates="resume")


class SearchProfile(Base):
    """A saved search plus the policy applied to its results."""

    __tablename__ = "search_profiles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str] = mapped_column(Text, unique=True)
    keywords: Mapped[str] = mapped_column(Text)
    location: Mapped[str] = mapped_column(Text, default="")
    remote_only: Mapped[bool] = mapped_column(Boolean, default=False)
    easy_apply_only: Mapped[bool] = mapped_column(Boolean, default=True)
    experience_levels: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    job_types: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    date_posted: Mapped[str] = mapped_column(Text, default="past_week")
    exclude_companies: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    exclude_title_keywords: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    min_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    max_applications_per_run: Mapped[int | None] = mapped_column(Integer, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())

    runs: Mapped[list[Run]] = relationship(back_populates="search_profile")


class Run(Base):
    """One execution of search → score → apply."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    search_profile_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("search_profiles.id", ondelete="SET NULL"), nullable=True
    )
    status: Mapped[str] = mapped_column(Text, default="pending")
    trigger: Mapped[str] = mapped_column(Text, default="manual")
    pause_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    jobs_scanned: Mapped[int] = mapped_column(Integer, default=0)
    jobs_scored: Mapped[int] = mapped_column(Integer, default=0)
    applications_created: Mapped[int] = mapped_column(Integer, default=0)
    applications_submitted: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())

    search_profile: Mapped[SearchProfile | None] = relationship(back_populates="runs")
    applications: Mapped[list[Application]] = relationship(back_populates="run")

    TERMINAL_STATUSES = frozenset({"completed", "failed", "cancelled"})

    @property
    def is_terminal(self) -> bool:
        return self.status in self.TERMINAL_STATUSES


class Job(Base):
    """A job posting. ``linkedin_job_id`` is the dedupe key."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    linkedin_job_id: Mapped[str] = mapped_column(Text, unique=True)
    url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    company: Mapped[str] = mapped_column(Text, default="")
    location: Mapped[str] = mapped_column(Text, default="")
    workplace_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    employment_type: Mapped[str | None] = mapped_column(Text, nullable=True)
    seniority: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")
    easy_apply: Mapped[bool] = mapped_column(Boolean, default=False)
    applicant_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    posted_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())
    first_seen_run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    raw: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    scores: Mapped[list[JobScore]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    application: Mapped[Application | None] = relationship(
        back_populates="job", uselist=False, cascade="all, delete-orphan"
    )


class JobScore(Base):
    """Compatibility of one job against one resume."""

    __tablename__ = "job_scores"
    __table_args__ = (
        UniqueConstraint("job_id", "resume_id", name="job_scores_job_id_resume_id_key"),
        CheckConstraint("score BETWEEN 0 AND 100", name="job_scores_score_check"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id", ondelete="CASCADE"))
    resume_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("resumes.id", ondelete="CASCADE"))
    score: Mapped[int] = mapped_column(Integer)
    verdict: Mapped[str] = mapped_column(Text)
    reasoning: Mapped[str] = mapped_column(Text, default="")
    matched_skills: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    missing_skills: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    concerns: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    provider: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="scores")
    resume: Mapped[Resume] = relationship(back_populates="scores")


class Application(Base):
    """An application to a job. Unique per job — the never-apply-twice guard."""

    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    job_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("jobs.id", ondelete="CASCADE"))
    run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("runs.id", ondelete="SET NULL"), nullable=True
    )
    resume_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("resumes.id", ondelete="SET NULL"), nullable=True
    )
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="pending_review")
    cover_letter: Mapped[str | None] = mapped_column(Text, nullable=True)
    cover_letter_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    cover_letter_flags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    review_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    skip_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    screenshot_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    reviewed_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    submitted_at: Mapped[datetime | None] = mapped_column(_TS, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())

    job: Mapped[Job] = relationship(back_populates="application")
    run: Mapped[Run | None] = relationship(back_populates="applications")
    answers: Mapped[list[ApplicationAnswer]] = relationship(
        back_populates="application", cascade="all, delete-orphan"
    )


class ApplicationAnswer(Base):
    """A screening question, its answer, and the provenance of that answer."""

    __tablename__ = "application_answers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    application_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("applications.id", ondelete="CASCADE")
    )
    question: Mapped[str] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(Text, default="text")
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(Text, default="unknown")
    confidence: Mapped[Decimal | None] = mapped_column(Numeric(3, 2), nullable=True)
    requires_review: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())

    application: Mapped[Application] = relationship(back_populates="answers")


class Event(Base):
    """Append-only audit log."""

    __tablename__ = "events"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("runs.id", ondelete="CASCADE"), nullable=True
    )
    application_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("applications.id", ondelete="CASCADE"), nullable=True
    )
    job_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("jobs.id", ondelete="SET NULL"), nullable=True
    )
    level: Mapped[str] = mapped_column(Text, default="info")
    event_type: Mapped[str] = mapped_column(Text)
    message: Mapped[str] = mapped_column(Text, default="")
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())


class NotificationLog(Base):
    """Delivery record per notification channel."""

    __tablename__ = "notification_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    channel: Mapped[str] = mapped_column(Text)
    event_type: Mapped[str] = mapped_column(Text, default="")
    subject: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(Text, default="sent")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())


class AppState(Base):
    """Small key/value store for cursors and session validity flags."""

    __tablename__ = "app_state"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    updated_at: Mapped[datetime] = mapped_column(_TS, server_default=func.now())
