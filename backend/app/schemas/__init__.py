"""Pydantic schemas: AI response validation and the HTTP API contract."""

from app.schemas.ai import (
    CoverLetterResult,
    JobScoreResult,
    QuestionAnswerResult,
    ResumeSelection,
)

__all__ = [
    "CoverLetterResult",
    "JobScoreResult",
    "QuestionAnswerResult",
    "ResumeSelection",
]
