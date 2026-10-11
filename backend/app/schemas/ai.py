"""
Pydantic schemas for AI responses.

Purpose
-------
Model output is untrusted input. These schemas are the boundary where it stops
being a string and becomes typed, range-checked data safe to act on.

Failure scenarios
-----------------
* Score out of range, unknown verdict, wrong types → ``ValidationError``,
  which :meth:`AIProvider.complete_json` converts into a correction turn and,
  if that fails, an :class:`AIResponseError` the caller degrades from.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Verdict = Literal["strong_match", "possible_match", "weak_match", "not_a_match"]
AnswerSource = Literal["profile", "resume", "ai", "manual", "default", "unknown"]


class JobScoreResult(BaseModel):
    """A model's compatibility assessment of one job against one resume."""

    score: int = Field(ge=0, le=100)
    verdict: Verdict
    reasoning: str = ""
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)

    @field_validator("matched_skills", "missing_skills", "concerns", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> object:
        """Accept a bare string where a list was specified.

        Smaller models routinely return ``"Python, SQL"`` instead of a list;
        rejecting that would burn a retry for no benefit.
        """
        if v is None:
            return []
        if isinstance(v, str):
            return [part.strip() for part in v.split(",") if part.strip()]
        return v

    @field_validator("matched_skills", "missing_skills", "concerns")
    @classmethod
    def _cap_length(cls, v: list[str]) -> list[str]:
        """Bound list size so a runaway response cannot bloat a database row."""
        return [item[:200] for item in v[:40]]

    @field_validator("reasoning")
    @classmethod
    def _cap_reasoning(cls, v: str) -> str:
        return v[:4000]


class ResumeSelection(BaseModel):
    """Which CV variant the model recommends for a role."""

    resume_label: str
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    reasoning: str = ""

    @field_validator("reasoning")
    @classmethod
    def _cap(cls, v: str) -> str:
        return v[:1000]


class CoverLetterResult(BaseModel):
    """A generated cover letter plus the claims it makes."""

    cover_letter: str
    claims_made: list[str] = Field(default_factory=list)
    warning: str = ""

    @field_validator("claims_made", mode="before")
    @classmethod
    def _coerce_list(cls, v: object) -> object:
        if v is None:
            return []
        if isinstance(v, str):
            return [v] if v.strip() else []
        return v

    @field_validator("cover_letter")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        stripped = v.strip()
        if len(stripped) < 50:
            raise ValueError("Cover letter is implausibly short (< 50 characters)")
        return stripped[:8000]


class QuestionAnswerResult(BaseModel):
    """A model's proposed answer to a screening question."""

    answer: str = ""
    needs_human: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source: AnswerSource = "unknown"
    reasoning: str = ""

    @field_validator("answer")
    @classmethod
    def _cap_answer(cls, v: str) -> str:
        return v.strip()[:2000]

    @field_validator("reasoning")
    @classmethod
    def _cap_reasoning(cls, v: str) -> str:
        return v[:1000]
