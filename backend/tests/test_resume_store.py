"""
Tests for resume ingestion.

Resume extraction is the source of truth for every AI claim, so a
silently-empty resume is a correctness bug, not a UX one.
"""

from __future__ import annotations

import io

import pytest

from app.services.resume_store import (
    ResumeExtractionError,
    _normalise_whitespace,
    _sanitise_filename,
    extract_text,
    guess_skills,
)


class TestResumeExtraction:
    def test_plain_text(self) -> None:
        body = "Jane Doe\n" + ("Senior Python engineer with Django experience. " * 10)
        assert "Django" in extract_text(body.encode(), "cv.txt")

    def test_markdown(self) -> None:
        body = "# Jane Doe\n\n" + ("Backend engineer, Python, PostgreSQL, Docker. " * 10)
        assert "PostgreSQL" in extract_text(body.encode(), "cv.md")

    def test_unsupported_extension_lists_what_is_supported(self) -> None:
        with pytest.raises(ResumeExtractionError, match=r"\.pdf"):
            extract_text(b"data", "cv.rtf")

    def test_too_little_text_is_treated_as_a_scanned_document(self) -> None:
        """Silently storing an empty resume would make every score meaningless."""
        with pytest.raises(ResumeExtractionError, match="scanned image|OCR"):
            extract_text(b"Jane Doe", "cv.txt")

    def test_unreadable_pdf_raises_a_clear_error(self) -> None:
        with pytest.raises(ResumeExtractionError, match="Could not read PDF"):
            extract_text(b"this is definitely not a pdf", "cv.pdf")

    def test_real_pdf_round_trip(self) -> None:
        """Build a minimal one-page PDF and read it back."""
        pytest.importorskip("pypdf")
        from pypdf import PdfWriter

        # An empty page yields no text, which must be rejected — that is the
        # scanned-document case and exactly the behaviour we want.
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        buffer = io.BytesIO()
        writer.write(buffer)

        with pytest.raises(ResumeExtractionError):
            extract_text(buffer.getvalue(), "blank.pdf")


class TestResumeHelpers:
    def test_whitespace_normalisation(self) -> None:
        messy = "Line  one\r\n\r\n\r\n\r\nLine   two\t\tend  "
        cleaned = _normalise_whitespace(messy)
        assert "\r" not in cleaned
        assert "\n\n\n" not in cleaned
        assert "Line one" in cleaned

    @pytest.mark.parametrize(
        ("raw", "expect_absent"),
        [
            ("../../etc/passwd", ".."),
            ("cv;rm -rf /.pdf", ";"),
            ("my resume.pdf", " "),
        ],
    )
    def test_filenames_are_sanitised(self, raw: str, expect_absent: str) -> None:
        assert expect_absent not in _sanitise_filename(raw)

    def test_sanitised_filename_is_never_empty(self) -> None:
        assert _sanitise_filename("???") == "resume"

    def test_skill_guessing(self) -> None:
        text = "Experienced with Python, Django, PostgreSQL, Docker, Kubernetes and AWS."
        skills = guess_skills(text)
        assert "python" in skills
        assert "postgresql" in skills
        assert "kubernetes" in skills

    def test_skill_guessing_is_bounded(self) -> None:
        assert len(guess_skills("python " * 1000, limit=5)) <= 5
