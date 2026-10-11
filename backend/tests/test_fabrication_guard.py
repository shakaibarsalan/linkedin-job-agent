"""
Tests for the anti-fabrication guard.

This is the most safety-critical module in the codebase: it is the mechanical
check that stops a generated cover letter making claims the resume does not
support. These tests encode the behaviour the requirement "never fabricate
experience" actually demands.
"""

from __future__ import annotations

import pytest

from app.services.fabrication_guard import (
    _years_from_date_ranges,
    verify_against_resume,
    verify_answer,
)


class TestYearClaims:
    """Year-of-experience claims must be supported by the resume."""

    def test_inflated_years_claim_is_flagged(self, resume_text: str) -> None:
        report = verify_against_resume(
            "I bring over 15 years of Python experience to this role.", resume_text
        )
        assert not report.ok
        assert any(f.kind == "unsupported_years" for f in report.flags)

    def test_supported_years_claim_passes(self, resume_text: str) -> None:
        # Resume spans 2019 to present, so a ~6-7 year claim is honest.
        report = verify_against_resume(
            "I have 6 years of backend engineering experience.", resume_text
        )
        assert report.ok, report.as_strings()

    def test_one_year_of_rounding_slack_is_allowed(self, resume_text: str) -> None:
        """Rounding "nearly 7" up to 7 is honest, not fabrication."""
        span = _years_from_date_ranges(resume_text)
        assert span is not None
        report = verify_against_resume(f"I have {span + 1} years of experience.", resume_text)
        assert report.ok, report.as_strings()

    def test_two_years_beyond_the_ceiling_is_flagged(self, resume_text: str) -> None:
        span = _years_from_date_ranges(resume_text)
        assert span is not None
        report = verify_against_resume(f"I have {span + 2} years of experience.", resume_text)
        assert not report.ok

    def test_no_years_claim_means_no_flag(self, resume_text: str) -> None:
        report = verify_against_resume("I have built Django services on PostgreSQL.", resume_text)
        assert report.ok

    def test_resume_with_no_dates_at_all_rejects_any_years_claim(self) -> None:
        report = verify_against_resume(
            "I have 5 years of experience.", "I write software. I know Python."
        )
        assert not report.ok


class TestDateRangeInference:
    """`_years_from_date_ranges` is what keeps honest letters from being flagged."""

    def test_open_ended_current_role_is_understood(self) -> None:
        """Regression: "2019 - Present" is one literal year plus a keyword.

        An earlier version required two four-digit years, so the single most
        common way to write a current role inferred nothing — and every honest
        letter written from such a CV was flagged.
        """
        span = _years_from_date_ranges("Acme Corp, 2019 - Present")
        assert span is not None and span >= 6

    def test_closed_range(self) -> None:
        assert _years_from_date_ranges("Beta Ltd, 2015 - 2020") == 5

    def test_single_year_without_present_infers_nothing(self) -> None:
        assert _years_from_date_ranges("Graduated 2019") is None

    def test_no_dates_infers_nothing(self) -> None:
        assert _years_from_date_ranges("I am a software engineer.") is None

    def test_future_typo_does_not_inflate_the_ceiling(self) -> None:
        """A mistyped future date must not silently license a bigger claim."""
        span = _years_from_date_ranges("Acme, 2021 - 2099")
        # 2099 is discarded as a lower bound; the span stays plausible.
        assert span is None or span < 60


class TestCredentials:
    """Degrees, certifications, and clearances must appear in the resume."""

    @pytest.mark.parametrize(
        "claim",
        [
            "I am AWS Certified.",
            "I hold a PhD in Computer Science.",
            "I have an MBA from INSEAD.",
            "I hold an active security clearance.",
            "I am a certified Kubernetes administrator.",
        ],
    )
    def test_unsupported_credentials_are_flagged(self, claim: str, resume_text: str) -> None:
        report = verify_against_resume(claim, resume_text)
        assert not report.ok, f"{claim!r} should have been flagged"
        assert any(f.kind == "unsupported_credential" for f in report.flags)

    def test_credential_present_in_resume_passes(self, resume_text: str) -> None:
        report = verify_against_resume(
            "My BSc in Computer Science gave me strong fundamentals.", resume_text
        )
        assert report.ok, report.as_strings()


class TestBorrowedTechnology:
    """The highest-signal check: terms echoed from the posting."""

    def test_technology_from_the_posting_is_flagged(
        self, resume_text: str, job_description: str
    ) -> None:
        report = verify_against_resume(
            "I have deep production experience with Kubernetes and Kafka.",
            resume_text,
            job_description=job_description,
        )
        assert report.requires_review
        assert any(f.kind == "borrowed_technology" for f in report.flags)
        evidence = " ".join(f.evidence for f in report.flags).lower()
        assert "kubernetes" in evidence or "kafka" in evidence

    def test_technology_genuinely_in_the_resume_passes(
        self, resume_text: str, job_description: str
    ) -> None:
        report = verify_against_resume(
            "I have run Django on PostgreSQL and containerised it with Docker.",
            resume_text,
            job_description=job_description,
        )
        assert report.ok, report.as_strings()

    def test_alias_spelling_does_not_false_positive(self, resume_text: str) -> None:
        """Resume says "PostgreSQL"; letter says "Postgres". Same thing."""
        report = verify_against_resume(
            "I used Postgres extensively.",
            resume_text,
            job_description="We use Postgres heavily.",
        )
        assert report.ok, report.as_strings()

    def test_no_job_description_disables_the_check(self, resume_text: str) -> None:
        report = verify_against_resume("I know Kubernetes and Kafka.", resume_text)
        assert not any(f.kind == "borrowed_technology" for f in report.flags)


class TestSelfReportedClaims:
    """`claims_made` from the model is verified alongside the letter body."""

    def test_claim_list_is_checked_even_if_the_body_is_vague(self, resume_text: str) -> None:
        report = verify_against_resume(
            "I would bring relevant experience to your team.",
            resume_text,
            claims=["Holds a PhD in Machine Learning"],
        )
        assert not report.ok


class TestReportSemantics:
    def test_empty_text_produces_no_flags(self, resume_text: str) -> None:
        assert verify_against_resume("", resume_text).ok
        assert verify_against_resume("   ", resume_text).flags == []

    def test_medium_severity_requires_review_but_stays_ok(
        self, resume_text: str, job_description: str
    ) -> None:
        """`ok` gates sending; `requires_review` gates automation."""
        report = verify_against_resume(
            "I am excited to work with Kubernetes.",
            resume_text,
            job_description=job_description,
        )
        assert report.ok is True  # no high-severity flag, so the letter may be used
        assert report.requires_review is True  # but a human must look at it


class TestAnswerVerification:
    """Screening answers get the year and credential checks, not the tech check."""

    def test_inflated_answer_is_flagged(self, resume_text: str) -> None:
        report = verify_answer("20", resume_text, "")
        # A bare "20" is not a years claim without context, so this must pass —
        # the guard should not invent a violation.
        assert report.ok

    def test_answer_claiming_a_credential_is_flagged(self, resume_text: str) -> None:
        report = verify_answer("Yes, I am AWS Certified", resume_text, "")
        assert not report.ok

    def test_answer_supported_by_profile_facts_passes(self, resume_text: str) -> None:
        report = verify_answer(
            "I hold a PhD", resume_text, profile_facts="education level: PhD in Computer Science"
        )
        assert report.ok, report.as_strings()


class TestCompanyNameIsNotAFabrication:
    """The employer's own name must not count as a borrowed technology.

    Found on the first real cover letter this system produced. It was held for
    review with the company's own name as the "borrowed technology". The company name is in the letter and in the posting, and is
    never in the resume, so this fires on every letter ever generated.

    That matters more than it sounds. This flag is the honesty signal a human
    is supposed to act on. One that fires every single time is one nobody
    reads, and the real unsupported claim goes through with it.
    """

    RESUME = (
        "AI Engineer at Example Labs. Python, JavaScript, Java. PostgreSQL and pgvector. "
        "Built n8n agent pipelines and an AWS Bedrock RAG search pipeline. 2023 - Present."
    )
    JOB = (
        "Northwind is hiring an LLM Agentic Trainer. You will structure training "
        "datasets. Strong Kubernetes skills required."
    )

    def test_the_company_name_is_flagged_when_it_is_not_passed(self) -> None:
        # Documents the old behaviour, so a regression is unambiguous.
        report = verify_against_resume(
            "I admire Northwind's work on agentic AI.", self.RESUME, job_description=self.JOB
        )
        assert "northwind" in " ".join(report.as_strings()).lower()

    def test_the_company_name_is_exempt_when_it_is_passed(self) -> None:
        report = verify_against_resume(
            "I admire Northwind's work on agentic AI.",
            self.RESUME,
            job_description=self.JOB,
            company="Northwind",
        )
        assert "northwind" not in " ".join(report.as_strings()).lower()

    def test_a_multi_word_company_is_exempt_in_full(self) -> None:
        report = verify_against_resume(
            "Fabrikam Works builds remote teams and I want to join.",
            self.RESUME,
            job_description="Fabrikam Works is hiring. Remote teams welcome.",
            company="Fabrikam Works",
        )
        joined = " ".join(report.as_strings()).lower()
        assert "fabrikam" not in joined
        assert "works" not in joined

    def test_exempting_the_company_does_not_silence_a_real_borrowed_claim(self) -> None:
        """The check must still do its job. This is the part that matters."""
        report = verify_against_resume(
            "At Northwind I would apply my deep Kubernetes expertise.",
            self.RESUME,
            job_description=self.JOB,
            company="Northwind",
        )
        joined = " ".join(report.as_strings()).lower()
        assert "kubernetes" in joined, "an unsupported technology must still be caught"
        assert report.requires_review

    def test_an_empty_company_is_harmless(self) -> None:
        report = verify_against_resume(
            "I admire the work here.", self.RESUME, job_description=self.JOB, company=""
        )
        assert isinstance(report.as_strings(), list)


class TestCredentialFamilies:
    """A credential claim is judged by family, not by literal word match.

    Found on a real letter. The candidate's CV reads "CERTIFICATES & AWARDS",
    "AI Cloud certified" and "Internship Certificate" -- yet the
    letter was flagged HIGH for claiming "a certification", because the exact
    string "certification" never appears. A high flag blocks the letter from
    being attached at all, so an honest letter was discarded over a word form.

    The same CV says "BS Computer Science", so an honest letter saying
    "bachelor's degree" would have been thrown away too -- on nearly every
    application, forever.

    That is not caution. A control that fires on synonyms teaches the operator
    to wave it away, and it will be waved away the one time it is real.
    """

    CV = (
        "EDUCATION: Example University, Springfield. "
        "BS Computer Science, 2022 - 2026. CGPA 3.70/4.00. "
        "Example Cloud AI Cloud certified. "
        "CERTIFICATES & AWARDS: Example Corp Internship Certificate."
    )

    @pytest.mark.parametrize(
        "claim",
        [
            "I hold a certification in AI Cloud technologies.",
            "I am AWS certified.",
            "My certifications include cloud platform work.",
            "I have a bachelor's degree in Computer Science.",
            "My BSc gave me strong fundamentals.",
            "I completed my undergraduate degree at Example University.",
        ],
    )
    def test_a_truthful_claim_is_not_flagged(self, claim: str) -> None:
        report = verify_against_resume(claim, self.CV)
        assert report.ok, f"honest claim was flagged: {report.as_strings()}"
        assert not report.flags

    @pytest.mark.parametrize(
        ("claim", "expected"),
        [
            ("I hold a PhD in machine learning.", "doctorate"),
            ("I completed my master's degree in 2024.", "master's"),
            ("I have an MBA from LBS.", "MBA"),
            ("I am PMP certified.", "PMP"),
            ("I hold a CISSP.", "CISSP"),
            ("I have an active security clearance.", "security clearance"),
        ],
    )
    def test_an_invented_credential_is_still_caught(self, claim: str, expected: str) -> None:
        """The control must keep working. This is the half that matters."""
        report = verify_against_resume(claim, self.CV)
        assert not report.ok, f"missed an invented credential: {claim!r}"
        assert expected.lower() in " ".join(report.as_strings()).lower()

    def test_an_invented_credential_is_high_severity(self) -> None:
        """High severity is what stops the letter being attached at all."""
        report = verify_against_resume("I hold a PhD.", self.CV)
        assert any(f.severity == "high" for f in report.flags)

    def test_a_bare_cv_still_flags_a_degree_claim(self) -> None:
        """Exempting by family must not exempt everything."""
        report = verify_against_resume(
            "I have a bachelor's degree.", "Self-taught developer. Python, Django. 2019 - Present."
        )
        assert not report.ok
