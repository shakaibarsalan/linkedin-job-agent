"""
Tests for profile-based answer resolution and the question-answering chain.

The rule under test: an answer is only submitted when it traces to a fact the
user wrote down or a model answer that then passed verification. Everything
else must reach a human.
"""

from __future__ import annotations

import pathlib
import textwrap

import pytest

from app.services.ai.providers import RulesProvider
from app.services.profile import Profile, ProfileError
from app.services.question_answering import FormQuestion, QuestionAnswerer, summarise


@pytest.fixture
def profile_file(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "answers.yaml"
    path.write_text(
        textwrap.dedent(
            """
            facts:
              full_name: Jane Doe
              email: jane@example.com
              phone: "+44 7700 900000"
              work_authorized: true
              requires_sponsorship: false
              willing_to_relocate: false
              notice_period: "1 month"
              salary_expectation: "70000"
              education_level: "Bachelor's degree"
              city: London
              country: United Kingdom

            skill_years:
              python: 6
              django: 5
              postgresql: 6
              aws: 3
              kubernetes: 1.5

            exact_answers:
              "Do you have a valid driving licence?": "Yes"

            defaults:
              checkbox: "No"
            """
        ).strip(),
        encoding="utf-8",
    )
    return path


class TestProfileLoading:
    def test_loads_all_sections(self, profile_file: pathlib.Path) -> None:
        profile = Profile.load(profile_file)
        assert profile.data["full_name"] == "Jane Doe"
        assert profile.skill_years["python"] == 6.0
        assert len(profile.exact) == 1

    def test_missing_file_yields_an_empty_profile(self, tmp_path: pathlib.Path) -> None:
        """A missing profile must degrade, not crash — questions go to a human."""
        profile = Profile.load(tmp_path / "nope.yaml")
        assert profile.data == {}
        assert profile.lookup("Anything?") is None

    def test_malformed_yaml_raises(self, tmp_path: pathlib.Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("facts:\n  - [unclosed", encoding="utf-8")
        with pytest.raises(ProfileError):
            Profile.load(bad)

    def test_non_mapping_top_level_raises(self, tmp_path: pathlib.Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ProfileError):
            Profile.load(bad)


class TestProfileLookup:
    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("Are you legally authorized to work in the UK?", "Yes"),
            ("Will you now or in the future require sponsorship?", "No"),
            ("What is your notice period?", "1 month"),
            ("What are your salary expectations?", "70000"),
            ("Are you willing to relocate?", "No"),
            ("What is your email address?", "jane@example.com"),
            ("What is your phone number?", "+44 7700 900000"),
            ("What city do you live in?", "London"),
            ("What is your highest level of education?", "Bachelor's degree"),
        ],
    )
    def test_pattern_matches(
        self, profile_file: pathlib.Path, question: str, expected: str
    ) -> None:
        hit = Profile.load(profile_file).lookup(question)
        assert hit is not None, f"{question!r} did not match"
        assert hit.value == expected

    @pytest.mark.parametrize(
        ("question", "expected"),
        [
            ("How many years of experience do you have with Python?", "6"),
            ("Years of experience with Django", "5"),
            ("How many years of professional experience do you have using AWS?", "3"),
            ("How many years of hands-on experience with Kubernetes?", "1.5"),
        ],
    )
    def test_skill_years(self, profile_file: pathlib.Path, question: str, expected: str) -> None:
        hit = Profile.load(profile_file).lookup(question, question_type="numeric")
        assert hit is not None, f"{question!r} did not match"
        assert hit.value == expected

    def test_exact_answer_wins(self, profile_file: pathlib.Path) -> None:
        hit = Profile.load(profile_file).lookup("Do you have a valid driving licence?")
        assert hit is not None
        assert hit.matched_by == "exact"

    def test_unknown_question_returns_none(self, profile_file: pathlib.Path) -> None:
        assert Profile.load(profile_file).lookup("What is your favourite band?") is None

    def test_unconfigured_skill_returns_none(self, profile_file: pathlib.Path) -> None:
        """Never invent a years figure for a skill the user did not list."""
        hit = Profile.load(profile_file).lookup(
            "How many years of experience do you have with COBOL?"
        )
        assert hit is None

    def test_longest_skill_match_wins(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "p.yaml"
        path.write_text("skill_years:\n  aws: 3\n  aws lambda: 1\n", encoding="utf-8")
        hit = Profile.load(path).lookup("How many years of experience do you have with AWS Lambda?")
        assert hit is not None
        assert hit.value == "1"

    def test_non_numeric_answer_is_rejected_for_a_numeric_field(
        self, profile_file: pathlib.Path
    ) -> None:
        """Submitting "1 month" into a number field fails the form."""
        hit = Profile.load(profile_file).lookup(
            "What is your notice period?", question_type="numeric"
        )
        assert hit is None

    def test_select_answer_must_be_one_of_the_options(self, profile_file: pathlib.Path) -> None:
        profile = Profile.load(profile_file)

        good = profile.lookup(
            "Are you legally authorized to work?",
            question_type="select",
            options=["Yes", "No"],
        )
        assert good is not None and good.value == "Yes"

        bad = profile.lookup(
            "Are you legally authorized to work?",
            question_type="select",
            options=["Definitely", "Absolutely not"],
        )
        assert bad is None

    def test_boolean_facts_render_as_yes_no(self, profile_file: pathlib.Path) -> None:
        profile = Profile.load(profile_file)
        assert profile.lookup("Are you legally authorized to work?").value == "Yes"  # type: ignore[union-attr]

    def test_prompt_facts_render_without_secrets(self, profile_file: pathlib.Path) -> None:
        rendered = Profile.load(profile_file).as_prompt_facts()
        assert "full name: Jane Doe" in rendered
        assert "python: 6 years" in rendered


class TestQuestionAnswerer:
    """The full chain: profile → AI → human."""

    async def test_profile_hit_never_calls_the_model(
        self, profile_file: pathlib.Path, resume_text: str
    ) -> None:
        answerer = QuestionAnswerer(
            profile=Profile.load(profile_file),
            provider=RulesProvider(),  # would raise if asked to generate
            resume_text=resume_text,
        )
        answered = await answerer.answer(FormQuestion(question="What is your notice period?"))
        assert answered.resolved
        assert answered.source == "profile"
        assert answered.confidence == 1.0

    async def test_unknown_question_reaches_a_human(
        self, profile_file: pathlib.Path, resume_text: str
    ) -> None:
        """RulesProvider refuses to generate, so this must land on a human."""
        answerer = QuestionAnswerer(
            profile=Profile.load(profile_file),
            provider=RulesProvider(),
            resume_text=resume_text,
        )
        answered = await answerer.answer(
            FormQuestion(question="Describe a time you resolved a team conflict.")
        )
        assert not answered.resolved
        assert answered.answer is None
        assert answered.requires_review

    async def test_summarise_counts_unresolved(
        self, profile_file: pathlib.Path, resume_text: str
    ) -> None:
        answerer = QuestionAnswerer(
            profile=Profile.load(profile_file),
            provider=RulesProvider(),
            resume_text=resume_text,
        )
        answers = await answerer.answer_all(
            [
                FormQuestion(question="What is your email address?"),
                FormQuestion(question="What is your notice period?"),
                FormQuestion(question="Why do you want to work here?"),
            ]
        )
        resolved, unresolved, texts = summarise(answers)
        assert resolved == 2
        assert unresolved == 1
        assert "Why do you want to work here?" in texts[0]


class TestSkillMatchingIsExact:
    """A short skill name must not answer for every question that contains it.

    Found while building a real profile. With ``r: 1`` and ``go: 1`` present --
    both taken straight from the candidate's own CV -- the substring test
    ``skill in phrase`` produced:

        Kubernetes -> 1 year     (the "r" inside Kubernetes)
        MongoDB    -> 1 year     (the "go" inside MongoDB)
        Rust, Salesforce, Angular -> 1 year each

    None of those were in the profile. That is an invented experience claim
    typed into a numeric field on a real job application, under the
    candidate's name.

    It is also invisible to the fabrication guard, which verifies generated
    prose against the resume. ``skill_years`` is treated as fact the candidate
    supplied, so nothing downstream second-guesses it. Exactness here is the
    only control there is.
    """

    @pytest.fixture
    def profile(self, tmp_path: pathlib.Path) -> Profile:
        path = tmp_path / "answers.yaml"
        path.write_text(
            "skill_years:\n"
            "  r: 1\n"
            "  go: 1\n"
            "  python: 2\n"
            "  aws: 1.5\n"
            "  aws lambda: 1\n"
            "  c++: 3\n"
            "  c#: 2\n"
            "  node.js: 2\n"
            "  ci/cd: 1\n"
            "  github actions: 1\n",
            encoding="utf-8",
        )
        return Profile.load(path)

    @pytest.mark.parametrize(
        "skill",
        ["Kubernetes", "MongoDB", "Rust", "Salesforce", "Angular", "Terraform", "Ruby", "Groovy"],
    )
    def test_a_skill_not_in_the_profile_is_never_answered(
        self, profile: Profile, skill: str
    ) -> None:
        answer = profile.lookup(f"How many years of experience do you have with {skill}?")
        assert answer is None, (
            f"claimed {answer.value if answer else ''} years of {skill}, "
            "which the candidate never listed"
        )

    @pytest.mark.parametrize(
        ("skill", "expected"),
        [
            ("Python", "2"),
            ("AWS", "1.5"),
            ("R", "1"),
            ("Go", "1"),
        ],
    )
    def test_a_configured_skill_still_answers(
        self, profile: Profile, skill: str, expected: str
    ) -> None:
        answer = profile.lookup(f"How many years of experience do you have with {skill}?")
        assert answer is not None, f"{skill} is configured and must still answer"
        assert answer.value == expected

    def test_punctuated_skill_names_survive_boundary_matching(self, profile: Profile) -> None:
        """'+', '#', '.' and '/' must stay part of the term, not act as boundaries."""
        for skill, expected in (("C++", "3"), ("C#", "2"), ("Node.js", "2"), ("CI/CD", "1")):
            answer = profile.lookup(f"How many years of experience do you have with {skill}?")
            assert answer is not None, f"{skill} should match"
            assert answer.value == expected

    def test_multi_word_skills_still_match(self, profile: Profile) -> None:
        answer = profile.lookup("How many years of experience do you have with GitHub Actions?")
        assert answer is not None
        assert answer.value == "1"

    def test_the_longest_match_still_wins(self, profile: Profile) -> None:
        """'aws lambda' must beat 'aws' — boundary matching must not break this."""
        answer = profile.lookup("How many years of experience do you have with AWS Lambda?")
        assert answer is not None
        assert answer.value == "1", "should have matched 'aws lambda', not 'aws'"

    def test_go_does_not_leak_into_django(self, profile: Profile) -> None:
        """Django contains 'go'. It is a different technology entirely."""
        assert profile.lookup("How many years of experience do you have with Django?") is None
