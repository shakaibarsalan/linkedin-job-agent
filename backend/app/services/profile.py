"""
Candidate profile — the authoritative answer source.

Purpose
-------
Holds the facts a screening form asks for that a resume does not state: work
authorisation, notice period, salary expectation, willingness to relocate,
per-skill years of experience. These are answered from *your written facts*,
never inferred by a model, because getting them wrong has legal and financial
consequences.

How it works
------------
A YAML file at ``PROFILE_PATH`` (default ``/data/profile/answers.yaml``).
:meth:`Profile.lookup` resolves a question in three stages, most trustworthy
first:

1. **Exact match** on a normalised question string.
2. **Pattern match** — curated regexes for the ~30 questions that make up the
   long tail of job application forms.
3. **Skill-years lookup** — "How many years of experience do you have with X?"
   resolved against the ``skill_years`` map.

Only if all three miss does the question go to the AI layer, and even then the
model is instructed to return ``needs_human`` rather than guess.

Configuration
-------------
``PROFILE_PATH`` — YAML file. See ``config/answers.example.yaml``.

Failure scenarios
-----------------
* File missing → an empty profile; every question falls through to AI or human.
  Logged at warning level on startup, not fatal.
* Malformed YAML → :class:`ProfileError` at load; nothing is answered from a
  half-parsed file.
"""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass, field
from typing import Any

import yaml

from app.logging_config import get_logger

log = get_logger(__name__)


class ProfileError(RuntimeError):
    """Raised when the profile file exists but cannot be used."""


@dataclass(slots=True)
class ProfileAnswer:
    """An answer resolved from the profile, with its provenance."""

    value: str
    source: str = "profile"
    confidence: float = 1.0
    matched_by: str = ""


def _normalise_question(question: str) -> str:
    """Lowercase, strip punctuation and whitespace runs, for stable matching."""
    text = question.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Curated patterns for the questions application forms actually ask.
# Each maps a regex to a profile key. Order matters: the first match wins, so
# more specific patterns come first.
_QUESTION_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Work authorisation / visa
    (
        re.compile(r"legally authoriz|legally authoris|authorized to work|authorised to work"),
        "work_authorized",
    ),
    (
        re.compile(
            r"require (visa )?sponsor|need sponsor|sponsorship (now|in the future|required)"
        ),
        "requires_sponsorship",
    ),
    (re.compile(r"\bvisa status\b|work permit|right to work"), "work_authorization_status"),
    (re.compile(r"\bcitizen\b|citizenship"), "citizenship"),
    (re.compile(r"security clearance"), "security_clearance"),
    # Location & logistics
    (re.compile(r"willing to relocat|open to relocat|able to relocat"), "willing_to_relocate"),
    (
        re.compile(r"willing to (travel|commute)|able to commute|comfortable commuting"),
        "willing_to_commute",
    ),
    (
        re.compile(
            r"currently (live|located|reside)|current location|where are you (based|located)"
        ),
        "location",
    ),
    (re.compile(r"\bremote\b.*\b(work|comfortable|open)\b|work remotely"), "open_to_remote"),
    (re.compile(r"\bhybrid\b|\bon.?site\b.*\b(work|comfortable|open)\b"), "open_to_onsite"),
    # Compensation & availability
    (
        re.compile(
            r"salary expectation|expected (salary|compensation)|desired salary|compensation expectation"
        ),
        "salary_expectation",
    ),
    (re.compile(r"current salary|present salary"), "current_salary"),
    (
        re.compile(
            r"notice period|when can you (start|join)|earliest start|availability to start|start date"
        ),
        "notice_period",
    ),
    (re.compile(r"hourly rate|day rate|contract rate"), "hourly_rate"),
    # Identity & contact
    (re.compile(r"\b(phone|mobile|contact) number\b"), "phone"),
    (re.compile(r"\bemail address\b|\byour email\b"), "email"),
    (re.compile(r"\blinkedin (profile|url)\b"), "linkedin_url"),
    (re.compile(r"\b(github|portfolio|website|personal site)\b"), "portfolio_url"),
    (re.compile(r"\bfirst name\b"), "first_name"),
    (re.compile(r"\blast name\b|\bsurname\b"), "last_name"),
    (re.compile(r"\bfull name\b|\byour name\b"), "full_name"),
    (re.compile(r"\b(city|town)\b"), "city"),
    (re.compile(r"\bcountry\b"), "country"),
    (re.compile(r"\bpostal code\b|\bzip code\b|\bpostcode\b"), "postal_code"),
    # Education & background
    (
        re.compile(
            r"highest (level of )?education|education level|degree.*(completed|obtained|hold)"
        ),
        "education_level",
    ),
    (re.compile(r"\buniversity\b|\bcollege\b|\bschool\b.*attend"), "university"),
    (re.compile(r"\bgpa\b|grade point"), "gpa"),
    # Demographics — usually voluntary; answered only if explicitly configured.
    (re.compile(r"\bgender\b"), "gender"),
    (re.compile(r"\bethnicit|\brace\b"), "ethnicity"),
    (re.compile(r"\bveteran\b|military service"), "veteran_status"),
    (re.compile(r"\bdisabilit"), "disability_status"),
    # Consent / acknowledgement
    # Widened after a real form went unanswered. LinkedIn's consent step reads
    # "By clicking Yes you agree to our Privacy Policy and data processing",
    # and "our" is not "the", so the old pattern missed it and the required
    # radio group was left blank -- stopping the application at page 3 of 4.
    (
        re.compile(
            r"agree to.{0,20}?(terms|privacy|policy|processing)"
            r"|consent to|acknowledge|privacy policy"
        ),
        "consent_acknowledge",
    ),
    (re.compile(r"background check|drug (test|screen)"), "consent_background_check"),
]

# "How many years ... with|of|in X" — captures the skill so it can be looked up
# in `skill_years`.
_SKILL_YEARS_RE = re.compile(
    r"(?:how many |number of )?years?(?: of)?(?: professional| commercial| work(?:ing)?| hands.on)?"
    r"(?: experience)?(?: do you have)?(?: with| in| using| of)?\s+(?P<skill>[a-z0-9+#.\s/&-]{2,60})",
    re.IGNORECASE,
)


def _skill_matches(skill: str, phrase: str) -> bool:
    """True if ``skill`` occurs in ``phrase`` as a whole term.

    Word-boundary matching, not substring. ``"r" in "kubernetes"`` is True and
    ``"go" in "mongodb"`` is True, which is how a profile listing R and Go came
    to claim a year of Kubernetes, MongoDB, Rust, Salesforce and Angular.

    Boundaries are non-alphanumeric, so '+', '#' and '.' stay part of the term
    and "c++", "c#", "node.js" and "ci/cd" still match. Multi-word skills such
    as "aws lambda" work unchanged: the boundary sits at each end of the whole
    phrase, not between its words.
    """
    if not skill:
        return False
    return (
        re.search(
            rf"(?<![a-z0-9]){re.escape(skill)}(?![a-z0-9])",
            phrase,
            re.IGNORECASE,
        )
        is not None
    )


@dataclass(slots=True)
class Profile:
    """Loaded candidate facts."""

    data: dict[str, Any] = field(default_factory=dict)
    exact: dict[str, str] = field(default_factory=dict)
    skill_years: dict[str, float] = field(default_factory=dict)
    defaults: dict[str, str] = field(default_factory=dict)
    source_path: str = ""

    # ── Loading ───────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: str | pathlib.Path) -> Profile:
        """Load a profile from YAML. Returns an empty profile if absent."""
        p = pathlib.Path(path)
        if not p.exists():
            log.warning(
                "profile.not_found",
                path=str(p),
                impact="Screening questions will fall through to AI or human review",
            )
            return cls(source_path=str(p))

        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ProfileError(f"Profile file {p} is not valid YAML: {exc}") from exc

        if not isinstance(raw, dict):
            raise ProfileError(f"Profile file {p} must contain a YAML mapping at the top level")

        facts = raw.get("facts") or {}
        if not isinstance(facts, dict):
            raise ProfileError("'facts' must be a mapping")

        exact_raw = raw.get("exact_answers") or {}
        exact = {_normalise_question(str(k)): str(v) for k, v in exact_raw.items()}

        skills_raw = raw.get("skill_years") or {}
        skill_years: dict[str, float] = {}
        for skill, years in skills_raw.items():
            try:
                skill_years[str(skill).lower().strip()] = float(years)
            except (TypeError, ValueError):
                log.warning("profile.bad_skill_years", skill=skill, value=years)

        defaults = {str(k): str(v) for k, v in (raw.get("defaults") or {}).items()}

        profile = cls(
            data={str(k): v for k, v in facts.items()},
            exact=exact,
            skill_years=skill_years,
            defaults=defaults,
            source_path=str(p),
        )
        log.info(
            "profile.loaded",
            path=str(p),
            facts=len(profile.data),
            exact_answers=len(profile.exact),
            skills=len(profile.skill_years),
        )
        return profile

    # ── Lookup ────────────────────────────────────────────────────────

    def lookup(
        self, question: str, *, question_type: str = "text", options: list[str] | None = None
    ) -> ProfileAnswer | None:
        """Resolve ``question`` from configured facts, or return ``None``.

        Args:
            question: The raw question text from the form.
            question_type: ``text`` | ``numeric`` | ``boolean`` | ``select`` | ...
            options: Permitted values for select/radio inputs. A resolved answer
                that is not among them is rejected rather than submitted.

        Returns:
            A :class:`ProfileAnswer`, or ``None`` if nothing matched.
        """
        normalised = _normalise_question(question)
        if not normalised:
            return None

        # 1. Exact match — highest confidence.
        if normalised in self.exact:
            return self._validated(
                ProfileAnswer(self.exact[normalised], matched_by="exact"), options, question_type
            )

        # 2. Skill-years, before generic patterns: "years of experience with
        #    Python" would otherwise be swallowed by a broader pattern.
        skill_answer = self._lookup_skill_years(question)
        if skill_answer is not None:
            return self._validated(skill_answer, options, question_type)

        # 3. Curated patterns.
        for pattern, key in _QUESTION_PATTERNS:
            if pattern.search(normalised):
                value = self.data.get(key)
                if value is None:
                    continue
                return self._validated(
                    ProfileAnswer(_stringify(value), matched_by=f"pattern:{key}"),
                    options,
                    question_type,
                )

        return None

    def _lookup_skill_years(self, question: str) -> ProfileAnswer | None:
        """Resolve "how many years with X" against ``skill_years``."""
        if "year" not in question.lower():
            return None

        match = _SKILL_YEARS_RE.search(question)
        if not match:
            return None

        phrase = match.group("skill").lower().strip(" ?.:;,")
        if not phrase:
            return None

        # Longest configured skill matching the phrase wins, so "aws lambda"
        # beats "aws" when both are configured.
        #
        # Matching is on WORD BOUNDARIES, not raw substrings. A plain
        # ``skill in phrase`` test looks harmless until you configure a short
        # skill name: with "r: 1" and "go: 1" in the profile, this claimed
        #
        #     Kubernetes -> 1 year   (matched the "r" in Kubernetes)
        #     MongoDB    -> 1 year   (matched the "go" in MongoDB)
        #     Salesforce -> 1 year   Angular -> 1 year   Rust -> 1 year
        #
        # none of which were in the profile at all. That is a fabricated
        # experience claim typed into a numeric field on a real application,
        # and the fabrication guard cannot catch it: skill_years is treated as
        # fact the candidate supplied, which is exactly why it must be exact.
        #
        # An unmatched skill returns None and the operator is asked. Being
        # asked is always better than being wrong here.
        best_key: str | None = None
        for skill in self.skill_years:
            if not _skill_matches(skill, phrase):
                continue
            if best_key is None or len(skill) > len(best_key):
                best_key = skill

        if best_key is None:
            return None

        years = self.skill_years[best_key]
        rendered = str(int(years)) if years == int(years) else f"{years:g}"
        return ProfileAnswer(rendered, matched_by=f"skill_years:{best_key}")

    def _validated(
        self, answer: ProfileAnswer, options: list[str] | None, question_type: str
    ) -> ProfileAnswer | None:
        """Reject an answer that cannot legally be submitted for this input."""
        if question_type == "numeric":
            # A non-numeric value in a numeric field fails the form, so treat it
            # as unresolved rather than submitting garbage.
            if not re.fullmatch(r"\d+(\.\d+)?", answer.value.strip()):
                log.warning(
                    "profile.answer_not_numeric",
                    value=answer.value[:50],
                    matched_by=answer.matched_by,
                )
                return None
            return answer

        if options:
            # Exact match first, then case-insensitive, then substring.
            for option in options:
                if option == answer.value:
                    return answer
            lowered = answer.value.lower().strip()
            for option in options:
                if option.lower().strip() == lowered:
                    return ProfileAnswer(
                        option, answer.source, answer.confidence, answer.matched_by
                    )
            for option in options:
                if lowered and lowered in option.lower():
                    return ProfileAnswer(
                        option, answer.source, 0.8, f"{answer.matched_by}+fuzzy_option"
                    )
            log.info(
                "profile.answer_not_in_options",
                value=answer.value[:50],
                options=[o[:40] for o in options[:8]],
            )
            return None

        return answer

    # ── Rendering for prompts ─────────────────────────────────────────

    def as_prompt_facts(self) -> str:
        """Render the profile as authoritative facts for an AI prompt."""
        if not self.data and not self.skill_years:
            return ""

        lines = [
            f"{key.replace('_', ' ')}: {_stringify(value)}"
            for key, value in sorted(self.data.items())
        ]
        if self.skill_years:
            lines.append("")
            lines.append("Years of experience by skill:")
            lines += [
                f"  {skill}: {years:g} years" for skill, years in sorted(self.skill_years.items())
            ]
        return "\n".join(lines)

    def default_for(self, question_type: str) -> str | None:
        """A configured default for an input type, if the user set one."""
        return self.defaults.get(question_type)


def _stringify(value: Any) -> str:
    """Render a YAML scalar the way a form expects it."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float) and value == int(value):
        return str(int(value))
    return str(value)
