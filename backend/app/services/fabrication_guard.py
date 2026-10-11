"""
Mechanical anti-fabrication verification.

Purpose
-------
"Never fabricate experience" is a hard requirement. A prompt instruction is a
mitigation, not a control — models drift, get jailbroken by odd job
descriptions, and quietly borrow requirements from the posting to sound like a
match. This module independently verifies generated text against the resume,
with no model in the loop.

How it works
------------
Three checks, in descending order of confidence:

1. **Year claims** — every ``N years`` assertion in the text is compared with
   the largest duration the resume supports. Claiming more is a hard failure.
2. **Credentials** — degree, certification, and clearance keywords present in
   the text but absent from the resume are a hard failure.
3. **Borrowed technologies** — the highest-signal check. A term that appears in
   *both the letter and the job description* but *not in the resume* is almost
   always the model echoing a requirement back as if the candidate had it.

Anything flagged at ``high`` severity forces the application into human review
regardless of ``REVIEW_MODE``. This is deliberately un-bypassable by config.

Failure scenarios
-----------------
* **False positive** (e.g. the resume says "Postgres", the letter says
  "PostgreSQL") → an alias table covers the common cases; anything else costs
  one human review. That is the correct direction to be wrong in.
* **False negative** — possible; this is a safety net, not a proof. Human
  review remains the primary control, which is why ``REVIEW_MODE=always`` is
  the default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from app.logging_config import get_logger

log = get_logger(__name__)

Severity = Literal["high", "medium", "low"]


@dataclass(slots=True)
class FabricationFlag:
    """One suspected unsupported claim."""

    kind: str
    severity: Severity
    detail: str
    evidence: str = ""

    def __str__(self) -> str:
        return f"[{self.severity}] {self.kind}: {self.detail}"


@dataclass(slots=True)
class FabricationReport:
    """Outcome of verifying generated text against a resume."""

    flags: list[FabricationFlag] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when nothing at ``high`` severity was found."""
        return not any(f.severity == "high" for f in self.flags)

    @property
    def requires_review(self) -> bool:
        """True when a human must look at this before it is sent."""
        return bool(self.flags)

    def as_strings(self) -> list[str]:
        return [str(f) for f in self.flags]


# ── Normalisation ─────────────────────────────────────────────────────

# Terms that mean the same thing but are spelled differently on CVs vs postings.
# Keeps the borrowed-technology check from firing on cosmetic mismatches.
_ALIASES: dict[str, str] = {
    "postgresql": "postgres",
    "postgre": "postgres",
    "psql": "postgres",
    "js": "javascript",
    "ts": "typescript",
    "node": "nodejs",
    "node.js": "nodejs",
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "nextjs": "next",
    "next.js": "next",
    "k8s": "kubernetes",
    "gcp": "googlecloud",
    "google cloud": "googlecloud",
    "amazon web services": "aws",
    "ms sql": "mssql",
    "sql server": "mssql",
    "c sharp": "csharp",
    "c#": "csharp",
    "c++": "cplusplus",
    "cpp": "cplusplus",
    "golang": "go",
    "ci/cd": "cicd",
    "rest api": "rest",
    "restful": "rest",
    "ml": "machinelearning",
    "machine learning": "machinelearning",
    "ai": "artificialintelligence",
    "nlp": "naturallanguageprocessing",
    "tf": "tensorflow",
    "sklearn": "scikitlearn",
    "scikit-learn": "scikitlearn",
    "github actions": "githubactions",
    "gitlab ci": "gitlabci",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9+#._-]{1,30}")


def _normalise(token: str) -> str:
    """Fold a token to its canonical comparison form."""
    lowered = token.lower().strip(".,;:!?()[]{}\"'")
    if lowered in _ALIASES:
        return _ALIASES[lowered]
    # Strip trailing punctuation-ish characters that survive tokenisation, but
    # keep '+' and '#' since they are load-bearing in C++ and C#.
    stripped = lowered.rstrip("._-")
    return _ALIASES.get(stripped, stripped)


def _token_set(text: str) -> set[str]:
    """Normalised token set, including multi-word aliases."""
    lowered = (text or "").lower()
    tokens = {_normalise(t) for t in _TOKEN_RE.findall(lowered)}
    # Multi-word aliases ("machine learning") are not single tokens, so scan
    # for them explicitly.
    for phrase, canonical in _ALIASES.items():
        if " " in phrase and phrase in lowered:
            tokens.add(canonical)
    return {t for t in tokens if t}


# ── Check 1: year claims ──────────────────────────────────────────────

_YEARS_RE = re.compile(
    r"(\d{1,2})\s*(?:\+|plus)?\s*(?:or more\s*)?year(?:s)?(?:\s+of)?",
    re.IGNORECASE,
)


def _max_years_in(text: str) -> int | None:
    """Largest ``N years`` figure stated in ``text``."""
    values = [int(m.group(1)) for m in _YEARS_RE.finditer(text or "")]
    return max(values) if values else None


def _years_from_date_ranges(text: str) -> int | None:
    """Infer total career span from ``2019 - 2024`` style date ranges.

    Many CVs never write "N years" — they list employment date ranges. Without
    this, every honest letter that totals up experience would be flagged.
    Uses earliest-to-latest span, which is generous by design: over-crediting
    the resume costs a false negative, under-crediting costs a false positive
    on an honest letter, and the former is the safer error here.

    A single year plus "Present" is sufficient — "2019 - Present" is an
    extremely common way to write a current role, and requiring two literal
    years would flag every honest letter written from such a CV.
    """
    years = [int(y) for y in re.findall(r"\b(19[7-9]\d|20\d{2})\b", text or "")]
    if not years:
        return None

    ongoing = bool(
        re.search(r"\b(present|current|now|ongoing|to date)\b", text or "", re.IGNORECASE)
    )
    this_year = datetime.now(UTC).year

    if ongoing:
        latest = this_year
    elif len(years) >= 2:
        latest = max(years)
    else:
        # One year, no "Present": not enough to establish a range.
        return None

    # Ignore future-dated typos, which would otherwise inflate the ceiling.
    span = latest - min(y for y in years if y <= this_year)
    return span if 0 < span < 60 else None


def _check_years(text: str, resume: str) -> list[FabricationFlag]:
    claimed = _max_years_in(text)
    if claimed is None:
        return []

    supported = _max_years_in(resume)
    inferred = _years_from_date_ranges(resume)
    ceiling = max(v for v in (supported, inferred, 0) if v is not None)

    if ceiling == 0:
        return [
            FabricationFlag(
                kind="unsupported_years",
                severity="high",
                detail=(
                    f"Text claims {claimed} years of experience, but the resume states "
                    f"no duration and no date ranges to infer one from."
                ),
                evidence=f"{claimed} years",
            )
        ]

    # One year of slack absorbs honest rounding ("nearly 5" → "5").
    if claimed > ceiling + 1:
        return [
            FabricationFlag(
                kind="unsupported_years",
                severity="high",
                detail=(
                    f"Text claims {claimed} years of experience; the resume supports "
                    f"at most {ceiling}."
                ),
                evidence=f"{claimed} years",
            )
        ]
    return []


# ── Check 2: credentials ──────────────────────────────────────────────

# Credentials, grouped into FAMILIES.
#
# Each entry is (label, what-counts-as-claiming-it, what-counts-as-having-it).
# A claim is only a fabrication when the resume shows no evidence of the whole
# family — not merely when it uses a different word for the same fact.
#
# Matching each keyword against itself, which is what this did before, is too
# literal to be safe in either direction. Measured on a real letter: the
# candidate's CV reads "CERTIFICATES & AWARDS", "AI Cloud certified",
# "Internship Certificate", and the letter was still flagged HIGH for
# claiming "a certification" — so the letter was discarded outright. The same
# CV says "BS Computer Science", so an honest letter saying "bachelor's
# degree" would have been thrown away too.
#
# A high-severity flag blocks the letter from ever being attached. Spending
# that on a synonym mismatch is not caution, it is noise that trains the
# operator to wave the flag away — including the time it is real.
_CREDENTIAL_FAMILIES: list[tuple[str, str, str]] = [
    (
        "a doctorate",
        r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b",
        r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b",
    ),
    (
        "a master's degree",
        r"\bmaster'?s\b|\bm\.?\s?sc\b|\bm\.?\s?s\.?\b(?=\s|$)|\bmasters\b",
        r"\bmaster'?s\b|\bm\.?\s?sc\b|\bm\.?\s?s\.?\b(?=\s|$)|\bmasters\b|\bmphil\b",
    ),
    (
        "an MBA",
        r"\bm\.?b\.?a\b",
        r"\bm\.?b\.?a\b",
    ),
    (
        "a bachelor's degree",
        r"\bbachelor'?s?\b|\bb\.?\s?sc\b|\bb\.?\s?eng\b|\bundergraduate degree\b",
        # "BS Computer Science" and "BE" are how most CVs actually write it.
        r"\bbachelor'?s?\b|\bb\.?\s?sc\b|\bb\.?\s?eng\b|\bb\.?\s?s\.?\b(?=\s|$)"
        r"|\bb\.?\s?e\.?\b(?=\s|$)|\bundergraduate\b|\bdegree\b",
    ),
    (
        "a certification",
        r"\bcertified\b|\bcertification\b|\bcertifications\b",
        # Any word from the same family is evidence: a CV listing
        # "CERTIFICATES & AWARDS" plainly documents certifications.
        r"\bcertified\b|\bcertificat",
    ),
    ("a PMP certification", r"\bpmp\b", r"\bpmp\b"),
    ("a CISSP certification", r"\bcissp\b", r"\bcissp\b"),
    ("a CCNA certification", r"\bccna\b", r"\bccna\b"),
    ("a security clearance", r"\bsecurity clearance\b", r"\bsecurity clearance\b|\bcleared\b"),
    (
        "a professional licence",
        r"\blicensed\b|\blicence\b|\blicense\b",
        r"\blicens|\blicenc",
    ),
]


def _check_credentials(text: str, resume: str) -> list[FabricationFlag]:
    flags: list[FabricationFlag] = []
    lowered_text = (text or "").lower()
    lowered_resume = (resume or "").lower()

    for label, claim_pattern, evidence_pattern in _CREDENTIAL_FAMILIES:
        if not re.search(claim_pattern, lowered_text):
            continue
        if re.search(evidence_pattern, lowered_resume):
            continue
        flags.append(
            FabricationFlag(
                kind="unsupported_credential",
                severity="high",
                detail=f"Text claims {label}, which does not appear in the resume.",
                evidence=label,
            )
        )
    return flags


# ── Check 3: borrowed technologies ────────────────────────────────────

# Words that legitimately appear in a letter about a job without being a claim
# about the candidate. Suppresses the bulk of false positives.
_LETTER_NOISE = frozenset(
    """
    i my me we our you your the a an and or but if then so as at by for from in into of on
    to with without within is are was were be been being do does did doing have has had
    will would can could may might should must not no this that these those there here
    role position job opportunity team company organisation organization work working
    experience experienced background skills skill years year time strong excellent
    interested interest interesting excited exciting apply application applying candidate
    hiring manager sincerely regards thank thanks please contact discuss conversation
    build building built develop developing developed design designing designed deliver
    delivering delivered improve improving improved lead leading led manage managing
    managed support supporting supported create creating created help helping helped
    such very much more most well good great best able ability across during while where
    which what when who whom whose how why also than about over under between through
    currently recently particularly especially directly closely
    """.split()
)


def _check_borrowed_technologies(
    text: str,
    resume: str,
    job_description: str,
    *,
    company: str = "",
    limit: int = 8,
) -> list[FabricationFlag]:
    """Flag terms taken from the job description that the resume does not support."""
    if not job_description:
        return []

    letter_tokens = _token_set(text)
    resume_tokens = _token_set(resume)
    job_tokens = _token_set(job_description)

    # The employer's own name is in the letter and in the posting, and is never
    # in the resume -- so without this it is flagged as a borrowed technology on
    # every single letter. The first real letter this system produced was held
    # for review over the name of the company it was written to.
    #
    # That is not a small annoyance. This flag is the honesty signal a human is
    # meant to act on, and a signal that fires every time is one nobody reads.
    # Addressing a company by name is the opposite of a fabricated claim.
    company_tokens = _token_set(company) if company else set()

    suspicious = (letter_tokens & job_tokens) - resume_tokens - _LETTER_NOISE - company_tokens

    # Keep only tokens that look like a technology or proper noun rather than
    # ordinary English: appears capitalised mid-sentence, or contains a digit,
    # '+', or '#'.
    original_words = set(_TOKEN_RE.findall(text or ""))
    capitalised = {_normalise(w) for w in original_words if w[:1].isupper()}

    def looks_technical(token: str) -> bool:
        return (
            token in capitalised
            or any(ch.isdigit() for ch in token)
            or "+" in token
            or "#" in token
        )

    candidates = sorted(t for t in suspicious if looks_technical(t) and len(t) > 2)

    if not candidates:
        return []

    shown = candidates[:limit]
    return [
        FabricationFlag(
            kind="borrowed_technology",
            severity="medium",
            detail=(
                "These terms appear in the letter and in the job description but not in "
                f"the resume, which suggests they were echoed from the posting: "
                f"{', '.join(shown)}"
            ),
            evidence=", ".join(shown),
        )
    ]


# ── Public API ────────────────────────────────────────────────────────


def verify_against_resume(
    text: str,
    resume_text: str,
    *,
    job_description: str = "",
    company: str = "",
    claims: list[str] | None = None,
) -> FabricationReport:
    """Verify generated ``text`` against ``resume_text``.

    Args:
        text: The generated content (typically a cover letter).
        resume_text: Ground truth about the candidate.
        job_description: Enables the borrowed-technology check. Optional but
            strongly recommended — it is the highest-signal check available.
        company: The employer's name, exempted from the borrowed-technology
            check. Pass it whenever it is known: without it the company's own
            name is flagged on every letter, and a flag that always fires stops
            being read.
        claims: The model's self-reported claims, checked in addition to the
            letter body. A model that lists a claim it cannot support is caught
            even if the letter phrases it obliquely.

    Returns:
        A :class:`FabricationReport`. ``ok`` is False if anything is
        ``high`` severity.
    """
    if not text.strip():
        return FabricationReport()

    # Self-reported claims are appended so they go through the same checks.
    corpus = text if not claims else text + "\n" + "\n".join(claims)

    flags: list[FabricationFlag] = []
    flags += _check_years(corpus, resume_text)
    flags += _check_credentials(corpus, resume_text)
    flags += _check_borrowed_technologies(text, resume_text, job_description, company=company)

    report = FabricationReport(flags=flags)

    if flags:
        log.warning(
            "fabrication.flags_raised",
            count=len(flags),
            high_severity=sum(1 for f in flags if f.severity == "high"),
            kinds=sorted({f.kind for f in flags}),
        )
    return report


def verify_answer(answer: str, resume_text: str, profile_facts: str) -> FabricationReport:
    """Verify a screening-question answer against the resume and profile.

    Narrower than :func:`verify_against_resume`: a short factual answer only
    needs the year and credential checks. The borrowed-technology check does
    not apply because the answer will naturally echo the question's wording.
    """
    if not answer.strip():
        return FabricationReport()

    ground_truth = f"{resume_text}\n{profile_facts}"
    flags = _check_years(answer, ground_truth) + _check_credentials(answer, ground_truth)
    return FabricationReport(flags=flags)
