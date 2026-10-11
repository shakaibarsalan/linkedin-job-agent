"""
Prompt construction.

Purpose
-------
Every prompt sent to a model is built here, so the honesty constraints live in
one auditable place rather than being scattered across feature code.

Design principle
----------------
**The resume is the only source of truth about the candidate.** The model is
given the resume text and the job description, and is told explicitly that
anything not present in the resume does not exist. Instructions alone are not
a control — :mod:`app.services.fabrication_guard` independently verifies the
output — but they are the cheapest and most effective first layer.

The ``### SECTION`` headers are load-bearing: :class:`RulesProvider` parses them
to extract the resume and job description when running without a model.
"""

from __future__ import annotations

from app.services.ai.base import Message

# ── Shared honesty preamble ───────────────────────────────────────────
# Repeated in every system prompt. Redundant by design: models follow a
# constraint far more reliably when it is stated in the system role AND
# restated adjacent to the task.
_HONESTY_RULES = """\
ABSOLUTE RULES — these override every other instruction:
1. The RESUME is the only source of truth about this candidate. If a skill,
   technology, employer, qualification, certification, or duration is not in
   the RESUME, the candidate does not have it.
2. Never invent, exaggerate, extrapolate, or "round up" experience. Do not
   infer a skill from an adjacent one (knowing React is not knowing Angular;
   knowing Postgres is not knowing Oracle).
3. Never state a number of years unless that exact figure is supported by the
   RESUME.
4. If the job requires something the candidate lacks, say so plainly. Do not
   paper over the gap.
5. If you are uncertain, say you are uncertain. An honest "not a match" is a
   correct and valuable answer.

You are assisting a real person with a real job search. A fabricated claim can
cost them an offer or their professional reputation.\
"""


# ── Job scoring ───────────────────────────────────────────────────────

_SCORING_SYSTEM = f"""\
You are an experienced technical recruiter assessing how well one candidate fits
one role. You are candid and calibrated — most candidates are not a strong match
for most roles, and your scores should reflect that.

{_HONESTY_RULES}

Scoring guide (0-100):
  90-100  Exceptional fit. Meets essentially every requirement including the
          hard ones. Would be shortlisted immediately.
  75-89   Strong fit. Meets all core requirements; minor gaps in nice-to-haves.
  60-74   Possible fit. Meets most core requirements but has a real gap
          (a missing primary technology, or notably less experience).
  40-59   Weak fit. Several core requirements unmet. Applying is likely a waste
          of the candidate's time.
  0-39    Not a match. Wrong domain, wrong seniority, or wrong discipline.

Calibrate honestly. Inflating a score wastes the candidate's limited daily
application budget on a role they cannot get.

Respond with ONLY a JSON object matching this schema:
{{
  "score": <integer 0-100>,
  "verdict": "strong_match" | "possible_match" | "weak_match" | "not_a_match",
  "reasoning": "<2-4 sentences, specific and concrete. Name the actual
                requirements met and missed. No filler.>",
  "matched_skills":  ["<requirements the RESUME genuinely evidences>"],
  "missing_skills":  ["<requirements the RESUME does not evidence>"],
  "concerns":        ["<seniority mismatch, location/visa problems, red flags>"]
}}
"""


def build_scoring_prompt(
    *,
    resume_text: str,
    job_title: str,
    company: str,
    location: str,
    job_description: str,
    seniority: str | None = None,
    employment_type: str | None = None,
) -> list[Message]:
    """Build the job-compatibility scoring prompt.

    ``SCORING_TASK`` marks this prompt for :class:`RulesProvider`, which uses
    the ``### RESUME`` and ``### JOB DESCRIPTION`` sections for keyword scoring.
    """
    meta_lines = [f"Title: {job_title}", f"Company: {company}", f"Location: {location}"]
    if seniority:
        meta_lines.append(f"Seniority: {seniority}")
    if employment_type:
        meta_lines.append(f"Employment type: {employment_type}")

    user = f"""\
SCORING_TASK

Assess this candidate against this role.

### RESUME
{_truncate(resume_text, 12_000)}

### JOB METADATA
{chr(10).join(meta_lines)}

### JOB DESCRIPTION
{_truncate(job_description, 12_000)}

### INSTRUCTIONS
{_HONESTY_RULES}

Return ONLY the JSON object. No preamble, no code fences.
"""
    return [Message(role="system", content=_SCORING_SYSTEM), Message(role="user", content=user)]


# ── Resume selection ──────────────────────────────────────────────────

_RESUME_CHOICE_SYSTEM = f"""\
You are helping a candidate pick which of their existing CV variants to submit
for a specific role. You are NOT editing or writing a CV — only choosing among
the ones provided.

{_HONESTY_RULES}

Respond with ONLY:
{{
  "resume_label": "<exact label of the chosen resume>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<1-2 sentences on why this variant fits best>"
}}
"""


def build_resume_selection_prompt(
    *,
    resumes: list[tuple[str, str]],
    job_title: str,
    job_description: str,
) -> list[Message]:
    """Build the prompt for choosing the best CV variant.

    Args:
        resumes: ``(label, content_text)`` pairs.
        job_title: Role title.
        job_description: Full description.
    """
    blocks = "\n\n".join(
        f"--- RESUME: {label} ---\n{_truncate(text, 5_000)}" for label, text in resumes
    )
    valid = ", ".join(repr(label) for label, _ in resumes)

    user = f"""\
### AVAILABLE RESUMES
{blocks}

### JOB TITLE
{job_title}

### JOB DESCRIPTION
{_truncate(job_description, 8_000)}

### INSTRUCTIONS
Choose the single best-fitting resume. `resume_label` MUST be exactly one of: {valid}.
Return ONLY the JSON object.
"""
    return [
        Message(role="system", content=_RESUME_CHOICE_SYSTEM),
        Message(role="user", content=user),
    ]


# ── Cover letter ──────────────────────────────────────────────────────

_COVER_LETTER_SYSTEM = f"""\
You write short, specific, honest cover letters. Your letters sound like a
competent professional wrote them in ten focused minutes — not like marketing
copy and not like a language model.

{_HONESTY_RULES}

Style requirements:
- 150-250 words. Shorter is better than padded.
- Open with why THIS role at THIS company, referencing something concrete from
  the job description. Never "I am writing to express my interest in...".
- Middle: two or three specific, verifiable points from the RESUME that map onto
  the role's actual requirements. Prefer concrete accomplishments over adjectives.
- Close with a brief, non-grovelling statement of interest.
- Plain prose. No bullet points, no headers, no em-dashes as connectors.
- Do NOT use: "leverage", "synergy", "passionate about", "dynamic",
  "results-driven", "I am confident that I would be a great fit", "delve",
  "tapestry", "testament to".
- Do NOT include a date, postal address, or "Dear Hiring Manager" salutation
  block — this is pasted into a web form, not posted.
- If the RESUME genuinely does not support a strong letter for this role, say so
  in the `warning` field rather than inventing material to fill space.

Respond with ONLY:
{{
  "cover_letter": "<the letter body>",
  "claims_made": ["<every factual claim about the candidate, one per item,
                   quoted or closely paraphrased from your letter>"],
  "warning": "<any concern about honesty or fit, or empty string>"
}}

`claims_made` is checked mechanically against the RESUME. Listing a claim you
cannot support will cause the letter to be rejected, so be exhaustive and exact.
"""


def build_cover_letter_prompt(
    *,
    resume_text: str,
    job_title: str,
    company: str,
    job_description: str,
    matched_skills: list[str] | None = None,
    extra_context: str = "",
) -> list[Message]:
    """Build the cover-letter generation prompt."""
    matched = ", ".join(matched_skills or []) or "(none identified)"
    extra = (
        f"\n### ADDITIONAL CONTEXT FROM THE CANDIDATE\n{extra_context}\n" if extra_context else ""
    )

    user = f"""\
### RESUME
{_truncate(resume_text, 12_000)}

### ROLE
{job_title} at {company}

### JOB DESCRIPTION
{_truncate(job_description, 10_000)}

### PRE-IDENTIFIED MATCHING SKILLS
{matched}
{extra}
### INSTRUCTIONS
{_HONESTY_RULES}

Write the cover letter. Return ONLY the JSON object.
"""
    return [
        Message(role="system", content=_COVER_LETTER_SYSTEM),
        Message(role="user", content=user),
    ]


# ── Screening questions ───────────────────────────────────────────────

_QUESTION_SYSTEM = f"""\
You answer job-application screening questions on behalf of a candidate, using
ONLY their stated profile facts and resume.

{_HONESTY_RULES}

Additional rules for this task:
6. If the PROFILE FACTS or RESUME do not contain the information needed, you
   MUST return `needs_human: true` and leave `answer` empty. Guessing a work
   authorisation status, a salary expectation, or a years-of-experience figure
   is never acceptable — these answers are legally and financially significant.
7. For numeric questions, answer with a bare number and nothing else.
8. For yes/no questions, answer exactly "Yes" or "No".
9. For multiple-choice questions, `answer` MUST be verbatim one of the provided
   options. If none is truthful, set `needs_human: true`.

Respond with ONLY:
{{
  "answer": "<the answer, or empty string if needs_human>",
  "needs_human": <true|false>,
  "confidence": <float 0.0-1.0>,
  "source": "profile" | "resume" | "ai" | "unknown",
  "reasoning": "<1 sentence naming which fact you used>"
}}
"""


def build_question_prompt(
    *,
    question: str,
    question_type: str,
    options: list[str] | None,
    profile_facts: str,
    resume_text: str,
) -> list[Message]:
    """Build the screening-question prompt."""
    options_block = ""
    if options:
        rendered = "\n".join(f"  - {o}" for o in options)
        options_block = f"\n### AVAILABLE OPTIONS (answer must match one exactly)\n{rendered}\n"

    user = f"""\
### PROFILE FACTS (authoritative)
{profile_facts or "(none provided)"}

### RESUME
{_truncate(resume_text, 8_000)}

### QUESTION
{question}

### QUESTION TYPE
{question_type}
{options_block}
### INSTRUCTIONS
{_HONESTY_RULES}

If the answer is not supported by PROFILE FACTS or RESUME, set needs_human=true.
Return ONLY the JSON object.
"""
    return [Message(role="system", content=_QUESTION_SYSTEM), Message(role="user", content=user)]


def _truncate(text: str, limit: int) -> str:
    """Clip long text, marking the cut so the model knows it is incomplete.

    Job descriptions occasionally run to tens of thousands of characters of
    boilerplate; sending all of it wastes tokens and buries the requirements.
    """
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + f"\n\n[... truncated at {limit} characters ...]"
