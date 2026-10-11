"""
Concrete AI providers.

Purpose
-------
Three implementations of :class:`~app.services.ai.base.AIProvider`:

* :class:`OpenAICompatibleProvider` — any ``/v1/chat/completions`` endpoint
  (OpenAI, Anthropic-compatible gateways, OpenRouter, Groq, vLLM, LM Studio).
* :class:`OllamaProvider` — a local Ollama daemon, for a fully offline deployment.
* :class:`RulesProvider` — no LLM at all. Deterministic, free, always available.
  Used as the automatic fallback when a real provider is down, which is what
  keeps a run going instead of failing it.

How it works
------------
Both network providers share a retry policy: exponential backoff with jitter on
timeouts, connection errors, 429, and 5xx; immediate failure on 4xx (retrying an
auth error just burns time).

Failure scenarios
-----------------
* 401/403 → :class:`AIUnavailableError` immediately, no retry. Check ``AI_API_KEY``.
* 429 → honours ``Retry-After`` when present, otherwise backs off.
* Timeout → retried up to ``AI_MAX_RETRIES``, then raises.
* Ollama model not pulled → 404 with a message naming the model; pull it first.
"""

from __future__ import annotations

import asyncio
import random
import re
from typing import Any

import httpx

from app.logging_config import get_logger
from app.services.ai.base import (
    AIProvider,
    AIUnavailableError,
    Completion,
    Message,
)

log = get_logger(__name__)

# Status codes worth retrying: rate limit + transient server-side faults.
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class _HTTPProviderBase(AIProvider):
    """Shared HTTP client lifecycle and retry policy."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = "",
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1500,
        timeout: float = 90.0,
        max_retries: int = 3,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=15.0),
            follow_redirects=True,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _post_with_retry(
        self, url: str, *, json: dict[str, Any], headers: dict[str, str]
    ) -> dict[str, Any]:
        """POST with backoff on transient failures. Raises AIUnavailableError."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(url, json=json, headers=headers)

                if response.status_code in _RETRYABLE_STATUS:
                    retry_after = _parse_retry_after(response)
                    last_error = AIUnavailableError(
                        f"{self.name} returned {response.status_code}: {response.text[:200]}"
                    )
                    if attempt == self.max_retries:
                        break
                    delay = retry_after if retry_after is not None else _backoff(attempt)
                    log.warning(
                        "ai.retrying",
                        provider=self.name,
                        status=response.status_code,
                        attempt=attempt + 1,
                        sleep_seconds=round(delay, 1),
                    )
                    await asyncio.sleep(delay)
                    continue

                if response.status_code >= 400:
                    # 4xx that isn't rate limiting: config error. Retrying wastes time.
                    raise AIUnavailableError(
                        f"{self.name} rejected the request with "
                        f"{response.status_code}: {response.text[:300]}"
                    )

                return response.json()  # type: ignore[no-any-return]

            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                delay = _backoff(attempt)
                log.warning(
                    "ai.retrying",
                    provider=self.name,
                    error=type(exc).__name__,
                    attempt=attempt + 1,
                    sleep_seconds=round(delay, 1),
                )
                await asyncio.sleep(delay)

        raise AIUnavailableError(
            f"{self.name} unavailable after {self.max_retries + 1} attempts: {last_error}"
        ) from last_error


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter, capped at 30 s."""
    jitter = random.uniform(0.0, 1.0)  # noqa: S311 — pacing jitter, not crypto
    return min(30.0, float(2**attempt) + jitter)


def _parse_retry_after(response: httpx.Response) -> float | None:
    """Read a ``Retry-After`` header if the server sent a sane one."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, min(120.0, float(raw)))
    except ValueError:
        return None


class OpenAICompatibleProvider(_HTTPProviderBase):
    """Any endpoint speaking the OpenAI ``/chat/completions`` schema."""

    name = "openai_compatible"

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            data = await self._post_with_retry(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
        except AIUnavailableError as exc:
            # Some gateways reject response_format outright. One retry without
            # it turns a hard failure into a soft one; we validate anyway.
            if json_mode and "response_format" in str(exc):
                log.info("ai.retry_without_json_mode", provider=self.name)
                payload.pop("response_format", None)
                data = await self._post_with_retry(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
            else:
                raise

        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise AIUnavailableError(
                f"Unexpected response shape from {self.base_url}: {str(data)[:300]}"
            ) from exc

        # `.get`, not `[...]`. When a reasoning model exhausts its budget on
        # hidden thinking, Gemini omits the "content" key ENTIRELY rather than
        # returning an empty string:
        #     {"finish_reason": "length", "message": {"role": "assistant"}}
        # Subscripting raised KeyError and masked the real cause behind a
        # generic "unexpected response shape". Falling through to the empty
        # check below produces an error that actually names the fix.
        text = (choice.get("message") or {}).get("content") or ""

        usage = data.get("usage") or {}

        # Reasoning models (Gemini 3.x, o-series, and friends) spend the token
        # budget on hidden thinking before emitting anything visible. Exhaust it
        # and the API returns HTTP 200, finish_reason "stop", and an EMPTY
        # string — no error anywhere. Left unhandled that surfaces much later as
        # a confusing "no parseable JSON", so name the real cause here.
        if not text.strip():
            total = usage.get("total_tokens")
            completion_tokens = usage.get("completion_tokens")
            requested = self.max_tokens if max_tokens is None else max_tokens
            raise AIUnavailableError(
                f"{self.model} returned an empty response "
                f"(finish_reason={choice.get('finish_reason')!r}, "
                f"visible tokens={completion_tokens}, total={total}, "
                f"max_tokens={requested}). "
                "For a reasoning model the budget also covers hidden thinking "
                "tokens — raise AI_MAX_TOKENS."
            )

        return Completion(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )


class OllamaProvider(_HTTPProviderBase):
    """A local Ollama daemon (``/api/chat``)."""

    name = "ollama"

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"

        data = await self._post_with_retry(
            f"{self.base_url}/api/chat", json=payload, headers={"Content-Type": "application/json"}
        )

        try:
            text = data["message"]["content"] or ""
        except (KeyError, TypeError) as exc:
            raise AIUnavailableError(
                f"Unexpected Ollama response shape: {str(data)[:300]}"
            ) from exc

        return Completion(
            text=text,
            provider=self.name,
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )


# ── Rules provider ────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[a-z0-9+#.]+")

# Tokens that carry no signal when comparing a CV to a job description.
_STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those with without within into onto
    for from to of in on at by as is are was were be been being do does did doing have
    has had having will would shall should can could may might must not no nor so such
    you your we our they their he she it its i me my us them who whom which what when
    where why how all any both each few more most other some only own same very just
    about above after again against because before below between during further here
    once out over under up down off through too also across per via
    work working works experience experienced role position job opportunity team teams
    company candidate candidates applicant applicants required requirements requirement
    responsibilities responsibility skills skill ability abilities strong excellent good
    great plus preferred nice must year years month months day days new
    """.split()
)


def _tokenize(text: str) -> set[str]:
    """Lowercase word set with stopwords removed."""
    return {
        w
        for w in _WORD_RE.findall(text.lower())
        if len(w) > 1 and w not in _STOPWORDS and not w.isdigit()
    }


class RulesProvider(AIProvider):
    """Deterministic, LLM-free provider.

    Purpose
    -------
    Two jobs: (1) the configured mode for a zero-cost, zero-egress deployment,
    and (2) the automatic fallback when a network provider is down, so a run
    degrades to keyword matching instead of dying.

    Behaviour
    ---------
    * **Scoring** — Jaccard-style overlap between resume tokens and job-description
      tokens, scaled into 0-100. Genuinely useful for filtering, and it never
      hallucinates.
    * **Cover letters** — refuses. There is no honest way to write a tailored
      letter without a language model, and a templated one is worse than none.
      The caller marks the application for human review.
    * **Questions** — refuses. Answers must come from the profile file or a
      human, never from a guess.
    """

    name = "rules"

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Handle only the scoring task; refuse anything requiring generation."""
        user_text = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if "SCORING_TASK" in user_text:
            return Completion(
                text=self._score(user_text), provider=self.name, model="keyword-overlap-v1"
            )

        raise AIUnavailableError(
            "The 'rules' provider cannot generate free text. Configure a real "
            "AI provider, or let this application fall through to human review."
        )

    @staticmethod
    def _score(user_text: str) -> str:
        """Score by token overlap between the RESUME and JOB DESCRIPTION blocks."""
        resume = _section(user_text, "RESUME")
        description = _section(user_text, "JOB DESCRIPTION")

        resume_tokens = _tokenize(resume)
        job_tokens = _tokenize(description)

        if not job_tokens or not resume_tokens:
            return (
                '{"score": 0, "verdict": "not_a_match", "reasoning": '
                '"Insufficient text to compare.", "matched_skills": [], '
                '"missing_skills": [], "concerns": ["Empty resume or job description"]}'
            )

        matched = sorted(resume_tokens & job_tokens)
        missing = sorted(job_tokens - resume_tokens)

        # Coverage of the job's vocabulary by the resume. Squashed with a 0.55
        # divisor because even an ideal CV rarely exceeds ~55% raw overlap —
        # job posts are full of boilerplate that no resume mirrors.
        coverage = len(matched) / len(job_tokens)
        score = max(0, min(100, round(coverage / 0.55 * 100)))

        if score >= 80:
            verdict = "strong_match"
        elif score >= 60:
            verdict = "possible_match"
        elif score >= 35:
            verdict = "weak_match"
        else:
            verdict = "not_a_match"

        import json as _json

        return _json.dumps(
            {
                "score": score,
                "verdict": verdict,
                "reasoning": (
                    f"Keyword overlap only (no language model): the resume covers "
                    f"{len(matched)} of {len(job_tokens)} distinct job-description terms "
                    f"({coverage:.0%} raw coverage). This is a coarse filter, not a "
                    f"judgement of fit — review before applying."
                ),
                "matched_skills": matched[:25],
                "missing_skills": missing[:25],
                "concerns": ["Scored without a language model; human review recommended."],
            }
        )

    async def health(self) -> bool:
        return True


def _section(text: str, header: str) -> str:
    """Extract a ``### HEADER`` block from a prompt built by app.services.ai.prompts."""
    pattern = re.compile(
        rf"###\s*{re.escape(header)}\s*\n(.*?)(?=\n###\s|\Z)", re.DOTALL | re.IGNORECASE
    )
    match = pattern.search(text)
    return match.group(1) if match else ""
