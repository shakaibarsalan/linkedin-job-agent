"""
AI provider interface.

Purpose
-------
Decouples every AI-consuming feature (scoring, cover letters, question
answering) from any particular vendor, so the deployment can move between a
hosted API, a local Ollama model, and a deterministic no-LLM mode without
touching feature code.

How it works
------------
:class:`AIProvider` defines one primitive — ``complete()`` — plus a
``complete_json()`` helper that enforces a Pydantic schema on the response.
Model output is untrusted input: it is parsed, validated, and range-checked
before it can influence a decision.

Failure scenarios
-----------------
* Provider unreachable / rate-limited → :class:`AIUnavailableError`. Callers
  degrade rather than crash: scoring falls back to the rules provider, cover
  letter generation is skipped and the application is routed to human review.
* Model returns prose around the JSON → ``extract_json`` recovers the object.
* Model returns invalid JSON or fails validation → :class:`AIResponseError`
  after ``AI_MAX_RETRIES`` attempts, each one re-prompting with the error.
"""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.logging_config import get_logger

log = get_logger(__name__)

TModel = TypeVar("TModel", bound=BaseModel)


class AIError(RuntimeError):
    """Base class for AI layer failures."""


class AIUnavailableError(AIError):
    """Provider could not be reached, timed out, or refused the request.

    Callers should degrade gracefully — never fail the whole run over this.
    """


class AIResponseError(AIError):
    """Provider responded, but the content could not be used."""


@dataclass(slots=True)
class Message:
    """One chat message."""

    role: str  # "system" | "user" | "assistant"
    content: str


@dataclass(slots=True)
class Completion:
    """A provider response plus the metadata we persist for auditability."""

    text: str
    provider: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


# Matches a fenced code block so ```json ... ``` wrappers can be stripped.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of a model response.

    Handles the three shapes models actually emit: bare JSON, JSON inside a
    fenced code block, and JSON with a sentence of preamble.

    Raises:
        AIResponseError: if no parseable JSON object or array is present.
    """
    candidate = text.strip()

    if not candidate:
        raise AIResponseError("Empty response from model")

    # 1. Straight parse — the common case with well-behaved models.
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block.
    fence = _FENCE_RE.search(candidate)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 3. First balanced {...} or [...] span. Scanning for balance rather than
    #    regexing is what makes nested objects work.
    for opener, closer in (("{", "}"), ("[", "]")):
        start = candidate.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(candidate)):
            ch = candidate[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : idx + 1])
                    except json.JSONDecodeError:
                        break

    raise AIResponseError(f"No parseable JSON in model response: {candidate[:200]!r}")


class AIProvider(abc.ABC):
    """Abstract chat-completion provider."""

    name: str = "base"

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        """Return a completion for ``messages``.

        Args:
            messages: Conversation, system message first.
            temperature: Sampling temperature; provider default if ``None``.
            max_tokens: Response cap; provider default if ``None``.
            json_mode: Request structured JSON output where the provider
                supports it. Callers must still validate — this is a hint.

        Raises:
            AIUnavailableError: transport, auth, or rate-limit failure.
        """

    async def complete_json(
        self,
        messages: list[Message],
        schema: type[TModel],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
    ) -> tuple[TModel, Completion]:
        """Complete and validate against a Pydantic model.

        On a validation failure the error is fed back to the model as a
        correction turn, which recovers most malformed responses in one retry.

        Raises:
            AIResponseError: if validation still fails after ``max_retries``.
            AIUnavailableError: propagated from :meth:`complete`.
        """
        conversation = list(messages)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            completion = await self.complete(
                conversation,
                temperature=temperature,
                max_tokens=max_tokens,
                json_mode=True,
            )
            try:
                payload = extract_json(completion.text)
                return schema.model_validate(payload), completion
            except (AIResponseError, ValidationError) as exc:
                last_error = exc
                log.warning(
                    "ai.json_validation_failed",
                    provider=self.name,
                    attempt=attempt + 1,
                    error=str(exc)[:300],
                )
                if attempt == max_retries:
                    break
                conversation = [
                    *messages,
                    Message(role="assistant", content=completion.text[:2000]),
                    Message(
                        role="user",
                        content=(
                            "That response was not valid for the required schema. "
                            f"Error: {str(exc)[:500]}\n\n"
                            "Reply with ONLY the corrected JSON object. "
                            "No prose, no code fences."
                        ),
                    ),
                ]

        raise AIResponseError(
            f"Model did not produce schema-valid JSON after {max_retries + 1} attempts: "
            f"{last_error}"
        ) from last_error

    # Generous on purpose. A reasoning model spends this budget on hidden
    # thinking before emitting anything visible, so a "cheap" probe of 8 tokens
    # returns finish_reason="length" with zero visible output and the health
    # check reports a perfectly healthy provider as down. Measured on
    # gemini-3.6-flash: a two-word answer consumed ~70 tokens of thinking.
    _HEALTH_MAX_TOKENS = 512

    async def health(self) -> bool:
        """Liveness probe. Never raises."""
        try:
            await self.complete(
                [Message(role="user", content="Reply with the single word: ok")],
                max_tokens=self._HEALTH_MAX_TOKENS,
            )
            return True
        except Exception:  # noqa: BLE001 — health checks must not propagate
            log.warning("ai.health_check_failed", provider=self.name, exc_info=True)
            return False

    async def aclose(self) -> None:
        """Release provider resources. Default is a no-op."""
        return None
