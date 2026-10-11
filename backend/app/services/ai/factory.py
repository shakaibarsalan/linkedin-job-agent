"""
AI provider construction and degradation.

Purpose
-------
Builds the configured provider and wraps it so that a provider outage degrades
the system instead of breaking it.

How it works
------------
:class:`ResilientProvider` delegates to the primary provider. If the primary
raises :class:`AIUnavailableError`, it retries through :class:`RulesProvider`
when the task is one the rules engine can actually do (scoring), and re-raises
otherwise (generation). It also trips a short circuit breaker so that a dead
endpoint is not re-dialled on every job in a 100-job run.

Failure scenarios
-----------------
* Primary down, scoring requested → keyword scoring, ``degraded=True`` recorded
  on the score row so it is obvious in the dashboard why numbers look flat.
* Primary down, cover letter requested → :class:`AIUnavailableError` propagates;
  the orchestrator routes the application to ``needs_manual``.
* Breaker open → primary is skipped entirely until the cooldown elapses.
"""

from __future__ import annotations

import time

from app.config import Settings, get_settings
from app.logging_config import get_logger
from app.services.ai.base import AIProvider, AIUnavailableError, Completion, Message
from app.services.ai.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    RulesProvider,
)

log = get_logger(__name__)

# How long to stop dialling a provider after it trips the breaker.
_BREAKER_COOLDOWN_SECONDS = 120.0
_BREAKER_FAILURE_THRESHOLD = 3


def build_provider(settings: Settings | None = None) -> AIProvider:
    """Construct the provider named by ``AI_PROVIDER``."""
    s = settings or get_settings()

    if s.ai_provider == "rules":
        log.info("ai.provider_selected", provider="rules")
        return RulesProvider()

    primary: AIProvider
    if s.ai_provider == "ollama":
        primary = OllamaProvider(
            base_url=s.ai_base_url,
            model=s.ai_model,
            temperature=s.ai_temperature,
            max_tokens=s.ai_max_tokens,
            timeout=s.ai_timeout_seconds,
            max_retries=s.ai_max_retries,
        )
    else:
        primary = OpenAICompatibleProvider(
            base_url=s.ai_base_url,
            api_key=s.ai_api_key,
            model=s.ai_model,
            temperature=s.ai_temperature,
            max_tokens=s.ai_max_tokens,
            timeout=s.ai_timeout_seconds,
            max_retries=s.ai_max_retries,
        )

    log.info("ai.provider_selected", provider=primary.name, model=s.ai_model)
    return ResilientProvider(primary=primary, fallback=RulesProvider())


class ResilientProvider(AIProvider):
    """Primary provider with a rules-based fallback and a circuit breaker."""

    def __init__(self, *, primary: AIProvider, fallback: AIProvider) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = primary.name
        self._consecutive_failures = 0
        self._breaker_opened_at: float | None = None
        # Set when the last completion came from the fallback, so callers can
        # record that a score was computed without a model.
        self.last_call_degraded = False

    @property
    def breaker_open(self) -> bool:
        """True while the primary is being skipped."""
        if self._breaker_opened_at is None:
            return False
        if time.monotonic() - self._breaker_opened_at >= _BREAKER_COOLDOWN_SECONDS:
            log.info("ai.breaker_reset", provider=self.primary.name)
            self._breaker_opened_at = None
            self._consecutive_failures = 0
            return False
        return True

    def _record_failure(self) -> None:
        self._consecutive_failures += 1
        if (
            self._consecutive_failures >= _BREAKER_FAILURE_THRESHOLD
            and self._breaker_opened_at is None
        ):
            self._breaker_opened_at = time.monotonic()
            log.error(
                "ai.breaker_opened",
                provider=self.primary.name,
                failures=self._consecutive_failures,
                cooldown_seconds=_BREAKER_COOLDOWN_SECONDS,
            )

    def _record_success(self) -> None:
        self._consecutive_failures = 0
        self._breaker_opened_at = None

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> Completion:
        self.last_call_degraded = False

        if not self.breaker_open:
            try:
                completion = await self.primary.complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
                self._record_success()
                return completion
            except AIUnavailableError as exc:
                self._record_failure()
                log.warning("ai.primary_failed", provider=self.primary.name, error=str(exc)[:300])
        else:
            log.debug("ai.breaker_skip", provider=self.primary.name)

        # Fallback. RulesProvider itself raises AIUnavailableError for any task
        # it cannot honestly perform, which is the behaviour we want: no silent
        # substitution of a guess for a generated answer.
        completion = await self.fallback.complete(
            messages, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode
        )
        self.last_call_degraded = True
        log.warning("ai.degraded_to_rules", task="scoring")
        return completion

    async def health(self) -> bool:
        return await self.primary.health()

    async def aclose(self) -> None:
        await self.primary.aclose()
        await self.fallback.aclose()
