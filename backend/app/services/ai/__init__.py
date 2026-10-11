"""Pluggable AI layer: provider interface, concrete providers, prompts."""

from app.services.ai.base import (
    AIError,
    AIProvider,
    AIResponseError,
    AIUnavailableError,
    Completion,
    Message,
    extract_json,
)
from app.services.ai.factory import ResilientProvider, build_provider
from app.services.ai.providers import (
    OllamaProvider,
    OpenAICompatibleProvider,
    RulesProvider,
)

__all__ = [
    "AIError",
    "AIProvider",
    "AIResponseError",
    "AIUnavailableError",
    "Completion",
    "Message",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "ResilientProvider",
    "RulesProvider",
    "build_provider",
    "extract_json",
]
