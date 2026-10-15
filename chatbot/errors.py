"""Exception hierarchy for the chatbot.

Every error the chatbot raises on purpose derives from ChatbotError, so the
CLI can catch one type and still show a precise message.
"""

from __future__ import annotations

__all__ = [
    "ChatbotError",
    "ConfigError",
    "InvalidInputError",
    "ProviderError",
]


class ChatbotError(Exception):
    """Base class for all expected chatbot errors."""


class ConfigError(ChatbotError):
    """Raised when configuration or environment variables are invalid."""


class InvalidInputError(ChatbotError):
    """Raised when user input or a value fails validation."""


class ProviderError(ChatbotError):
    """Raised when an AI provider request fails or returns bad data."""
