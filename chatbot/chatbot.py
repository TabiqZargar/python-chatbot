"""Chatbot logic: validates input, talks to a provider, records history.

This module has no knowledge of any specific AI backend — it only uses the
ChatProvider interface — which is what makes providers interchangeable.
"""

from __future__ import annotations

from chatbot.errors import ChatbotError, InvalidInputError, ProviderError
from chatbot.history import ConversationHistory
from chatbot.providers.base import ChatProvider

__all__ = ["Chatbot", "MAX_MESSAGE_LENGTH"]

MAX_MESSAGE_LENGTH = 2000


class Chatbot:
    """One chat session: a provider plus its conversation history."""

    def __init__(
        self,
        provider: ChatProvider,
        history: ConversationHistory | None = None,
        max_message_length: int = MAX_MESSAGE_LENGTH,
    ) -> None:
        self._provider = provider
        self._history = history if history is not None else ConversationHistory()
        self._max_message_length = max_message_length

    @property
    def provider(self) -> ChatProvider:
        return self._provider

    @property
    def history(self) -> ConversationHistory:
        return self._history

    def chat(self, message: str) -> str:
        """Send ``message`` and return the reply.

        The history is only updated after a successful reply, so a failed
        provider call never leaves a half-recorded turn behind.
        """
        clean = self.validate_message(message)
        try:
            reply = self._provider.send(clean, self._history.messages)
        except ChatbotError:
            raise
        except Exception as exc:
            raise ProviderError(
                f"{self._provider.name} failed unexpectedly: {exc}"
            ) from exc
        if not isinstance(reply, str) or not reply.strip():
            raise ProviderError(f"{self._provider.name} returned an empty reply")
        reply = reply.strip()
        self._history.add("user", clean)
        self._history.add("assistant", reply)
        return reply

    def validate_message(self, message: str) -> str:
        """Return the cleaned message or raise InvalidInputError."""
        if not isinstance(message, str):
            raise InvalidInputError(f"message must be text, got {type(message).__name__}")
        cleaned = message.strip()
        if not cleaned:
            raise InvalidInputError("message must not be empty")
        if len(cleaned) > self._max_message_length:
            raise InvalidInputError(
                f"message is too long ({len(cleaned)} characters; "
                f"limit is {self._max_message_length})"
            )
        return cleaned

    def clear(self) -> None:
        """Forget the whole conversation."""
        self._history.clear()

    def render_history(self) -> list[str]:
        """Human-readable, numbered lines of the conversation so far."""
        return [
            f"{index}. {message.role}: {message.content}"
            for index, message in enumerate(self._history.messages, start=1)
        ]
