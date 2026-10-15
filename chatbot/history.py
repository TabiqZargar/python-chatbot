"""Conversation history: an ordered, size-limited list of messages."""

from __future__ import annotations

from dataclasses import dataclass

from chatbot.errors import ConfigError, InvalidInputError

__all__ = ["Message", "ConversationHistory"]

VALID_ROLES = ("user", "assistant")


@dataclass(frozen=True)
class Message:
    """One turn in the conversation."""

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Return the OpenAI-style {"role": ..., "content": ...} form."""
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    """Stores messages for one session, oldest first.

    ``limit`` caps how many *user* messages are kept. When the cap is
    exceeded, the oldest turns are dropped so earlier context is trimmed
    gracefully instead of growing forever.
    """

    def __init__(self, limit: int = 50) -> None:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ConfigError(f"history limit must be a positive integer, got {limit!r}")
        self._limit = limit
        self._messages: list[Message] = []

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def messages(self) -> tuple[Message, ...]:
        """Immutable snapshot of the history (safe to keep or iterate)."""
        return tuple(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def add(self, role: str, content: str) -> Message:
        """Append a message after validating it, then trim to the limit."""
        if role not in VALID_ROLES:
            raise InvalidInputError(
                f"role must be one of {VALID_ROLES}, got {role!r}"
            )
        if not isinstance(content, str) or not content.strip():
            raise InvalidInputError("message content must be a non-empty string")
        message = Message(role=role, content=content)
        self._messages.append(message)
        self._trim()
        return message

    def to_dicts(self) -> list[dict[str, str]]:
        """Return provider-ready message dictionaries."""
        return [message.to_dict() for message in self._messages]

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    def _user_count(self) -> int:
        return sum(1 for message in self._messages if message.role == "user")

    def _trim(self) -> None:
        while self._user_count() > self._limit and self._messages:
            self._messages.pop(0)
        while self._messages and self._messages[0].role != "user":
            self._messages.pop(0)
