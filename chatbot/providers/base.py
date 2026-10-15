"""Provider interface: the contract every AI backend must satisfy.

The chatbot logic only knows this interface, so swapping a local model for a
cloud API (or vice versa) never requires touching chatbot or CLI code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from chatbot.history import Message

__all__ = ["ChatProvider"]


class ChatProvider(ABC):
    """An AI backend that turns a message + history into a reply."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable provider name shown to the user."""

    @abstractmethod
    def send(self, message: str, history: list[Message] | tuple[Message, ...]) -> str:
        """Return a reply to ``message`` given prior ``history``.

        Raises ProviderError when the backend fails or returns bad data.
        """
