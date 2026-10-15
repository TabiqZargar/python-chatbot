"""Built-in offline provider: no network, no API key, deterministic.

Lets the chatbot run out of the box and makes tests independent of any
external service. Replies are simple rule-based responses that still use the
conversation history, so multi-turn behaviour is real.
"""

from __future__ import annotations

import random

from chatbot.errors import ProviderError
from chatbot.history import Message
from chatbot.providers.base import ChatProvider

__all__ = ["OfflineProvider"]

_GREETINGS = ("hello", "hi", "hey", "good morning", "good evening", "greetings")

_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("how are you", "how do you do"),
        "I'm just a local script, but I'm running fine — and I remember "
        "what we've said so far in this session.",
    ),
    (
        ("your name", "who are you"),
        "I'm a beginner-friendly Python chatbot running the offline provider.",
    ),
    (
        ("help", "what can you do"),
        "You can chat with me, and I'll reply from my history-aware rules. "
        "Type /help to see the available commands.",
    ),
    (
        ("bye", "goodbye", "see you"),
        "Goodbye! Type /exit any time to leave the chat.",
    ),
)


class OfflineProvider(ChatProvider):
    """Deterministic rule-based provider used by default and in tests."""

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "offline"

    def send(self, message: str, history: list[Message] | tuple[Message, ...]) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ProviderError("offline provider received an empty message")
        text = message.strip().lower()

        for keywords, reply in _RULES:
            if any(keyword in text for keyword in keywords):
                return reply

        if any(greeting in text for greeting in _GREETINGS):
            return (
                "Hello! I'm an offline demo chatbot. "
                "This is turn {} of our conversation."
                .format(self._turn_count(history))
            )

        if text.endswith("?"):
            return (
                "That's a good question, but I'm the offline provider and have "
                "no model behind me. Set CHATBOT_PROVIDER=openai to talk to a "
                "real model."
            )

        return self._fallback(message, history)

    @staticmethod
    def _turn_count(history: list[Message] | tuple[Message, ...]) -> int:
        return sum(1 for message in history if message.role == "user") + 1

    def _fallback(self, message: str, history: list[Message] | tuple[Message, ...]) -> str:
        turn = self._turn_count(history)
        if turn == 1:
            return (
                f'You said: "{message.strip()}". I\'m the offline demo — '
                "keep chatting and I'll keep the history!"
            )
        last = history[-1].content if history else ""
        return (
            f'You said: "{message.strip()}". So far you have sent {turn} messages'
            + (f', and before this one the last thing you said was: "{last}".' if last else ".")
        )
