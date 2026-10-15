"""Tests for the Chatbot core: multi-turn history and error handling."""

from __future__ import annotations

import pytest

from chatbot.chatbot import MAX_MESSAGE_LENGTH, Chatbot
from chatbot.errors import InvalidInputError, ProviderError
from chatbot.history import ConversationHistory, Message
from chatbot.providers.base import ChatProvider
from chatbot.providers.offline import OfflineProvider


class StubProvider(ChatProvider):
    """Minimal provider whose behaviour each test can control."""

    def __init__(self, reply: str = "ok", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[tuple[str, tuple[Message, ...]]] = []

    @property
    def name(self) -> str:
        return "stub"

    def send(self, message: str, history: list[Message] | tuple[Message, ...]) -> str:
        self.calls.append((message, tuple(history)))
        if self.error is not None:
            raise self.error
        return self.reply


def make_bot(limit: int = 50, provider: ChatProvider | None = None) -> Chatbot:
    return Chatbot(provider or OfflineProvider(), history=ConversationHistory(limit=limit))


def test_multi_turn_conversation_is_preserved() -> None:
    bot = make_bot()

    first = bot.chat("hello")
    second = bot.chat("tell me more")

    assert isinstance(first, str) and first
    assert isinstance(second, str) and second
    roles = [m.role for m in bot.history.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert bot.history.messages[0].content == "hello"
    assert bot.history.messages[2].content == "tell me more"


def test_second_call_receives_first_turn_in_history() -> None:
    provider = StubProvider()
    bot = Chatbot(provider, history=ConversationHistory(limit=10))

    bot.chat("first message")
    bot.chat("second message")

    _, history_seen = provider.calls[1]
    assert [(m.role, m.content) for m in history_seen] == [
        ("user", "first message"),
        ("assistant", "ok"),
    ]


@pytest.mark.parametrize("message", ["", "   ", None, 42])
def test_invalid_messages_rejected(message: object) -> None:
    bot = make_bot()
    with pytest.raises(InvalidInputError):
        bot.chat(message)  # type: ignore[arg-type]
    assert len(bot.history) == 0


def test_too_long_message_rejected() -> None:
    bot = make_bot()
    with pytest.raises(InvalidInputError):
        bot.chat("x" * (MAX_MESSAGE_LENGTH + 1))
    assert len(bot.history) == 0


def test_message_is_trimmed_before_use() -> None:
    provider = StubProvider()
    bot = Chatbot(provider, history=ConversationHistory(limit=10))

    bot.chat("  spaced out  ")

    assert provider.calls[0][0] == "spaced out"
    assert bot.history.messages[0].content == "spaced out"


def test_provider_error_propagates_and_history_unchanged() -> None:
    provider = StubProvider(error=ProviderError("upstream down"))
    bot = Chatbot(provider, history=ConversationHistory(limit=10))

    with pytest.raises(ProviderError, match="upstream down"):
        bot.chat("hello")

    assert len(bot.history) == 0


def test_unexpected_provider_exception_is_wrapped() -> None:
    provider = StubProvider(error=RuntimeError("kaboom"))
    bot = Chatbot(provider, history=ConversationHistory(limit=10))

    with pytest.raises(ProviderError) as excinfo:
        bot.chat("hello")

    assert "kaboom" in str(excinfo.value)
    assert len(bot.history) == 0


@pytest.mark.parametrize("reply", ["", "   "])
def test_empty_provider_reply_is_an_error(reply: str) -> None:
    bot = Chatbot(StubProvider(reply=reply), history=ConversationHistory(limit=10))

    with pytest.raises(ProviderError):
        bot.chat("hello")

    assert len(bot.history) == 0


def test_clear_and_render_history() -> None:
    bot = make_bot()
    bot.chat("hello")
    lines = bot.render_history()

    assert lines[0].startswith("1. user: hello")

    bot.clear()
    assert bot.render_history() == []
    assert len(bot.history) == 0


def test_history_limit_is_enforced_end_to_end() -> None:
    bot = make_bot(limit=1)

    bot.chat("first")
    bot.chat("second")

    assert len(bot.history) == 2
    assert bot.history.messages[0].content == "second"
    assert bot.history.messages[0].role == "user"
