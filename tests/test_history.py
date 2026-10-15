"""Tests for ConversationHistory."""

from __future__ import annotations

import pytest

from chatbot.errors import ConfigError, InvalidInputError
from chatbot.history import ConversationHistory, Message


def test_add_keeps_order_and_records_roles() -> None:
    history = ConversationHistory(limit=10)
    history.add("user", "hello")
    history.add("assistant", "hi there")

    assert [m.role for m in history.messages] == ["user", "assistant"]
    assert [m.content for m in history.messages] == ["hello", "hi there"]
    assert len(history) == 2


def test_messages_snapshot_is_stable() -> None:
    history = ConversationHistory(limit=10)
    history.add("user", "first")
    snapshot = history.messages

    history.add("assistant", "second")

    assert len(snapshot) == 1
    assert len(history.messages) == 2


def test_to_dicts_uses_openai_shape() -> None:
    history = ConversationHistory(limit=10)
    history.add("user", "hello")

    assert history.to_dicts() == [{"role": "user", "content": "hello"}]


def test_message_to_dict() -> None:
    assert Message("user", "x").to_dict() == {"role": "user", "content": "x"}


@pytest.mark.parametrize("role", ["system", "bot", "", "USER"])
def test_invalid_role_rejected(role: str) -> None:
    history = ConversationHistory(limit=10)
    with pytest.raises(InvalidInputError):
        history.add(role, "hello")


@pytest.mark.parametrize("content", ["", "   ", 123, None])
def test_invalid_content_rejected(content: object) -> None:
    history = ConversationHistory(limit=10)
    with pytest.raises(InvalidInputError):
        history.add("user", content)  # type: ignore[arg-type]


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_limit_rejected(limit: object) -> None:
    with pytest.raises(ConfigError):
        ConversationHistory(limit=limit)  # type: ignore[arg-type]


def test_limit_trims_oldest_turns() -> None:
    history = ConversationHistory(limit=2)
    for text in ("u1", "u2", "u3", "u4"):
        history.add("user", text)
        history.add("assistant", f"a-{text}")

    user_messages = [m for m in history.messages if m.role == "user"]
    assert [m.content for m in user_messages] == ["u3", "u4"]
    assert history.messages[0].role == "user"


def test_trims_orphaned_leading_assistant_message() -> None:
    history = ConversationHistory(limit=1)
    history.add("user", "u1")
    history.add("assistant", "a1")
    history.add("user", "u2")

    assert [m.content for m in history.messages] == ["u2"]
    assert history.messages[0].role == "user"


def test_clear_empties_history() -> None:
    history = ConversationHistory(limit=5)
    history.add("user", "hello")
    history.clear()

    assert len(history) == 0
    assert history.to_dicts() == []
