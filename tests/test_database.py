"""SQLite persistence tests: conversations, messages, preferences, cascades."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.errors import InvalidInputError, StorageError
from database.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "chat.db")
    try:
        yield database
    finally:
        database.close()


def test_schema_is_created_and_database_persists(tmp_path: Path) -> None:
    path = tmp_path / "persistent.db"
    first = Database(path)
    conversation = first.create_conversation("Saved", "qwen2.5:3b")
    first.add_message(conversation.id, "user", "hello")
    first.add_message(conversation.id, "assistant", "hi")
    first.set_preferences({"model": "qwen2.5:3b"})
    first.close()

    second = Database(path)
    try:
        loaded = second.get_conversation(conversation.id)
        assert loaded is not None
        assert loaded.title == "Saved"
        assert loaded.message_count == 2
        assert [m.content for m in second.get_messages(conversation.id)] == [
            "hello",
            "hi",
        ]
        assert second.get_preferences()["model"] == "qwen2.5:3b"
    finally:
        second.close()


def test_create_and_list_conversations_newest_first(db: Database) -> None:
    first = db.create_conversation("First", "model-a")
    second = db.create_conversation("Second", "model-b")
    db.add_message(first.id, "user", "bump first")

    conversations = db.list_conversations()

    assert [c.id for c in conversations] == [first.id, second.id]
    assert conversations[0].message_count == 1
    assert conversations[1].message_count == 0


def test_message_order_and_roles(db: Database) -> None:
    conversation = db.create_conversation("Chat", "model")
    db.add_message(conversation.id, "user", "one")
    db.add_message(conversation.id, "assistant", "two")
    db.add_message(conversation.id, "user", "three")

    messages = db.get_messages(conversation.id)

    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert [m.content for m in messages] == ["one", "two", "three"]
    assert all(m.id is not None for m in messages)


def test_rename_and_settings_update(db: Database) -> None:
    conversation = db.create_conversation("Old", "model-a")

    assert db.rename_conversation(conversation.id, "New title") is True
    assert db.update_conversation_settings(
        conversation.id, "model-b", "Be terse.", 1.25, 222
    ) is True

    loaded = db.get_conversation(conversation.id)
    assert loaded is not None
    assert loaded.title == "New title"
    assert loaded.model == "model-b"
    assert loaded.system_prompt == "Be terse."
    assert loaded.temperature == 1.25
    assert loaded.max_tokens == 222


def test_delete_conversation_cascades_messages(db: Database) -> None:
    conversation = db.create_conversation("Remove me", "model")
    db.add_message(conversation.id, "user", "hello")

    assert db.delete_conversation(conversation.id) is True
    assert db.get_conversation(conversation.id) is None
    assert db.get_messages(conversation.id) == []
    assert db.delete_conversation(conversation.id) is False


def test_clear_messages_keeps_conversation(db: Database) -> None:
    conversation = db.create_conversation("Keep me", "model")
    db.add_message(conversation.id, "user", "hello")
    db.add_message(conversation.id, "assistant", "hi")

    assert db.clear_messages(conversation.id) == 2
    assert db.get_conversation(conversation.id) is not None
    assert db.get_messages(conversation.id) == []


def test_remove_last_assistant_message(db: Database) -> None:
    conversation = db.create_conversation("Chat", "model")
    db.add_message(conversation.id, "user", "question")
    db.add_message(conversation.id, "assistant", "answer")
    db.add_message(conversation.id, "user", "follow-up")

    assert db.remove_last_assistant_message(conversation.id) is None
    db.add_message(conversation.id, "assistant", "second answer")

    removed = db.remove_last_assistant_message(conversation.id)
    assert removed is not None
    assert removed.content == "second answer"
    assert [m.content for m in db.get_messages(conversation.id)] == [
        "question",
        "answer",
        "follow-up",
    ]


def test_preferences_upsert_and_read(db: Database) -> None:
    db.set_preferences({"model": "a", "temperature": "0.5"})
    db.set_preferences({"temperature": "1.0"})

    assert db.get_preferences() == {"model": "a", "temperature": "1.0"}


def test_sql_injection_text_is_stored_literally(db: Database) -> None:
    payload = "'); DROP TABLE conversations; --"
    conversation = db.create_conversation(payload, "model")
    db.add_message(conversation.id, "user", payload)

    loaded = db.get_conversation(conversation.id)
    assert loaded is not None
    assert loaded.title == payload
    assert db.get_messages(conversation.id)[0].content == payload


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("create_conversation", ("", "model")),
        ("create_conversation", ("title", "  ")),
        ("add_message", (1, "system", "hello")),
        ("add_message", (1, "user", "   ")),
        ("rename_conversation", (1, "")),
        ("update_conversation_settings", (1, "model", "", 3.0, 10)),
        ("update_conversation_settings", (1, "model", "", 0.5, 0)),
        ("delete_conversation", (0,)),
    ],
)
def test_invalid_inputs_are_rejected(
    db: Database, method: str, args: tuple[object, ...]
) -> None:
    conversation = db.create_conversation("Valid", "model")
    actual_args = tuple(conversation.id if value == 1 else value for value in args)

    with pytest.raises(InvalidInputError):
        getattr(db, method)(*actual_args)


def test_adding_to_missing_conversation_is_an_error(db: Database) -> None:
    with pytest.raises(StorageError):
        db.add_message(999, "user", "hello")
