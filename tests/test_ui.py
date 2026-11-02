"""UI construction and display-helper tests (no browser or network)."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import gradio as gr
import pytest

from chatbot.engine import ChatEngine
from config import load_settings
from database.db import Database
from database.models import ChatMessage, Conversation
from ui.components import (
    THEME,
    build_app,
    conversation_choices,
    messages_for_chatbot,
)


class OfflineClient:
    base_url = "http://127.0.0.1:11434"

    def __init__(self) -> None:
        self.network_calls = 0

    def check_connection(self) -> Any:
        self.network_calls += 1
        raise AssertionError("build_app must not contact Ollama")

    def list_models(self) -> Any:
        self.network_calls += 1
        raise AssertionError("build_app must not contact Ollama")

    def stream_chat(
        self, *args: Any, **kwargs: Any
    ) -> Iterator[str]:
        yield "unused"


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "ui.db")
    try:
        yield database
    finally:
        database.close()


def test_build_app_does_not_contact_ollama(db: Database) -> None:
    client = OfflineClient()
    engine = ChatEngine(db, client)  # type: ignore[arg-type]
    settings = load_settings(environ={}, dotenv_path=None)

    demo = build_app(engine, settings)

    assert isinstance(demo, gr.Blocks)
    assert client.network_calls == 0
    assert isinstance(THEME, gr.themes.Base)


def test_build_app_restores_most_recent_conversation(
    db: Database, tmp_path: Path
) -> None:
    engine = ChatEngine(db, OfflineClient())  # type: ignore[arg-type]
    older = db.create_conversation("Older", "qwen2.5:3b")
    newer = db.create_conversation("Newer", "llama3.2:3b")
    db.add_message(older.id, "user", "old")
    db.add_message(newer.id, "user", "new")
    db.add_message(newer.id, "assistant", "reply")
    demo = build_app(engine, load_settings(environ={}, dotenv_path=None))

    assert isinstance(demo, gr.Blocks)


def test_conversation_choices_labels_and_values() -> None:
    conversations = [
        Conversation(
            id=2,
            title="Short",
            model="m",
            system_prompt="",
            temperature=0.7,
            max_tokens=10,
            created_at="2026-01-02T03:04:05.000000+00:00",
            updated_at="2026-01-02T03:04:05.000000+00:00",
            message_count=1,
        ),
        Conversation(
            id=3,
            title="x" * 100,
            model="m",
            system_prompt="",
            temperature=0.7,
            max_tokens=10,
            created_at="2026-01-02T03:04:05.000000+00:00",
            updated_at="2026-01-02T03:04:05.000000+00:00",
            message_count=4,
        ),
    ]

    choices = conversation_choices(conversations)

    assert [value for _, value in choices] == [2, 3]
    assert choices[0][0].startswith("Short · 1 message · ")
    assert choices[1][0].startswith("x" * 41 + "… · 4 messages · ")


def test_blank_title_is_untitled() -> None:
    conversation = Conversation(
        id=1,
        title="   ",
        model="m",
        system_prompt="",
        temperature=0.7,
        max_tokens=10,
        created_at="",
        updated_at="",
        message_count=0,
    )

    assert conversation_choices([conversation])[0][0].startswith("Untitled · 0 messages")


def test_messages_for_chatbot_filters_unknown_content() -> None:
    messages = [
        ChatMessage(id=1, conversation_id=1, role="user", content="hello", created_at=""),
        ChatMessage(id=2, conversation_id=1, role="assistant", content="hi", created_at=""),
        ChatMessage(id=3, conversation_id=1, role="system", content="hidden", created_at=""),
        ChatMessage(id=4, conversation_id=1, role="user", content=None, created_at=""),  # type: ignore[arg-type]
        {"role": "user", "content": "plain dict"},
    ]

    assert messages_for_chatbot(messages) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
