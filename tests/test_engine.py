"""Engine tests: streaming persistence, regeneration, context, and errors."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from chatbot.engine import ChatEngine, ChatOptions
from chatbot.errors import InvalidInputError, OllamaError
from database.db import Database


class FakeOllama:
    """Deterministic streaming client that records every request."""

    base_url = "http://127.0.0.1:11434"

    def __init__(
        self,
        chunks: Sequence[str] = ("Hello",),
        error: Exception | None = None,
    ) -> None:
        self.chunks = list(chunks)
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop_requested: Any = None,
    ) -> Iterator[str]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        for chunk in self.chunks:
            yield chunk
        if self.error is not None:
            raise self.error


@pytest.fixture
def db(tmp_path: Path) -> Database:
    database = Database(tmp_path / "engine.db")
    try:
        yield database
    finally:
        database.close()


def make_engine(
    db: Database,
    chunks: Sequence[str] = ("Hello",),
    error: Exception | None = None,
    history_limit: int = 30,
) -> tuple[ChatEngine, FakeOllama]:
    client = FakeOllama(chunks, error)
    return ChatEngine(db, client, history_limit=history_limit), client


OPTIONS = ChatOptions(
    model="qwen2.5:3b",
    system_prompt="Be concise.",
    temperature=0.3,
    max_tokens=256,
)


def test_first_message_creates_conversation_and_persists_turn(db: Database) -> None:
    engine, client = make_engine(db, chunks=("Hello", " world"))

    updates = list(engine.stream_reply(None, "Hi there", OPTIONS))

    assert [update.text for update in updates] == [
        "Hello",
        "Hello world",
        "Hello world",
    ]
    assert updates[-1].done is True
    conversation_id = updates[-1].conversation_id
    conversation = db.get_conversation(conversation_id)
    assert conversation is not None
    assert conversation.title == "Hi there"
    assert conversation.model == OPTIONS.model
    assert conversation.system_prompt == OPTIONS.system_prompt
    messages = db.get_messages(conversation_id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "Hi there"),
        ("assistant", "Hello world"),
    ]
    assert client.calls[0]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hi there"},
    ]


def test_multi_turn_payload_includes_full_context(db: Database) -> None:
    engine, client = make_engine(db, chunks=("First reply",))
    first = list(engine.stream_reply(None, "First question", OPTIONS))
    conversation_id = first[-1].conversation_id

    client.chunks = ["Second reply"]
    list(engine.stream_reply(conversation_id, "Second question", OPTIONS))

    assert client.calls[1]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First reply"},
        {"role": "user", "content": "Second question"},
    ]


def test_regenerate_replaces_previous_assistant_reply(db: Database) -> None:
    engine, client = make_engine(db, chunks=("Old answer",))
    first = list(engine.stream_reply(None, "Question", OPTIONS))
    conversation_id = first[-1].conversation_id

    client.chunks = ["New answer"]
    updates = list(
        engine.stream_reply(conversation_id, None, OPTIONS, regenerate=True)
    )

    assert updates[-1].text == "New answer"
    messages = db.get_messages(conversation_id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "Question"),
        ("assistant", "New answer"),
    ]
    # The old assistant reply must not be sent back as context.
    assert client.calls[1]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Question"},
    ]


def test_closing_generator_saves_partial_reply(db: Database) -> None:
    engine, _ = make_engine(db, chunks=("Part", "ial"))
    generator = engine.stream_reply(None, "Hello", OPTIONS)

    first = next(generator)
    assert first.text == "Part"
    generator.close()

    conversation_id = first.conversation_id
    messages = db.get_messages(conversation_id)
    assert [(m.role, m.content) for m in messages] == [
        ("user", "Hello"),
        ("assistant", "Part"),
    ]


def test_error_before_content_restores_previous_reply_on_regenerate(
    db: Database,
) -> None:
    engine, client = make_engine(db, chunks=("Old answer",))
    first = list(engine.stream_reply(None, "Question", OPTIONS))
    conversation_id = first[-1].conversation_id

    client.chunks = []
    client.error = OllamaError("Ollama is down")
    updates = list(
        engine.stream_reply(conversation_id, None, OPTIONS, regenerate=True)
    )

    assert "Ollama is down" in updates[-1].error
    assert updates[-1].text == "Old answer"
    assert [m.content for m in db.get_messages(conversation_id)] == [
        "Question",
        "Old answer",
    ]


def test_error_after_partial_saves_partial_and_reports_error(db: Database) -> None:
    engine, _ = make_engine(
        db, chunks=("Par",), error=OllamaError("connection lost")
    )

    updates = list(engine.stream_reply(None, "Hello", OPTIONS))

    assert updates[-1].error == "connection lost"
    assert updates[-1].text == "Par"
    conversation_id = updates[-1].conversation_id
    assert [m.content for m in db.get_messages(conversation_id)] == [
        "Hello",
        "Par",
    ]


def test_whitespace_only_reply_is_an_error(db: Database) -> None:
    engine, _ = make_engine(db, chunks=("   ",))

    updates = list(engine.stream_reply(None, "Hello", OPTIONS))

    assert "empty response" in updates[-1].error
    conversation_id = updates[-1].conversation_id
    assert [m.role for m in db.get_messages(conversation_id)] == ["user"]


@pytest.mark.parametrize("message", ["", "   ", None, 42])
def test_invalid_user_message_creates_nothing(db: Database, message: object) -> None:
    engine, _ = make_engine(db)

    with pytest.raises(InvalidInputError):
        list(engine.stream_reply(None, message, OPTIONS))  # type: ignore[arg-type]

    assert db.list_conversations() == []


def test_missing_model_is_rejected(db: Database) -> None:
    engine, _ = make_engine(db)
    options = ChatOptions(model="", system_prompt="", temperature=0.7, max_tokens=10)

    with pytest.raises(InvalidInputError, match="No model selected"):
        list(engine.stream_reply(None, "Hello", options))
    assert db.list_conversations() == []


def test_regenerate_requires_a_user_message(db: Database) -> None:
    engine, client = make_engine(db)
    conversation = db.create_conversation("Empty", OPTIONS.model)
    db.add_message(conversation.id, "assistant", "orphan")

    with pytest.raises(InvalidInputError, match="previous user message"):
        list(
            engine.stream_reply(
                conversation.id, None, OPTIONS, regenerate=True
            )
        )
    assert client.calls == []
    assert [m.content for m in db.get_messages(conversation.id)] == ["orphan"]


def test_history_limit_trims_old_turns_from_payload(db: Database) -> None:
    engine, client = make_engine(db, chunks=("a1",), history_limit=1)
    first = list(engine.stream_reply(None, "u1", OPTIONS))
    conversation_id = first[-1].conversation_id

    client.chunks = ["a2"]
    list(engine.stream_reply(conversation_id, "u2", OPTIONS))

    assert client.calls[1]["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "u2"},
    ]


def test_preferences_roundtrip_and_defaults(db: Database) -> None:
    engine, _ = make_engine(db)
    assert engine.load_preferences(OPTIONS) == OPTIONS

    engine.save_preferences(
        ChatOptions(
            model="llama3.2:3b",
            system_prompt="Terse.",
            temperature=1.1,
            max_tokens=99,
        )
    )

    assert engine.load_preferences(OPTIONS) == ChatOptions(
        model="llama3.2:3b",
        system_prompt="Terse.",
        temperature=1.1,
        max_tokens=99,
    )


def test_corrupt_preferences_fall_back_to_defaults(db: Database) -> None:
    engine, _ = make_engine(db)
    db.set_preferences(
        {"temperature": "not-a-number", "max_tokens": "-4", "model": ""}
    )

    loaded = engine.load_preferences(OPTIONS)

    assert loaded.model == OPTIONS.model
    assert loaded.temperature == OPTIONS.temperature
    assert loaded.max_tokens == OPTIONS.max_tokens


def test_clear_rename_and_delete(db: Database) -> None:
    engine, _ = make_engine(db, chunks=("answer",))
    first = list(engine.stream_reply(None, "question", OPTIONS))
    conversation_id = first[-1].conversation_id

    assert engine.rename_conversation(conversation_id, "Renamed") is True
    assert engine.clear_conversation(conversation_id) == 2
    assert engine.get_messages(conversation_id) == []
    assert engine.delete_conversation(conversation_id) is True
    assert engine.get_conversation(conversation_id) is None


def test_reply_helper_raises_on_stream_error(db: Database) -> None:
    engine, _ = make_engine(db, chunks=(), error=OllamaError("boom"))

    with pytest.raises(OllamaError, match="boom"):
        engine.reply(None, "Hello", OPTIONS)


def test_stop_requested_is_forwarded_to_client(db: Database) -> None:
    class StopAwareClient(FakeOllama):
        def __init__(self) -> None:
            super().__init__(chunks=("never",))
            self.stop_seen: Any = None

        def stream_chat(self, model, messages, temperature=0.7, max_tokens=1024, stop_requested=None):
            self.stop_seen = stop_requested
            for chunk in self.chunks:
                if stop_requested is None or not stop_requested():
                    yield chunk

    client = StopAwareClient()
    engine = ChatEngine(db, client)
    flag = {"stop": True}

    updates = list(
        engine.stream_reply(
            None, "Hello", OPTIONS, stop_requested=lambda: flag["stop"]
        )
    )

    assert client.stop_seen is not None
    # The fake yields nothing, so the engine reports an empty reply.
    assert "empty response" in updates[-1].error
