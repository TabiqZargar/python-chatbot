"""SQLite persistence for conversations, messages, and preferences.

Every query is parameterized. A single connection is shared between Gradio's
worker threads and serialized with a re-entrant lock; WAL mode keeps readers
from blocking the writer.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Mapping

from chatbot.errors import InvalidInputError, StorageError
from database.models import ChatMessage, Conversation, utc_now_iso

__all__ = ["Database"]

VALID_ROLES = ("user", "assistant")
MAX_TITLE_LENGTH = 200
MAX_CONTENT_LENGTH = 1_000_000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT NOT NULL DEFAULT '',
    temperature REAL NOT NULL DEFAULT 0.7,
    max_tokens INTEGER NOT NULL DEFAULT 1024,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS preferences (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class Database:
    """Thread-safe SQLite repository for the chatbot."""

    def __init__(self, path: str | Path = "chatbot.db") -> None:
        self._path = str(path)
        if self._path != ":memory:":
            parent = Path(self._path).expanduser().resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(
                self._path, check_same_thread=False, timeout=10.0
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA busy_timeout = 10000")
            if self._path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            with self._conn:
                self._conn.executescript(_SCHEMA)
        except sqlite3.Error as exc:
            raise StorageError(f"Could not open the database: {exc}") from exc

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def path(self) -> str:
        return self._path

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # --- conversations -------------------------------------------------

    def create_conversation(
        self,
        title: str,
        model: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> Conversation:
        clean_title = _validate_title(title)
        clean_model = _validate_model(model)
        clean_prompt = _validate_prompt(system_prompt)
        _validate_temperature(temperature)
        _validate_max_tokens(max_tokens)
        now = utc_now_iso()
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    """
                    INSERT INTO conversations
                        (title, model, system_prompt, temperature, max_tokens,
                         created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_title,
                        clean_model,
                        clean_prompt,
                        float(temperature),
                        int(max_tokens),
                        now,
                        now,
                    ),
                )
                conversation_id = int(cursor.lastrowid)
        except sqlite3.Error as exc:
            raise StorageError(f"Could not create the conversation: {exc}") from exc
        return Conversation(
            id=conversation_id,
            title=clean_title,
            model=clean_model,
            system_prompt=clean_prompt,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            created_at=now,
            updated_at=now,
            message_count=0,
        )

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        try:
            with self._lock:
                row = self._conn.execute(
                    """
                    SELECT c.*, COUNT(m.id) AS message_count
                    FROM conversations AS c
                    LEFT JOIN messages AS m ON m.conversation_id = c.id
                    WHERE c.id = ?
                    GROUP BY c.id
                    """,
                    (_validate_id(conversation_id),),
                ).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read the conversation: {exc}") from exc
        return _conversation_from_row(row) if row else None

    def list_conversations(self) -> list[Conversation]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT c.*, COUNT(m.id) AS message_count
                    FROM conversations AS c
                    LEFT JOIN messages AS m ON m.conversation_id = c.id
                    GROUP BY c.id
                    ORDER BY c.updated_at DESC, c.id DESC
                    """
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not list conversations: {exc}") from exc
        return [_conversation_from_row(row) for row in rows]

    def rename_conversation(self, conversation_id: int, title: str) -> bool:
        clean_title = _validate_title(title)
        now = utc_now_iso()
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                    (clean_title, now, _validate_id(conversation_id)),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not rename the conversation: {exc}") from exc
        return cursor.rowcount > 0

    def update_conversation_settings(
        self,
        conversation_id: int,
        model: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> bool:
        clean_model = _validate_model(model)
        clean_prompt = _validate_prompt(system_prompt)
        _validate_temperature(temperature)
        _validate_max_tokens(max_tokens)
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    """
                    UPDATE conversations
                    SET model = ?, system_prompt = ?, temperature = ?, max_tokens = ?
                    WHERE id = ?
                    """,
                    (
                        clean_model,
                        clean_prompt,
                        float(temperature),
                        int(max_tokens),
                        _validate_id(conversation_id),
                    ),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not update conversation settings: {exc}") from exc
        return cursor.rowcount > 0

    def touch_conversation(self, conversation_id: int) -> bool:
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (utc_now_iso(), _validate_id(conversation_id)),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not update the conversation: {exc}") from exc
        return cursor.rowcount > 0

    def delete_conversation(self, conversation_id: int) -> bool:
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM conversations WHERE id = ?",
                    (_validate_id(conversation_id),),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not delete the conversation: {exc}") from exc
        return cursor.rowcount > 0

    # --- messages ------------------------------------------------------

    def add_message(
        self, conversation_id: int, role: str, content: str
    ) -> ChatMessage:
        clean_role = _validate_role(role)
        clean_content = _validate_content(content)
        now = utc_now_iso()
        try:
            with self._lock, self._conn:
                exists = self._conn.execute(
                    "SELECT 1 FROM conversations WHERE id = ?",
                    (_validate_id(conversation_id),),
                ).fetchone()
                if not exists:
                    raise StorageError(
                        f"Conversation {conversation_id} does not exist."
                    )
                cursor = self._conn.execute(
                    """
                    INSERT INTO messages
                        (conversation_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (conversation_id, clean_role, clean_content, now),
                )
                message_id = int(cursor.lastrowid)
                self._conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not save the message: {exc}") from exc
        return ChatMessage(
            id=message_id,
            conversation_id=conversation_id,
            role=clean_role,
            content=clean_content,
            created_at=now,
        )

    def get_messages(self, conversation_id: int) -> list[ChatMessage]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id ASC
                    """,
                    (_validate_id(conversation_id),),
                ).fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read messages: {exc}") from exc
        return [_message_from_row(row) for row in rows]

    def delete_message(self, message_id: int) -> bool:
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM messages WHERE id = ?", (_validate_id(message_id),)
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not delete the message: {exc}") from exc
        return cursor.rowcount > 0

    def remove_last_assistant_message(self, conversation_id: int) -> ChatMessage | None:
        """Delete and return the latest message only when it is an assistant reply."""
        try:
            with self._lock, self._conn:
                row = self._conn.execute(
                    """
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (_validate_id(conversation_id),),
                ).fetchone()
                if row is None or row["role"] != "assistant":
                    return None
                self._conn.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
                return _message_from_row(row)
        except sqlite3.Error as exc:
            raise StorageError(
                f"Could not remove the previous assistant message: {exc}"
            ) from exc

    def clear_messages(self, conversation_id: int) -> int:
        try:
            with self._lock, self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM messages WHERE conversation_id = ?",
                    (_validate_id(conversation_id),),
                )
                self._conn.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (utc_now_iso(), conversation_id),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not clear the conversation: {exc}") from exc
        return cursor.rowcount

    # --- preferences ---------------------------------------------------

    def get_preferences(self) -> dict[str, str]:
        try:
            with self._lock:
                rows = self._conn.execute("SELECT key, value FROM preferences").fetchall()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read preferences: {exc}") from exc
        return {str(row["key"]): str(row["value"]) for row in rows}

    def set_preferences(self, values: Mapping[str, str]) -> None:
        clean: dict[str, str] = {}
        for key, value in values.items():
            if not isinstance(key, str) or not key.strip():
                raise InvalidInputError("Preference keys must be non-empty strings.")
            if not isinstance(value, str):
                raise InvalidInputError(
                    f"Preference {key!r} must have a text value."
                )
            clean[key] = value
        if not clean:
            return
        try:
            with self._lock, self._conn:
                self._conn.executemany(
                    """
                    INSERT INTO preferences (key, value)
                    VALUES (?, ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    list(clean.items()),
                )
        except sqlite3.Error as exc:
            raise StorageError(f"Could not save preferences: {exc}") from exc


def _conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation(
        id=int(row["id"]),
        title=str(row["title"]),
        model=str(row["model"]),
        system_prompt=str(row["system_prompt"]),
        temperature=float(row["temperature"]),
        max_tokens=int(row["max_tokens"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        message_count=int(row["message_count"]),
    )


def _message_from_row(row: sqlite3.Row) -> ChatMessage:
    return ChatMessage(
        id=int(row["id"]),
        conversation_id=int(row["conversation_id"]),
        role=str(row["role"]),
        content=str(row["content"]),
        created_at=str(row["created_at"]),
    )


def _validate_id(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InvalidInputError(f"Invalid id: {value!r}")
    return value


def _validate_title(title: object) -> str:
    if not isinstance(title, str) or not title.strip():
        raise InvalidInputError("Conversation title must not be empty.")
    clean = title.strip()
    if len(clean) > MAX_TITLE_LENGTH:
        raise InvalidInputError(
            f"Conversation title is too long (limit is {MAX_TITLE_LENGTH})."
        )
    return clean


def _validate_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        raise InvalidInputError("Choose a model before sending a message.")
    return model.strip()


def _validate_prompt(prompt: object) -> str:
    if not isinstance(prompt, str):
        raise InvalidInputError("System prompt must be text.")
    return prompt


def _validate_temperature(value: object) -> None:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not 0 <= float(value) <= 2
    ):
        raise InvalidInputError("Temperature must be between 0 and 2.")


def _validate_max_tokens(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InvalidInputError("Maximum tokens must be a positive integer.")


def _validate_role(role: object) -> str:
    if role not in VALID_ROLES:
        raise InvalidInputError(f"Message role must be one of {VALID_ROLES}.")
    return str(role)


def _validate_content(content: object) -> str:
    if not isinstance(content, str) or not content.strip():
        raise InvalidInputError("Message content must not be empty.")
    if len(content) > MAX_CONTENT_LENGTH:
        raise InvalidInputError(
            f"Message content is too long (limit is {MAX_CONTENT_LENGTH})."
        )
    return content
