"""Data models shared by the database, engine, and UI layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping

__all__ = [
    "ChatMessage",
    "Conversation",
    "UserPreferences",
    "utc_now_iso",
    "format_timestamp",
]


def utc_now_iso() -> str:
    """Current UTC time as a sortable ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass(frozen=True)
class Conversation:
    """A stored conversation and the settings it was last used with."""

    id: int
    title: str
    model: str
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    created_at: str = ""
    updated_at: str = ""
    message_count: int = 0


@dataclass(frozen=True)
class ChatMessage:
    """One persisted user or assistant message."""

    id: int | None
    conversation_id: int
    role: str
    content: str
    created_at: str = ""


@dataclass(frozen=True)
class UserPreferences:
    """Defaults restored when the app starts."""

    model: str = "qwen2.5:3b"
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    extra: Mapping[str, str] = field(default_factory=dict)

    def to_mapping(self) -> dict[str, str]:
        values = {
            "model": self.model,
            "system_prompt": self.system_prompt,
            "temperature": str(self.temperature),
            "max_tokens": str(self.max_tokens),
        }
        values.update(self.extra)
        return values


def format_timestamp(value: str) -> str:
    """Format an ISO timestamp for the sidebar, falling back to the raw value."""
    if not value:
        return ""
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return value
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone().strftime("%Y-%m-%d %H:%M")
