"""Pure conversation helpers: validation, titles, and model payloads."""

from __future__ import annotations

from collections.abc import Sequence

from chatbot.errors import InvalidInputError
from database.models import ChatMessage

__all__ = [
    "DEFAULT_MAX_MESSAGE_LENGTH",
    "TITLE_MAX_LENGTH",
    "build_model_messages",
    "make_title",
    "trim_messages",
    "validate_message",
]

DEFAULT_MAX_MESSAGE_LENGTH = 8000
TITLE_MAX_LENGTH = 60


def validate_message(
    message: object, max_length: int = DEFAULT_MAX_MESSAGE_LENGTH
) -> str:
    """Return a cleaned user message or raise :class:`InvalidInputError`."""
    if not isinstance(message, str):
        raise InvalidInputError("Message must be text.")
    cleaned = message.strip()
    if not cleaned:
        raise InvalidInputError("Type a message before sending.")
    if len(cleaned) > max_length:
        raise InvalidInputError(
            f"Message is too long ({len(cleaned)} characters; limit is {max_length})."
        )
    return cleaned


def make_title(message: str, max_length: int = TITLE_MAX_LENGTH) -> str:
    """Turn the first user message into a short, single-line conversation title."""
    first_line = " ".join(message.strip().split())
    if not first_line:
        return "New conversation"
    if len(first_line) <= max_length:
        return first_line
    shortened = first_line[: max_length - 1].rstrip()
    if " " in shortened:
        shortened = shortened.rsplit(" ", 1)[0]
    return f"{shortened}…"


def trim_messages(
    messages: Sequence[ChatMessage], history_limit: int
) -> list[ChatMessage]:
    """Cap retained context at ``history_limit`` user turns.

    The oldest turns are removed first, and any leading assistant message is
    dropped so the model payload always starts with a user turn.
    """
    kept = list(messages)
    while sum(1 for item in kept if item.role == "user") > history_limit and kept:
        kept.pop(0)
    while kept and kept[0].role != "user":
        kept.pop(0)
    return kept


def build_model_messages(
    messages: Sequence[ChatMessage],
    system_prompt: str = "",
    history_limit: int = 50,
) -> list[dict[str, str]]:
    """Build the ``/api/chat`` message payload from stored history."""
    payload: list[dict[str, str]] = []
    if system_prompt:
        payload.append({"role": "system", "content": system_prompt})
    payload.extend(
        {"role": item.role, "content": item.content}
        for item in trim_messages(messages, history_limit)
    )
    return payload
