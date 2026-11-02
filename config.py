"""Application configuration for the local Gradio chatbot.

Settings are read from real environment variables first, then from an optional
``.env`` file. Nothing here talks to a cloud service: the only network endpoint
is the local Ollama server configured by ``OLLAMA_BASE_URL``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chatbot.config import parse_dotenv
from chatbot.errors import ConfigError

__all__ = [
    "AppSettings",
    "DEFAULT_OLLAMA_BASE_URL",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_DATABASE_PATH",
    "load_settings",
]

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"
DEFAULT_OLLAMA_TIMEOUT = 120.0
DEFAULT_DATABASE_PATH = "chatbot.db"
DEFAULT_HISTORY_LIMIT = 30
DEFAULT_MAX_MESSAGE_LENGTH = 8000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1024
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate assistant. "
    "Answer clearly, use Markdown when it improves readability, "
    "and say when you are unsure instead of inventing facts."
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_DOTENV = Path(".env")


@dataclass(frozen=True)
class AppSettings:
    """Validated settings for the local chatbot application."""

    ollama_base_url: str = DEFAULT_OLLAMA_BASE_URL
    default_model: str = DEFAULT_OLLAMA_MODEL
    ollama_timeout: float = DEFAULT_OLLAMA_TIMEOUT
    database_path: str = DEFAULT_DATABASE_PATH
    history_limit: int = DEFAULT_HISTORY_LIMIT
    max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH
    default_temperature: float = DEFAULT_TEMPERATURE
    default_max_tokens: int = DEFAULT_MAX_TOKENS
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    share: bool = False


def load_settings(
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | str | None = DEFAULT_DOTENV,
) -> AppSettings:
    """Build :class:`AppSettings`, raising :class:`ConfigError` on bad values."""
    source: Mapping[str, str] = os.environ if environ is None else environ
    file_values = parse_dotenv(dotenv_path) if dotenv_path is not None else {}

    def raw(key: str, default: str = "") -> str:
        if key in source:
            return str(source[key])
        return file_values.get(key, default)

    base_url = raw("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise ConfigError(
            "OLLAMA_BASE_URL must start with http:// or https://, "
            f"got {base_url!r}"
        )

    default_model = raw("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL).strip()
    if not default_model:
        raise ConfigError("OLLAMA_MODEL must not be empty")

    ollama_timeout = _parse_positive_float(
        raw("OLLAMA_TIMEOUT"), DEFAULT_OLLAMA_TIMEOUT, "OLLAMA_TIMEOUT"
    )
    history_limit = _parse_positive_int(
        raw("CHATBOT_HISTORY_LIMIT"), DEFAULT_HISTORY_LIMIT, "CHATBOT_HISTORY_LIMIT"
    )
    max_message_length = _parse_positive_int(
        raw("CHATBOT_MAX_MESSAGE_LENGTH"),
        DEFAULT_MAX_MESSAGE_LENGTH,
        "CHATBOT_MAX_MESSAGE_LENGTH",
    )
    max_tokens = _parse_positive_int(
        raw("CHATBOT_MAX_TOKENS"), DEFAULT_MAX_TOKENS, "CHATBOT_MAX_TOKENS"
    )
    temperature = _parse_temperature(
        raw("CHATBOT_TEMPERATURE"), DEFAULT_TEMPERATURE, "CHATBOT_TEMPERATURE"
    )

    database_path = raw("CHATBOT_DB_PATH", DEFAULT_DATABASE_PATH).strip()
    if not database_path:
        raise ConfigError("CHATBOT_DB_PATH must not be empty")

    host = raw("CHATBOT_UI_HOST", DEFAULT_HOST).strip()
    if not host:
        raise ConfigError("CHATBOT_UI_HOST must not be empty")

    port = _parse_positive_int(
        raw("CHATBOT_UI_PORT"), DEFAULT_PORT, "CHATBOT_UI_PORT"
    )
    if port > 65535:
        raise ConfigError(f"CHATBOT_UI_PORT must be at most 65535, got {port}")

    share = _parse_bool(raw("CHATBOT_UI_SHARE"), False, "CHATBOT_UI_SHARE")

    # Environment values are loaded verbatim; unquoted .env whitespace is trimmed.
    system_prompt = raw("CHATBOT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

    return AppSettings(
        ollama_base_url=base_url,
        default_model=default_model,
        ollama_timeout=ollama_timeout,
        database_path=database_path,
        history_limit=history_limit,
        max_message_length=max_message_length,
        default_temperature=temperature,
        default_max_tokens=max_tokens,
        default_system_prompt=system_prompt,
        host=host,
        port=port,
        share=share,
    )


def _parse_positive_int(raw: str, default: int, name: str) -> int:
    text = raw.strip()
    if not text:
        return default
    try:
        value = int(text)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ConfigError(f"{name} must be at least 1, got {value}")
    return value


def _parse_positive_float(raw: str, default: float, name: str) -> float:
    text = raw.strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise ConfigError(f"{name} must be greater than 0, got {value}")
    return value


def _parse_temperature(raw: str, default: float, name: str) -> float:
    text = raw.strip()
    if not text:
        return default
    try:
        value = float(text)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc
    if not 0 <= value <= 2:
        raise ConfigError(f"{name} must be between 0 and 2, got {value}")
    return value


def _parse_bool(raw: str, default: bool, name: str) -> bool:
    text = raw.strip().lower()
    if not text:
        return default
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")
