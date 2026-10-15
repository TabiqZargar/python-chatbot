"""Configuration: load settings from environment variables and an optional .env file.

Rules:
- Real environment variables always win over values in .env.
- Secrets live only in the environment / .env — never in source code.
- Invalid values raise ConfigError with a message that says what to fix.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from chatbot.errors import ConfigError

__all__ = ["Settings", "load_settings", "parse_dotenv", "SUPPORTED_PROVIDERS"]

SUPPORTED_PROVIDERS = ("offline", "openai")

DEFAULT_DOTENV = Path(".env")

DEFAULT_PROVIDER = "offline"
DEFAULT_HISTORY_LIMIT = 50
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30.0


def parse_dotenv(path: Path | str) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file into a dict.

    Supports blank lines, # comments, `export` prefixes and quoted values.
    Returns an empty dict if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    """Validated runtime configuration."""

    provider: str = DEFAULT_PROVIDER
    history_limit: int = DEFAULT_HISTORY_LIMIT
    openai_api_key: str = ""
    openai_base_url: str = DEFAULT_BASE_URL
    openai_model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT


def load_settings(
    environ: Mapping[str, str] | None = None,
    dotenv_path: Path | str | None = DEFAULT_DOTENV,
) -> Settings:
    """Build Settings from ``environ`` (defaults to os.environ) plus a .env file.

    Raises ConfigError when a value is missing or invalid.

    For the API key, OPENAI_API_KEY takes precedence over GEMINI_API_KEY when
    both are set.
    """
    source: Mapping[str, str] = os.environ if environ is None else environ
    file_values = parse_dotenv(dotenv_path) if dotenv_path is not None else {}

    def raw(key: str, default: str = "") -> str:
        if key in source:
            return str(source[key])
        return file_values.get(key, default)

    provider = raw("CHATBOT_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"CHATBOT_PROVIDER must be one of {SUPPORTED_PROVIDERS}, "
            f"got {provider!r}"
        )

    history_limit = _parse_positive_int(raw("CHATBOT_HISTORY_LIMIT"), DEFAULT_HISTORY_LIMIT, "CHATBOT_HISTORY_LIMIT")
    timeout = _parse_positive_float(raw("CHATBOT_TIMEOUT"), DEFAULT_TIMEOUT, "CHATBOT_TIMEOUT")

    base_url = raw("OPENAI_BASE_URL", DEFAULT_BASE_URL).strip().rstrip("/")
    model = raw("OPENAI_MODEL", DEFAULT_MODEL).strip()
    api_key = raw("OPENAI_API_KEY").strip() or raw("GEMINI_API_KEY").strip()

    if provider == "openai":
        if not base_url:
            raise ConfigError("OPENAI_BASE_URL must not be empty when provider is 'openai'")
        if not base_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"OPENAI_BASE_URL must start with http:// or https://, got {base_url!r}"
            )
        if not model:
            raise ConfigError("OPENAI_MODEL must not be empty when provider is 'openai'")

    return Settings(
        provider=provider,
        history_limit=history_limit,
        openai_api_key=api_key,
        openai_base_url=base_url,
        openai_model=model,
        timeout=timeout,
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
