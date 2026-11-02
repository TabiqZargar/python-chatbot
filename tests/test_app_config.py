"""Tests for the new Gradio application's validated configuration."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.errors import ConfigError
from config import (
    DEFAULT_DATABASE_PATH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_TIMEOUT,
    AppSettings,
    load_settings,
)


def test_defaults_need_no_environment() -> None:
    settings = load_settings(environ={}, dotenv_path=None)

    assert settings == AppSettings()
    assert settings.ollama_base_url == DEFAULT_OLLAMA_BASE_URL
    assert settings.default_model == DEFAULT_OLLAMA_MODEL
    assert settings.ollama_timeout == DEFAULT_OLLAMA_TIMEOUT
    assert settings.database_path == DEFAULT_DATABASE_PATH
    assert settings.default_max_tokens == DEFAULT_MAX_TOKENS
    assert settings.share is False


def test_environment_overrides_and_url_normalisation() -> None:
    settings = load_settings(
        environ={
            "OLLAMA_BASE_URL": "http://localhost:11434///",
            "OLLAMA_MODEL": " llama3.2:3b ",
            "OLLAMA_TIMEOUT": "15.5",
            "CHATBOT_DB_PATH": "custom/chat.db",
            "CHATBOT_HISTORY_LIMIT": "7",
            "CHATBOT_MAX_MESSAGE_LENGTH": "500",
            "CHATBOT_TEMPERATURE": "0",
            "CHATBOT_MAX_TOKENS": "64",
            "CHATBOT_UI_HOST": "0.0.0.0",
            "CHATBOT_UI_PORT": "8080",
            "CHATBOT_UI_SHARE": "yes",
        },
        dotenv_path=None,
    )

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.default_model == "llama3.2:3b"
    assert settings.ollama_timeout == 15.5
    assert settings.database_path == "custom/chat.db"
    assert settings.history_limit == 7
    assert settings.max_message_length == 500
    assert settings.default_temperature == 0.0
    assert settings.default_max_tokens == 64
    assert settings.host == "0.0.0.0"
    assert settings.port == 8080
    assert settings.share is True


def test_dotenv_is_used_when_environment_is_absent(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# comment",
                "export OLLAMA_BASE_URL='http://127.0.0.1:9999'",
                'OLLAMA_MODEL="qwen2.5:7b"',
                "CHATBOT_TEMPERATURE=1.5",
                "CHATBOT_SYSTEM_PROMPT=  keep spacing  ",
            ]
        ),
        encoding="utf-8",
    )

    settings = load_settings(environ={}, dotenv_path=env_file)

    assert settings.ollama_base_url == "http://127.0.0.1:9999"
    assert settings.default_model == "qwen2.5:7b"
    assert settings.default_temperature == 1.5
    assert settings.default_system_prompt == "keep spacing"


def test_real_environment_beats_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OLLAMA_MODEL=from-file\nOLLAMA_BASE_URL=http://127.0.0.1:1111\n",
        encoding="utf-8",
    )

    settings = load_settings(
        environ={"OLLAMA_MODEL": "from-env"}, dotenv_path=env_file
    )

    assert settings.default_model == "from-env"
    assert settings.ollama_base_url == "http://127.0.0.1:1111"


def test_legacy_provider_settings_are_ignored() -> None:
    settings = load_settings(
        environ={
            "CHATBOT_PROVIDER": "offline",
            "OPENAI_API_KEY": "secret",
            "OPENAI_MODEL": "gpt-4o",
        },
        dotenv_path=None,
    )

    assert settings.default_model == DEFAULT_OLLAMA_MODEL
    assert not hasattr(settings, "provider")


def test_missing_dotenv_is_fine(tmp_path: Path) -> None:
    settings = load_settings(
        environ={}, dotenv_path=tmp_path / "does-not-exist.env"
    )
    assert settings == AppSettings()


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("OLLAMA_BASE_URL", "localhost:11434", "http://"),
        ("OLLAMA_BASE_URL", "", "http://"),
        ("OLLAMA_MODEL", "   ", "must not be empty"),
        ("OLLAMA_TIMEOUT", "0", "greater than 0"),
        ("OLLAMA_TIMEOUT", "soon", "must be a number"),
        ("CHATBOT_HISTORY_LIMIT", "0", "at least 1"),
        ("CHATBOT_HISTORY_LIMIT", "many", "must be an integer"),
        ("CHATBOT_MAX_MESSAGE_LENGTH", "-1", "at least 1"),
        ("CHATBOT_TEMPERATURE", "-0.1", "between 0 and 2"),
        ("CHATBOT_TEMPERATURE", "2.1", "between 0 and 2"),
        ("CHATBOT_TEMPERATURE", "hot", "must be a number"),
        ("CHATBOT_MAX_TOKENS", "0", "at least 1"),
        ("CHATBOT_DB_PATH", "  ", "must not be empty"),
        ("CHATBOT_UI_HOST", "", "must not be empty"),
        ("CHATBOT_UI_PORT", "0", "at least 1"),
        ("CHATBOT_UI_PORT", "65536", "at most 65535"),
        ("CHATBOT_UI_PORT", "eighty", "must be an integer"),
        ("CHATBOT_UI_SHARE", "maybe", "true or false"),
    ],
)
def test_invalid_values_raise_config_error(
    key: str, value: str, match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        load_settings(environ={key: value}, dotenv_path=None)


@pytest.mark.parametrize("value", ["0", "0.0", "2", "2.0"])
def test_temperature_boundaries_are_valid(value: str) -> None:
    settings = load_settings(
        environ={"CHATBOT_TEMPERATURE": value}, dotenv_path=None
    )
    assert settings.default_temperature == float(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1", True),
        ("TRUE", True),
        ("on", True),
        ("0", False),
        ("false", False),
        ("off", False),
        ("", False),
    ],
)
def test_share_boolean_parsing(value: str, expected: bool) -> None:
    settings = load_settings(
        environ={"CHATBOT_UI_SHARE": value}, dotenv_path=None
    )
    assert settings.share is expected
