"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from chatbot.config import (
    DEFAULT_HISTORY_LIMIT,
    SUPPORTED_PROVIDERS,
    load_settings,
    parse_dotenv,
)
from chatbot.errors import ConfigError


def test_defaults_when_nothing_is_set() -> None:
    settings = load_settings(environ={}, dotenv_path=None)

    assert settings.provider == "offline"
    assert settings.history_limit == DEFAULT_HISTORY_LIMIT
    assert settings.timeout > 0


def test_environment_overrides_defaults() -> None:
    settings = load_settings(
        environ={
            "CHATBOT_PROVIDER": "openai",
            "OPENAI_MODEL": "llama3",
            "OPENAI_BASE_URL": "http://localhost:11434/v1",
            "CHATBOT_HISTORY_LIMIT": "5",
            "CHATBOT_TIMEOUT": "2.5",
        },
        dotenv_path=None,
    )

    assert settings.provider == "openai"
    assert settings.openai_model == "llama3"
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.history_limit == 5
    assert settings.timeout == 2.5


def test_openai_api_key_is_optional_for_local_servers() -> None:
    settings = load_settings(environ={"CHATBOT_PROVIDER": "openai"}, dotenv_path=None)

    assert settings.provider == "openai"
    assert settings.openai_api_key == ""


def test_invalid_provider_rejected() -> None:
    with pytest.raises(ConfigError) as excinfo:
        load_settings(environ={"CHATBOT_PROVIDER": "gpt5"}, dotenv_path=None)

    assert "CHATBOT_PROVIDER" in str(excinfo.value)
    for provider in SUPPORTED_PROVIDERS:
        assert provider in str(excinfo.value)


@pytest.mark.parametrize("value", ["0", "-3", "abc"])
def test_invalid_history_limit_rejected(value: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(environ={"CHATBOT_HISTORY_LIMIT": value}, dotenv_path=None)


@pytest.mark.parametrize("value", ["0", "-1", "soon"])
def test_invalid_timeout_rejected(value: str) -> None:
    with pytest.raises(ConfigError):
        load_settings(environ={"CHATBOT_TIMEOUT": value}, dotenv_path=None)


def test_openai_requires_valid_base_url() -> None:
    with pytest.raises(ConfigError):
        load_settings(
            environ={"CHATBOT_PROVIDER": "openai", "OPENAI_BASE_URL": "ftp://nope"},
            dotenv_path=None,
        )


def test_openai_requires_model() -> None:
    with pytest.raises(ConfigError):
        load_settings(
            environ={"CHATBOT_PROVIDER": "openai", "OPENAI_MODEL": ""},
            dotenv_path=None,
        )


def test_parse_dotenv_handles_comments_quotes_and_export(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "# a comment",
                "CHATBOT_PROVIDER=openai",
                'OPENAI_MODEL="gpt-4o-mini"',
                "export OPENAI_BASE_URL='http://localhost:1234/v1'",
                "",
                "NOT_A_PAIR",
            ]
        ),
        encoding="utf-8",
    )

    values = parse_dotenv(env_file)

    assert values["CHATBOT_PROVIDER"] == "openai"
    assert values["OPENAI_MODEL"] == "gpt-4o-mini"
    assert values["OPENAI_BASE_URL"] == "http://localhost:1234/v1"
    assert "NOT_A_PAIR" not in values


def test_dotenv_file_is_used_when_environment_is_empty(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CHATBOT_PROVIDER=openai\nOPENAI_MODEL=local-model\n", encoding="utf-8")

    settings = load_settings(environ={}, dotenv_path=env_file)

    assert settings.provider == "openai"
    assert settings.openai_model == "local-model"


def test_environment_beats_dotenv(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("CHATBOT_PROVIDER=openai\n", encoding="utf-8")

    settings = load_settings(environ={"CHATBOT_PROVIDER": "offline"}, dotenv_path=env_file)

    assert settings.provider == "offline"


def test_missing_dotenv_file_is_ignored(tmp_path: Path) -> None:
    assert parse_dotenv(tmp_path / "does-not-exist") == {}
    settings = load_settings(environ={}, dotenv_path=tmp_path / "does-not-exist")
    assert settings.provider == "offline"
