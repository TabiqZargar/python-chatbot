"""Requesty through its OpenAI-compatible endpoint: config, requests, parsing, errors.

Requesty is an OpenAI-compatible router (one base URL + one key for many models),
so it reuses OpenAICompatibleProvider — no Requesty-specific client exists. All
HTTP goes to the local mock server from conftest.py: no network, no real API key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from chatbot.config import load_settings
from chatbot.errors import ProviderError
from chatbot.history import ConversationHistory
from chatbot.providers import build_provider
from chatbot.providers.openai_compatible import OpenAICompatibleProvider

REQUESTY_BASE_URL = "https://router.requesty.ai/v1"
REQUESTY_MODEL = "google/gemma-4-31b-it"
REQUESTY_KEY = "test-requesty-key"


def requesty_env(**overrides: str) -> dict[str, str]:
    env = {
        "CHATBOT_PROVIDER": "openai",
        "OPENAI_BASE_URL": REQUESTY_BASE_URL,
        "OPENAI_MODEL": REQUESTY_MODEL,
        "OPENAI_API_KEY": REQUESTY_KEY,
    }
    env.update(overrides)
    return env


def _mock_env(server: Any, **overrides: str) -> dict[str, str]:
    host, port = server.server_address
    env = requesty_env(OPENAI_BASE_URL=f"http://{host}:{port}/v1")
    env.update(overrides)
    return env


# --- configuration ---


def test_requesty_env_loads_settings() -> None:
    settings = load_settings(environ=requesty_env(), dotenv_path=None)

    assert settings.provider == "openai"
    assert settings.openai_api_key == REQUESTY_KEY
    assert settings.openai_base_url == REQUESTY_BASE_URL
    assert settings.openai_model == REQUESTY_MODEL


def test_requesty_key_read_from_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

    settings = load_settings(environ={}, dotenv_path=env_file)

    assert settings.openai_api_key == "file-key"


def test_requesty_base_url_trailing_slash_is_normalized() -> None:
    settings = load_settings(
        environ=requesty_env(OPENAI_BASE_URL=REQUESTY_BASE_URL + "/"), dotenv_path=None
    )

    assert settings.openai_base_url == REQUESTY_BASE_URL
    provider = build_provider(settings)
    assert provider.endpoint == f"{REQUESTY_BASE_URL}/chat/completions"


def test_requesty_settings_build_the_reused_openai_provider() -> None:
    settings = load_settings(environ=requesty_env(), dotenv_path=None)

    provider = build_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.endpoint == f"{REQUESTY_BASE_URL}/chat/completions"
    assert REQUESTY_MODEL in provider.name


def test_offline_remains_the_default_without_requesty_config() -> None:
    settings = load_settings(environ={}, dotenv_path=None)

    assert settings.provider == "offline"
    assert settings.openai_api_key == ""


# --- request formatting (mocked HTTP) ---


def test_requesty_request_formatting(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {
        "choices": [{"message": {"role": "assistant", "content": "Gemma says hi"}}]
    }
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))
    history = ConversationHistory(limit=50)
    history.add("user", "prior")
    history.add("assistant", "reply")

    reply = provider.send("hello gemma", history.messages)

    assert reply == "Gemma says hi"
    request = server.last_request
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == f"Bearer {REQUESTY_KEY}"
    assert request["payload"]["model"] == REQUESTY_MODEL
    assert request["payload"]["messages"] == [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "hello gemma"},
    ]


# --- response parsing (mocked HTTP) ---


def test_requesty_openai_shaped_response_is_parsed(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1760000000,
        "model": REQUESTY_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "  42.  "},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    }
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    reply = provider.send("the answer?", ())

    assert reply == "42."


# --- error handling (mocked HTTP) ---


def test_requesty_invalid_key_reports_401_with_hint(openai_server: Any) -> None:
    server = openai_server
    server.reply_status = 401
    server.reply_body = {"error": {"message": "invalid api key"}}
    provider = build_provider(
        load_settings(environ=_mock_env(server, OPENAI_API_KEY="bad"), dotenv_path=None)
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    message = str(excinfo.value)
    assert "401" in message
    assert "invalid api key" in message
    assert "OPENAI_API_KEY" in message


def test_requesty_unexpected_response_shape_is_a_provider_error(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {"unexpected": "shape"}
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "no choices" in str(excinfo.value)


def test_requesty_non_json_response_is_a_provider_error(openai_server: Any) -> None:
    server = openai_server
    server.reply_raw = b"<html>upstream error</html>"
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "invalid JSON" in str(excinfo.value)
