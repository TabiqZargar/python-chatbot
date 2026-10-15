"""Gemini through its OpenAI-compatible endpoint: config, requests, parsing, errors.

Reuses OpenAICompatibleProvider — no Gemini-specific client exists. All HTTP
goes to the local mock server from conftest.py: no network, no real API key.
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

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_KEY = "test-gemini-key"


def gemini_env(**overrides: str) -> dict[str, str]:
    env = {
        "CHATBOT_PROVIDER": "openai",
        "OPENAI_BASE_URL": GEMINI_BASE_URL,
        "OPENAI_MODEL": GEMINI_MODEL,
        "GEMINI_API_KEY": GEMINI_KEY,
    }
    env.update(overrides)
    return env


def _mock_env(server: Any, **overrides: str) -> dict[str, str]:
    host, port = server.server_address
    env = gemini_env(OPENAI_BASE_URL=f"http://{host}:{port}/v1")
    env.update(overrides)
    return env


# --- configuration ---


def test_gemini_env_loads_settings_via_key_alias() -> None:
    settings = load_settings(environ=gemini_env(), dotenv_path=None)

    assert settings.provider == "openai"
    assert settings.openai_api_key == GEMINI_KEY
    assert settings.openai_base_url == GEMINI_BASE_URL
    assert settings.openai_model == GEMINI_MODEL


def test_openai_api_key_takes_precedence_over_gemini_key() -> None:
    settings = load_settings(
        environ=gemini_env(OPENAI_API_KEY="openai-key"), dotenv_path=None
    )

    assert settings.openai_api_key == "openai-key"


def test_gemini_key_read_from_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("GEMINI_API_KEY=file-key\n", encoding="utf-8")

    settings = load_settings(environ={}, dotenv_path=env_file)

    assert settings.openai_api_key == "file-key"


def test_gemini_base_url_trailing_slash_is_normalized() -> None:
    settings = load_settings(
        environ=gemini_env(OPENAI_BASE_URL=GEMINI_BASE_URL + "/"), dotenv_path=None
    )

    assert settings.openai_base_url == GEMINI_BASE_URL
    provider = build_provider(settings)
    assert provider.endpoint == f"{GEMINI_BASE_URL}/chat/completions"


def test_gemini_settings_build_the_reused_openai_provider() -> None:
    settings = load_settings(environ=gemini_env(), dotenv_path=None)

    provider = build_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.endpoint == f"{GEMINI_BASE_URL}/chat/completions"
    assert GEMINI_MODEL in provider.name


def test_offline_remains_the_default_without_gemini_config() -> None:
    settings = load_settings(environ={}, dotenv_path=None)

    assert settings.provider == "offline"
    assert settings.openai_api_key == ""


# --- request formatting (mocked HTTP) ---


def test_gemini_request_formatting(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {
        "choices": [{"message": {"role": "assistant", "content": "Gemini says hi"}}]
    }
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))
    history = ConversationHistory(limit=50)
    history.add("user", "prior")
    history.add("assistant", "reply")

    reply = provider.send("hello gemini", history.messages)

    assert reply == "Gemini says hi"
    request = server.last_request
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == f"Bearer {GEMINI_KEY}"
    assert request["payload"]["model"] == GEMINI_MODEL
    assert request["payload"]["messages"] == [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "hello gemini"},
    ]


# --- response parsing (mocked HTTP) ---


def test_gemini_openai_shaped_response_is_parsed(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1760000000,
        "model": GEMINI_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "  Paris.  "},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
    }
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    reply = provider.send("capital of France?", ())

    assert reply == "Paris."


# --- error handling (mocked HTTP) ---


def test_gemini_invalid_key_reports_401_with_hint(openai_server: Any) -> None:
    server = openai_server
    server.reply_status = 401
    server.reply_body = {"error": {"message": "API key not valid. Please pass a valid API key."}}
    provider = build_provider(
        load_settings(environ=_mock_env(server, GEMINI_API_KEY="bad"), dotenv_path=None)
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    message = str(excinfo.value)
    assert "401" in message
    assert "API key not valid" in message
    assert "GEMINI_API_KEY" in message


def test_gemini_unexpected_response_shape_is_a_provider_error(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {"candidates": [], "promptFeedback": {"blockReason": "OTHER"}}
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "no choices" in str(excinfo.value)


def test_gemini_non_json_response_is_a_provider_error(openai_server: Any) -> None:
    server = openai_server
    server.reply_raw = b"<html>upstream error</html>"
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "invalid JSON" in str(excinfo.value)
