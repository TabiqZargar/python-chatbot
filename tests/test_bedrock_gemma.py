"""Gemma 4 on Amazon Bedrock: config, requests, parsing, errors.

Gemma 4 on Bedrock is served only on the bedrock-mantle endpoint behind an
OpenAI-compatible Chat Completions API, so it reuses OpenAICompatibleProvider
— no Bedrock-specific client or boto3 exists. All HTTP goes to the local mock
server from conftest.py: no network, no real API key, no real AWS call.
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

BEDROCK_BASE_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
BEDROCK_MODEL = "google.gemma-4-31b"
BEDROCK_KEY = "test-bedrock-api-key"


def bedrock_env(**overrides: str) -> dict[str, str]:
    env = {
        "CHATBOT_PROVIDER": "openai",
        "OPENAI_BASE_URL": BEDROCK_BASE_URL,
        "OPENAI_MODEL": BEDROCK_MODEL,
        "OPENAI_API_KEY": BEDROCK_KEY,
    }
    env.update(overrides)
    return env


def _mock_env(server: Any, **overrides: str) -> dict[str, str]:
    host, port = server.server_address
    env = bedrock_env(OPENAI_BASE_URL=f"http://{host}:{port}/openai/v1")
    env.update(overrides)
    return env


# --- configuration ---


def test_bedrock_env_loads_settings() -> None:
    settings = load_settings(environ=bedrock_env(), dotenv_path=None)

    assert settings.provider == "openai"
    assert settings.openai_api_key == BEDROCK_KEY
    assert settings.openai_base_url == BEDROCK_BASE_URL
    assert settings.openai_model == BEDROCK_MODEL


def test_bedrock_key_read_from_dotenv_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=file-key\n", encoding="utf-8")

    settings = load_settings(environ={}, dotenv_path=env_file)

    assert settings.openai_api_key == "file-key"


def test_bedrock_base_url_trailing_slash_is_normalized() -> None:
    settings = load_settings(
        environ=bedrock_env(OPENAI_BASE_URL=BEDROCK_BASE_URL + "/"), dotenv_path=None
    )

    assert settings.openai_base_url == BEDROCK_BASE_URL
    provider = build_provider(settings)
    assert provider.endpoint == f"{BEDROCK_BASE_URL}/chat/completions"


def test_bedrock_settings_build_the_reused_openai_provider() -> None:
    settings = load_settings(environ=bedrock_env(), dotenv_path=None)

    provider = build_provider(settings)

    assert isinstance(provider, OpenAICompatibleProvider)
    # The /openai/v1 prefix is required for Gemma 4 on bedrock-mantle.
    assert provider.endpoint == f"{BEDROCK_BASE_URL}/chat/completions"
    assert BEDROCK_MODEL in provider.name


def test_offline_remains_the_default_without_bedrock_config() -> None:
    settings = load_settings(environ={}, dotenv_path=None)

    assert settings.provider == "offline"


# --- request formatting (mocked HTTP) ---


def test_bedrock_request_formatting(openai_server: Any) -> None:
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
    assert request["path"] == "/openai/v1/chat/completions"
    assert request["authorization"] == f"Bearer {BEDROCK_KEY}"
    assert request["payload"]["model"] == BEDROCK_MODEL
    assert request["payload"]["messages"] == [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "hello gemma"},
    ]


# --- response parsing (mocked HTTP) ---


def test_bedrock_openai_shaped_response_is_parsed(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1760000000,
        "model": BEDROCK_MODEL,
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


def test_bedrock_invalid_key_reports_401_with_hint(openai_server: Any) -> None:
    server = openai_server
    server.reply_status = 401
    server.reply_body = {
        "error": {"message": "The security token included in the request is invalid."}
    }
    provider = build_provider(
        load_settings(environ=_mock_env(server, OPENAI_API_KEY="bad"), dotenv_path=None)
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    message = str(excinfo.value)
    assert "401" in message
    assert "security token" in message
    assert "OPENAI_API_KEY" in message


def test_bedrock_unexpected_response_shape_is_a_provider_error(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {"unexpected": "shape"}
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "no choices" in str(excinfo.value)


def test_bedrock_non_json_response_is_a_provider_error(openai_server: Any) -> None:
    server = openai_server
    server.reply_raw = b"<html>upstream error</html>"
    provider = build_provider(load_settings(environ=_mock_env(server), dotenv_path=None))

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "invalid JSON" in str(excinfo.value)
