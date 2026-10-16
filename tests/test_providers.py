"""Tests for providers: offline, factory, and the OpenAI-compatible HTTP client."""

from __future__ import annotations

import urllib.error
import urllib.request
from typing import Any

import pytest

from chatbot.config import Settings
from chatbot.errors import ConfigError, ProviderError
from chatbot.history import ConversationHistory
from chatbot.providers import build_provider
from chatbot.providers.offline import OfflineProvider
from chatbot.providers.openai_compatible import OpenAICompatibleProvider


def make_history(*turns: tuple[str, str]) -> ConversationHistory:
    history = ConversationHistory(limit=50)
    for role, content in turns:
        history.add(role, content)
    return history


def test_offline_provider_is_deterministic_and_history_aware() -> None:
    provider = OfflineProvider()
    history = make_history(("user", "hello"), ("assistant", "hi"))

    first = provider.send("hello", history.messages)
    second = provider.send("hello", history.messages)

    assert first == second
    assert isinstance(first, str) and first


def test_offline_provider_uses_history_for_fallback() -> None:
    provider = OfflineProvider()
    empty_history = make_history()
    rich_history = make_history(("user", "first"), ("assistant", "reply"))

    first_reply = provider.send("something new", empty_history.messages)
    later_reply = provider.send("something new", rich_history.messages)

    assert first_reply != later_reply


@pytest.mark.parametrize("message", ["", "   ", None])
def test_offline_provider_rejects_empty_message(message: object) -> None:
    provider = OfflineProvider()
    with pytest.raises(ProviderError):
        provider.send(message, ())  # type: ignore[arg-type]


def test_factory_builds_offline_provider() -> None:
    provider = build_provider(Settings(provider="offline"))
    assert isinstance(provider, OfflineProvider)


def test_factory_builds_openai_provider() -> None:
    provider = build_provider(
        Settings(
            provider="openai",
            openai_base_url="http://localhost:11434/v1",
            openai_model="llama3",
        )
    )
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.endpoint == "http://localhost:11434/v1/chat/completions"


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ConfigError):
        build_provider(Settings(provider="magic"))


def test_endpoint_handles_trailing_slash_and_full_path() -> None:
    assert (
        OpenAICompatibleProvider(base_url="http://x/v1/").endpoint
        == "http://x/v1/chat/completions"
    )
    assert (
        OpenAICompatibleProvider(base_url="http://x/v1/chat/completions").endpoint
        == "http://x/v1/chat/completions"
    )


def test_provider_rejects_invalid_base_url() -> None:
    with pytest.raises(ProviderError):
        OpenAICompatibleProvider(base_url="localhost:8000")


def _url(server: Any) -> str:
    host, port = server.server_address
    return f"http://{host}:{port}/v1"


def test_openai_provider_success(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {"choices": [{"message": {"role": "assistant", "content": "  hello!  "}}]}
    provider = OpenAICompatibleProvider(api_key="secret", base_url=_url(server), model="test-model")
    history = make_history(("user", "prior"), ("assistant", "reply"))

    reply = provider.send("current question", history.messages)

    assert reply == "hello!"
    request = server.last_request
    assert request["path"] == "/v1/chat/completions"
    assert request["authorization"] == "Bearer secret"
    assert request["payload"]["model"] == "test-model"
    assert request["payload"]["messages"] == [
        {"role": "user", "content": "prior"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "current question"},
    ]


def test_openai_provider_omits_authorization_without_key(openai_server: Any) -> None:
    server = openai_server
    provider = OpenAICompatibleProvider(api_key="", base_url=_url(server), model="m")

    provider.send("hi", ())

    assert server.last_request["authorization"] is None


def test_openai_provider_http_error_includes_hint(openai_server: Any) -> None:
    server = openai_server
    server.reply_status = 401
    server.reply_body = {"error": {"message": "invalid api key"}}
    provider = OpenAICompatibleProvider(api_key="bad", base_url=_url(server), model="m")

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    message = str(excinfo.value)
    assert "401" in message
    assert "invalid api key" in message
    assert "OPENAI_API_KEY" in message


def test_openai_provider_missing_choices(openai_server: Any) -> None:
    server = openai_server
    server.reply_body = {"unexpected": "shape"}

    provider = OpenAICompatibleProvider(base_url=_url(server), model="m")

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "no choices" in str(excinfo.value)


def test_openai_provider_handles_non_json_body(openai_server: Any) -> None:
    server = openai_server
    server.reply_raw = b"<html>not json</html>"

    provider = OpenAICompatibleProvider(base_url=_url(server), model="m")

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "invalid JSON" in str(excinfo.value)


def test_openai_provider_connection_refused(openai_server: Any) -> None:
    server = openai_server
    host, port = server.server_address
    base_url = f"http://{host}:{port}/v1"
    server.shutdown()
    server.server_close()

    provider = OpenAICompatibleProvider(base_url=base_url, model="m", retry_backoff=0.0)
    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "could not reach" in str(excinfo.value)


def test_openai_provider_timeout(openai_server: Any) -> None:
    server = openai_server
    server.reply_delay = 1.0

    provider = OpenAICompatibleProvider(base_url=_url(server), model="m", timeout=0.05)
    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "timed out" in str(excinfo.value)
    assert server.request_count == 1


def test_response_body_over_limit_is_rejected(openai_server: Any) -> None:
    server = openai_server
    server.reply_raw = b"x" * 4096

    provider = OpenAICompatibleProvider(
        base_url=_url(server), model="m", max_response_bytes=1024
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "exceeded" in str(excinfo.value)


def test_safe_connection_refused_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> Any:
        attempts.append(request)
        raise urllib.error.URLError(ConnectionRefusedError("refused"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:1/v1", model="m", max_retries=2, retry_backoff=0.0
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "could not reach" in str(excinfo.value)
    assert len(attempts) == 3


def test_timeout_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[urllib.request.Request] = []

    def fake_urlopen(request: urllib.request.Request, timeout: float = 0.0) -> Any:
        attempts.append(request)
        raise TimeoutError("slow")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        base_url="http://127.0.0.1:1/v1", model="m", max_retries=2, retry_backoff=0.0
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert "timed out" in str(excinfo.value)
    assert len(attempts) == 1


@pytest.mark.parametrize("status", [429, 500, 503])
def test_http_status_errors_are_not_retried(openai_server: Any, status: int) -> None:
    server = openai_server
    server.reply_status = status

    provider = OpenAICompatibleProvider(
        base_url=_url(server), model="m", max_retries=3, retry_backoff=0.0
    )

    with pytest.raises(ProviderError) as excinfo:
        provider.send("hi", ())

    assert str(status) in str(excinfo.value)
    assert server.request_count == 1
