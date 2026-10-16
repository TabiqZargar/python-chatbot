"""Shared test fixtures: a fake provider and a local OpenAI-compatible server."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from chatbot.errors import ProviderError
from chatbot.history import Message
from chatbot.providers.base import ChatProvider


class FakeProvider(ChatProvider):
    """Test double: records calls and returns scripted replies or errors."""

    def __init__(
        self,
        replies: list[str] | None = None,
        error: Exception | None = None,
        name: str = "fake",
    ) -> None:
        self._replies = list(replies or [])
        self._error = error
        self._name = name
        self.calls: list[tuple[str, tuple[Message, ...]]] = []

    @property
    def name(self) -> str:
        return self._name

    def send(self, message: str, history: list[Message] | tuple[Message, ...]) -> str:
        self.calls.append((message, tuple(history)))
        if self._error is not None:
            raise self._error
        if self._replies:
            return self._replies.pop(0)
        return f"echo: {message}"


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        server = self.server
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        server.request_count += 1
        server.last_request = {
            "path": self.path,
            "authorization": self.headers.get("Authorization"),
            "payload": parsed,
        }
        payload = json.dumps(server.reply_body).encode("utf-8")
        if server.reply_raw is not None:
            payload = server.reply_raw
        if server.reply_delay:
            time.sleep(server.reply_delay)
        self.send_response(server.reply_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: Any) -> None:
        pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.reply_status = 200
        self.reply_delay = 0.0
        self.reply_raw: bytes | None = None
        self.reply_body: dict[str, Any] = {
            "choices": [{"message": {"role": "assistant", "content": "hi from mock"}}]
        }
        self.last_request: dict[str, Any] | None = None
        self.request_count = 0


@pytest.fixture
def openai_server() -> Any:
    """A local OpenAI-compatible HTTP server; configurable per test."""
    server = _Server()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.fixture
def failing_provider() -> FakeProvider:
    return FakeProvider(error=ProviderError("boom"))
