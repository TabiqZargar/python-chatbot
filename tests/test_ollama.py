"""Offline tests for the stdlib Ollama HTTP client.

A local mock server speaks the handful of Ollama endpoints the client uses, so
these tests never touch the network or require an installed model.
"""

from __future__ import annotations

import json
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from chatbot.errors import InvalidInputError, OllamaError
from chatbot.ollama_client import OllamaClient


class _OllamaHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    @property
    def state(self) -> "_OllamaServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        state = self.state
        state.request_count += 1
        if self.path == "/api/version":
            self._json(state.version_status, state.version_body)
        elif self.path == "/api/tags":
            self._json(state.tags_status, state.tags_body)
        else:
            self._json(404, {"error": f"unknown path {self.path}"})

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        state = self.state
        state.request_count += 1
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            parsed = None
        state.requests.append(
            {
                "path": self.path,
                "body": parsed,
                "raw": raw,
                "content_type": self.headers.get("Content-Type"),
            }
        )
        if self.path != "/api/chat":
            self._json(404, {"error": f"unknown path {self.path}"})
            return
        if state.chat_delay:
            time.sleep(state.chat_delay)
        if state.chat_status != 200:
            self._json(state.chat_status, state.chat_body)
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        if state.chat_lines is not None:
            for line in state.chat_lines:
                self.wfile.write(line)
                self.wfile.flush()
                if state.chunk_delay:
                    time.sleep(state.chunk_delay)
            return
        for event in state.chat_events:
            self.wfile.write(json.dumps(event).encode("utf-8") + b"\n")
            self.wfile.flush()
            if state.chunk_delay:
                time.sleep(state.chunk_delay)


class _OllamaServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _OllamaHandler)
        self.version_status = 200
        self.version_body: dict[str, Any] = {"version": "0.9.9"}
        self.tags_status = 200
        self.tags_body: dict[str, Any] = {
            "models": [
                {
                    "name": "qwen2.5:3b",
                    "size": 2 * 1024**3,
                    "modified_at": "2026-01-01T00:00:00Z",
                    "details": {
                        "family": "qwen2",
                        "parameter_size": "3B",
                        "quantization_level": "Q4_K_M",
                    },
                },
                {"name": "llama3.2:3b", "size": 1024**3},
            ]
        }
        self.chat_status = 200
        self.chat_body: dict[str, Any] = {"error": "boom"}
        self.chat_events: list[dict[str, Any]] = [
            {"message": {"role": "assistant", "content": "Hel"}},
            {"message": {"role": "assistant", "content": "lo"}},
            {"done": True},
        ]
        self.chat_lines: list[bytes] | None = None
        self.chat_delay = 0.0
        self.chunk_delay = 0.0
        self.requests: list[dict[str, Any]] = []
        self.request_count = 0


@pytest.fixture
def ollama_server() -> Any:
    server = _OllamaServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _client(server: Any, **kwargs: Any) -> OllamaClient:
    host, port = server.server_address
    return OllamaClient(base_url=f"http://{host}:{port}", timeout=2.0, **kwargs)


def _closed_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_check_connection_reports_version_and_models(ollama_server: Any) -> None:
    status = _client(ollama_server).check_connection()

    assert status.connected is True
    assert status.version == "0.9.9"
    assert [model.name for model in status.models] == ["llama3.2:3b", "qwen2.5:3b"]
    assert status.error == ""


def test_check_connection_reports_unavailable_server() -> None:
    client = OllamaClient(base_url=f"http://127.0.0.1:{_closed_port()}", timeout=0.5)

    status = client.check_connection()

    assert status.connected is False
    assert "Could not connect to Ollama" in status.error


def test_list_models_parses_details(ollama_server: Any) -> None:
    models = _client(ollama_server).list_models()

    qwen = next(model for model in models if model.name == "qwen2.5:3b")
    assert qwen.size == 2 * 1024**3
    assert qwen.size_gb == 2.0
    assert qwen.family == "qwen2"
    assert qwen.parameter_size == "3B"
    assert qwen.quantization == "Q4_K_M"
    assert "2.0 GB" in qwen.label


def test_list_models_rejects_unexpected_shape(ollama_server: Any) -> None:
    ollama_server.tags_body = {"models": "not-a-list"}

    with pytest.raises(OllamaError, match="unexpected model list"):
        _client(ollama_server).list_models()


def test_stream_chat_yields_chunks_and_sends_expected_payload(
    ollama_server: Any,
) -> None:
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]

    chunks = list(
        _client(ollama_server).stream_chat(
            "qwen2.5:3b", messages, temperature=0.25, max_tokens=128
        )
    )

    assert chunks == ["Hel", "lo"]
    request = ollama_server.requests[-1]
    assert request["path"] == "/api/chat"
    assert request["content_type"] == "application/json"
    assert request["body"] == {
        "model": "qwen2.5:3b",
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.25, "num_predict": 128},
    }


def test_chat_convenience_joins_and_strips(ollama_server: Any) -> None:
    assert _client(ollama_server).chat("qwen2.5:3b", [{"role": "user", "content": "hi"}]) == "Hello"


def test_model_not_found_http_error_has_pull_hint(ollama_server: Any) -> None:
    ollama_server.chat_status = 404
    ollama_server.chat_body = {"error": "model 'qwen2.5:3b' not found"}

    with pytest.raises(OllamaError) as excinfo:
        list(
            _client(ollama_server).stream_chat(
                "qwen2.5:3b", [{"role": "user", "content": "hi"}]
            )
        )

    message = str(excinfo.value)
    assert "ollama pull qwen2.5:3b" in message
    assert "not available" in message


def test_stream_error_event_is_raised(ollama_server: Any) -> None:
    ollama_server.chat_events = [{"error": "model 'missing' not found"}]

    with pytest.raises(OllamaError, match="ollama pull missing"):
        list(
            _client(ollama_server).stream_chat(
                "missing", [{"role": "user", "content": "hi"}]
            )
        )


def test_malformed_json_line_is_an_ollama_error(ollama_server: Any) -> None:
    ollama_server.chat_lines = [b"{not json}\n"]

    with pytest.raises(OllamaError, match="malformed"):
        list(
            _client(ollama_server).stream_chat(
                "m", [{"role": "user", "content": "hi"}]
            )
        )


def test_non_utf8_stream_is_an_ollama_error(ollama_server: Any) -> None:
    ollama_server.chat_lines = [b"\xff\xfe\n"]

    with pytest.raises(OllamaError, match="UTF-8"):
        list(
            _client(ollama_server).stream_chat(
                "m", [{"role": "user", "content": "hi"}]
            )
        )


def test_truncated_stream_is_an_ollama_error(ollama_server: Any) -> None:
    ollama_server.chat_events = [{"message": {"content": "partial"}}]

    with pytest.raises(OllamaError, match="before it finished"):
        list(
            _client(ollama_server).stream_chat(
                "m", [{"role": "user", "content": "hi"}]
            )
        )


def test_stream_without_events_is_an_ollama_error(ollama_server: Any) -> None:
    ollama_server.chat_events = []

    with pytest.raises(OllamaError, match="empty response"):
        list(
            _client(ollama_server).stream_chat(
                "m", [{"role": "user", "content": "hi"}]
            )
        )


def test_timeout_is_reported(ollama_server: Any) -> None:
    ollama_server.chat_delay = 0.6
    client = _client(ollama_server)
    client = OllamaClient(base_url=client.base_url, timeout=0.1)

    with pytest.raises(OllamaError, match="did not respond"):
        list(
            client.stream_chat("m", [{"role": "user", "content": "hi"}])
        )


def test_connection_refused_is_reported() -> None:
    client = OllamaClient(base_url=f"http://127.0.0.1:{_closed_port()}", timeout=0.5)

    with pytest.raises(OllamaError, match="ollama serve"):
        list(
            client.stream_chat("m", [{"role": "user", "content": "hi"}])
        )


def test_stream_size_limit_is_enforced(ollama_server: Any) -> None:
    ollama_server.chat_events = [
        {"message": {"content": "x" * 200}},
        {"done": True},
    ]
    client = _client(ollama_server, max_stream_bytes=100)

    with pytest.raises(OllamaError, match="exceeded"):
        list(client.stream_chat("m", [{"role": "user", "content": "hi"}]))


def test_stop_requested_ends_iteration(ollama_server: Any) -> None:
    chunks = list(
        _client(ollama_server).stream_chat(
            "m",
            [{"role": "user", "content": "hi"}],
            stop_requested=lambda: True,
        )
    )

    assert chunks == []


def test_http_500_mentions_memory_or_loading(ollama_server: Any) -> None:
    ollama_server.chat_status = 500
    ollama_server.chat_body = {"error": "out of memory"}

    with pytest.raises(OllamaError) as excinfo:
        list(
            _client(ollama_server).stream_chat(
                "m", [{"role": "user", "content": "hi"}]
            )
        )

    assert "HTTP 500" in str(excinfo.value)
    assert "out of memory" in str(excinfo.value)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "system", "content": "ok"}],
        [{"role": "tool", "content": "nope"}],
        [{"role": "user", "content": "   "}],
        ["not a mapping"],
    ],
)
def test_invalid_messages_are_rejected(ollama_server: Any, messages: list[Any]) -> None:
    with pytest.raises(InvalidInputError):
        list(_client(ollama_server).stream_chat("m", messages))


def test_empty_model_is_rejected() -> None:
    with pytest.raises(InvalidInputError):
        list(OllamaClient().stream_chat("  ", [{"role": "user", "content": "hi"}]))


def test_invalid_options_are_rejected(ollama_server: Any) -> None:
    client = _client(ollama_server)
    with pytest.raises(InvalidInputError):
        list(
            client.stream_chat(
                "m", [{"role": "user", "content": "hi"}], temperature=3.0
            )
        )
    with pytest.raises(InvalidInputError):
        list(
            client.stream_chat(
                "m", [{"role": "user", "content": "hi"}], max_tokens=0
            )
        )
    with pytest.raises(InvalidInputError):
        list(
            client.stream_chat(
                "m", [{"role": "user", "content": "hi"}], temperature="hot"
            )
        )


def test_base_url_validation() -> None:
    with pytest.raises(OllamaError):
        OllamaClient(base_url="localhost:11434")
    with pytest.raises(OllamaError):
        OllamaClient(timeout=0)
