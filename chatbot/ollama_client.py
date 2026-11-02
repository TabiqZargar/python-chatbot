"""Stdlib-only client for the local Ollama HTTP API.

Only the configured Ollama endpoint is ever contacted. There is deliberately no
cloud fallback and no API key handling in this module.
"""

from __future__ import annotations

import http.client
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

from chatbot.errors import InvalidInputError, OllamaError

__all__ = [
    "MAX_LINE_BYTES",
    "MAX_STREAM_BYTES",
    "OllamaClient",
    "OllamaModel",
    "OllamaStatus",
]

MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_STREAM_BYTES = 50 * 1024 * 1024
MAX_ERROR_BYTES = 64 * 1024
VALID_ROLES = ("system", "user", "assistant")


@dataclass(frozen=True)
class OllamaModel:
    """One model installed in the local Ollama store."""

    name: str
    size: int = 0
    modified_at: str = ""
    family: str = ""
    parameter_size: str = ""
    quantization: str = ""

    @property
    def size_gb(self) -> float:
        return self.size / (1024**3) if self.size else 0.0

    @property
    def label(self) -> str:
        if self.size_gb:
            return f"{self.name} ({self.size_gb:.1f} GB)"
        return self.name


@dataclass(frozen=True)
class OllamaStatus:
    """Result of a connection check, suitable for direct display in the UI."""

    connected: bool
    version: str = ""
    models: tuple[OllamaModel, ...] = ()
    error: str = ""


class OllamaClient:
    """Small HTTP client for ``/api/version``, ``/api/tags``, and ``/api/chat``."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        max_stream_bytes: int = MAX_STREAM_BYTES,
    ) -> None:
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise OllamaError(
                f"OLLAMA_BASE_URL must start with http:// or https://, got {base_url!r}"
            )
        if timeout <= 0:
            raise OllamaError("OLLAMA_TIMEOUT must be greater than 0.")
        if max_stream_bytes < 1:
            raise OllamaError("max_stream_bytes must be positive.")
        self._base_url = base_url.rstrip("/")
        self._timeout = float(timeout)
        self._max_stream_bytes = max_stream_bytes

    @property
    def base_url(self) -> str:
        return self._base_url

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    # --- discovery -----------------------------------------------------

    def check_connection(self) -> OllamaStatus:
        """Return connection state without raising, so the UI can render it."""
        version = ""
        try:
            data = self._get_json("/api/version", timeout=min(self._timeout, 5.0))
            if isinstance(data, dict) and isinstance(data.get("version"), str):
                version = data["version"]
        except OllamaError as exc:
            try:
                models = self.list_models()
            except OllamaError:
                return OllamaStatus(connected=False, error=str(exc))
            return OllamaStatus(connected=True, models=tuple(models))
        try:
            models = self.list_models()
        except OllamaError as exc:
            return OllamaStatus(
                connected=True, version=version, error=str(exc)
            )
        return OllamaStatus(
            connected=True, version=version, models=tuple(models)
        )

    def list_models(self) -> list[OllamaModel]:
        """Return models currently installed locally, sorted by name."""
        data = self._get_json("/api/tags")
        if not isinstance(data, dict):
            raise OllamaError("Ollama returned an unexpected model list.")
        raw_models = data.get("models", [])
        if not isinstance(raw_models, list):
            raise OllamaError("Ollama returned an unexpected model list.")
        models: list[OllamaModel] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("model")
            if not isinstance(name, str) or not name.strip():
                continue
            details = item.get("details")
            if not isinstance(details, dict):
                details = {}
            models.append(
                OllamaModel(
                    name=name.strip(),
                    size=_as_int(item.get("size")),
                    modified_at=str(item.get("modified_at") or ""),
                    family=str(details.get("family") or ""),
                    parameter_size=str(details.get("parameter_size") or ""),
                    quantization=str(details.get("quantization_level") or ""),
                )
            )
        return sorted(models, key=lambda model: model.name.lower())

    def model_names(self) -> list[str]:
        return [model.name for model in self.list_models()]

    # --- chat ----------------------------------------------------------

    def stream_chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[str]:
        """Yield assistant text chunks from Ollama's streaming chat API."""
        clean_model = _validate_model(model)
        payload_messages = _validate_messages(messages)
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as exc:
            raise InvalidInputError("Temperature must be between 0 and 2.") from exc
        if not 0 <= temperature_value <= 2:
            raise InvalidInputError("Temperature must be between 0 and 2.")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise InvalidInputError("Maximum tokens must be a positive integer.")

        body = json.dumps(
            {
                "model": clean_model,
                "messages": payload_messages,
                "stream": True,
                "options": {
                    "temperature": temperature_value,
                    "num_predict": int(max_tokens),
                },
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self._url("/api/chat"),
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        received_content = False
        received_done = False
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                for raw_line in _bounded_lines(response, self._max_stream_bytes):
                    if stop_requested is not None and stop_requested():
                        return
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        text = line.decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise OllamaError(
                            "Ollama returned a stream that was not valid UTF-8."
                        ) from exc
                    try:
                        event = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise OllamaError(
                            "Ollama returned a malformed streaming response."
                        ) from exc
                    if not isinstance(event, dict):
                        raise OllamaError(
                            "Ollama returned an unexpected streaming response."
                        )
                    if event.get("error"):
                        raise OllamaError(_clean_error(event["error"], clean_model))
                    message = event.get("message")
                    content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, str) and content:
                        received_content = True
                        yield content
                    if event.get("done") is True:
                        received_done = True
                        break
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            raise OllamaError(_http_error_message(exc.code, detail, clean_model)) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise OllamaError(_timeout_message(self._timeout)) from exc
            raise OllamaError(
                f"Could not connect to Ollama at {self._base_url}. "
                "Is it installed and running? Start it with `ollama serve`."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError(_timeout_message(self._timeout)) from exc
        except ConnectionError as exc:
            raise OllamaError(
                f"The connection to Ollama at {self._base_url} was interrupted."
            ) from exc
        except http.client.HTTPException as exc:
            raise OllamaError(
                f"Ollama sent an incomplete HTTP response: {exc}"
            ) from exc

        if not received_done:
            if received_content:
                raise OllamaError("Ollama closed the stream before it finished.")
            raise OllamaError("Ollama returned an empty response.")

    def chat(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        """Non-streaming convenience wrapper used by tests and scripts."""
        return "".join(
            self.stream_chat(model, messages, temperature, max_tokens)
        ).strip()

    # --- HTTP helpers --------------------------------------------------

    def _get_json(self, path: str, timeout: float | None = None) -> object:
        request = urllib.request.Request(self._url(path), method="GET")
        try:
            with urllib.request.urlopen(
                request, timeout=timeout if timeout is not None else self._timeout
            ) as response:
                body = _read_bounded(response, MAX_ERROR_BYTES)
        except urllib.error.HTTPError as exc:
            detail = _read_error_body(exc)
            raise OllamaError(
                f"Ollama returned HTTP {exc.code} for {path}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise OllamaError(_timeout_message(self._timeout)) from exc
            raise OllamaError(
                f"Could not connect to Ollama at {self._base_url}. "
                "Is it installed and running? Start it with `ollama serve`."
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError(_timeout_message(self._timeout)) from exc
        except ConnectionError as exc:
            raise OllamaError(
                f"The connection to Ollama at {self._base_url} was interrupted."
            ) from exc
        except http.client.HTTPException as exc:
            raise OllamaError(
                f"Ollama sent an incomplete HTTP response: {exc}"
            ) from exc
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc


def _validate_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        raise InvalidInputError("Choose a model before sending a message.")
    return model.strip()


def _validate_messages(
    messages: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    if not isinstance(messages, (list, tuple)) or not messages:
        raise InvalidInputError("A chat request needs at least one message.")
    cleaned: list[dict[str, str]] = []
    for item in messages:
        if not isinstance(item, Mapping):
            raise InvalidInputError("Chat messages must be mappings.")
        role = item.get("role")
        content = item.get("content")
        if role not in VALID_ROLES:
            raise InvalidInputError(f"Invalid chat role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise InvalidInputError("Chat message content must not be empty.")
        cleaned.append({"role": role, "content": content})
    if not any(item["role"] == "user" for item in cleaned):
        raise InvalidInputError("A chat request needs at least one user message.")
    return cleaned


def _bounded_lines(response: object, max_total_bytes: int) -> Iterator[bytes]:
    total = 0
    while True:
        line = response.readline(MAX_LINE_BYTES + 1)  # type: ignore[attr-defined]
        if not line:
            return
        total += len(line)
        if total > max_total_bytes:
            raise OllamaError(
                f"Ollama response exceeded the {max_total_bytes}-byte limit."
            )
        if len(line) > MAX_LINE_BYTES:
            raise OllamaError(
                f"Ollama sent a line larger than the {MAX_LINE_BYTES}-byte limit."
            )
        yield line


def _read_bounded(response: object, limit: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(8192, limit - total + 1))  # type: ignore[attr-defined]
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise OllamaError(f"Ollama response exceeded the {limit}-byte limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _read_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        raw = _read_bounded(exc, MAX_ERROR_BYTES)
        text = raw.decode("utf-8", errors="replace").strip()
    except Exception:
        return str(getattr(exc, "reason", exc))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text[:500]
    if isinstance(parsed, dict) and parsed.get("error"):
        error = parsed["error"]
        if isinstance(error, dict):
            return str(error.get("message") or error)
        return str(error)
    return text[:500] or str(getattr(exc, "reason", exc))


def _http_error_message(status: int, detail: str, model: str) -> str:
    lowered = detail.lower()
    if status == 404 or "not found" in lowered or "not installed" in lowered:
        return (
            f"Model {model!r} is not available in Ollama. "
            f"Run `ollama pull {model}` and try again. Details: {detail}"
        )
    if status in (401, 403):
        return f"Ollama refused the request (HTTP {status}): {detail}"
    if status >= 500:
        return (
            f"Ollama failed with HTTP {status}. The model may have run out of "
            f"memory or failed to load. Details: {detail}"
        )
    return f"Ollama returned HTTP {status}: {detail}"


def _timeout_message(timeout: float) -> str:
    return (
        f"Ollama did not respond within {timeout:g} seconds. "
        "The model may still be loading; try again or increase OLLAMA_TIMEOUT."
    )


def _clean_error(error: object, model: str) -> str:
    detail = str(error)
    if "not found" in detail.lower() or "not installed" in detail.lower():
        return (
            f"Model {model!r} is not available in Ollama. "
            f"Run `ollama pull {model}` and try again. Details: {detail}"
        )
    return f"Ollama error: {detail}"


def _as_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
