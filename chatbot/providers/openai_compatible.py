"""Provider for any OpenAI-compatible chat completions endpoint (stdlib only).

Works with OpenAI itself and with other services that speak the same API, e.g.:
- Ollama:      OPENAI_BASE_URL=http://localhost:11434/v1
- LM Studio:   OPENAI_BASE_URL=http://localhost:1234/v1
- Gemini:      OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
               (auth may come from GEMINI_API_KEY; see chatbot.config)
- vLLM / others with /v1/chat/completions

No third-party HTTP library is used — only urllib from the standard library.
The provider is dormant until you set CHATBOT_PROVIDER=openai.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from chatbot.errors import ProviderError
from chatbot.history import Message
from chatbot.providers.base import ChatProvider

__all__ = ["OpenAICompatibleProvider", "MAX_RESPONSE_BYTES"]

MAX_RESPONSE_BYTES = 10 * 1024 * 1024

_READ_CHUNK_BYTES = 64 * 1024
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_RETRY_BACKOFF = 0.5


class OpenAICompatibleProvider(ChatProvider):
    """POSTs conversations to {base_url}/chat/completions."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout: float = 30.0,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_backoff: float = _DEFAULT_RETRY_BACKOFF,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ProviderError(f"invalid base URL: {base_url!r}")
        if max_response_bytes < 1:
            raise ProviderError(
                f"max_response_bytes must be at least 1, got {max_response_bytes}"
            )
        if max_retries < 0:
            raise ProviderError(f"max_retries must not be negative, got {max_retries}")
        if retry_backoff < 0:
            raise ProviderError(
                f"retry_backoff must not be negative, got {retry_backoff}"
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_response_bytes = max_response_bytes
        self._max_retries = max_retries
        self._retry_backoff = retry_backoff

    @property
    def name(self) -> str:
        return f"openai-compatible ({self._model} @ {self._base_url})"

    @property
    def endpoint(self) -> str:
        url = self._base_url
        if url.endswith("/chat/completions"):
            return url
        return f"{url}/chat/completions"

    def send(self, message: str, history: list[Message] | tuple[Message, ...]) -> str:
        if not isinstance(message, str) or not message.strip():
            raise ProviderError("openai provider received an empty message")

        messages = [item.to_dict() for item in history]
        messages.append({"role": "user", "content": message.strip()})
        payload = json.dumps({"model": self._model, "messages": messages}).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        request = urllib.request.Request(self.endpoint, data=payload, headers=headers, method="POST")

        attempts = 0
        while True:
            attempts += 1
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    body = _read_bounded(response, self._max_response_bytes).decode("utf-8")
            except urllib.error.HTTPError as exc:
                detail = _read_error_body(exc, self._max_response_bytes)
                hint = _hint_for_status(exc.code)
                raise ProviderError(
                    f"provider returned HTTP {exc.code}: {detail}{hint}"
                ) from exc
            except urllib.error.URLError as exc:
                reason = getattr(exc, "reason", exc)
                if _is_safe_to_retry(reason) and attempts <= self._max_retries:
                    time.sleep(self._retry_backoff * (2 ** (attempts - 1)))
                    continue
                raise ProviderError(f"could not reach {self._base_url}: {reason}") from exc
            except TimeoutError as exc:
                raise ProviderError(
                    f"request to {self._base_url} timed out after {self._timeout}s"
                ) from exc

            return _extract_reply(body, self._model)


def _read_bounded(stream: object, limit: int) -> bytes:
    """Read at most ``limit`` bytes, aborting as soon as the limit is exceeded.

    The body is pulled in small chunks so an oversized or endless response is
    never buffered in full. Raises ProviderError on overflow.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = stream.read(min(_READ_CHUNK_BYTES, limit - total + 1))  # type: ignore[attr-defined]
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise ProviderError(f"response body exceeded the {limit}-byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _is_safe_to_retry(reason: object) -> bool:
    """True only for failures that provably happened before any response.

    A refused connection means the TCP handshake never completed, so the server
    cannot have processed the request. Everything else — timeouts, mid-stream
    disconnects, HTTP status errors — is ambiguous or already a server response
    and is never retried automatically.
    """
    return isinstance(reason, ConnectionRefusedError)


def _extract_reply(body: str, model: str) -> str:
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ProviderError(f"{model} returned invalid JSON") from exc
    choices = data.get("choices") if isinstance(data, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ProviderError(f"{model} response contained no choices")
    first = choices[0]
    content = None
    if isinstance(first, dict):
        message = first.get("message")
        if isinstance(message, dict):
            content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProviderError(f"{model} response was missing a text reply")
    return content.strip()


def _read_error_body(exc: urllib.error.HTTPError, limit: int = MAX_RESPONSE_BYTES) -> str:
    try:
        detail = _read_bounded(exc, limit).decode("utf-8", errors="replace").strip()
        parsed = json.loads(detail)
        if isinstance(parsed, dict):
            error = parsed.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return error["message"]
            if isinstance(error, str):
                return error
        return detail[:300] or exc.reason
    except ProviderError:
        return f"error body exceeded the {limit}-byte limit"
    except Exception:
        return str(exc.reason)


def _hint_for_status(status: int) -> str:
    if status in (401, 403):
        return "\nHint: check OPENAI_API_KEY (or GEMINI_API_KEY when using Gemini; omit the key for local servers that need none)."
    if status == 404:
        return "\nHint: check OPENAI_BASE_URL — it must point at the API root (e.g. .../v1)."
    if status == 429:
        return "\nHint: rate limit or quota exceeded — wait and try again."
    return ""
