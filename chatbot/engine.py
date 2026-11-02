"""Chat orchestration: validation, Ollama streaming, and SQLite persistence."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from chatbot.conversation import (
    DEFAULT_MAX_MESSAGE_LENGTH,
    build_model_messages,
    make_title,
    validate_message,
)
from chatbot.errors import ChatbotError, InvalidInputError, OllamaError, StorageError
from chatbot.ollama_client import OllamaClient, OllamaModel, OllamaStatus
from database.db import Database
from database.models import ChatMessage, Conversation, UserPreferences

__all__ = ["ChatEngine", "ChatOptions", "StreamUpdate"]


@dataclass(frozen=True)
class ChatOptions:
    """Per-request generation settings."""

    model: str
    system_prompt: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024


@dataclass(frozen=True)
class StreamUpdate:
    """One streaming update for the UI layer."""

    conversation_id: int
    text: str
    done: bool = False
    error: str = ""
    user_message: str = ""


class ChatEngine:
    """Coordinates the Ollama client, conversation rules, and the database."""

    def __init__(
        self,
        database: Database,
        client: OllamaClient,
        history_limit: int = 30,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
    ) -> None:
        if history_limit < 1:
            raise InvalidInputError("History limit must be at least 1.")
        if max_message_length < 1:
            raise InvalidInputError("Maximum message length must be at least 1.")
        self._db = database
        self._client = client
        self._history_limit = history_limit
        self._max_message_length = max_message_length

    @property
    def database(self) -> Database:
        return self._db

    @property
    def client(self) -> OllamaClient:
        return self._client

    # --- discovery -----------------------------------------------------

    def connection_status(self) -> OllamaStatus:
        return self._client.check_connection()

    def installed_models(self) -> list[OllamaModel]:
        return self._client.list_models()

    # --- conversations -------------------------------------------------

    def list_conversations(self) -> list[Conversation]:
        return self._db.list_conversations()

    def get_conversation(self, conversation_id: int) -> Conversation | None:
        return self._db.get_conversation(conversation_id)

    def get_messages(self, conversation_id: int) -> list[ChatMessage]:
        return self._db.get_messages(conversation_id)

    def rename_conversation(self, conversation_id: int, title: str) -> bool:
        return self._db.rename_conversation(conversation_id, title)

    def delete_conversation(self, conversation_id: int) -> bool:
        return self._db.delete_conversation(conversation_id)

    def clear_conversation(self, conversation_id: int) -> int:
        return self._db.clear_messages(conversation_id)

    # --- preferences ---------------------------------------------------

    def load_preferences(self, defaults: ChatOptions) -> ChatOptions:
        stored = self._db.get_preferences()
        model = stored.get("model") or defaults.model
        system_prompt = stored.get("system_prompt", defaults.system_prompt)
        try:
            temperature = float(stored.get("temperature", defaults.temperature))
        except (TypeError, ValueError):
            temperature = defaults.temperature
        if not 0 <= temperature <= 2:
            temperature = defaults.temperature
        try:
            max_tokens = int(stored.get("max_tokens", defaults.max_tokens))
        except (TypeError, ValueError):
            max_tokens = defaults.max_tokens
        if max_tokens < 1:
            max_tokens = defaults.max_tokens
        return ChatOptions(
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def save_preferences(self, options: ChatOptions) -> None:
        self._db.set_preferences(
            {
                "model": options.model,
                "system_prompt": options.system_prompt,
                "temperature": str(options.temperature),
                "max_tokens": str(options.max_tokens),
            }
        )

    def preferences_as_user(self, options: ChatOptions) -> UserPreferences:
        return UserPreferences(
            model=options.model,
            system_prompt=options.system_prompt,
            temperature=options.temperature,
            max_tokens=options.max_tokens,
        )

    # --- generation ----------------------------------------------------

    def stream_reply(
        self,
        conversation_id: int | None,
        user_message: str | None,
        options: ChatOptions,
        regenerate: bool = False,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Iterator[StreamUpdate]:
        """Stream a reply and persist both sides of the turn.

        The user message is saved before generation starts. The assistant
        message is saved only when a complete reply arrives, or as a partial
        reply if generation is stopped or fails midway.
        """
        clean_model = _validate_model(options.model)
        _validate_options(options)

        removed_assistant: ChatMessage | None = None
        if regenerate:
            if conversation_id is None:
                raise InvalidInputError("Select a conversation to regenerate.")
            conversation = self._db.get_conversation(conversation_id)
            if conversation is None:
                raise InvalidInputError("That conversation no longer exists.")
            history = self._db.get_messages(conversation_id)
            if not history:
                raise InvalidInputError(
                    "There is no previous user message to regenerate."
                )
            if history[-1].role == "assistant":
                if len(history) < 2 or history[-2].role != "user":
                    raise InvalidInputError(
                        "There is no previous user message to regenerate."
                    )
                removed_assistant = self._db.remove_last_assistant_message(
                    conversation_id
                )
                history = history[:-1]
            elif history[-1].role != "user":
                raise InvalidInputError(
                    "There is no previous user message to regenerate."
                )
            prompt_for_ui = history[-1].content
            self._db.update_conversation_settings(
                conversation_id,
                clean_model,
                options.system_prompt,
                options.temperature,
                options.max_tokens,
            )
        else:
            if conversation_id is None:
                clean_message = self._validate_user_message(user_message)
                conversation = self._db.create_conversation(
                    title=make_title(clean_message),
                    model=clean_model,
                    system_prompt=options.system_prompt,
                    temperature=options.temperature,
                    max_tokens=options.max_tokens,
                )
                conversation_id = conversation.id
            else:
                conversation = self._db.get_conversation(conversation_id)
                if conversation is None:
                    raise InvalidInputError("That conversation no longer exists.")
                clean_message = self._validate_user_message(user_message)
                self._db.update_conversation_settings(
                    conversation_id,
                    clean_model,
                    options.system_prompt,
                    options.temperature,
                    options.max_tokens,
                )
            self._db.add_message(conversation_id, "user", clean_message)
            prompt_for_ui = clean_message
            history = self._db.get_messages(conversation_id)

        payload = build_model_messages(
            history, options.system_prompt, self._history_limit
        )
        if not payload or payload[-1]["role"] == "system":
            raise InvalidInputError("There is no message to send.")

        accumulated = ""
        saved = False
        try:
            for chunk in self._client.stream_chat(
                model=clean_model,
                messages=payload,
                temperature=options.temperature,
                max_tokens=options.max_tokens,
                stop_requested=stop_requested,
            ):
                accumulated += chunk
                yield StreamUpdate(
                    conversation_id=conversation_id,
                    text=accumulated,
                    user_message=prompt_for_ui,
                )

            if not accumulated.strip():
                raise OllamaError("The model returned an empty response.")
            saved_reply = accumulated.strip()
            self._db.add_message(conversation_id, "assistant", saved_reply)
            saved = True
            yield StreamUpdate(
                conversation_id=conversation_id,
                text=saved_reply,
                done=True,
                user_message=prompt_for_ui,
            )
        except GeneratorExit:
            self._save_partial(conversation_id, accumulated, saved)
            raise
        except ChatbotError as exc:
            self._save_partial(conversation_id, accumulated, saved)
            restored = self._restore_if_empty(
                conversation_id, removed_assistant, accumulated
            )
            yield StreamUpdate(
                conversation_id=conversation_id,
                text=restored if restored is not None else accumulated,
                done=True,
                error=_friendly_error(exc),
                user_message=prompt_for_ui,
            )
        except Exception as exc:
            self._save_partial(conversation_id, accumulated, saved)
            restored = self._restore_if_empty(
                conversation_id, removed_assistant, accumulated
            )
            yield StreamUpdate(
                conversation_id=conversation_id,
                text=restored if restored is not None else accumulated,
                done=True,
                error=f"Unexpected error: {exc}",
                user_message=prompt_for_ui,
            )

    def reply(
        self,
        conversation_id: int | None,
        user_message: str,
        options: ChatOptions,
        regenerate: bool = False,
    ) -> str:
        """Consume a stream and return the final text, raising on failure."""
        text = ""
        error = ""
        for update in self.stream_reply(
            conversation_id, user_message, options, regenerate=regenerate
        ):
            text = update.text
            error = update.error
        if error:
            raise OllamaError(error)
        return text

    def _validate_user_message(self, message: object) -> str:
        return validate_message(message, self._max_message_length)

    def _save_partial(
        self, conversation_id: int, accumulated: str, already_saved: bool
    ) -> None:
        if already_saved or not accumulated.strip():
            return
        try:
            self._db.add_message(conversation_id, "assistant", accumulated.strip())
        except StorageError:
            # A cleanup save must never mask the original error/cancellation.
            pass

    def _restore_if_empty(
        self,
        conversation_id: int,
        removed: ChatMessage | None,
        accumulated: str,
    ) -> str | None:
        if removed is None or accumulated.strip():
            return None
        try:
            self._db.add_message(conversation_id, "assistant", removed.content)
        except StorageError:
            return None
        return removed.content


def _validate_model(model: object) -> str:
    if not isinstance(model, str) or not model.strip():
        raise InvalidInputError(
            "No model selected. Choose an installed Ollama model first."
        )
    return model.strip()


def _validate_options(options: ChatOptions) -> None:
    if not isinstance(options.system_prompt, str):
        raise InvalidInputError("System prompt must be text.")
    try:
        temperature = float(options.temperature)
    except (TypeError, ValueError) as exc:
        raise InvalidInputError("Temperature must be a number.") from exc
    if not 0 <= temperature <= 2:
        raise InvalidInputError("Temperature must be between 0 and 2.")
    if (
        not isinstance(options.max_tokens, int)
        or isinstance(options.max_tokens, bool)
        or options.max_tokens < 1
    ):
        raise InvalidInputError("Maximum tokens must be a positive integer.")


def _friendly_error(exc: ChatbotError) -> str:
    message = str(exc).strip()
    return message or "The request failed."
