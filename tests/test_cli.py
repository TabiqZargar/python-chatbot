"""Tests for the terminal chat loop and CLI entrypoint."""

from __future__ import annotations

from collections.abc import Iterable

import pytest

from chatbot.chatbot import MAX_MESSAGE_LENGTH, Chatbot
from chatbot.cli import chat_loop, main
from chatbot.history import ConversationHistory
from chatbot.providers.offline import OfflineProvider


class _ScriptedInput:
    def __init__(self, lines: Iterable[str], end: type[BaseException] = EOFError) -> None:
        self._lines = list(lines)
        self._end = end
        self._index = 0

    def __call__(self, prompt: str = "") -> str:
        if self._index >= len(self._lines):
            raise self._end()
        line = self._lines[self._index]
        self._index += 1
        return line


def run_loop(lines: list[str], bot: Chatbot | None = None, end: type[BaseException] = EOFError):
    bot = bot or Chatbot(OfflineProvider(), history=ConversationHistory(limit=10))
    outputs: list[str] = []
    code = chat_loop(bot, input_fn=_ScriptedInput(lines, end=end), print_fn=outputs.append)
    return code, outputs


def test_chat_and_exit() -> None:
    code, outputs = run_loop(["hello", "/exit"])

    assert code == 0
    assert any(line.startswith("Bot> ") for line in outputs)
    assert "Goodbye!" in outputs


@pytest.mark.parametrize("exit_cmd", ["/exit", "/quit", "/q", "/EXIT"])
def test_exit_commands(exit_cmd: str) -> None:
    code, _ = run_loop([exit_cmd])
    assert code == 0


def test_eof_exits_cleanly() -> None:
    code, outputs = run_loop([], end=EOFError)
    assert code == 0
    assert "Goodbye!" in outputs


def test_keyboard_interrupt_exits_cleanly() -> None:
    code, outputs = run_loop([], end=KeyboardInterrupt)
    assert code == 0
    assert "Goodbye!" in outputs


def test_empty_input_is_rejected_but_loop_continues() -> None:
    code, outputs = run_loop(["   ", "/exit"])

    assert code == 0
    assert any("type something" in line for line in outputs)
    assert "Goodbye!" in outputs


def test_unknown_command_is_reported() -> None:
    code, outputs = run_loop(["/bogus", "/exit"])

    assert code == 0
    assert any("Unknown command" in line for line in outputs)


def test_help_lists_commands() -> None:
    _, outputs = run_loop(["/help", "/exit"])
    joined = "\n".join(outputs)

    assert "/help" in joined
    assert "/history" in joined
    assert "/clear" in joined
    assert "/exit" in joined


def test_history_command_reports_empty_then_records() -> None:
    _, outputs = run_loop(["/history", "hello", "/history", "/exit"])

    assert "(conversation is empty)" in outputs
    assert any(line.startswith("1. user: hello") for line in outputs)


def test_clear_command_forgets_conversation() -> None:
    bot = Chatbot(OfflineProvider(), history=ConversationHistory(limit=10))
    _, outputs = run_loop(["hello", "/clear", "/history", "/exit"], bot=bot)

    assert "Conversation cleared." in outputs
    assert "(conversation is empty)" in outputs
    assert len(bot.history) == 0


def test_provider_error_is_shown_and_loop_continues(failing_provider) -> None:
    bot = Chatbot(failing_provider, history=ConversationHistory(limit=10))

    code, outputs = run_loop(["hello", "/exit"], bot=bot)

    assert code == 0
    assert any(line.startswith("Error:") for line in outputs)
    assert "Goodbye!" in outputs


def test_invalid_input_is_shown_and_loop_continues() -> None:
    code, outputs = run_loop(["x" * (MAX_MESSAGE_LENGTH + 1), "/exit"])

    assert code == 0
    assert any(line.startswith("Invalid input:") for line in outputs)


def test_main_reports_configuration_errors(monkeypatch, capsys) -> None:
    monkeypatch.setenv("CHATBOT_PROVIDER", "not-a-provider")

    code = main()

    captured = capsys.readouterr()
    assert code == 1
    assert "Configuration error" in captured.err
