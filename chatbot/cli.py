"""Terminal chat interface.

Commands inside the chat: /help, /history, /clear, /exit (also /quit, /q).
Ctrl-C and Ctrl-D both exit cleanly.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from chatbot.chatbot import Chatbot
from chatbot.config import load_settings
from chatbot.errors import ChatbotError, ConfigError, InvalidInputError
from chatbot.history import ConversationHistory
from chatbot.providers import build_provider

__all__ = ["main", "chat_loop"]

EXIT_COMMANDS = frozenset({"exit", "quit", "q"})
HELP_TEXT = """\
Available commands:
  /help     show this help
  /history  show the conversation so far
  /clear    forget the conversation
  /exit     leave the chat (also: /quit, /q)
Anything else you type is sent to the chatbot."""

BANNER_TEMPLATE = """\
Python Chatbot — provider: {provider}
Type /help for commands, /exit to quit.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Load configuration, build the chatbot, and start the chat loop."""
    try:
        settings = load_settings()
        provider = build_provider(settings)
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print("Check your environment variables and .env file (see .env.example).", file=sys.stderr)
        return 1

    bot = Chatbot(
        provider,
        history=ConversationHistory(limit=settings.history_limit),
    )
    print(BANNER_TEMPLATE.format(provider=provider.name))
    return chat_loop(bot)


def chat_loop(
    bot: Chatbot,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> int:
    """Run the read-eval-print loop until the user exits.

    Returns 0 on a normal exit. All expected errors are shown to the user
    and the session keeps running.
    """
    while True:
        try:
            raw = input_fn("You> ")
        except (EOFError, KeyboardInterrupt):
            print_fn("")
            print_fn("Goodbye!")
            return 0
        text = raw.strip()

        if not text:
            print_fn("Please type something, or /help to see the commands.")
            continue

        if text.startswith("/"):
            command = text[1:].lower()
            if command in EXIT_COMMANDS:
                print_fn("Goodbye!")
                return 0
            if command == "help":
                print_fn(HELP_TEXT)
                continue
            if command == "history":
                lines = bot.render_history()
                print_fn("\n".join(lines) if lines else "(conversation is empty)")
                continue
            if command == "clear":
                bot.clear()
                print_fn("Conversation cleared.")
                continue
            print_fn(f"Unknown command: {text}. Type /help to see the commands.")
            continue

        try:
            reply = bot.chat(text)
        except InvalidInputError as exc:
            print_fn(f"Invalid input: {exc}")
            continue
        except ChatbotError as exc:
            print_fn(f"Error: {exc}")
            continue
        print_fn(f"Bot> {reply}")


if __name__ == "__main__":
    raise SystemExit(main())
