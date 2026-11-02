"""Launch the local Ollama chatbot UI.

Run with:
    python app.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from chatbot.engine import ChatEngine
from chatbot.errors import ChatbotError, ConfigError, StorageError
from chatbot.ollama_client import OllamaClient
from config import load_settings
from database.db import Database
from ui.components import CSS, THEME, build_app


def main() -> int:
    try:
        settings = load_settings()
        database = Database(settings.database_path)
        client = OllamaClient(
            base_url=settings.ollama_base_url,
            timeout=settings.ollama_timeout,
        )
        engine = ChatEngine(
            database=database,
            client=client,
            history_limit=settings.history_limit,
            max_message_length=settings.max_message_length,
        )
        demo = build_app(engine, settings)
    except (ConfigError, StorageError, ChatbotError) as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        print(
            "Check .env and the environment variables (see .env.example).",
            file=sys.stderr,
        )
        return 1

    print(
        f"Local AI Chatbot\n"
        f"  Ollama: {settings.ollama_base_url}\n"
        f"  Model:  {settings.default_model}\n"
        f"  Data:   {settings.database_path}\n"
        f"  URL:    http://{settings.host}:{settings.port}\n"
    )
    try:
        demo.launch(
            server_name=settings.host,
            server_port=settings.port,
            share=settings.share,
            inbrowser=False,
            show_error=True,
            footer_links=[],
            theme=THEME,
            css=CSS,
            quiet=False,
        )
    except KeyboardInterrupt:
        print("\nShutting down.")
    except OSError as exc:
        print(f"Could not start the web server: {exc}", file=sys.stderr)
        print(
            f"Is port {settings.port} already in use? "
            "Set CHATBOT_UI_PORT to a free port.",
            file=sys.stderr,
        )
        return 1
    finally:
        database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
