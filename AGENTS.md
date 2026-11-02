# AGENTS.md

## Repo state
Two working interfaces share one repository:
- **Web app (recommended):** `app.py` + Gradio UI in `ui/`, backed by the local
  Ollama server, with SQLite persistence through `database/`.
- **Legacy terminal app:** `main.py` / `python -m chatbot`, using the original
  provider abstraction in `chatbot/providers/`.

`.env` is git-ignored; `.env.example` is the committed template. The new web app
ignores the legacy `CHATBOT_PROVIDER` / `OPENAI_*` settings.

## Commands (run from repo root)
- Tests: `.venv/bin/python -m pytest` (fully offline, no Ollama/model/API key).
- Single test: `.venv/bin/python -m pytest tests/test_engine.py::test_closing_generator_saves_partial_reply`
- Run the web app: `.venv/bin/python app.py` (defaults to http://127.0.0.1:7860).
- Run the legacy terminal app: `.venv/bin/python main.py` or `.venv/bin/python -m chatbot`.
- Install web dependencies: `.venv/bin/pip install -r requirements.txt`.
- No install/editable-install step: `pyproject.toml` sets `pythonpath = ["."]`
  so pytest resolves the packages from the repo root.
- No linter, formatter, or typechecker is configured — do not invent one or
  assume `ruff`/`mypy` exist.

## Dependencies
- Web runtime: `gradio>=6.0,<7` only. Install from `requirements.txt`.
- Ollama, SQLite, and the legacy OpenAI-compatible provider use only the stdlib.
- `pytest` is the only development dependency.
- Do not add an HTTP client, the `openai` package, an SDK, or a paid dependency
  unless the user approves it.

## Architecture (new web app)
- `app.py` loads `config.load_settings()`, opens a `Database`, creates an
  `OllamaClient` and `ChatEngine`, builds the UI, then calls
  `demo.launch(..., theme=THEME, css=CSS)`. Gradio 6 moved `theme`/`css` from
  the `Blocks` constructor to `launch()`.
- `config.py` (root) is the **new app** config; `chatbot/config.py` remains the
  **legacy terminal** config. Do not merge or repurpose them silently.
- `chatbot/engine.py` records the user message before generation. It saves the
  assistant message only after a successful reply, or saves partial text when
  generation is stopped or fails midway. Regeneration removes the last assistant
  message first and restores it if the new attempt produced no text.
- `chatbot/ollama_client.py` only contacts `OLLAMA_BASE_URL`; there is no cloud
  fallback and no mock reply in the web app. It streams newline-delimited JSON
  from `/api/chat` with bounded reads (`MAX_LINE_BYTES`, `MAX_STREAM_BYTES`).
  Errors become `OllamaError`; raw urllib/socket exceptions must not escape.
- `database/db.py` shares one SQLite connection across Gradio worker threads
  behind an `RLock`, enables WAL and foreign keys, uses parameterized queries,
  and cascades message deletion when a conversation is deleted.
- `database/models.py` defines persisted dataclasses; timestamps are UTC ISO
  strings. `chatbot/conversation.py` handles validation, titles, and context
  trimming (caps user turns and always starts with a user message).
- `ui/components.py` owns the Gradio layout and handlers. Use messages-format
  `gr.Chatbot` (no `type=` argument), `gr.Sidebar`, and `gr.update`. The Stop
  button uses `cancels=[send_event, submit_event, regen_event]`; the cancelled
  generator is closed so the engine can persist a partial reply.
- The app must never auto-download models. Show `ollama pull <model>` help
  instead. Do not add cloud fallbacks to make the UI appear functional.

## Configuration (new web app)
- `config.load_settings()` reads real environment variables first, then `.env`
  via `chatbot.config.parse_dotenv`. Real env vars win.
- Variables: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`,
  `CHATBOT_HISTORY_LIMIT`, `CHATBOT_MAX_MESSAGE_LENGTH`,
  `CHATBOT_TEMPERATURE`, `CHATBOT_MAX_TOKENS`, `CHATBOT_SYSTEM_PROMPT`,
  `CHATBOT_DB_PATH`, `CHATBOT_UI_HOST`, `CHATBOT_UI_PORT`, `CHATBOT_UI_SHARE`.
- Temperature `0.0` is valid (deterministic); the valid range is `0`–`2`.
- Invalid values raise `ConfigError`. System prompts are loaded from real env
  values verbatim; unquoted `.env` whitespace is trimmed by the parser.
- Default bind address is `127.0.0.1`. `CHATBOT_UI_SHARE=true` creates a public
  Gradio link; keep it false for local-only use.

## Legacy architecture (preserve unless asked)
- `chatbot/chatbot.py` knows only the `ChatProvider` ABC. It records history
  after a successful reply, so a failed provider call leaves history unchanged.
- Adding a provider = implement `ChatProvider`, wire it in
  `chatbot/providers/__init__.py:build_provider`, and add its name to
  `chatbot/config.py:SUPPORTED_PROVIDERS`. Default is `offline`.
- `ConversationHistory(limit=N)` caps user messages and trims oldest turns,
  always leaving the list starting with a `user` message.
- `OpenAICompatibleProvider` caps response bodies at 10 MiB via bounded reads.
  It retries only pre-response connection refusals; timeouts, mid-stream
  failures, HTTP errors, 429s, 5xx, and non-UTF-8 bodies are never retried and
  surface as `ProviderError`.
- The CLI (`chatbot/cli.py`) catches `KeyboardInterrupt` around both the input
  prompt and `bot.chat()`. Keep `chat_loop(bot, input_fn=..., print_fn=...)`
  injectable for tests.
- Gemini, Requesty, and Google Gemma 4 on Amazon Bedrock are config-only reuse
  of the `openai` provider. Do not add provider classes or SDKs for them.

## Testing notes
- `tests/test_ollama.py` starts a local mock Ollama HTTP server; no network.
  Keep streaming, timeout, malformed-response, model-not-found, and size-limit
  coverage offline.
- `tests/test_database.py`, `tests/test_engine.py`, and `tests/test_ui.py` use
  temporary SQLite files. `build_app` must not contact Ollama; test that with a
  client that raises on network calls.
- Existing legacy tests (provider, CLI, history, config) must continue to pass.
