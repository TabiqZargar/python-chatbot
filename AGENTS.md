# AGENTS.md

## Repo state
Working Python chatbot (package `chatbot/`, entrypoints `main.py` and
`python -m chatbot`). Git repo is initialized with no commits yet; `.env` is
git-ignored, `.env.example` is the committed template.

## Commands (run from repo root)
- Tests: `.venv/bin/python -m pytest` (pytest is preinstalled in `.venv`; the suite is
  fully offline and runs in a few seconds).
- Single test: `.venv/bin/python -m pytest tests/test_chatbot.py::test_multi_turn_conversation_is_preserved`
- Run the chatbot: `.venv/bin/python main.py` or `.venv/bin/python -m chatbot`
- No install/editable-install step: the package is not `pip install`ed. `pyproject.toml`
  sets `pythonpath = ["."]` so pytest resolves `chatbot` from the repo root.
- No linter, formatter, or typechecker is configured — do not invent one or assume
  `ruff`/`mypy` exist.

## Dependencies
- Zero third-party *runtime* dependencies (stdlib only). `pytest` is the only dev dep.
- The OpenAI-compatible provider uses `urllib` from the stdlib — do not add an HTTP
  client or the `openai` package unless the user approves a paid dependency.

## Architecture (non-obvious wiring)
- `chatbot/chatbot.py` (Chatbot) only knows the `ChatProvider` ABC in
  `chatbot/providers/base.py`; it records history **after** a successful reply, so a
  failed provider call leaves history unchanged.
- Adding a provider = implement `ChatProvider`, wire it in
  `chatbot/providers/__init__.py:build_provider`, and add its name to
  `chatbot/config.py:SUPPORTED_PROVIDERS`. Default is `offline` (deterministic, no key).
- `ConversationHistory(limit=N)` caps **user** messages and trims oldest turns, always
  leaving the list starting with a `user` message.

## Configuration
- Loaded by `chatbot/config.py:load_settings` from `os.environ` then `.env`; real env
  vars override `.env`. Secrets never go in code. Invalid values raise `ConfigError`.
- `OPENAI_API_KEY` may be empty (local servers like Ollama/LM Studio need none).
- Gemini is supported by config-only reuse of the `openai` provider
  (`OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`); the
  `GEMINI_API_KEY` env var is a fallback alias used only when `OPENAI_API_KEY` is
  empty. Do not add a separate Gemini provider class or the Google SDK.

## Testing notes
- `tests/conftest.py` starts a local `http.server` mock for OpenAI-compatible tests —
  no network or API key required. Provider errors are exercised with `FakeProvider`.
- CLI is tested via `chat_loop(bot, input_fn=..., print_fn=...)`; keep that injectable
  signature when editing `chatbot/cli.py`.
