# AGENTS.md

## Repo state
Working Python chatbot (package `chatbot/`, entrypoints `main.py` and
`python -m chatbot`). `.env` is git-ignored, `.env.example` is the committed
template.

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
- `OpenAICompatibleProvider` caps every response body at 10 MiB (`MAX_RESPONSE_BYTES`)
  via chunked, bounded reads — never `response.read()` unbounded. It retries **only**
  pre-response connection refusals (`ConnectionRefusedError`); timeouts, mid-stream
  failures (`ConnectionError`, `http.client.HTTPException`), HTTP errors, 429s, 5xx,
  and non-UTF-8 bodies are never retried — all surface as `ProviderError`, never raw
  exceptions. `max_response_bytes`, `max_retries`, and `retry_backoff` are
  constructor-only knobs (not env config).
- The CLI (`chatbot/cli.py`) catches `KeyboardInterrupt` around both the input prompt
  **and** `bot.chat()` so Ctrl-C during an in-flight API request exits cleanly instead
  of crashing with a traceback.

## Configuration
- Loaded by `chatbot/config.py:load_settings` from `os.environ` then `.env`; real env
  vars override `.env`. Secrets never go in code. Invalid values raise `ConfigError`.
- `OPENAI_API_KEY` may be empty (local servers like Ollama/LM Studio need none).
- `CHATBOT_SYSTEM_PROMPT` defaults to empty; when set, `OpenAICompatibleProvider`
  prepends it as a `{"role": "system", ...}` message before history. The value is
  loaded **unstripped** (whitespace/Unicode are meaningful), and the offline
  provider ignores it entirely.
- Gemini is supported by config-only reuse of the `openai` provider
  (`OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`); the
  `GEMINI_API_KEY` env var is a fallback alias used only when `OPENAI_API_KEY` is
  empty. Do not add a separate Gemini provider class or the Google SDK.
- Google Gemma 4 on Amazon Bedrock is likewise config-only: it lives only on the
  `bedrock-mantle` endpoint behind an OpenAI-compatible API, so set
  `CHATBOT_PROVIDER=openai`,
  `OPENAI_BASE_URL=https://bedrock-mantle.<region>.api.aws/openai/v1` (the
  `/openai/v1` prefix is required), `OPENAI_MODEL=google.gemma-4-31b`, and put a
  Bedrock API key in `OPENAI_API_KEY`. Do not add boto3/botocore, a Bedrock provider
  class, or a Converse/SigV4 path — the stdlib bearer-token request already works.
- Requesty is likewise config-only: it is an OpenAI-compatible multi-model router, so
  set `CHATBOT_PROVIDER=openai`, `OPENAI_BASE_URL=https://router.requesty.ai/v1`,
  `OPENAI_MODEL` to a Requesty model id (e.g. `google/gemma-4-31b-it` to pin a
  deployment, or a bare name like `gemma-4-31b-it` for automatic failover), and put a
  Requesty key in `OPENAI_API_KEY`. Do not add a Requesty provider class or SDK.

## Testing notes
- `tests/conftest.py` starts a local `http.server` mock for OpenAI-compatible tests —
  no network or API key required. Provider errors are exercised with `FakeProvider`.
- CLI is tested via `chat_loop(bot, input_fn=..., print_fn=...)`; keep that injectable
  signature when editing `chatbot/cli.py`.
