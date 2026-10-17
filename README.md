# Python AI Chatbot

A beginner-friendly, terminal-based AI chatbot written in Python. It keeps
track of the whole conversation during a session and talks to a pluggable
"provider" backend, so you can start with the built-in offline demo and later
switch to a real model (local or API) without touching the chatbot logic..

## Features

- Terminal (command-line) chat interface.
- Multi-turn conversations with in-session history.
- Provider abstraction: chatbot logic is fully separate from AI backends.
- Built-in **offline** provider — runs with no API key and no extra packages.
- Optional **OpenAI-compatible** provider using only the standard library,
  which works with OpenAI, Google Gemini's OpenAI-compatible endpoint,
  Google Gemma 4 on Amazon Bedrock, and local servers such as
  Ollama / LM Studio / vLLM.
- Configuration and secrets come from environment variables / `.env` — no
  secrets in code.
- Graceful error handling, input validation, and in-chat commands.
- Automated tests with `pytest`, including history and error-handling tests.

## Project structure

```
.
├── main.py                     # entrypoint: python main.py
├── pyproject.toml              # metadata + pytest config
├── .env.example                # template for your .env (safe to commit)
├── .gitignore                  # ignores .env, caches, venv
├── chatbot/
│   ├── config.py               # env + .env loading and validation
│   ├── errors.py               # ChatbotError hierarchy
│   ├── history.py              # ConversationHistory (size-limited, ordered)
│   ├── chatbot.py              # core logic: validate -> provider -> record
│   ├── cli.py                  # terminal loop + commands
│   ├── __main__.py             # python -m chatbot
│   └── providers/
│       ├── base.py             # ChatProvider interface
│       ├── offline.py          # built-in, deterministic, no network
│       ├── openai_compatible.py# any OpenAI-compatible HTTP endpoint
│       └── __init__.py         # build_provider(settings) factory
└── tests/                      # pytest suite
```

## Requirements

- Python 3.10 or newer (this project was developed on Python 3.14).
- No third-party packages are needed to _run_ the chatbot.
- `pytest` is needed to run the tests.

## Setup

The project already has a virtual environment at `.venv/`. Activate it:

```bash
source .venv/bin/activate
```

Install the test dependency (only needed to run tests):

```bash
pip install pytest
```

Create your local configuration file (never commit the real `.env`):

```bash
cp .env.example .env
```

The defaults use the offline provider, so the chatbot works immediately with
no key.

## Configuration

All settings are read from environment variables, falling back to values in
`.env`. Real environment variables always win over `.env`.

| Variable                | Default                     | Purpose                                                    |
| ----------------------- | --------------------------- | ---------------------------------------------------------- |
| `CHATBOT_PROVIDER`      | `offline`                   | `offline` or `openai` (any OpenAI-compatible API).         |
| `OPENAI_API_KEY`        | _(empty)_                   | API key; optional for local servers that need none.        |
| `GEMINI_API_KEY`        | _(empty)_                   | Alias for the key; used only if `OPENAI_API_KEY` is empty. |
| `OPENAI_BASE_URL`       | `https://api.openai.com/v1` | API root URL.                                              |
| `OPENAI_MODEL`          | `gpt-4o-mini`               | Model name sent to the endpoint.                           |
| `CHATBOT_TIMEOUT`       | `30`                        | Request timeout in seconds.                                |
| `CHATBOT_HISTORY_LIMIT` | `50`                        | Max user messages kept in history.                         |

Never commit a real `.env` — it is listed in `.gitignore`.

## Usage

Run the chatbot from the project root:

```bash
python main.py
# or
python -m chatbot
```

Inside the chat:

| Command    | Action                               |
| ---------- | ------------------------------------ |
| `/help`    | Show available commands.             |
| `/history` | Show the conversation so far.        |
| `/clear`   | Forget the conversation.             |
| `/exit`    | Leave the chat (also `/quit`, `/q`). |

`Ctrl-C` and `Ctrl-D` also exit cleanly. Blank input is ignored with a hint.

### Using a real model (optional)

Point the chatbot at any OpenAI-compatible endpoint. For a local Ollama
server:

```bash
export CHATBOT_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=llama3
python main.py
```

The OpenAI-compatible provider uses only Python's standard library
(`urllib`) — no extra package is installed. **No paid dependency is added by
default.** If you point `OPENAI_BASE_URL` at a paid service such as OpenAI and
set `OPENAI_API_KEY`, that service may charge you; use a local server or your
own budget to avoid costs.

### Using Google Gemini (optional)

Gemini offers an official OpenAI-compatible endpoint, so it works with the
existing `openai` provider — no extra provider, package, or Google SDK.

```bash
export CHATBOT_PROVIDER=openai
export OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
export OPENAI_MODEL=gemini-3.5-flash
export GEMINI_API_KEY=your-key-from-ai-studio
python main.py
```

Or set the same four variables in your ignored `.env` file (see `.env.example`).

- Get a key at [Google AI Studio](https://aistudio.google.com/apikey); keep it
  out of version control.
- `GEMINI_API_KEY` is a convenience alias for `OPENAI_API_KEY`. If both are
  set, `OPENAI_API_KEY` wins.
- Model IDs change over time — use a current Gemini model name from the
  [Gemini docs](https://ai.google.dev/gemini-api/docs/openai).
- The Gemini API is a paid, metered service; requests to it may incur charges
  on your Google account.

### Using Google Gemma 4 on Amazon Bedrock (optional)

Gemma 4 models on Amazon Bedrock are served only on the `bedrock-mantle`
endpoint behind an OpenAI-compatible Chat Completions API, so they work with
the existing `openai` provider — no boto3, AWS SDK, or extra package is
needed. Two gotchas from the Bedrock docs: use the `/openai/v1` path prefix
(not `/v1`), and authenticate with a **Bedrock API key** (this stdlib
provider sends plain bearer-token auth; it does not SigV4-sign requests).

```bash
export CHATBOT_PROVIDER=openai
export OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/openai/v1
export OPENAI_MODEL=google.gemma-4-31b
export OPENAI_API_KEY=your-bedrock-api-key
python main.py
```

Or set the same four variables in your ignored `.env` file (see `.env.example`).

- Create a Bedrock API key in the [Amazon Bedrock
  console](https://console.aws.amazon.com/bedrock/home#/api-keys/long-term/create)
  and enable model access for Gemma 4; keep the key out of version control.
- Model IDs: `google.gemma-4-31b` (dense, 256K context),
  `google.gemma-4-26b-a4b` (MoE, cheaper), `google.gemma-4-e2b` (smallest).
- Regions with Gemma 4: `us-east-1`, `us-east-2`, `us-west-2`,
  `eu-central-1` (In-Region only — no cross-region profiles). Change the
  region in the base URL accordingly.
- Bedrock API keys are region-scoped: a key generated in `us-east-1` fails in
  other regions with `Credential should be scoped to a valid region`.
- Bedrock is a paid, metered service; requests may incur charges on your AWS
  account.

## Testing

Run the whole suite:

```bash
pytest
```

Run one file or one test:

```bash
pytest tests/test_history.py
pytest tests/test_chatbot.py::test_provider_error_propagates_and_history_unchanged
```

The tests use a local mock HTTP server (standard library) for the
OpenAI-compatible provider, so they need no network and no API key.

## Error handling

- Invalid input (empty or too long) is rejected with a friendly message and
  the conversation continues.
- Provider failures (network, HTTP status, bad JSON, timeouts) are reported as
  errors and never corrupt history — a failed turn is not recorded.
- Invalid configuration (bad provider name, limits, or URL) stops startup with
  a clear message instead of a traceback.
