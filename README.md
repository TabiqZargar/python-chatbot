# Local AI Chatbot

A private, local-first AI chatbot with a polished Gradio web interface. It runs
models through [Ollama](https://ollama.com) on your own machine, streams replies
token by token, and stores conversations in a local SQLite database. There is no
cloud API, no account, and no telemetry.

The original terminal chatbot is still included and still works. The new web app
is the recommended interface.

## Features

- **Fully local inference** through Ollama; no cloud fallback.
- **Streaming replies** with Markdown rendering and copy buttons.
- **Persistent conversations** in SQLite: titles, messages, model, and settings.
- **Multi-turn context** with a configurable history limit.
- **Model picker** populated from models installed in Ollama.
- **System prompt presets** plus custom prompts, temperature, and token limits.
- **Conversation management**: new chat, rename, delete, clear, and regenerate.
- **Stop generation**, with partial replies saved instead of lost.
- **Connection checks** with setup help instead of silent failures or mock replies.
- **Responsive light/dark UI** with a collapsible conversation sidebar.
- **No model downloads from the app**; you choose and pull models yourself.

## Quick start

### 1. Install Ollama

Install Ollama from <https://ollama.com/download>, or use your package manager.

```bash
ollama serve
```

Ollama listens on `http://127.0.0.1:11434` by default.

### 2. Pull a model

The app never downloads models automatically. Pull one in a terminal:

```bash
# Small models suited to machines with about 4 GB of RAM
ollama pull qwen2.5:1.5b
ollama pull llama3.2:1b

# A good balance if you have a little more memory
ollama pull qwen2.5:3b
ollama pull llama3.2:3b
```

Check what is installed at any time:

```bash
ollama list
```

### 3. Install the app

The repository already has a virtual environment at `.venv/`:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The Ollama client and SQLite storage use only the Python standard library.
Gradio is the only runtime dependency; pytest is included for development.

### 4. Configure (optional)

```bash
cp .env.example .env
```

The defaults work with a standard local Ollama install. Edit `.env` only if you
want a different model, port, database path, or generation defaults. Real
environment variables override values from `.env`.

### 5. Run

```bash
python app.py
```

Then open <http://127.0.0.1:7860>.

The sidebar shows Ollama connection state and installed models. If Ollama is not
running, the app explains how to start it and offers a **Retry connection**
button.

## Configuration

All new-app settings are read from environment variables first, then `.env`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Local Ollama server URL. |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model selected at startup. Must be installed. |
| `OLLAMA_TIMEOUT` | `120` | Read/connection timeout in seconds. |
| `CHATBOT_HISTORY_LIMIT` | `30` | Maximum user turns sent as context. |
| `CHATBOT_MAX_MESSAGE_LENGTH` | `8000` | Maximum characters in one user message. |
| `CHATBOT_TEMPERATURE` | `0.7` | Sampling temperature, `0.0`–`2.0`. |
| `CHATBOT_MAX_TOKENS` | `1024` | Maximum tokens in one reply. |
| `CHATBOT_SYSTEM_PROMPT` | helpful general prompt | Default system prompt. Empty means none. |
| `CHATBOT_DB_PATH` | `chatbot.db` | SQLite database location. |
| `CHATBOT_UI_HOST` | `127.0.0.1` | Web server bind address. Keep local by default. |
| `CHATBOT_UI_PORT` | `7860` | Web server port. |
| `CHATBOT_UI_SHARE` | `false` | Create a public Gradio share link. Use only intentionally. |

Invalid values stop startup with a clear `ConfigError` message rather than a
traceback.

### Legacy `.env` values

The original terminal app uses `CHATBOT_PROVIDER`, `OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `OPENAI_MODEL`, and `CHATBOT_TIMEOUT`. The new web app ignores
those names; it only talks to `OLLAMA_BASE_URL`. They are documented at the
bottom of `.env.example` for the legacy terminal interface.

## Using the web app

- **New chat** clears the composer and starts a fresh conversation without
  changing your model or settings.
- **Conversation history** loads any saved chat; its model and settings are
  restored with it.
- **Rename / Delete / Clear messages** manage the selected conversation.
- **Settings** changes model, system prompt, temperature, and maximum tokens.
- **Regenerate** replaces the latest assistant reply using the same user message.
- **Stop** ends an in-flight generation and keeps whatever text arrived.
- **Toggle light / dark theme** switches the interface theme.

Conversations are saved after every successful or partial reply. Closing the app
does not lose your history.

## Storage

Conversations live in `chatbot.db` by default:

- `conversations` stores the title, model, system prompt, temperature, and token
  limit for each chat.
- `messages` stores user and assistant messages in order. Deleting a
  conversation deletes its messages.
- `preferences` stores the last-used defaults.

The database is local and private. To start over, stop the app and delete
`chatbot.db` (or point `CHATBOT_DB_PATH` somewhere else). SQLite WAL sidecar
files may also exist (`chatbot.db-wal`, `chatbot.db-shm`); they are covered by
`.gitignore`.

## Testing

The full suite is offline and needs no Ollama install, model, or API key:

```bash
.venv/bin/python -m pytest
```

Run one file or test:

```bash
.venv/bin/python -m pytest tests/test_engine.py
.venv/bin/python -m pytest tests/test_engine.py::test_closing_generator_saves_partial_reply
```

The Ollama tests start a local mock HTTP server. UI-construction tests verify
that building the app performs no network calls.

## Project structure

```text
.
├── app.py                      # new Gradio web app entrypoint
├── config.py                   # validated web-app settings
├── database/
│   ├── db.py                   # thread-safe SQLite repository
│   └── models.py               # Conversation, ChatMessage, preferences
├── ui/
│   └── components.py           # Gradio layout and event handlers
├── chatbot/
│   ├── engine.py               # streaming orchestration + persistence
│   ├── ollama_client.py        # stdlib Ollama HTTP/streaming client
│   ├── conversation.py         # validation, titles, context trimming
│   ├── prompts.py              # prompt presets and setup help
│   ├── errors.py               # shared exception hierarchy
│   ├── config.py               # legacy terminal settings
│   ├── chatbot.py              # legacy terminal chatbot logic
│   ├── cli.py                  # legacy terminal loop
│   ├── history.py              # legacy in-session history
│   ├── __main__.py             # python -m chatbot
│   └── providers/              # legacy pluggable providers
├── tests/                      # offline pytest suite
├── main.py                     # legacy terminal entrypoint
├── requirements.txt            # gradio + pytest
└── .env.example                # configuration template
```

## How it works

1. The UI builds a `ChatOptions` payload from the selected model and settings.
2. `ChatEngine` saves the user message, trims old context, then calls
   `OllamaClient`.
3. `OllamaClient` streams newline-delimited JSON from Ollama's `/api/chat`
   endpoint and yields text chunks.
4. The UI renders chunks as they arrive. On completion, stop, or error, the
   engine saves the full or partial assistant reply.
5. `Database` persists everything in SQLite behind a re-entrant lock so Gradio's
   worker threads cannot corrupt a transaction.

There is deliberately no mock response in the web app: if Ollama is unavailable,
the UI says so and does not pretend to answer.

## Troubleshooting

**"Ollama is not reachable"**

Start the server with `ollama serve`, confirm `ollama list` works in a terminal,
then click **Retry connection**.

**"Model ... is not available"**

The app does not download models. Run `ollama pull <model>` for the exact name
shown in the error, then retry.

**The first reply is slow**

The first request may load several gigabytes into memory. Wait for the model to
load; later replies are faster. `OLLAMA_TIMEOUT` controls how long the client
waits.

**Out-of-memory or server errors**

Use a smaller quantized model, for example `qwen2.5:1.5b` or `llama3.2:1b`.
Close other memory-heavy applications. A 7B model generally needs more than this
machine's available RAM.

**Port 7860 is already in use**

Set another port:

```bash
CHATBOT_UI_PORT=7861 python app.py
```

## Legacy terminal chatbot

The original terminal interface remains available:

```bash
.venv/bin/python main.py
# or
.venv/bin/python -m chatbot
```

It uses the provider abstraction in `chatbot/providers/` and its own settings in
`chatbot/config.py`. The default `offline` provider is deterministic and needs no
key. It can also talk to OpenAI-compatible endpoints, including local Ollama,
Gemini, Requesty, and Google Gemma 4 on Amazon Bedrock.

Example: use the legacy terminal app with a local Ollama model:

```bash
export CHATBOT_PROVIDER=openai
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=qwen2.5:3b
python main.py
```

Terminal commands are `/help`, `/history`, `/clear`, and `/exit` (also `/quit`
or `/q`). The legacy app keeps history only for the current session; the new web
app persists it in SQLite.

### Legacy provider notes

- **Gemini**: set `CHATBOT_PROVIDER=openai`,
  `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai`, a
  Gemini model in `OPENAI_MODEL`, and `GEMINI_API_KEY` (or
  `OPENAI_API_KEY`).
- **Requesty**: set `CHATBOT_PROVIDER=openai`,
  `OPENAI_BASE_URL=https://router.requesty.ai/v1`, a Requesty model id in
  `OPENAI_MODEL`, and your key in `OPENAI_API_KEY`.
- **Google Gemma 4 on Amazon Bedrock**: set `CHATBOT_PROVIDER=openai`,
  `OPENAI_BASE_URL=https://bedrock-mantle.<region>.api.aws/openai/v1`,
  `OPENAI_MODEL=google.gemma-4-31b`, and a Bedrock API key in
  `OPENAI_API_KEY`. The `/openai/v1` prefix is required.

These are paid, metered services when pointed at a cloud provider. The new web
app does not use them; it is local-only.
