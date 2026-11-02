"""Prompts and user-facing help text for the local chatbot."""

from __future__ import annotations

from config import DEFAULT_SYSTEM_PROMPT

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "SYSTEM_PROMPT_PRESETS",
    "ollama_unavailable_help",
    "no_models_help",
    "model_pull_hint",
]

SYSTEM_PROMPT_PRESETS: dict[str, str] = {
    "General assistant": DEFAULT_SYSTEM_PROMPT,
    "Concise": (
        "You are a concise assistant. Answer in as few words as possible while "
        "remaining accurate and complete."
    ),
    "Code mentor": (
        "You are a senior software engineer. Give correct, runnable code and "
        "brief explanations. Point out important edge cases and security issues."
    ),
    "Tutor": (
        "You are a patient tutor. Explain ideas step by step, check the learner's "
        "understanding, and use small examples."
    ),
}


def ollama_unavailable_help(base_url: str) -> str:
    """Markdown shown when the local Ollama server cannot be reached."""
    return (
        f"**Ollama is not reachable at `{base_url}`.**\n\n"
        "1. Install Ollama from <https://ollama.com/download> (or your package manager).\n"
        "2. Start the server: `ollama serve`\n"
        "3. Download a model you want to use, for example: `ollama pull qwen2.5:3b`\n"
        "4. Click **Retry connection**.\n\n"
        "This app never falls back to a cloud API."
    )


def no_models_help() -> str:
    """Markdown shown when Ollama is running but has no installed models."""
    return (
        "**Ollama is running, but no models are installed.**\n\n"
        "Download a model in a terminal, then click **Retry connection**:\n\n"
        "```bash\n"
        "ollama pull qwen2.5:3b   # lightweight, about 2 GB\n"
        "ollama pull llama3.2:3b  # conversational, about 2 GB\n"
        "ollama pull gemma3:4b    # a little larger\n"
        "```\n\n"
        "Models are downloaded only when you run `ollama pull`; this app never "
        "downloads them for you."
    )


def model_pull_hint(model: str) -> str:
    return f"Run `ollama pull {model}` in a terminal, then retry."
