"""Gradio interface for the local Ollama chatbot."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import gradio as gr

from chatbot.conversation import validate_message
from chatbot.engine import ChatEngine, ChatOptions, StreamUpdate
from chatbot.errors import ChatbotError, InvalidInputError
from chatbot.prompts import (
    SYSTEM_PROMPT_PRESETS,
    no_models_help,
    ollama_unavailable_help,
)
from config import AppSettings
from database.models import Conversation, format_timestamp

__all__ = [
    "CSS",
    "THEME",
    "build_app",
    "conversation_choices",
    "messages_for_chatbot",
]

CSS = """
:root {
    --chat-max-width: 920px;
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}
#app-header h1 {
    margin: 0;
    font-size: 1.45rem;
    letter-spacing: -0.02em;
}
#app-header p {
    margin: 0.15rem 0 0;
    opacity: 0.7;
    font-size: 0.9rem;
}
#status {
    min-height: 1.6rem;
    font-size: 0.88rem;
    opacity: 0.85;
}
#status p { margin: 0.2rem 0; }
#chat-window {
    border-radius: 14px;
}
#chat-window .message-row {
    padding-top: 0.2rem;
    padding-bottom: 0.2rem;
}
#sidebar-brand {
    font-weight: 650;
    font-size: 1.05rem;
    letter-spacing: -0.01em;
}
#sidebar-subtitle {
    opacity: 0.65;
    font-size: 0.82rem;
    margin-top: -0.4rem;
}
#conversation-list {
    max-height: 34vh;
    overflow-y: auto;
}
#composer textarea {
    font-size: 0.98rem;
}
@media (max-width: 760px) {
    .gradio-container { padding: 0.5rem !important; }
    #app-header h1 { font-size: 1.2rem; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[
        "ui-sans-serif",
        "system-ui",
        "-apple-system",
        "BlinkMacSystemFont",
        "Segoe UI",
        "sans-serif",
    ],
    font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"],
)


def conversation_choices(
    conversations: Sequence[Conversation],
) -> list[tuple[str, int]]:
    """Labels for the sidebar radio, newest first."""
    choices: list[tuple[str, int]] = []
    for conversation in conversations:
        title = conversation.title.strip() or "Untitled"
        if len(title) > 42:
            title = f"{title[:41]}…"
        count = conversation.message_count
        unit = "message" if count == 1 else "messages"
        stamp = format_timestamp(conversation.updated_at)
        suffix = f" · {count} {unit}"
        if stamp:
            suffix += f" · {stamp}"
        choices.append((f"{title}{suffix}", conversation.id))
    return choices


def messages_for_chatbot(messages: Sequence[object]) -> list[dict[str, str]]:
    """Convert stored messages into Gradio's messages format."""
    result: list[dict[str, str]] = []
    for message in messages:
        role = getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role in ("user", "assistant") and isinstance(content, str):
            result.append({"role": role, "content": content})
    return result


def build_app(engine: ChatEngine, settings: AppSettings) -> gr.Blocks:
    """Build the Gradio app. No network calls happen during construction."""
    default_options = engine.load_preferences(
        ChatOptions(
            model=settings.default_model,
            system_prompt=settings.default_system_prompt,
            temperature=settings.default_temperature,
            max_tokens=settings.default_max_tokens,
        )
    )
    initial_conversations = engine.list_conversations()
    initial_id = initial_conversations[0].id if initial_conversations else None
    if initial_id is not None:
        conversation = initial_conversations[0]
        options = ChatOptions(
            model=conversation.model,
            system_prompt=conversation.system_prompt,
            temperature=conversation.temperature,
            max_tokens=conversation.max_tokens,
        )
        initial_messages = messages_for_chatbot(engine.get_messages(initial_id))
        initial_title = conversation.title
    else:
        options = default_options
        initial_messages = []
        initial_title = ""

    # --- handlers ------------------------------------------------------

    def on_load(current_id: int | None, preferred_model: str | None):
        conversations = engine.list_conversations()
        choices = conversation_choices(conversations)
        selected = current_id
        if selected is not None and not any(
            conversation.id == selected for conversation in conversations
        ):
            selected = None
        status, model_update = _connection_snapshot(engine, preferred_model)
        return (
            status,
            model_update,
            gr.update(choices=choices, value=selected),
            selected,
        )

    def on_retry(preferred_model: str | None):
        return _connection_snapshot(engine, preferred_model)

    def on_preset(preset: str | None):
        if preset and preset in SYSTEM_PROMPT_PRESETS:
            return SYSTEM_PROMPT_PRESETS[preset]
        return gr.update()

    def on_new_chat():
        return (
            [],
            None,
            "",
            "New chat. Your model and settings are preserved.",
            gr.update(value=None),
            "",
        )

    def on_select(conversation_id: int | None):
        if conversation_id is None:
            return (
                [],
                None,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "",
                "New chat.",
            )
        conversation = engine.get_conversation(conversation_id)
        if conversation is None:
            return (
                [],
                None,
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                "",
                "**Error:** That conversation no longer exists.",
            )
        messages = messages_for_chatbot(engine.get_messages(conversation_id))
        return (
            messages,
            conversation_id,
            gr.update(value=conversation.model),
            gr.update(value=conversation.system_prompt),
            gr.update(value=conversation.temperature),
            gr.update(value=conversation.max_tokens),
            conversation.title,
            f"Loaded **{conversation.title}**. You can continue the conversation.",
        )

    def on_rename(conversation_id: int | None, title: str):
        if conversation_id is None:
            raise gr.Error("Select a conversation first.")
        try:
            ok = engine.rename_conversation(conversation_id, title)
        except ChatbotError as exc:
            raise gr.Error(str(exc)) from exc
        if not ok:
            raise gr.Error("That conversation no longer exists.")
        choices = conversation_choices(engine.list_conversations())
        return (
            gr.update(choices=choices, value=conversation_id),
            gr.update(value=title.strip()),
            "Conversation renamed.",
        )

    def on_delete(conversation_id: int | None):
        if conversation_id is None:
            raise gr.Error("Select a conversation first.")
        try:
            deleted = engine.delete_conversation(conversation_id)
        except ChatbotError as exc:
            raise gr.Error(str(exc)) from exc
        if not deleted:
            raise gr.Error("That conversation no longer exists.")
        choices = conversation_choices(engine.list_conversations())
        return (
            [],
            None,
            "Conversation deleted.",
            gr.update(choices=choices, value=None),
            "",
            "",
        )

    def on_clear(conversation_id: int | None):
        try:
            if conversation_id is not None:
                engine.clear_conversation(conversation_id)
        except ChatbotError as exc:
            raise gr.Error(str(exc)) from exc
        choices = conversation_choices(engine.list_conversations())
        value = conversation_id if conversation_id is not None else None
        return (
            [],
            "Conversation cleared.",
            gr.update(choices=choices, value=value),
        )

    def on_save_preferences(
        system_prompt: str,
        temperature: float,
        max_tokens: int,
        model: str | None,
    ):
        options = ChatOptions(
            model=(model or settings.default_model),
            system_prompt=system_prompt or "",
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        try:
            engine.save_preferences(options)
        except ChatbotError as exc:
            raise gr.Error(str(exc)) from exc
        return "Preferences saved."

    def on_send(
        message: str,
        history: list[dict[str, str]] | None,
        conversation_id: int | None,
        model: str | None,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[tuple[object, ...]]:
        try:
            clean = validate_message(message, settings.max_message_length)
        except InvalidInputError as exc:
            yield (
                _normalise_history(history),
                conversation_id,
                message,
                f"**Error:** {exc}",
                gr.update(),
                gr.update(),
            )
            return
        if not (model or "").strip():
            yield (
                _normalise_history(history),
                conversation_id,
                message,
                "**Error:** Choose an installed model before sending.",
                gr.update(),
                gr.update(),
            )
            return
        options = ChatOptions(
            model=model or "",
            system_prompt=system_prompt or "",
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        try:
            engine.save_preferences(options)
        except ChatbotError as exc:
            yield (
                _normalise_history(history),
                conversation_id,
                message,
                f"**Error:** {exc}",
                gr.update(),
                gr.update(),
            )
            return

        ui_history = _normalise_history(history)
        ui_history.append({"role": "user", "content": clean})
        ui_history.append({"role": "assistant", "content": ""})
        updates = engine.stream_reply(conversation_id, clean, options)
        yield from _consume_stream(engine, updates, ui_history, conversation_id, options.model)

    def on_regenerate(
        history: list[dict[str, str]] | None,
        conversation_id: int | None,
        model: str | None,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> Iterator[tuple[object, ...]]:
        ui_history = _normalise_history(history)
        if conversation_id is None:
            yield (
                ui_history,
                conversation_id,
                "",
                "**Error:** Select a conversation to regenerate.",
                gr.update(),
                gr.update(),
            )
            return
        if not (model or "").strip():
            yield (
                ui_history,
                conversation_id,
                "",
                "**Error:** Choose an installed model before regenerating.",
                gr.update(),
                gr.update(),
            )
            return
        if ui_history and ui_history[-1]["role"] == "assistant":
            ui_history.pop()
        if not ui_history or ui_history[-1]["role"] != "user":
            yield (
                ui_history,
                conversation_id,
                "",
                "**Error:** There is no previous user message to regenerate.",
                gr.update(),
                gr.update(),
            )
            return
        options = ChatOptions(
            model=(model or ""),
            system_prompt=system_prompt or "",
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        try:
            engine.save_preferences(options)
        except ChatbotError as exc:
            yield (
                ui_history,
                conversation_id,
                "",
                f"**Error:** {exc}",
                gr.update(),
                gr.update(),
            )
            return
        ui_history.append({"role": "assistant", "content": ""})
        updates = engine.stream_reply(
            conversation_id, None, options, regenerate=True
        )
        yield from _consume_stream(engine, updates, ui_history, conversation_id, options.model)

    def on_stop():
        return "Generation stopped. Any partial reply was saved."

    # --- layout --------------------------------------------------------

    with gr.Blocks(
        title="Local AI Chatbot",
        analytics_enabled=False,
        fill_height=True,
    ) as demo:
        current_id = gr.State(initial_id)

        with gr.Sidebar(open=True, width=330, label="Chats"):
            gr.Markdown(
                '<div id="sidebar-brand">Local AI Chatbot</div>'
                '<div id="sidebar-subtitle">Private inference with Ollama</div>'
            )
            new_chat_btn = gr.Button("New chat", variant="primary", size="md")
            conversation_list = gr.Radio(
                choices=conversation_choices(initial_conversations),
                value=initial_id,
                label="Conversation history",
                type="value",
                elem_id="conversation-list",
            )
            with gr.Row():
                rename_box = gr.Textbox(
                    value=initial_title,
                    placeholder="Rename selected chat",
                    show_label=False,
                    container=False,
                    scale=4,
                )
                rename_btn = gr.Button("Rename", size="sm", scale=1)
            with gr.Row():
                delete_btn = gr.Button("Delete", variant="stop", size="sm")
                clear_sidebar_btn = gr.Button("Clear messages", size="sm")

            gr.Markdown("---")
            connection_status = gr.Markdown("Checking Ollama…", elem_id="status")
            model_selector = gr.Dropdown(
                choices=[],
                value=options.model,
                label="Model",
                info="Installed Ollama models",
                allow_custom_value=True,
                interactive=True,
            )
            retry_btn = gr.Button("Retry connection", size="sm")

            with gr.Accordion("Settings", open=False):
                preset_selector = gr.Dropdown(
                    choices=["Custom", *SYSTEM_PROMPT_PRESETS.keys()],
                    value="Custom",
                    label="System prompt preset",
                )
                system_prompt = gr.Textbox(
                    value=options.system_prompt,
                    label="System prompt",
                    lines=4,
                    max_lines=10,
                )
                temperature = gr.Slider(
                    minimum=0,
                    maximum=2,
                    value=options.temperature,
                    step=0.05,
                    label="Temperature",
                    info="Lower is more focused; higher is more creative.",
                )
                max_tokens = gr.Slider(
                    minimum=64,
                    maximum=4096,
                    value=max(64, min(4096, options.max_tokens)),
                    step=64,
                    label="Maximum response tokens",
                )
                save_prefs_btn = gr.Button("Save preferences", size="sm")

            theme_btn = gr.Button("Toggle light / dark theme", size="sm")

        with gr.Column(scale=1):
            gr.Markdown(
                "<div id=\"app-header\"><h1>Local AI Chatbot</h1>"
                "<p>Everything stays on this machine. No cloud API is used.</p></div>"
            )
            chatbot = gr.Chatbot(
                value=initial_messages,
                label="Chat",
                height=560,
                layout="bubble",
                group_consecutive_messages=False,
                buttons=["copy"],
                elem_id="chat-window",
                placeholder=(
                    "### Start a conversation\n"
                    "Pick an installed Ollama model, then send a message."
                ),
            )
            status = gr.Markdown("Ready.", elem_id="status")

            with gr.Row(elem_id="composer"):
                message_box = gr.Textbox(
                    placeholder="Message a local model…",
                    show_label=False,
                    container=False,
                    lines=1,
                    max_lines=8,
                    autofocus=True,
                    scale=9,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            with gr.Row():
                stop_btn = gr.Button("Stop", variant="stop", size="sm")
                regenerate_btn = gr.Button("Regenerate", size="sm")
                clear_main_btn = gr.Button("Clear conversation", size="sm")

        # --- events ----------------------------------------------------

        load_event = demo.load(
            fn=on_load,
            inputs=[current_id, model_selector],
            outputs=[connection_status, model_selector, conversation_list, current_id],
            show_progress="hidden",
        )
        demo.load(
            fn=None,
            inputs=None,
            outputs=None,
            js=(
                "() => {"
                "try {"
                "const stored = localStorage.getItem('local-chatbot-theme');"
                "if (stored === 'dark') document.body.classList.add('dark');"
                "if (stored === 'light') document.body.classList.remove('dark');"
                "} catch (e) {}"
                "}"
            ),
            api_visibility="private",
        )
        retry_btn.click(
            fn=on_retry,
            inputs=[model_selector],
            outputs=[connection_status, model_selector],
            show_progress="hidden",
        )
        preset_selector.change(
            fn=on_preset,
            inputs=[preset_selector],
            outputs=[system_prompt],
            show_progress="hidden",
        )
        save_prefs_btn.click(
            fn=on_save_preferences,
            inputs=[system_prompt, temperature, max_tokens, model_selector],
            outputs=[status],
            show_progress="hidden",
        )
        new_chat_btn.click(
            fn=on_new_chat,
            inputs=None,
            outputs=[
                chatbot,
                current_id,
                message_box,
                status,
                conversation_list,
                rename_box,
            ],
            show_progress="hidden",
        )
        conversation_list.change(
            fn=on_select,
            inputs=[conversation_list],
            outputs=[
                chatbot,
                current_id,
                model_selector,
                system_prompt,
                temperature,
                max_tokens,
                rename_box,
                status,
            ],
            show_progress="hidden",
        )
        rename_btn.click(
            fn=on_rename,
            inputs=[current_id, rename_box],
            outputs=[conversation_list, rename_box, status],
            show_progress="hidden",
        )
        delete_btn.click(
            fn=on_delete,
            inputs=[current_id],
            outputs=[
                chatbot,
                current_id,
                status,
                conversation_list,
                rename_box,
                message_box,
            ],
            show_progress="hidden",
        )
        clear_sidebar_btn.click(
            fn=on_clear,
            inputs=[current_id],
            outputs=[chatbot, status, conversation_list],
            show_progress="hidden",
        )
        clear_main_btn.click(
            fn=on_clear,
            inputs=[current_id],
            outputs=[chatbot, status, conversation_list],
            show_progress="hidden",
        )

        send_outputs = [
            chatbot,
            current_id,
            message_box,
            status,
            conversation_list,
            rename_box,
        ]
        send_inputs = [
            message_box,
            chatbot,
            current_id,
            model_selector,
            system_prompt,
            temperature,
            max_tokens,
        ]
        send_event = send_btn.click(
            fn=on_send,
            inputs=send_inputs,
            outputs=send_outputs,
            show_progress="hidden",
        )
        submit_event = message_box.submit(
            fn=on_send,
            inputs=send_inputs,
            outputs=send_outputs,
            show_progress="hidden",
        )
        regen_event = regenerate_btn.click(
            fn=on_regenerate,
            inputs=[
                chatbot,
                current_id,
                model_selector,
                system_prompt,
                temperature,
                max_tokens,
            ],
            outputs=send_outputs[:4] + [conversation_list, rename_box],
            show_progress="hidden",
        )
        stop_btn.click(
            fn=on_stop,
            inputs=None,
            outputs=[status],
            cancels=[send_event, submit_event, regen_event],
            show_progress="hidden",
        )
        theme_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=(
                "() => {"
                "document.body.classList.toggle('dark');"
                "const dark = document.body.classList.contains('dark');"
                "try { localStorage.setItem('local-chatbot-theme',"
                " dark ? 'dark' : 'light'); } catch (e) {}"
                "}"
            ),
            api_visibility="private",
        )

    return demo


def _consume_stream(
    engine: ChatEngine,
    updates: Iterator[StreamUpdate],
    ui_history: list[dict[str, str]],
    conversation_id: int | None,
    model: str,
) -> Iterator[tuple[object, ...]]:
    """Relay engine updates to Gradio while keeping the sidebar in sync."""
    list_update: dict[str, object] = gr.update()
    rename_update: dict[str, object] = gr.update()
    listed = False
    finished = False
    current = conversation_id
    try:
        for update in updates:
            current = update.conversation_id
            if update.error:
                if update.text:
                    ui_history[-1]["content"] = update.text
                elif (
                    ui_history
                    and ui_history[-1]["role"] == "assistant"
                    and not ui_history[-1]["content"]
                ):
                    ui_history.pop()
                status = f"**Error:** {update.error}"
            else:
                if ui_history:
                    ui_history[-1]["content"] = update.text
                status = "Ready." if update.done else f"Generating with `{model}`…"

            if not listed or update.done:
                conversation = engine.get_conversation(update.conversation_id)
                list_update = gr.update(
                    choices=conversation_choices(engine.list_conversations()),
                    value=update.conversation_id,
                )
                rename_update = gr.update(
                    value=conversation.title if conversation else ""
                )
                listed = True

            yield (
                ui_history,
                update.conversation_id,
                "",
                status,
                list_update,
                rename_update,
            )
        finished = True
    except ChatbotError as exc:
        if (
            ui_history
            and ui_history[-1]["role"] == "assistant"
            and not ui_history[-1]["content"]
        ):
            ui_history.pop()
        yield (
            ui_history,
            current,
            "",
            f"**Error:** {exc}",
            list_update,
            rename_update,
        )
        finished = True
    except Exception as exc:
        if (
            ui_history
            and ui_history[-1]["role"] == "assistant"
            and not ui_history[-1]["content"]
        ):
            ui_history.pop()
        yield (
            ui_history,
            current,
            "",
            f"**Error:** Unexpected error: {exc}",
            list_update,
            rename_update,
        )
        finished = True
    finally:
        if not finished:
            # Gradio cancelled this generator (Stop button): close the engine
            # generator so it can persist a partial reply.
            updates.close()


def _connection_snapshot(
    engine: ChatEngine, preferred_model: str | None
) -> tuple[str, dict[str, object]]:
    status = engine.connection_status()
    models = list(status.models)
    if not status.connected:
        detail = ollama_unavailable_help(engine.client.base_url)
        if status.error:
            detail += f"\n\nDetails: {status.error}"
        return detail, gr.update(choices=[], value=None, interactive=False)
    if not models:
        return no_models_help(), gr.update(
            choices=[], value=None, interactive=False
        )

    names = [model.name for model in models]
    selected = preferred_model if preferred_model in names else names[0]
    choices = [(model.label, model.name) for model in models]
    version = f" Ollama {status.version}." if status.version else ""
    message = (
        f"Connected to{version} {len(models)} model(s) installed. "
        "All inference runs locally."
    )
    if status.error:
        message += f"\n\n{status.error}"
    return message, gr.update(
        choices=choices, value=selected, interactive=True
    )


def _normalise_history(
    history: Sequence[dict[str, str]] | None,
) -> list[dict[str, str]]:
    cleaned: list[dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role in ("user", "assistant") and isinstance(content, str):
            cleaned.append({"role": role, "content": content})
    return cleaned
