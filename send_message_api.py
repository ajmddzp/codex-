from __future__ import annotations

from typing import Any
from pathlib import Path

from file_store import get_files
from history_store import append_message, get_conversation, list_messages, set_conversation_model
from model_client import (
    chat_completion,
    load_model_config,
    response_completion_with_uploaded_files,
)


CONFIG_PATH = Path(__file__).with_name("config.json")


def send_message(
    *,
    conversation_id: str,
    message: str,
    file_ids: list[str] | None,
    model: str | None,
) -> dict[str, Any]:
    conv_id = str(conversation_id or "").strip()
    text = str(message or "").strip()
    if not conv_id:
        raise ValueError("conversation_id_required")
    if not text:
        raise ValueError("message_is_empty")

    conversation = get_conversation(conversation_id=conv_id)
    default_model = load_model_config(config_path=CONFIG_PATH).model
    selected_model = (
        str(model or "").strip()
        or str(conversation.get("model") or "").strip()
        or default_model
    )
    selected_files = get_files(file_ids=file_ids or [])
    history_before = list_messages(conversation_id=conv_id)

    append_message(conversation_id=conv_id, role="user", content=text)
    model_history = _build_model_messages(history_before)

    if selected_files:
        reply_text, used_model = response_completion_with_uploaded_files(
            config_path=CONFIG_PATH,
            model=selected_model,
            history_messages=model_history,
            user_text=text,
            files=selected_files,
        )
    else:
        reply_text, used_model = chat_completion(
            config_path=CONFIG_PATH,
            model=selected_model,
            messages=[*model_history, {"role": "user", "content": text}],
        )

    set_conversation_model(conversation_id=conv_id, model=used_model)
    assistant_message = append_message(
        conversation_id=conv_id,
        role="assistant",
        content=reply_text,
    )
    return assistant_message


def _build_model_messages(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for item in history:
        role = str(item.get("role") or "").strip()
        content = str(item.get("model_content") or item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages
