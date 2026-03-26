from __future__ import annotations

from typing import Any
from pathlib import Path

from file_store import get_files
from history_store import append_message, get_conversation, list_messages, set_conversation_model
from model_client import chat_completion, load_model_config


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

    append_message(conversation_id=conv_id, role="user", content=text)

    # TODO: include real uploaded file contents in model prompt.
    _ = get_files(file_ids=file_ids or [])

    history = list_messages(conversation_id=conv_id)
    model_messages = _build_model_messages(history)
    reply_text, used_model = chat_completion(
        config_path=CONFIG_PATH,
        model=selected_model,
        messages=model_messages,
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
        content = str(item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        messages.append({"role": role, "content": content})
    return messages
