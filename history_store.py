from __future__ import annotations

from datetime import datetime
from threading import RLock
from typing import Any
import uuid

_LOCK = RLock()
_CONVERSATIONS: dict[str, dict[str, Any]] = {}
_MESSAGES: dict[str, list[dict[str, Any]]] = {}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def list_conversations() -> list[dict[str, Any]]:
    with _LOCK:
        conversations = [dict(item) for item in _CONVERSATIONS.values()]
    conversations.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return conversations


def create_conversation(*, title: str | None, model: str | None) -> dict[str, Any]:
    conversation = {
        "id": f"conv_{uuid.uuid4().hex[:8]}",
        "title": (title or "New Chat").strip() or "New Chat",
        "model": (model or "gpt-5.4").strip() or "gpt-5.4",
        "updated_at": _now_iso(),
    }
    with _LOCK:
        _CONVERSATIONS[conversation["id"]] = dict(conversation)
        _MESSAGES[conversation["id"]] = []
    return dict(conversation)


def get_conversation(*, conversation_id: str) -> dict[str, Any]:
    conv_id = str(conversation_id or "").strip()
    if not conv_id:
        raise ValueError("conversation_id_required")
    with _LOCK:
        conversation = _CONVERSATIONS.get(conv_id)
    if conversation is None:
        raise KeyError("conversation_not_found")
    return dict(conversation)


def list_messages(*, conversation_id: str) -> list[dict[str, Any]]:
    conv_id = str(conversation_id or "").strip()
    if not conv_id:
        raise ValueError("conversation_id_required")
    with _LOCK:
        if conv_id not in _CONVERSATIONS:
            raise KeyError("conversation_not_found")
        return [dict(item) for item in _MESSAGES.get(conv_id, [])]


def append_message(*, conversation_id: str, role: str, content: str) -> dict[str, Any]:
    conv_id = str(conversation_id or "").strip()
    if not conv_id:
        raise ValueError("conversation_id_required")
    text = str(content or "").strip()
    if not text:
        raise ValueError("message_is_empty")
    normalized_role = str(role or "").strip() or "assistant"
    message = {
        "id": f"m_{uuid.uuid4().hex[:8]}",
        "conversation_id": conv_id,
        "role": normalized_role,
        "content": text,
        "created_at": _now_iso(),
    }
    with _LOCK:
        if conv_id not in _CONVERSATIONS:
            raise KeyError("conversation_not_found")
        _MESSAGES.setdefault(conv_id, []).append(dict(message))
        _CONVERSATIONS[conv_id]["updated_at"] = _now_iso()
    return dict(message)


def set_conversation_model(*, conversation_id: str, model: str) -> dict[str, Any]:
    conv_id = str(conversation_id or "").strip()
    if not conv_id:
        raise ValueError("conversation_id_required")
    next_model = str(model or "").strip()
    if not next_model:
        raise ValueError("model_is_empty")

    with _LOCK:
        if conv_id not in _CONVERSATIONS:
            raise KeyError("conversation_not_found")
        _CONVERSATIONS[conv_id]["model"] = next_model
        _CONVERSATIONS[conv_id]["updated_at"] = _now_iso()
        updated = dict(_CONVERSATIONS[conv_id])
    return updated
