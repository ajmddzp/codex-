from __future__ import annotations

from history_store import set_conversation_model

def switch_model(*, conversation_id: str | None, model: str) -> str:
    next_model = str(model or "").strip()
    if not next_model:
        raise ValueError("model_is_empty")
    conv_id = str(conversation_id or "").strip()
    if conv_id:
        set_conversation_model(conversation_id=conv_id, model=next_model)
    return next_model
