from __future__ import annotations

from typing import Any

from history_store import list_messages as store_list_messages


def list_messages(*, conversation_id: str) -> list[dict[str, Any]]:
    return store_list_messages(conversation_id=conversation_id)
