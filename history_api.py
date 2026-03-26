from __future__ import annotations

from typing import Any

from history_store import list_conversations as store_list_conversations


def list_conversations() -> list[dict[str, Any]]:
    return store_list_conversations()
