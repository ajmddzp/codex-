from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping
import uuid

ALLOWED_ROLES = {"user", "assistant", "error", "system"}


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def render_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """
    Normalize message payload shape for frontend rendering.

    TODO:
    - Add markdown/code-block render metadata if your UI needs it.
    """
    role = str(message.get("role") or "assistant")
    if role not in ALLOWED_ROLES:
        role = "assistant"
    return {
        "id": str(message.get("id") or f"m_{uuid.uuid4().hex[:8]}"),
        "role": role,
        "content": str(message.get("content") or message.get("text") or ""),
        "created_at": str(message.get("created_at") or _now_iso()),
    }
