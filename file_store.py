from __future__ import annotations

from threading import RLock
from typing import Any
import uuid

_LOCK = RLock()
_FILES: dict[str, dict[str, Any]] = {}


def save_file(*, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]:
    safe_name = str(filename or "").strip() or "upload.bin"
    item = {
        "id": f"file_{uuid.uuid4().hex[:8]}",
        "name": safe_name,
        "size": len(content),
        "content_type": content_type,
    }
    with _LOCK:
        _FILES[item["id"]] = dict(item)
    return dict(item)


def get_files(*, file_ids: list[str]) -> list[dict[str, Any]]:
    ids = [str(item).strip() for item in (file_ids or []) if str(item).strip()]
    with _LOCK:
        return [dict(_FILES[file_id]) for file_id in ids if file_id in _FILES]
