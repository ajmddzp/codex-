from __future__ import annotations

from typing import Any

from file_store import save_file


def upload_file(*, filename: str, content: bytes, content_type: str | None) -> dict[str, Any]:
    return save_file(filename=filename, content=content, content_type=content_type)
