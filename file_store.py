from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from backend_types import UploadedFile


DEFAULT_UPLOAD_DIR = Path("uploads")
DEFAULT_FILE_INDEX_PATH = DEFAULT_UPLOAD_DIR / "files_index.json"


def parse_multipart_file(
    content_type: str,
    raw_body: bytes,
) -> tuple[str, bytes]:
    """
    Parse multipart/form-data body and return (filename, file_content).
    """
    boundary_match = re.search(r"boundary=([^;]+)", content_type or "", re.IGNORECASE)
    if not boundary_match:
        raise ValueError("missing multipart boundary")

    boundary = boundary_match.group(1).strip().strip('"')
    if not boundary:
        raise ValueError("empty multipart boundary")

    marker = b"--" + boundary.encode("utf-8")
    parts = raw_body.split(marker)
    for part in parts:
        chunk = part.strip()
        if not chunk or chunk == b"--":
            continue

        # Remove final terminator if present.
        if chunk.endswith(b"--"):
            chunk = chunk[:-2]
        chunk = chunk.strip(b"\r\n")

        head, sep, body = chunk.partition(b"\r\n\r\n")
        if not sep:
            continue

        headers_text = head.decode("latin1", errors="ignore")
        if "name=\"file\"" not in headers_text and "filename=" not in headers_text:
            continue

        filename_match = re.search(r'filename="([^"]*)"', headers_text)
        filename = filename_match.group(1).strip() if filename_match else "unnamed_file"
        content = body.rstrip(b"\r\n")
        return (filename or "unnamed_file", content)

    raise ValueError("multipart file field not found")


def save_uploaded_file(
    filename: str,
    content: bytes,
    upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> UploadedFile:
    """
    Save one uploaded file and return public metadata.
    """
    up_dir = Path(upload_dir)
    idx_path = Path(index_path)
    up_dir.mkdir(parents=True, exist_ok=True)
    idx_path.parent.mkdir(parents=True, exist_ok=True)

    safe_name = _sanitize_filename(filename)
    suffix = Path(safe_name).suffix
    file_id = f"file_{uuid.uuid4().hex[:16]}"
    stored_name = f"{file_id}{suffix}"
    stored_path = up_dir / stored_name
    stored_path.write_bytes(content)

    size = len(content)
    now = datetime.now(timezone.utc).isoformat()

    index = _read_index(idx_path)
    index[file_id] = {
        "id": file_id,
        "name": safe_name,
        "size": size,
        "path": str(stored_path),
        "created_at": now,
    }
    _write_index(idx_path, index)

    return {"id": file_id, "name": safe_name, "size": size}


def save_upload_request(
    content_type: str,
    raw_body: bytes,
    upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> UploadedFile:
    """
    Parse raw HTTP upload payload and save file in one call.
    """
    filename, content = parse_multipart_file(content_type, raw_body)
    return save_uploaded_file(filename, content, upload_dir=upload_dir, index_path=index_path)


def get_uploaded_file(
    file_id: str,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> UploadedFile | None:
    """
    Return uploaded file metadata by id.
    """
    record = _get_record(file_id, Path(index_path))
    if record is None:
        return None
    return {
        "id": str(record.get("id", file_id)),
        "name": str(record.get("name", "")),
        "size": _to_int_or_none(record.get("size")),
    }


def list_uploaded_files(
    file_ids: Iterable[str] | None = None,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> list[UploadedFile]:
    """
    Return uploaded files. If file_ids is set, return that subset only.
    """
    idx_path = Path(index_path)
    index = _read_index(idx_path)

    if file_ids is None:
        keys = list(index.keys())
    else:
        keys = [str(fid) for fid in file_ids if str(fid).strip()]

    out: list[UploadedFile] = []
    for fid in keys:
        record = index.get(fid)
        if not isinstance(record, dict):
            continue
        out.append(
            {
                "id": str(record.get("id", fid)),
                "name": str(record.get("name", "")),
                "size": _to_int_or_none(record.get("size")),
            }
        )
    return out


def delete_uploaded_file(
    file_id: str,
    upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> bool:
    """
    Delete uploaded file and metadata. Return True if deleted.
    """
    _ = Path(upload_dir)
    idx_path = Path(index_path)
    index = _read_index(idx_path)
    record = index.get(file_id)
    if not isinstance(record, dict):
        return False

    path_str = str(record.get("path", "")).strip()
    if path_str:
        file_path = Path(path_str)
        try:
            if file_path.exists():
                file_path.unlink()
        except OSError:
            pass

    index.pop(file_id, None)
    _write_index(idx_path, index)
    return True


def build_attachment_texts(
    file_ids: Iterable[str],
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
    *,
    max_chars_per_file: int = 4000,
    max_total_chars: int = 12000,
) -> list[dict[str, str | bool]]:
    """
    Read uploaded files by ids and return text blocks for model prompt.
    """
    idx_path = Path(index_path)
    index = _read_index(idx_path)
    remaining_total = max_total_chars
    blocks: list[dict[str, str | bool]] = []

    for raw_id in file_ids:
        file_id = str(raw_id).strip()
        if not file_id:
            continue
        record = index.get(file_id)
        if not isinstance(record, dict):
            continue

        name = str(record.get("name", file_id))
        path_str = str(record.get("path", "")).strip()
        if not path_str:
            continue

        file_path = Path(path_str)
        if not file_path.exists():
            continue

        try:
            raw = file_path.read_bytes()
        except OSError:
            continue

        text = _bytes_to_text(raw)
        if not text:
            text = "[binary or empty file omitted]"

        if remaining_total <= 0:
            break

        cap = min(max_chars_per_file, remaining_total)
        truncated = len(text) > cap
        snippet = text[:cap]
        remaining_total -= len(snippet)

        blocks.append(
            {
                "id": file_id,
                "name": name,
                "content": snippet,
                "truncated": truncated,
            }
        )

    return blocks


def _sanitize_filename(filename: str) -> str:
    base = Path(filename or "unnamed_file").name
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not safe:
        return "unnamed_file"
    if len(safe) > 120:
        stem = Path(safe).stem[:96]
        suffix = Path(safe).suffix[:20]
        return (stem or "file") + suffix
    return safe


def _read_index(index_path: Path) -> dict[str, dict]:
    if not index_path.exists():
        return {}
    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _write_index(index_path: Path, data: dict[str, dict]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_record(file_id: str, index_path: Path) -> dict | None:
    index = _read_index(index_path)
    record = index.get(file_id)
    return record if isinstance(record, dict) else None


def _to_int_or_none(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _bytes_to_text(raw: bytes) -> str:
    # Fast binary guard.
    if b"\x00" in raw[:2048]:
        return ""

    for encoding in ("utf-8", "utf-16", "gb18030", "latin1"):
        try:
            text = raw.decode(encoding, errors="ignore")
            if text.strip():
                return text
        except Exception:
            continue
    return ""
