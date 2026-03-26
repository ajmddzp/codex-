from __future__ import annotations

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
    raise NotImplementedError


def save_uploaded_file(
    filename: str,
    content: bytes,
    upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> UploadedFile:
    """
    Save one uploaded file and return public metadata.
    """
    raise NotImplementedError


def save_upload_request(
    content_type: str,
    raw_body: bytes,
    upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> UploadedFile:
    """
    Parse raw HTTP upload payload and save file in one call.
    """
    raise NotImplementedError


def get_uploaded_file(
    file_id: str,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> UploadedFile | None:
    """
    Return uploaded file metadata by id.
    """
    raise NotImplementedError


def list_uploaded_files(
    file_ids: Iterable[str] | None = None,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> list[UploadedFile]:
    """
    Return uploaded files. If file_ids is set, return that subset only.
    """
    raise NotImplementedError


def delete_uploaded_file(
    file_id: str,
    upload_dir: Path | str = DEFAULT_UPLOAD_DIR,
    index_path: Path | str = DEFAULT_FILE_INDEX_PATH,
) -> bool:
    """
    Delete uploaded file and metadata. Return True if deleted.
    """
    raise NotImplementedError
