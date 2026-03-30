from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import BoundedSemaphore, Lock, Semaphore
from typing import Any, Callable, TypeVar

from openai import OpenAI

MAX_IN_FLIGHT_REQUESTS = 1
MAX_QUEUED_REQUESTS = 24
QUEUE_WAIT_TIMEOUT_SECONDS = 45
FILE_REQUEST_WAIT_TIMEOUT_SECONDS = 60
MAX_RETRY_ATTEMPTS = 3
MAX_INLINE_IMAGE_BYTES = 8 * 1024 * 1024
REMOTE_FILE_CACHE_TTL_SECONDS = 30 * 60
MAX_REMOTE_FILE_CACHE_ITEMS = 1024

_MODEL_REQUEST_SLOTS = Semaphore(MAX_IN_FLIGHT_REQUESTS)
_MODEL_QUEUE_SLOTS = BoundedSemaphore(MAX_IN_FLIGHT_REQUESTS + MAX_QUEUED_REQUESTS)
_FILE_REQUEST_SLOT = Semaphore(1)
_REMOTE_FILE_CACHE_LOCK = Lock()
_REMOTE_FILE_CACHE: dict[str, dict[str, Any]] = {}
T = TypeVar("T")


class ModelBusyError(RuntimeError):
    pass


@dataclass(slots=True)
class ModelConfig:
    base_url: str
    api_key: str
    model: str
    candidate_models: list[str]


def load_model_config(*, config_path: Path) -> ModelConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"config_not_found: {config_path}")

    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid_config_json")
    llm = data.get("llm")
    if not isinstance(llm, dict):
        raise ValueError("invalid_llm_config")

    base_url = str(llm.get("base_url") or "").strip()
    api_key = str(llm.get("api_key") or "").strip()
    model = str(llm.get("model") or "").strip()
    candidate_models_raw = llm.get("candidate_models")
    candidate_models = [
        str(item).strip() for item in (candidate_models_raw or []) if str(item).strip()
    ]
    if not base_url:
        raise ValueError("llm.base_url_is_empty")
    if not api_key:
        raise ValueError("llm.api_key_is_empty")
    if not model:
        raise ValueError("llm.model_is_empty")
    if model not in candidate_models:
        candidate_models.insert(0, model)

    return ModelConfig(
        base_url=base_url,
        api_key=api_key,
        model=model,
        candidate_models=candidate_models,
    )


def _extract_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        return "".join(parts).strip()
    return str(content or "").strip()


def chat_completion(
    *,
    config_path: Path,
    model: str | None,
    messages: list[dict[str, str]],
) -> tuple[str, str]:
    cfg = load_model_config(config_path=config_path)

    if not messages:
        raise ValueError("messages_is_empty")

    selected_model = str(model or "").strip() or cfg.model
    try_models = _build_try_models(
        requested_model=selected_model,
        default_model=cfg.model,
        candidate_models=cfg.candidate_models,
    )
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    errors: list[str] = []
    with _acquire_model_capacity():
        for current_model in try_models:
            try:
                response = _run_with_retry(
                    lambda: client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                    ),
                )
                text = _extract_text(response)
                if text:
                    return text, current_model
                errors.append(f"{current_model}: empty_model_response")
            except Exception as exc:
                errors.append(f"{current_model}: {exc}")
                if _is_concurrency_limited_error(exc) or _is_provider_unavailable_error(
                    exc
                ):
                    break

    error_text = " | ".join(errors) if errors else "unknown_model_error"
    raise RuntimeError(f"model_request_failed: {error_text}")


def response_completion_with_uploaded_files(
    *,
    config_path: Path,
    model: str | None,
    history_messages: list[dict[str, str]],
    user_text: str,
    files: list[dict[str, Any]],
) -> tuple[str, str]:
    cfg = load_model_config(config_path=config_path)
    text = str(user_text or "").strip()
    if not text:
        raise ValueError("message_is_empty")

    selected_model = str(model or "").strip() or cfg.model
    try_models = _build_try_models(
        requested_model=selected_model,
        default_model=cfg.model,
        candidate_models=cfg.candidate_models,
    )
    client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
    errors: list[str] = []
    cache_scope = _build_cache_scope(cfg=cfg)
    with _acquire_file_message_slot():
        with _acquire_model_capacity():
            if _all_files_inline_image_capable(files):
                response_input = _build_responses_input_with_inline_images(
                    history_messages=history_messages,
                    user_text=text,
                    files=files,
                )
            else:
                remote_file_ids = [
                    _upload_file_for_responses(
                        client=client,
                        file_item=file_item,
                        cache_scope=cache_scope,
                    )
                    for file_item in files
                ]
                response_input = _build_responses_input(
                    history_messages=history_messages,
                    user_text=text,
                    remote_file_ids=remote_file_ids,
                )

            for current_model in try_models:
                try:
                    response = _run_with_retry(
                        lambda: client.responses.create(
                            model=current_model,
                            input=response_input,
                        ),
                    )
                    response_text = _extract_responses_text(response)
                    if response_text:
                        return response_text, current_model
                    errors.append(f"{current_model}: empty_model_response")
                except Exception as exc:
                    errors.append(f"{current_model}: {exc}")
                    if _is_concurrency_limited_error(exc) or _is_provider_unavailable_error(
                        exc
                    ):
                        break

    error_text = " | ".join(errors) if errors else "unknown_model_error"
    raise RuntimeError(f"model_request_failed: {error_text}")


def _upload_file_for_responses(
    *, client: OpenAI, file_item: dict[str, Any], cache_scope: str
) -> str:
    name = str(file_item.get("name") or "upload.bin")
    mime_type = _guess_mime_type(file_item)
    raw = file_item.get("_content")
    if not isinstance(raw, (bytes, bytearray)):
        raw = b""
    content = bytes(raw)
    if not content:
        raise ValueError(f"file_is_empty: {name}")
    cache_key = _build_remote_file_cache_key(
        cache_scope=cache_scope,
        name=name,
        mime_type=mime_type,
        content=content,
    )
    cached = _get_cached_remote_file_id(cache_key=cache_key)
    if cached:
        return cached

    try:
        uploaded = _run_with_retry(
            lambda: _files_create(
                client=client, content=content, name=name, purpose="user_data"
            )
        )
        file_id = str(getattr(uploaded, "id", "")).strip()
        if file_id:
            _set_cached_remote_file_id(cache_key=cache_key, file_id=file_id)
            return file_id
    except Exception as exc:
        if _is_concurrency_limited_error(exc):
            detail = _build_upstream_busy_detail(exc)
            raise ModelBusyError(detail) from exc
        if not _is_invalid_purpose_error(exc):
            raise RuntimeError(f"file_upload_failed({name}): {exc}") from exc

        try:
            uploaded = _run_with_retry(
                lambda: _files_create(
                    client=client,
                    content=content,
                    name=name,
                    purpose="assistants",
                )
            )
            file_id = str(getattr(uploaded, "id", "")).strip()
            if file_id:
                _set_cached_remote_file_id(cache_key=cache_key, file_id=file_id)
                return file_id
        except Exception as fallback_exc:
            if _is_concurrency_limited_error(fallback_exc):
                detail = _build_upstream_busy_detail(fallback_exc)
                raise ModelBusyError(detail) from fallback_exc
            raise RuntimeError(
                f"file_upload_failed({name}): {fallback_exc}"
            ) from fallback_exc

    raise RuntimeError(f"file_upload_failed({name}): empty_file_id")


def _build_responses_input(
    *,
    history_messages: list[dict[str, str]],
    user_text: str,
    remote_file_ids: list[str],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for item in history_messages:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        inputs.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            }
        )

    user_content: list[dict[str, str]] = [{"type": "input_text", "text": user_text}]
    for file_id in remote_file_ids:
        user_content.append({"type": "input_file", "file_id": file_id})
    inputs.append({"role": "user", "content": user_content})
    return inputs


def _extract_responses_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    output = getattr(response, "output", None)
    if not output:
        return ""

    parts: list[str] = []
    for item in output:
        content_items = getattr(item, "content", None)
        if content_items is None and isinstance(item, dict):
            content_items = item.get("content")
        if not content_items:
            continue
        for content_item in content_items:
            text_value = None
            if isinstance(content_item, dict):
                text_value = content_item.get("text")
            else:
                text_value = getattr(content_item, "text", None)
            if text_value:
                parts.append(str(text_value))
    return "".join(parts).strip()


def _build_responses_input_with_inline_images(
    *,
    history_messages: list[dict[str, str]],
    user_text: str,
    files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    inputs: list[dict[str, Any]] = []
    for item in history_messages:
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"system", "user", "assistant"}:
            continue
        if not content:
            continue
        inputs.append(
            {
                "role": role,
                "content": [{"type": "input_text", "text": content}],
            }
        )

    user_content: list[dict[str, str]] = [{"type": "input_text", "text": user_text}]
    for file_item in files:
        name = str(file_item.get("name") or "upload.bin")
        mime = _guess_mime_type(file_item)
        raw = file_item.get("_content")
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            raise ValueError(f"file_is_empty: {name}")
        content = bytes(raw)
        if len(content) > MAX_INLINE_IMAGE_BYTES:
            raise ValueError(f"image_too_large_for_inline: {name}")
        encoded = base64.b64encode(content).decode("ascii")
        user_content.append(
            {
                "type": "input_image",
                "image_url": f"data:{mime};base64,{encoded}",
            }
        )

    inputs.append({"role": "user", "content": user_content})
    return inputs


def _build_try_models(
    *, requested_model: str, default_model: str, candidate_models: list[str]
) -> list[str]:
    ordered = [requested_model, default_model, *candidate_models]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        name = str(item or "").strip()
        if not name:
            continue
        for variant in _model_name_variants(name):
            if variant in seen:
                continue
            seen.add(variant)
            deduped.append(variant)
    return deduped


def _model_name_variants(name: str) -> list[str]:
    variants = [name]
    lowered = name.lower()
    if lowered != name:
        variants.append(lowered)
    return variants


def _is_concurrency_limited_error(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = (
        "rate_limit_exceeded",
        "concurrent",
        "concurrent_sessions",
        "too many requests",
        "429",
    )
    return any(token in text for token in tokens)


def _is_provider_unavailable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = (
        "no_available_providers",
        "service_unavailable_error",
        "all providers",
        "503",
    )
    return any(token in text for token in tokens)


def _is_invalid_purpose_error(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = (
        "invalid purpose",
        "unsupported purpose",
        "invalid file purpose",
        "invalid_request_error",
    )
    return any(token in text for token in tokens)


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    tokens = (
        "service_unavailable_error",
        "no_available_providers",
        "timed out",
        "timeout",
        "connection reset",
        "connection error",
        "503",
    )
    return any(token in text for token in tokens)


def _run_with_retry(call: Callable[[], T]) -> T:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
        try:
            return call()
        except Exception as exc:
            last_error = exc
            if isinstance(exc, ModelBusyError):
                raise
            if attempt >= MAX_RETRY_ATTEMPTS or not _is_retryable_error(exc):
                raise
            time.sleep(0.8 * (2 ** (attempt - 1)))
    if last_error is not None:
        raise last_error
    raise RuntimeError("retry_failed_without_error")


def _files_create(*, client: OpenAI, content: bytes, name: str, purpose: str) -> Any:
    buffer = io.BytesIO(content)
    buffer.name = name
    return client.files.create(file=buffer, purpose=purpose)


def _guess_mime_type(file_item: dict[str, Any]) -> str:
    mime = str(file_item.get("content_type") or "").strip().lower()
    if mime:
        return mime
    name = str(file_item.get("name") or "").lower()
    if name.endswith(".png"):
        return "image/png"
    if name.endswith(".jpg") or name.endswith(".jpeg"):
        return "image/jpeg"
    if name.endswith(".gif"):
        return "image/gif"
    if name.endswith(".webp"):
        return "image/webp"
    return "application/octet-stream"


def _all_files_inline_image_capable(files: list[dict[str, Any]]) -> bool:
    if not files:
        return False
    for file_item in files:
        mime = _guess_mime_type(file_item)
        if not mime.startswith("image/"):
            return False
        raw = file_item.get("_content")
        if not isinstance(raw, (bytes, bytearray)) or not raw:
            return False
        if len(bytes(raw)) > MAX_INLINE_IMAGE_BYTES:
            return False
    return True


def _build_upstream_busy_detail(exc: Exception) -> str:
    reset_time = _extract_reset_time_from_error(exc)
    if reset_time:
        return f"upstream_concurrency_limited_until:{reset_time}"
    return "upstream_concurrency_limited"


def _extract_reset_time_from_error(exc: Exception) -> str | None:
    text = str(exc)
    match = re.search(
        r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)",
        text,
    )
    if not match:
        return None
    iso_text = match.group(1)
    try:
        dt = datetime.fromisoformat(iso_text.replace("Z", "+00:00")).astimezone(
            timezone.utc
        )
        return dt.isoformat().replace("+00:00", "Z")
    except Exception:
        return iso_text


@contextmanager
def _acquire_file_message_slot() -> Any:
    if not _FILE_REQUEST_SLOT.acquire(timeout=FILE_REQUEST_WAIT_TIMEOUT_SECONDS):
        raise ModelBusyError("server_busy_file_queue_timeout")
    try:
        yield
    finally:
        _FILE_REQUEST_SLOT.release()


@contextmanager
def _acquire_model_capacity() -> Any:
    if not _MODEL_QUEUE_SLOTS.acquire(blocking=False):
        raise ModelBusyError("server_busy_queue_full")

    in_flight_acquired = False
    try:
        if not _MODEL_REQUEST_SLOTS.acquire(timeout=QUEUE_WAIT_TIMEOUT_SECONDS):
            raise ModelBusyError("server_busy_queue_timeout")
        in_flight_acquired = True
        yield
    finally:
        if in_flight_acquired:
            _MODEL_REQUEST_SLOTS.release()
        _MODEL_QUEUE_SLOTS.release()


def _build_cache_scope(*, cfg: ModelConfig) -> str:
    seed = f"{cfg.base_url}|{cfg.api_key}".encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:20]


def _build_remote_file_cache_key(
    *, cache_scope: str, name: str, mime_type: str, content: bytes
) -> str:
    digest = hashlib.sha256(content).hexdigest()
    safe_name = name.strip().lower()
    safe_mime = mime_type.strip().lower()
    return f"{cache_scope}|{digest}|{len(content)}|{safe_mime}|{safe_name}"


def _get_cached_remote_file_id(*, cache_key: str) -> str | None:
    now = time.time()
    with _REMOTE_FILE_CACHE_LOCK:
        cached = _REMOTE_FILE_CACHE.get(cache_key)
        if not cached:
            return None
        expires_at = float(cached.get("expires_at", 0))
        if expires_at <= now:
            _REMOTE_FILE_CACHE.pop(cache_key, None)
            return None
        file_id = str(cached.get("file_id") or "").strip()
        if not file_id:
            _REMOTE_FILE_CACHE.pop(cache_key, None)
            return None
        cached["last_used"] = now
        return file_id


def _set_cached_remote_file_id(*, cache_key: str, file_id: str) -> None:
    now = time.time()
    entry = {
        "file_id": file_id,
        "expires_at": now + REMOTE_FILE_CACHE_TTL_SECONDS,
        "last_used": now,
    }
    with _REMOTE_FILE_CACHE_LOCK:
        _REMOTE_FILE_CACHE[cache_key] = entry
        _prune_remote_file_cache(now=now)


def _prune_remote_file_cache(*, now: float) -> None:
    expired_keys = [
        key
        for key, value in _REMOTE_FILE_CACHE.items()
        if float(value.get("expires_at", 0)) <= now
    ]
    for key in expired_keys:
        _REMOTE_FILE_CACHE.pop(key, None)

    if len(_REMOTE_FILE_CACHE) <= MAX_REMOTE_FILE_CACHE_ITEMS:
        return

    overflow = len(_REMOTE_FILE_CACHE) - MAX_REMOTE_FILE_CACHE_ITEMS
    sorted_items = sorted(
        _REMOTE_FILE_CACHE.items(),
        key=lambda item: float(item[1].get("last_used", 0)),
    )
    for idx in range(overflow):
        key = sorted_items[idx][0]
        _REMOTE_FILE_CACHE.pop(key, None)
