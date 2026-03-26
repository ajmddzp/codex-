from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib import request
from urllib.error import HTTPError, URLError

from backend_types import ModelMessage


DEFAULT_CONFIG_PATH = Path("config.json")


@dataclass(slots=True)
class ModelClientConfig:
    base_url: str
    api_key: str
    default_model: str
    candidate_models: list[str]
    timeout_seconds: float = 60.0


class ModelClientError(RuntimeError):
    pass


def load_model_config(config_path: Path | str = DEFAULT_CONFIG_PATH) -> ModelClientConfig:
    """
    Load LLM config from config.json.
    """
    path = Path(config_path)
    if not path.exists():
        raise ModelClientError(f"config file not found: {path}")

    try:
        root = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelClientError(f"invalid config json: {exc}") from exc

    llm = root.get("llm")
    if not isinstance(llm, dict):
        raise ModelClientError("missing 'llm' object in config")

    base_url = str(llm.get("base_url", "")).strip()
    api_key = str(llm.get("api_key", "")).strip()
    default_model = str(llm.get("model", "")).strip()
    candidates_raw = llm.get("candidate_models", [])
    candidate_models = [str(item).strip() for item in candidates_raw if str(item).strip()]

    if not default_model:
        default_model = candidate_models[0] if candidate_models else "gpt-5"
    if default_model and default_model not in candidate_models:
        candidate_models.insert(0, default_model)

    timeout_raw = llm.get("timeout_seconds", 60)
    try:
        timeout_seconds = float(timeout_raw)
    except Exception:
        timeout_seconds = 60.0

    if not base_url:
        raise ModelClientError("llm.base_url is required")
    if not api_key:
        raise ModelClientError("llm.api_key is required")

    return ModelClientConfig(
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        default_model=default_model,
        candidate_models=candidate_models,
        timeout_seconds=max(1.0, timeout_seconds),
    )


def list_candidate_models(config_path: Path | str = DEFAULT_CONFIG_PATH) -> list[str]:
    """
    Return configured candidate model names.
    """
    cfg = load_model_config(config_path)
    return cfg.candidate_models[:] if cfg.candidate_models else [cfg.default_model]


def merge_user_message_with_attachments(
    user_message: str,
    attachments: Sequence[dict[str, object]] | None,
    *,
    max_chars_total: int = 12000,
) -> str:
    """
    Merge uploaded file text blocks into the current user message.
    """
    text = user_message.strip()
    blocks = attachments or []
    if not blocks:
        return text

    out: list[str] = [text] if text else []
    out.append("以下是用户本次附带的文件内容：")

    used = 0
    for idx, block in enumerate(blocks, 1):
        name = str(block.get("name", f"file_{idx}"))
        content = str(block.get("content", ""))
        if not content:
            continue

        remain = max_chars_total - used
        if remain <= 0:
            break

        snippet = content[:remain]
        used += len(snippet)

        out.append(f"[文件 {idx}] {name}")
        out.append("```text")
        out.append(snippet)
        out.append("```")
        if bool(block.get("truncated", False)) or len(snippet) < len(content):
            out.append("(该文件内容已截断)")

    if used == 0:
        out.append("(未读取到可用文件文本内容)")

    return "\n".join(part for part in out if part).strip()


def chat_completion(
    *,
    config: ModelClientConfig,
    model: str,
    messages: Sequence[ModelMessage],
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, str] | None = None,
) -> str:
    """
    Call cloud model and return final assistant text.
    """
    payload = _build_payload(
        model=model or config.default_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
        metadata=metadata,
    )
    response_data = _post_json(
        url=f"{config.base_url}/chat/completions",
        payload=payload,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
    )
    return _extract_text_from_chat_completion(response_data)


def stream_chat_completion(
    *,
    config: ModelClientConfig,
    model: str,
    messages: Sequence[ModelMessage],
    temperature: float | None = None,
    max_tokens: int | None = None,
    metadata: dict[str, str] | None = None,
) -> Iterator[str]:
    """
    Stream cloud model output in chunks.
    """
    payload = _build_payload(
        model=model or config.default_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
        metadata=metadata,
    )
    yield from _stream_chat_completion(
        url=f"{config.base_url}/chat/completions",
        payload=payload,
        api_key=config.api_key,
        timeout_seconds=config.timeout_seconds,
    )


def compose_messages_for_request(
    *,
    system_prompt: str | None,
    history_messages: Iterable[ModelMessage],
    user_message: str,
) -> list[ModelMessage]:
    """
    Compose model input messages from system prompt, history and user message.
    """
    out: list[ModelMessage] = []

    if system_prompt and system_prompt.strip():
        out.append({"role": "system", "content": system_prompt.strip()})

    for message in history_messages:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()
        if role not in ("system", "user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content})

    user_text = user_message.strip()
    if user_text:
        out.append({"role": "user", "content": user_text})

    return out


def _build_payload(
    *,
    model: str,
    messages: Sequence[ModelMessage],
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    metadata: dict[str, str] | None,
) -> dict:
    if not messages:
        raise ModelClientError("messages must not be empty")

    payload: dict = {
        "model": model,
        "messages": list(messages),
        "stream": stream,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if metadata:
        payload["metadata"] = metadata
    return payload


def _post_json(
    *,
    url: str,
    payload: dict,
    api_key: str,
    timeout_seconds: float,
) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            raw = resp.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ModelClientError(f"model http error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ModelClientError(f"model network error: {exc}") from exc
    except Exception as exc:
        raise ModelClientError(f"model request failed: {exc}") from exc

    try:
        return json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ModelClientError(f"invalid model response json: {exc}") from exc


def _stream_chat_completion(
    *,
    url: str,
    payload: dict,
    api_key: str,
    timeout_seconds: float,
) -> Iterator[str]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            for line in resp:
                decoded = line.decode("utf-8", errors="ignore").strip()
                if not decoded or not decoded.startswith("data:"):
                    continue
                data_text = decoded[5:].strip()
                if data_text == "[DONE]":
                    break
                try:
                    data = json.loads(data_text)
                except Exception:
                    continue
                delta = _extract_delta_text(data)
                if delta:
                    yield delta
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise ModelClientError(f"model http error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ModelClientError(f"model network error: {exc}") from exc
    except Exception as exc:
        raise ModelClientError(f"model stream failed: {exc}") from exc


def _extract_text_from_chat_completion(data: dict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ModelClientError("model response missing choices")

    first = choices[0]
    message = first.get("message", {}) if isinstance(first, dict) else {}
    content = message.get("content")

    if isinstance(content, str):
        text = content.strip()
        return text if text else "(空回复)"

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        text = "".join(parts).strip()
        return text if text else "(空回复)"

    raise ModelClientError("model response missing message.content")


def _extract_delta_text(data: dict) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    delta = first.get("delta", {}) if isinstance(first, dict) else {}
    content = delta.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""
