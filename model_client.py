from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


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
    for current_model in try_models:
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=messages,
            )
            text = _extract_text(response)
            if text:
                return text, current_model
            errors.append(f"{current_model}: empty_model_response")
        except Exception as exc:
            errors.append(f"{current_model}: {exc}")

    error_text = " | ".join(errors) if errors else "unknown_model_error"
    raise RuntimeError(f"model_request_failed: {error_text}")


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
