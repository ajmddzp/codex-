from __future__ import annotations

import json
from pathlib import Path

DEFAULT_MODELS = ["gpt-5.4", "gpt-5.3-codex", "gpt-5.2"]


def list_models(*, config_path: Path) -> list[str]:
    """
    Return available models for frontend selector.

    TODO:
    - Replace with provider-side model discovery if needed.
    """
    if not config_path.exists():
        return DEFAULT_MODELS[:]

    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return DEFAULT_MODELS[:]

    if not isinstance(data, dict):
        return DEFAULT_MODELS[:]
    llm = data.get("llm")
    if not isinstance(llm, dict):
        return DEFAULT_MODELS[:]

    candidates = llm.get("candidate_models")
    primary = llm.get("model")
    models = [str(item) for item in (candidates or []) if str(item).strip()]
    primary_text = str(primary).strip() if primary is not None else ""
    if primary_text and primary_text not in models:
        models.insert(0, primary_text)
    return models or DEFAULT_MODELS[:]
