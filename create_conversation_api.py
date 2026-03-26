from __future__ import annotations

from typing import Any

from history_store import create_conversation as store_create_conversation

from model_client import load_model_config
from pathlib import Path


def create_conversation(*, title: str | None, model: str | None) -> dict[str, Any]:
    selected_model = str(model or "").strip()
    if not selected_model:
        config_path = Path(__file__).with_name("config.json")
        try:
            selected_model = load_model_config(config_path=config_path).model
        except Exception:
            selected_model = "gpt-5.4"
    return store_create_conversation(title=title, model=selected_model)
