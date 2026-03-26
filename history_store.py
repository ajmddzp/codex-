from __future__ import annotations

from pathlib import Path
from typing import Sequence

from backend_types import ChatMessage, Conversation, ModelMessage, Role


DEFAULT_HISTORY_DIR = Path("history")


def list_conversations(history_dir: Path | str = DEFAULT_HISTORY_DIR) -> list[Conversation]:
    """
    Return all conversations sorted by updated_at (newest first).
    """
    raise NotImplementedError


def create_conversation(
    title: str | None,
    model: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> Conversation:
    """
    Create a conversation record and initialize empty message storage.
    """
    raise NotImplementedError


def get_conversation(
    conversation_id: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> Conversation | None:
    """
    Return one conversation by id, or None if not found.
    """
    raise NotImplementedError


def update_conversation_model(
    conversation_id: str,
    model: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> Conversation:
    """
    Update the model for a conversation and return the updated record.
    """
    raise NotImplementedError


def touch_conversation(
    conversation_id: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> None:
    """
    Update updated_at timestamp for a conversation.
    """
    raise NotImplementedError


def list_messages(
    conversation_id: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> list[ChatMessage]:
    """
    Return all messages for a conversation in chronological order.
    """
    raise NotImplementedError


def append_message(
    conversation_id: str,
    role: Role,
    content: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
    *,
    message_id: str | None = None,
    created_at: str | None = None,
) -> ChatMessage:
    """
    Append one message to a conversation and return the stored object.
    """
    raise NotImplementedError


def append_many_messages(
    conversation_id: str,
    messages: Sequence[ChatMessage],
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> list[ChatMessage]:
    """
    Append multiple messages in one call.
    """
    raise NotImplementedError


def delete_conversation(
    conversation_id: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
) -> bool:
    """
    Delete one conversation and its messages. Return True if deleted.
    """
    raise NotImplementedError


def build_model_messages(
    conversation_id: str,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
    *,
    system_prompt: str | None = None,
    max_messages: int | None = None,
) -> list[ModelMessage]:
    """
    Build message list to send to the cloud model API.
    """
    raise NotImplementedError
