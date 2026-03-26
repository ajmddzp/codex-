from __future__ import annotations

from typing import Literal, TypedDict


Role = Literal["system", "user", "assistant", "error"]


class Conversation(TypedDict):
    id: str
    title: str
    model: str
    updated_at: str


class ChatMessage(TypedDict):
    id: str
    role: Role
    content: str
    created_at: str


class UploadedFile(TypedDict):
    id: str
    name: str
    size: int | None


class SendMessagePayload(TypedDict):
    conversationId: str
    message: str
    model: str
    fileIds: list[str]


class SwitchModelPayload(TypedDict):
    conversationId: str
    model: str


class ModelMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
