"""Serialize and deserialize agent conversation history for session resume."""
from __future__ import annotations

from typing import Any

from .llm_types import ChatMessage, ToolCall


def message_to_dict(message: ChatMessage) -> dict[str, Any]:
    data: dict[str, Any] = {
        "role": message.role,
        "content": message.content,
    }
    if message.tool_name:
        data["tool_name"] = message.tool_name
    if message.tool_call_id:
        data["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        data["tool_calls"] = [
            {"name": tc.name, "arguments": tc.arguments}
            for tc in message.tool_calls
        ]
    return data


def message_from_dict(data: dict[str, Any]) -> ChatMessage:
    tool_calls = None
    raw_calls = data.get("tool_calls")
    if isinstance(raw_calls, list):
        tool_calls = [
            ToolCall(name=str(tc.get("name", "")), arguments=dict(tc.get("arguments") or {}))
            for tc in raw_calls
            if isinstance(tc, dict)
        ]
    return ChatMessage(
        role=str(data.get("role", "user")),
        content=str(data.get("content", "")),
        tool_name=data.get("tool_name"),
        tool_call_id=data.get("tool_call_id"),
        tool_calls=tool_calls or None,
    )


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [message_to_dict(m) for m in messages]


def messages_from_dicts(data: list[dict[str, Any]]) -> list[ChatMessage]:
    return [message_from_dict(item) for item in data if isinstance(item, dict)]
