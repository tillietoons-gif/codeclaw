"""Shared LLM message and response types used by all backends."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation the model wants to make."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"ToolCall(name={self.name!r}, arguments={self.arguments!r})"


@dataclass
class ChatMessage:
    """One turn in the conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_api(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    }
                }
                for tc in self.tool_calls
            ]
        if self.tool_name:
            msg["name"] = self.tool_name
        if self.tool_call_id:
            msg["tool_call_id"] = self.tool_call_id
        return msg

    def to_ollama(self) -> dict[str, Any]:
        return self.to_api()


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]
    model: str
    done_reason: str
    thinking: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMError(RuntimeError):
    """Raised when an LLM backend returns a non-2xx response or network failure."""
