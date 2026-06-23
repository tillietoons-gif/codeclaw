"""Tool framework: base class, registry, and result type.

Each tool declares its JSON Schema (in OpenAI/Ollama format) and implements
`async run(args, ctx) -> ToolResult`. The registry turns the schemas into the
payload sent to Ollama and routes tool calls back to the right implementation.
"""
from __future__ import annotations

import abc
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Outcome of a tool invocation, sent back to the model as a `tool` message."""

    output: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_tool_message(self) -> str:
        if self.is_error:
            return f"ERROR: {self.output}"
        return self.output


class Tool(abc.ABC):
    """Base class for CodeClaw tools."""

    name: str = ""
    description: str = ""
    # JSON Schema (OpenAI function-calling format). Subclasses override.
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "required": [],
    }
    # Optional guard: when True, this tool requires explicit human approval
    # before the result is returned to the model. Wired up in `agent.py`.
    requires_approval: bool = False

    def to_ollama_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @abc.abstractmethod
    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:  # noqa: F821
        ...


@dataclass
class ToolContext:
    """Per-run context passed to every tool.

    Carries the working directory, an approval callback, and a logger. Tools
    that mutate state or run shell commands should consult `approval` for
    destructive operations.
    """

    cwd: str
    approval: Callable[[str, str], ApprovalDecision]  # (tool_name, summary) -> decision
    log: Callable[[str], None]  # for streaming progress to the user
    session_id: str = ""


class ApprovalDecision:
    APPROVE = "approve"
    REJECT = "reject"
    APPROVE_ALWAYS = "approve-always"

    def __init__(self, value: str, reason: str = "") -> None:
        self.value = value
        self.reason = reason

    @property
    def approved(self) -> bool:
        return self.value in (self.APPROVE, self.APPROVE_ALWAYS)


class ToolRegistry:
    """Holds the available tools and routes calls."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ValueError("Tool must have a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name!r}. Available: {sorted(self._tools)}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_ollama_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return sorted(self._tools)

    async def invoke(self, name: str, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        tool = self.get(name)
        try:
            return await tool.run(args, ctx)
        except Exception as exc:  # surface, don't crash the agent loop
            return ToolResult(output=f"{type(exc).__name__}: {exc}", is_error=True)


def parse_args(raw: Any) -> dict[str, Any]:
    """Best-effort coerce whatever the model sent into a dict."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {"_raw": raw}
        except json.JSONDecodeError:
            return {"_raw": raw}
    return {"_raw": str(raw)}
