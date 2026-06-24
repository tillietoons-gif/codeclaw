"""Thin async client for the local Ollama server.

Uses the native tool-calling API (`/api/chat` with `tools=...`) rather than
prompt-engineered JSON, because the configured `qwen2.5-coder` model advertises
the `tools` capability. Models without that capability will simply ignore the
`tools` field and return plain text; the agent loop falls back to a JSON-mode
extraction in that case.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """A single tool invocation the model wants to make."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: Any = None  # original payload, for debugging

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ToolCall(name={self.name!r}, arguments={self.arguments!r})"


@dataclass
class ChatMessage:
    """One turn in the conversation."""

    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    def to_ollama(self) -> dict[str, Any]:
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
        return msg


@dataclass
class ChatResponse:
    content: str
    tool_calls: list[ToolCall]
    model: str
    done_reason: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


class OllamaError(RuntimeError):
    """Raised when the Ollama server returns a non-2xx response."""


def _extract_tool_call_from_content(
    content: str,
    tool_names: set[str] | None = None,
) -> tuple[str, dict[str, Any]] | None:
    """Best-effort extract a tool call JSON object from free-form text.

    The model is expected to emit something like:
        {"name": "read_file", "arguments": {"path": "x.py"}}
    possibly wrapped in prose or markdown fences, and possibly followed by
    additional tool calls or commentary. We extract the first complete
    top-level JSON object, not the substring between the first `{` and the
    last `}` (that would glue multiple objects together).
    """
    if not content:
        return None
    text = content.strip()

    # Strip a single markdown ```json ... ``` fence if present.
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Find every top-level JSON object by scanning braces, then try to
    # parse each in order. Return the first that decodes to a tool-call
    # shape: {"name": str, "arguments": ...}. Some models also print:
    #
    #     exec
    #     {"command": "mkdir landing"}
    #
    # In that case, use the preceding line as the tool name and the object
    # itself as arguments.
    for start, end in _iter_top_level_json_objects(text):
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if "name" in obj and "arguments" in obj:
            name = str(obj["name"]).strip()
            args = obj["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            if not isinstance(args, dict):
                args = {"_raw": args}
            return name, args
        prefix_tool = _tool_name_before_json(text[:start], tool_names)
        if prefix_tool:
            return prefix_tool, obj
    return None


def _tool_name_before_json(prefix: str, tool_names: set[str] | None) -> str | None:
    if not tool_names:
        return None
    for line in reversed(prefix.splitlines()):
        candidate = line.strip().strip("`").strip()
        if not candidate:
            continue
        return candidate if candidate in tool_names else None
    return None


def _tool_names_from_schemas(tools: list[dict[str, Any]] | None) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        fn = (tool or {}).get("function") or {}
        name = fn.get("name")
        if name:
            names.add(str(name))
    return names


def _iter_top_level_json_objects(text: str):
    """Yield (start, end) char offsets of every top-level `{...}` in text.

    Tracks nesting depth and string quoting (with backslash escapes) so a
    brace inside a string doesn't fool us.
    """
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{":
            depth = 0
            j = i
            in_str = False
            escape = False
            while j < n:
                c = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_str = False
                else:
                    if c == '"':
                        in_str = True
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            yield i, j
                            i = j + 1
                            break
                j += 1
            else:
                # Unterminated; stop scanning.
                return
        else:
            i += 1


class OllamaClient:
    """Async HTTP wrapper. One instance per agent run is fine."""

    def __init__(self, host: str, timeout_s: float = 300.0):
        self.host = host.rstrip("/")
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)
        self._client = httpx.AsyncClient(timeout=self._timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def list_models(self) -> list[dict[str, Any]]:
        r = await self._client.get(f"{self.host}/api/tags")
        r.raise_for_status()
        return r.json().get("models", [])

    async def show_model(self, model: str) -> dict[str, Any]:
        r = await self._client.post(f"{self.host}/api/show", json={"model": model})
        if r.status_code != 200:
            try:
                detail = r.json()
            except json.JSONDecodeError:
                detail = r.text
            raise OllamaError(f"Ollama returned {r.status_code}: {detail}")
        return r.json()

    async def model_supports_tools(self, model: str) -> bool:
        try:
            models = await self.list_models()
        except httpx.HTTPError as exc:
            raise OllamaError(f"Failed to list models: {exc}") from exc
        for m in models:
            if m.get("name") == model or m.get("model") == model:
                caps = m.get("capabilities") or []
                if not caps:
                    try:
                        caps = (await self.show_model(model)).get("capabilities") or []
                    except (httpx.HTTPError, OllamaError):
                        caps = []
                return "tools" in caps
        return False

    async def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> ChatResponse:
        """Send a chat request and parse the response.

        When `tools` is provided the server may return `message.tool_calls`.
        When `json_mode` is true we ask the server to constrain output to JSON
        (used as a fallback when a model can't tool-call).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": [m.to_ollama() for m in messages],
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        if json_mode:
            payload["format"] = "json"

        try:
            r = await self._client.post(f"{self.host}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise OllamaError(f"Network error talking to Ollama at {self.host}: {exc}") from exc

        if r.status_code != 200:
            # Surface server-side error text verbatim; it's almost always useful.
            try:
                detail = r.json()
            except json.JSONDecodeError:
                detail = r.text
            raise OllamaError(f"Ollama returned {r.status_code}: {detail}")

        data = r.json()
        msg = data.get("message", {}) or {}
        content = msg.get("content", "") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", {}) or {}
            if isinstance(args, str):
                # Some servers serialize arguments as a JSON string.
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(name=name, arguments=args, raw=tc))

        # Fallback: some models (notably some qwen2.5-coder builds) emit
        # tool calls as a JSON object inside `content` rather than via the
        # structured `tool_calls` channel. If the structured channel is
        # empty, look for an inline JSON tool call and parse it.
        if not tool_calls and content:
            extracted = _extract_tool_call_from_content(content, _tool_names_from_schemas(tools))
            if extracted is not None:
                name, args = extracted
                tool_calls.append(ToolCall(name=name, arguments=args, raw=content))
                # Replace the visible content with a short marker so the
                # final-message branch (if any) doesn't surface raw JSON.
                content = ""

        return ChatResponse(
            content=content,
            tool_calls=tool_calls,
            model=data.get("model", model),
            done_reason=data.get("done_reason", "stop"),
            prompt_tokens=data.get("prompt_eval_count", 0),
            completion_tokens=data.get("eval_count", 0),
        )
