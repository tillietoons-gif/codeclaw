"""OpenAI-compatible chat completions client (vLLM, LM Studio, etc.)."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx

from .llm_types import ChatMessage, ChatResponse, LLMError, ToolCall
from .ollama import _extract_tool_call_from_content, _tool_names_from_schemas


class OpenAICompatClient:
    """Async HTTP wrapper for /v1/chat/completions."""

    def __init__(self, base_url: str, api_key: str = "", timeout_s: float = 300.0, model: str = ""):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.api_key = api_key
        self.default_model = model
        self._timeout = httpx.Timeout(timeout_s, connect=10.0)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(timeout=self._timeout, headers=headers)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> OpenAICompatClient:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def list_models(self) -> list[dict[str, Any]]:
        try:
            r = await self._client.get(f"{self.base_url}/models")
            r.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Failed to list models: {exc}") from exc
        data = r.json().get("data", [])
        return [{"name": m.get("id", ""), "model": m.get("id", "")} for m in data]

    async def show_model(self, model: str) -> dict[str, Any]:
        return {"capabilities": ["completion", "tools"], "model_info": {}}

    async def model_capabilities(self, model: str) -> list[str]:
        return ["completion", "tools"]

    async def model_supports_tools(self, model: str) -> bool:
        return True

    async def model_supports_thinking(self, model: str) -> bool:
        return False

    def _to_openai_messages(self, messages: list[ChatMessage]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for msg in messages:
            if msg.role == "tool":
                out.append({"role": "tool", "content": msg.content, "tool_call_id": msg.tool_call_id or msg.tool_name or "tool"})
                continue
            item: dict[str, Any] = {"role": msg.role, "content": msg.content}
            if msg.tool_calls:
                item["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)}} for i, tc in enumerate(msg.tool_calls)]
            out.append(item)
        return out

    async def chat(self, model: str, messages: list[ChatMessage], tools: list[dict[str, Any]] | None = None, temperature: float = 0.2, json_mode: bool = False, on_delta: Callable[[str, str], None] | None = None) -> ChatResponse:
        payload: dict[str, Any] = {"model": model or self.default_model, "messages": self._to_openai_messages(messages), "temperature": temperature, "stream": on_delta is not None}
        if tools:
            payload["tools"] = [{"type": "function", "function": t.get("function", t)} for t in tools]
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        if on_delta is not None:
            data = await self._chat_stream(payload, on_delta)
        else:
            try:
                r = await self._client.post(f"{self.base_url}/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                raise LLMError(f"Network error: {exc}") from exc
            if r.status_code != 200:
                try:
                    detail = r.json()
                except json.JSONDecodeError:
                    detail = r.text
                raise LLMError(f"API returned {r.status_code}: {detail}")
            data = r.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) or {}
        content = msg.get("content", "") or ""
        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            fn = (tc or {}).get("function") or {}
            name = fn.get("name", "")
            args = fn.get("arguments", {}) or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(name=name, arguments=args, raw=tc))
        if not tool_calls and content:
            extracted = _extract_tool_call_from_content(content, _tool_names_from_schemas(tools))
            if extracted is not None:
                name, args = extracted
                tool_calls.append(ToolCall(name=name, arguments=args, raw=content))
                content = ""
        usage = data.get("usage") or {}
        return ChatResponse(content=content, tool_calls=tool_calls, model=data.get("model", model), done_reason=choice.get("finish_reason", "stop"), prompt_tokens=usage.get("prompt_tokens", 0), completion_tokens=usage.get("completion_tokens", 0))

    async def _chat_stream(self, payload: dict[str, Any], on_delta: Callable[[str, str], None]) -> dict[str, Any]:
        content_parts: list[str] = []
        tool_calls_acc: dict[int, dict[str, Any]] = {}
        final: dict[str, Any] = {}
        try:
            async with self._client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    raise LLMError(f"API returned {response.status_code}: {text.decode('utf-8', errors='replace')}")
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_raw = line[6:].strip()
                    if chunk_raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_raw)
                    except json.JSONDecodeError:
                        continue
                    final = chunk
                    delta = ((chunk.get("choices") or [{}])[0]).get("delta") or {}
                    content = delta.get("content") or ""
                    if content:
                        content_parts.append(content)
                        on_delta("content", content)
                    for tc in delta.get("tool_calls") or []:
                        idx = tc.get("index", 0)
                        acc = tool_calls_acc.setdefault(idx, {"function": {"name": "", "arguments": ""}})
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            acc["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            acc["function"]["arguments"] += fn["arguments"]
        except httpx.HTTPError as exc:
            raise LLMError(f"Network error: {exc}") from exc
        tool_calls = []
        for tc in tool_calls_acc.values():
            name = tc["function"]["name"]
            args_raw = tc["function"]["arguments"]
            try:
                args = json.loads(args_raw) if args_raw else {}
            except json.JSONDecodeError:
                args = {"_raw": args_raw}
            tool_calls.append(ToolCall(name=name, arguments=args, raw=tc))
        return {"model": final.get("model", payload["model"]), "choices": [{"message": {"content": "".join(content_parts), "tool_calls": [{"function": {"name": t.name, "arguments": t.arguments}} for t in tool_calls]}, "finish_reason": "tool_calls" if tool_calls else "stop"}], "usage": final.get("usage") or {}}
