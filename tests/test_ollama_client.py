"""Unit tests for the Ollama HTTP client using a fake httpx transport.

We don't talk to a real server; we patch the underlying AsyncClient to return
canned responses and assert the parsing logic handles common shapes.
"""
from __future__ import annotations

import json

import httpx
import pytest

from codeclaw.ollama import ChatMessage, OllamaClient, OllamaError


def _transport(handler):
    return httpx.MockTransport(handler)


def _ok(status: int, payload: dict | str) -> httpx.Response:
    if isinstance(payload, dict):
        return httpx.Response(status, json=payload)
    return httpx.Response(status, text=payload)


@pytest.mark.asyncio
async def test_chat_parses_text_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = json.loads(request.content)
        assert body["model"] == "m"
        assert body["messages"][0]["role"] == "user"
        return _ok(200, {
            "model": "m",
            "done_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
            "prompt_eval_count": 10,
            "eval_count": 4,
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(
            model="m",
            messages=[ChatMessage("user", "hello")],
        )
    assert resp.content == "hi"
    assert resp.prompt_tokens == 10
    assert resp.completion_tokens == 4
    assert resp.tool_calls == []


@pytest.mark.asyncio
async def test_chat_parses_tool_calls():
    def handler(request):
        return _ok(200, {
            "model": "m",
            "done_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "x.py"},
                        }
                    }
                ],
            },
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(
            model="m",
            messages=[ChatMessage("user", "show me x.py")],
            tools=[{"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}],
        )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "x.py"}


@pytest.mark.asyncio
async def test_chat_handles_stringified_arguments():
    def handler(request):
        return _ok(200, {
            "model": "m",
            "done_reason": "tool_calls",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "exec", "arguments": '{"command": "ls"}'}}
                ],
            },
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(
            model="m", messages=[ChatMessage("user", "ls")],
        )
    assert resp.tool_calls[0].arguments == {"command": "ls"}


@pytest.mark.asyncio
async def test_chat_raises_on_http_error():
    def handler(request):
        return _ok(500, "boom")

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        with pytest.raises(OllamaError) as exc:
            await client.chat(model="m", messages=[ChatMessage("user", "x")])
    assert "500" in str(exc.value)


@pytest.mark.asyncio
async def test_list_models_parses_payload():
    def handler(request):
        assert request.url.path == "/api/tags"
        return _ok(200, {
            "models": [
                {"name": "qwen2.5-coder:32b", "capabilities": ["completion", "tools"]},
            ]
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        models = await client.list_models()
        assert models[0]["name"] == "qwen2.5-coder:32b"
        assert await client.model_supports_tools("qwen2.5-coder:32b") is True
        assert await client.model_supports_tools("not-installed") is False


@pytest.mark.asyncio
async def test_show_model_parses_capabilities():
    def handler(request):
        assert request.url.path == "/api/show"
        body = json.loads(request.content)
        assert body["model"] == "qwen2.5-coder:32b"
        return _ok(200, {
            "capabilities": ["completion", "tools"],
            "model_info": {"qwen2.context_length": 32768},
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        model = await client.show_model("qwen2.5-coder:32b")
    assert model["capabilities"] == ["completion", "tools"]
    assert model["model_info"]["qwen2.context_length"] == 32768


@pytest.mark.asyncio
async def test_model_supports_tools_falls_back_to_show_model():
    def handler(request):
        if request.url.path == "/api/tags":
            return _ok(200, {"models": [{"name": "m"}]})
        if request.url.path == "/api/show":
            return _ok(200, {"capabilities": ["completion", "tools"]})
        raise AssertionError(request.url.path)

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        assert await client.model_supports_tools("m") is True


@pytest.mark.asyncio
async def test_chat_falls_back_to_inline_json_tool_call():
    """Some models emit tool calls inside `content` as raw JSON. Verify the
    parser extracts them and clears the visible content."""
    def handler(request):
        return _ok(200, {
            "model": "m",
            "done_reason": "stop",
            "message": {
                "role": "assistant",
                "content": '{"name": "read_file", "arguments": {"path": "x.py"}}',
            },
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(
            model="m", messages=[ChatMessage("user", "show me x.py")],
        )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "read_file"
    assert resp.tool_calls[0].arguments == {"path": "x.py"}
    # Content is cleared so the final-message branch doesn't surface raw JSON.
    assert resp.content == ""


@pytest.mark.asyncio
async def test_chat_falls_back_with_markdown_fence_and_prose():
    def handler(request):
        return _ok(200, {
            "model": "m",
            "message": {
                "role": "assistant",
                "content": "Sure, calling the tool now.\n```json\n"
                           '{"name": "list_dir", "arguments": {"path": "."}}\n```\n',
            },
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(model="m", messages=[ChatMessage("user", "ls")])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "list_dir"
    assert resp.tool_calls[0].arguments == {"path": "."}


@pytest.mark.asyncio
async def test_chat_falls_back_from_tool_name_plus_json_args():
    def handler(request):
        return _ok(200, {
            "model": "m",
            "message": {
                "role": "assistant",
                "content": 'exec\n{"command": "mkdir landing", "timeout_s": 10}',
            },
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(
            model="m",
            messages=[ChatMessage("user", "create a folder")],
            tools=[{"type": "function", "function": {"name": "exec", "description": "", "parameters": {}}}],
        )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "exec"
    assert resp.tool_calls[0].arguments == {"command": "mkdir landing", "timeout_s": 10}
    assert resp.content == ""


@pytest.mark.asyncio
async def test_chat_no_tool_call_leaves_content_alone():
    def handler(request):
        return _ok(200, {
            "model": "m",
            "message": {"role": "assistant", "content": "Here is the answer."},
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(model="m", messages=[ChatMessage("user", "?")])
    assert resp.tool_calls == []
    assert resp.content == "Here is the answer."


@pytest.mark.asyncio
async def test_chat_picks_first_tool_call_when_multiple_inlined():
    """When the model emits multiple JSON tool calls inside `content`, the
    parser should pick the first one (the rest will come on subsequent
    turns once the first result is fed back)."""
    def handler(request):
        return _ok(200, {
            "model": "m",
            "message": {
                "role": "assistant",
                "content": (
                    "I'll do this in two steps.\n"
                    '{"name": "edit_file", "arguments": {"path": "a", "old_text": "x", "new_text": "y"}}\n'
                    '{"name": "exec", "arguments": {"command": "ls"}}'
                ),
            },
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(model="m", messages=[ChatMessage("user", "go")])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "edit_file"
    assert resp.tool_calls[0].arguments["path"] == "a"


@pytest.mark.asyncio
async def test_chat_ignores_braces_inside_strings():
    def handler(request):
        return _ok(200, {
            "model": "m",
            "message": {
                "role": "assistant",
                "content": 'note: {see docs} then call\n{"name": "list_dir", "arguments": {"path": "."}}',
            },
        })

    async with httpx.AsyncClient(transport=_transport(handler)) as http:
        client = OllamaClient.__new__(OllamaClient)
        client.host = "http://x"
        client._timeout = httpx.Timeout(5)
        client._client = http
        resp = await client.chat(model="m", messages=[ChatMessage("user", "go")])
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "list_dir"
