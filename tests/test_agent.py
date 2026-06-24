"""Unit tests for the agent loop and context trimming.

These tests use a fake OllamaClient to keep them deterministic and offline.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace

import pytest

from codeclaw.agent import CodeClawAgent, _approx_chars
from codeclaw.config import Settings
from codeclaw.ollama import ChatMessage, ChatResponse, ToolCall
from codeclaw.tools.base import ApprovalDecision


class FakeClient:
    """Records messages and serves scripted responses in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[list[ChatMessage]] = []
        self._closed = False

    async def close(self):
        self._closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        await self.close()

    async def chat(self, *, model, messages, tools=None, temperature=0.2, json_mode=False):
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


def _text_response(content: str) -> ChatResponse:
    return ChatResponse(
        content=content, tool_calls=[], model="fake",
        done_reason="stop", prompt_tokens=1, completion_tokens=1,
    )


def _thinking_response(thinking: str, content: str) -> ChatResponse:
    return ChatResponse(
        content=content, tool_calls=[], model="fake",
        done_reason="stop", thinking=thinking, prompt_tokens=1, completion_tokens=1,
    )


def _tool_response(name: str, args: dict) -> ChatResponse:
    return ChatResponse(
        content="", tool_calls=[ToolCall(name=name, arguments=args)],
        model="fake", done_reason="tool_calls",
        prompt_tokens=1, completion_tokens=1,
    )


@pytest.mark.asyncio
async def test_agent_terminates_on_text_response(tmp_path):
    settings = Settings()
    settings = replace(settings, project_dir=str(tmp_path), max_steps=5)
    client = FakeClient([_text_response("All done.")])
    agent = CodeClawAgent(settings=settings, client=client, log=lambda m: None)
    result = await agent.run("do the thing")
    assert result.completed
    assert result.final_message == "All done."
    assert result.reason == "done"


@pytest.mark.asyncio
async def test_agent_logs_thinking_when_present(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=5)
    client = FakeClient([_thinking_response("I need to inspect the request.", "All done.")])
    log_lines: list[str] = []
    agent = CodeClawAgent(settings=settings, client=client, log=log_lines.append)
    result = await agent.run("do the thing")

    assert result.completed
    assert "  [thinking]" in log_lines
    assert "  ? I need to inspect the request." in log_lines


@pytest.mark.asyncio
async def test_agent_executes_tool_then_finishes(tmp_path):
    (tmp_path / "note.txt").write_text("hello")
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=5)
    # Round 1: model asks to read a file. Round 2: model returns final text.
    client = FakeClient([
        _tool_response("read_file", {"path": "note.txt"}),
        _text_response("File contained: hello"),
    ])
    log_lines: list[str] = []
    agent = CodeClawAgent(
        settings=settings, client=client, log=log_lines.append,
    )
    result = await agent.run("read note.txt and report contents")
    assert result.completed
    assert result.reason == "done"
    # The agent should have called the tool, then synthesized the final answer.
    assert any("hello" in m.content for m in client.calls[-1])


@pytest.mark.asyncio
async def test_agent_continues_after_tool_error(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=5)
    client = FakeClient([
        _tool_response("read_file", {"path": "missing.txt"}),  # will error
        _text_response("Done; the file didn't exist."),
    ])
    agent = CodeClawAgent(settings=settings, client=client, log=lambda m: None)
    result = await agent.run("read missing file")
    assert result.completed
    # The error message should have been appended to the conversation.
    last = client.calls[-1]
    assert any("File not found" in m.content for m in last if m.role == "tool")


@pytest.mark.asyncio
async def test_agent_hits_max_steps(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=3)
    # Always return a tool call so the loop never naturally terminates.
    client = FakeClient([_tool_response("list_dir", {"path": "."})] * 3)
    agent = CodeClawAgent(settings=settings, client=client, log=lambda m: None)
    result = await agent.run("loop forever")
    assert not result.completed
    assert result.reason == "max_steps"


@pytest.mark.asyncio
async def test_destructive_action_requires_approval(tmp_path):
    (tmp_path / "f.txt").write_text("old")
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=5)
    client = FakeClient([
        _tool_response("write_file", {"path": "f.txt", "content": "new"}),
        _text_response("Wrote f.txt"),
    ])
    approvals: list[tuple[str, str]] = []

    async def approval(name, summary):
        approvals.append((name, summary))
        return ApprovalDecision(ApprovalDecision.APPROVE)

    agent = CodeClawAgent(settings=settings, client=client, approval=approval, log=lambda m: None)
    result = await agent.run("overwrite f.txt")
    assert result.completed
    assert approvals == [("write_file", "write_file: f.txt")]
    assert (tmp_path / "f.txt").read_text() == "new"


@pytest.mark.asyncio
async def test_rejected_action_returns_error_to_model(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=5)
    client = FakeClient([
        _tool_response("exec", {"command": "rm -rf /tmp/should-not-run"}),
        _text_response("OK, I'll avoid that."),
    ])

    async def deny(name, summary):
        return ApprovalDecision(ApprovalDecision.REJECT, reason="no thanks")

    agent = CodeClawAgent(settings=settings, client=client, approval=deny, log=lambda m: None)
    result = await agent.run("try a destructive command")
    assert result.completed
    # The tool result should be an error message in the final conversation.
    last = client.calls[-1]
    assert any("REJECTED" in m.content for m in last if m.role == "tool")


@pytest.mark.asyncio
async def test_pre_tool_hook_can_block_tool_use(tmp_path):
    settings_dir = tmp_path / ".codeclaw"
    settings_dir.mkdir()
    command = f"{sys.executable} -c \"import sys; print('blocked by policy'); sys.exit(4)\""
    (settings_dir / "settings.json").write_text(
        json.dumps({"hooks": {"PreToolUse": [{"type": "command", "command": command}]}}),
        encoding="utf-8",
    )
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=5)
    client = FakeClient([
        _tool_response("write_file", {"path": "f.txt", "content": "new"}),
        _text_response("I did not write the file."),
    ])

    async def approval(name, summary):
        raise AssertionError("approval should not run when PreToolUse blocks first")

    agent = CodeClawAgent(settings=settings, client=client, approval=approval, log=lambda m: None)
    result = await agent.run("write f.txt")

    assert result.completed
    assert not (tmp_path / "f.txt").exists()
    last = client.calls[-1]
    assert any("PreToolUse hook blocked" in m.content for m in last if m.role == "tool")
    assert any("blocked by policy" in m.content for m in last if m.role == "tool")


@pytest.mark.asyncio
async def test_auto_approval_runs_exec_without_pattern_filter(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path), max_steps=5)
    marker = tmp_path / "marker.txt"
    client = FakeClient([
        _tool_response("exec", {"command": f"echo ok > {marker}; echo rm -rf"}),
        _text_response("Done."),
    ])

    async def auto_approve(name, summary):
        return ApprovalDecision(ApprovalDecision.APPROVE, reason="--auto-approve")

    agent = CodeClawAgent(settings=settings, client=client, approval=auto_approve, log=lambda m: None)
    result = await agent.run("run a command")

    assert result.completed
    assert marker.read_text().strip() == "ok"


def test_approx_chars_counts_tool_calls_too():
    msgs = [
        ChatMessage("system", "sys"),
        ChatMessage("user", "do thing"),
        ChatMessage(
            "assistant", "", tool_calls=[ToolCall(name="read_file", arguments={"path": "x"})]
        ),
    ]
    assert _approx_chars(msgs) > len("sysdo thing")
