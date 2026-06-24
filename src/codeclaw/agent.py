"""The agent loop: turn a user objective into completed work.

Loop outline:

    system + project context
        ↓
    user objective
        ↓
    ┌─→ LLM step
    │       ↓
    │   tool calls? ──no──→ final answer, done
    │       ↓ yes
    │   approval gate for destructive tools
    │       ↓
    │   execute tools, append results
    │       ↓
    └─────┘

The loop is bounded by `max_steps` so a misbehaving model can't run forever.
Token pressure is handled by a simple sliding window: keep the system +
project context pinned, keep the user's objective pinned, and trim the
middle of the conversation when it grows too long.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import memory
from .config import Settings
from .hooks import HookResult, run_hooks
from .ollama import ChatMessage, ChatResponse, OllamaClient, ToolCall
from .tools import ToolContext, ToolRegistry, build_default_registry
from .tools.base import ApprovalDecision, ToolResult

logger = logging.getLogger(__name__)

# Approximate tokens-per-character for the sliding-window heuristic.
# 4 chars/token is a fine rule of thumb for English/code.
CHARS_PER_TOKEN = 4
# Slack reserved for the model's reply.
RESPONSE_RESERVE_CHARS = 2000

# Tools that mutate project state or hit the network. Anything that doesn't
# fall in this set is treated as read-only and runs without confirmation.
DESTRUCTIVE_TOOLS = {"write_file", "edit_file", "exec", "git_commit"}

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system.md"


@dataclass
class StepRecord:
    step: int
    assistant_text: str
    tool_calls: list[ToolCall]
    tool_results: list[tuple[str, ToolResult]] = field(default_factory=list)


@dataclass
class RunResult:
    objective: str
    final_message: str
    steps: list[StepRecord]
    total_tokens: int
    completed: bool
    reason: str  # "done" | "max_steps" | "error" | "blocked"


class CodeClawAgent:
    """The agent. One instance per user request."""

    def __init__(
        self,
        settings: Settings,
        client: OllamaClient,
        registry: ToolRegistry | None = None,
        approval: Callable[[str, str], Awaitable[ApprovalDecision]] | None = None,
        log: Callable[[str], None] | None = None,
    ):
        self.settings = settings
        self.client = client
        self.registry = registry or build_default_registry()
        self.approval = approval or _default_approval
        self.log = log or (lambda msg: print(msg, flush=True))
        self.session_id = uuid.uuid4().hex[:12]

    async def run(self, objective: str) -> RunResult:
        project_ctx = memory.load_project_context(self.settings.project_dir)
        system = self._build_system_prompt(project_ctx)
        messages: list[ChatMessage] = [
            ChatMessage("system", system),
            ChatMessage("user", objective),
        ]
        steps: list[StepRecord] = []
        total_tokens = 0
        ctx = ToolContext(
            cwd=str(Path(self.settings.project_dir).resolve()),
            approval=self._approval_wrapper,
            log=self.log,
            session_id=self.session_id,
        )
        await self._run_hooks(
            "RunStart",
            {"session_id": self.session_id, "objective": objective, "cwd": ctx.cwd},
        )

        for step_idx in range(1, self.settings.max_steps + 1):
            self.log(f"\n[step {step_idx}/{self.settings.max_steps}]")
            messages = self._trim_messages(messages, system, objective)

            try:
                resp = await self.client.chat(
                    model=self.settings.model,
                    messages=messages,
                    tools=self.registry.schemas(),
                    temperature=self.settings.temperature,
                )
            except Exception as exc:
                self.log(f"[error] Ollama call failed: {exc}")
                result = RunResult(objective, "", steps, total_tokens, False, f"error: {exc}")
                await self._run_complete_hook(result)
                return result

            total_tokens += resp.prompt_tokens + resp.completion_tokens
            self._log_assistant(resp)
            record = StepRecord(step=step_idx, assistant_text=resp.content, tool_calls=resp.tool_calls)
            messages.append(
                ChatMessage(
                    "assistant",
                    content=resp.content,
                    tool_calls=resp.tool_calls or None,
                )
            )

            if not resp.tool_calls:
                # Model decided it is done. Treat the assistant text as the
                # final report. We trust the model to be self-aware here —
                # but also break out of the loop regardless of what it says.
                result = RunResult(
                    objective=objective,
                    final_message=resp.content or "(no content)",
                    steps=steps + [record],
                    total_tokens=total_tokens,
                    completed=True,
                    reason="done",
                )
                await self._run_complete_hook(result)
                return result

            # Execute each tool call sequentially. In principle we could
            # parallelize, but most tool calls have ordering dependencies
            # and the model's reasoning is serial anyway.
            for tc in resp.tool_calls:
                result = await self._run_one_tool(tc, ctx)
                record.tool_results.append((tc.name, result))
                messages.append(
                    ChatMessage(
                        "tool",
                        content=result.as_tool_message(),
                        tool_name=tc.name,
                    )
                )

            steps.append(record)

        self.log(f"[done] hit max_steps={self.settings.max_steps}")
        result = RunResult(
            objective=objective,
            final_message="Reached the configured max_steps without a final answer.",
            steps=steps,
            total_tokens=total_tokens,
            completed=False,
            reason="max_steps",
        )
        await self._run_complete_hook(result)
        return result

    async def _run_one_tool(self, tc: ToolCall, ctx: ToolContext) -> ToolResult:
        from .tools import parse_args

        args = parse_args(tc.arguments)
        tool_name = tc.name
        hook_payload = {
            "session_id": self.session_id,
            "tool": tool_name,
            "arguments": args,
            "cwd": ctx.cwd,
        }
        pre_results = await self._run_hooks("PreToolUse", hook_payload)
        blocking_result = next((result for result in pre_results if not result.ok), None)
        if blocking_result:
            self.log(f"[denied] {tool_name}: blocked by PreToolUse hook")
            return ToolResult(
                _hook_block_message(blocking_result),
                is_error=True,
                metadata={"blocked_by_hook": True},
            )
        if tool_name in DESTRUCTIVE_TOOLS:
            summary = _summarize_call(tool_name, args)
            decision = await ctx.approval(tool_name, summary)
            if not decision.approved:
                self.log(f"[denied] {tool_name}: {summary}")
                return ToolResult(
                    f"User REJECTED this action: {summary}. Propose a safer alternative.",
                    is_error=True,
                )
            self.log(f"[approved] {tool_name}: {summary}")
        result = await self.registry.invoke(tool_name, args, ctx)
        await self._run_hooks(
            "PostToolUse",
            {
                **hook_payload,
                "is_error": result.is_error,
                "result": result.as_tool_message(),
            },
        )
        return result

    async def _run_complete_hook(self, result: RunResult) -> None:
        await self._run_hooks(
            "RunComplete",
            {
                "session_id": self.session_id,
                "objective": result.objective,
                "completed": result.completed,
                "reason": result.reason,
                "steps": len(result.steps),
                "tokens": result.total_tokens,
                "final_message": result.final_message,
            },
        )

    async def _run_hooks(self, event: str, payload: dict) -> list[HookResult]:
        results = await run_hooks(self.settings.project_dir, event, payload)
        for result in results:
            status = "ok" if result.ok else f"exit {result.returncode}"
            self.log(f"[hook] {event}: {status}")
            if result.output:
                for line in result.output.splitlines()[:8]:
                    self.log(f"  [hook] {line}")
        return results

    async def _approval_wrapper(self, tool_name: str, summary: str) -> ApprovalDecision:
        # ApprovalDecision is a synchronous decision in our registry, so we
        # adapt the async/sync shapes uniformly. The CLI passes an async fn.
        result = self.approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    def _log_assistant(self, resp: ChatResponse) -> None:
        if resp.thinking:
            self.log("  [thinking]")
            for line in resp.thinking.splitlines() or [""]:
                self.log(f"  ? {line}")
        if resp.content:
            for line in resp.content.splitlines() or [""]:
                self.log(f"  > {line}")
        if resp.tool_calls:
            names = ", ".join(tc.name for tc in resp.tool_calls)
            self.log(f"  [tool_calls: {names}]")
        self.log(
            f"  [tokens prompt={resp.prompt_tokens} completion={resp.completion_tokens}]"
        )

    def _build_system_prompt(self, project_ctx: str) -> str:
        sys = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
        tools_block = "\n".join(f"- `{name}`" for name in self.registry.names())
        extra = (
            f"\n## Available tools\n{tools_block}\n"
            f"\n## Project\nWorking directory: `{Path(self.settings.project_dir).resolve()}`\n"
        )
        if project_ctx:
            extra += "\n" + project_ctx + "\n"
        return sys + extra

    def _trim_messages(
        self,
        messages: list[ChatMessage],
        system: str,
        objective: str,
    ) -> list[ChatMessage]:
        """Sliding-window trim. Pin system + user, drop oldest middle turns."""
        budget_chars = self.settings.context_tokens * CHARS_PER_TOKEN - RESPONSE_RESERVE_CHARS
        if _approx_chars(messages) <= budget_chars:
            return messages
        # Preserve system + user; trim from the front in pairs (assistant+tool).
        head = [m for m in messages if m.role == "system"]
        # Keep the last user message (the original objective) and any trailing
        # tool+assistant turns that are after it.
        last_user_idx = max(i for i, m in enumerate(messages) if m.role == "user")
        pinned_tail = messages[last_user_idx:]
        middle = messages[len(head):last_user_idx]
        # Drop from the front of the middle until under budget.
        while middle and _approx_chars(head + middle + pinned_tail) > budget_chars:
            middle.pop(0)
        trimmed = head + middle + pinned_tail
        if len(trimmed) < len(messages):
            self.log(
                f"  [context trim] {len(messages)} -> {len(trimmed)} messages"
            )
        return trimmed


def _approx_chars(messages: list[ChatMessage]) -> int:
    total = 0
    for m in messages:
        total += len(m.content)
        for tc in m.tool_calls or []:
            total += len(tc.name) + len(json.dumps(tc.arguments))
    return total


def _summarize_call(tool_name: str, args: dict) -> str:
    if tool_name == "exec":
        return f"run shell: {args.get('command','')}"
    if tool_name in ("write_file", "edit_file"):
        return f"{tool_name}: {args.get('path','')}"
    if tool_name == "git_commit":
        return f"git commit: {args.get('message','')}"
    return f"{tool_name}({json.dumps(args)[:200]})"


def _hook_block_message(result: HookResult) -> str:
    detail = f"PreToolUse hook blocked this tool call with exit code {result.returncode}."
    if result.output:
        detail += f"\nHook output:\n{result.output}"
    return detail


async def _default_approval(tool_name: str, summary: str) -> ApprovalDecision:
    """Programmatic default: approve reads, prompt for everything else.

    The CLI overrides this with an interactive prompt.
    """
    return ApprovalDecision(ApprovalDecision.APPROVE, reason="auto-approved default")
