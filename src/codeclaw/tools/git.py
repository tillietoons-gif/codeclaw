"""Read-mostly git tools plus a single approval-gated commit tool."""
from __future__ import annotations

import asyncio
from pathlib import Path

from .base import Tool, ToolContext, ToolResult


async def _git(cwd: str, *args: str, timeout: int = 30) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", "timeout"
    return proc.returncode or 0, (out or b"").decode("utf-8", errors="replace"), (err or b"").decode("utf-8", errors="replace")


def _ensure_repo(cwd: str) -> tuple[bool, str]:
    if not (Path(cwd) / ".git").exists():
        return False, "not a git repository"
    return True, ""


class GitStatusTool(Tool):
    name = "git_status"
    description = "Return `git status --short` plus the current branch and HEAD."
    parameters = {"type": "object", "properties": {}, "required": []}

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        ok, why = _ensure_repo(ctx.cwd)
        if not ok:
            return ToolResult(why, is_error=True)
        rc, out, err = await _git(ctx.cwd, "status", "--short", "--branch")
        if rc != 0:
            return ToolResult(f"git status failed: {err}", is_error=True)
        return ToolResult(out or "(clean)")


class GitDiffTool(Tool):
    name = "git_diff"
    description = "Show `git diff` (working tree vs index by default; pass `staged=true` for staged changes)."
    parameters = {
        "type": "object",
        "properties": {
            "staged": {"type": "boolean", "description": "If true, show staged changes (`git diff --cached`)."},
            "pathspec": {"type": "string", "description": "Limit diff to a specific path."},
            "max_lines": {"type": "integer", "description": "Cap on output lines. Default 400."},
        },
    }

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        ok, why = _ensure_repo(ctx.cwd)
        if not ok:
            return ToolResult(why, is_error=True)
        cmd = ["diff"]
        if args.get("staged"):
            cmd.append("--cached")
        if args.get("pathspec"):
            cmd += ["--", args["pathspec"]]
        rc, out, err = await _git(ctx.cwd, *cmd)
        if rc != 0:
            return ToolResult(f"git diff failed: {err}", is_error=True)
        cap = int(args.get("max_lines", 400))
        lines = out.splitlines()
        if len(lines) > cap:
            out = "\n".join(lines[:cap]) + f"\n... [truncated {len(lines) - cap} more lines]"
        return ToolResult(out or "(no diff)")


class GitLogTool(Tool):
    name = "git_log"
    description = "Show recent commits: `git log --oneline -n <count>`."
    parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "description": "Number of commits to show. Default 10."},
        },
    }

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        ok, why = _ensure_repo(ctx.cwd)
        if not ok:
            return ToolResult(why, is_error=True)
        n = max(1, min(int(args.get("count", 10)), 50))
        rc, out, err = await _git(ctx.cwd, "log", "--oneline", f"-n{n}")
        if rc != 0:
            return ToolResult(f"git log failed: {err}", is_error=True)
        return ToolResult(out or "(no commits)")


class GitCommitTool(Tool):
    name = "git_commit"
    description = (
        "Stage all changes in the project directory and create a commit with "
        "the given message. Uses conventional commit style. Requires human "
        "approval before running."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message": {"type": "string", "description": "Commit message. Start with type: prefix (e.g. feat:, fix:, docs:)."},
        },
        "required": ["message"],
    }
    requires_approval = True

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        ok, why = _ensure_repo(ctx.cwd)
        if not ok:
            return ToolResult(why, is_error=True)
        message = args["message"]
        rc, _, err = await _git(ctx.cwd, "add", "-A")
        if rc != 0:
            return ToolResult(f"git add failed: {err}", is_error=True)
        rc, out, err = await _git(ctx.cwd, "commit", "-m", message)
        if rc != 0:
            return ToolResult(f"git commit failed: {err or out}", is_error=True)
        return ToolResult(out.strip() or "commit created")
