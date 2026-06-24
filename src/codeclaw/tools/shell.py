"""Shell execution tool with a safety gate."""
from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from .base import Tool, ToolContext, ToolResult


def _resolve_cwd(project_dir: str, cwd: str | None) -> Path:
    root = Path(project_dir).resolve()
    path = Path(cwd) if cwd else root
    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"cwd escapes project directory: {cwd}")
    return resolved


class ExecTool(Tool):
    name = "exec"
    description = (
        "Run a shell command in the project directory and return its combined "
        "stdout/stderr. Use this for: running tests, inspecting builds, "
        "querying git, listing files, installing deps, etc. The shell is "
        "non-interactive; do not invoke REPLs or anything that needs a TTY. "
        "Long-running commands should set a sensible `timeout_s`."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute. Avoid sudo, rm -rf, dd, mkfs, etc."},
            "timeout_s": {"type": "integer", "description": "Max seconds to wait. Default 60, max 600."},
            "cwd": {"type": "string", "description": "Override working directory. Default = project root."},
        },
        "required": ["command"],
    }
    requires_approval = True  # any shell exec is worth a beat of friction

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        cmd = args["command"]
        timeout = max(1, min(int(args.get("timeout_s", 60)), 600))
        try:
            cwd_path = _resolve_cwd(ctx.cwd, args.get("cwd"))
        except ValueError as exc:
            return ToolResult(str(exc), is_error=True)
        if not cwd_path.exists():
            return ToolResult(f"cwd does not exist: {cwd_path}", is_error=True)

        ctx.log(f"$ {cmd}")
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=str(cwd_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        except (OSError, ValueError) as exc:  # pragma: no cover
            return ToolResult(f"failed to spawn: {exc}", is_error=True)
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return ToolResult(
                f"command timed out after {timeout}s; killed.",
                is_error=True,
            )
        elapsed = time.monotonic() - t0
        text = stdout.decode("utf-8", errors="replace") if stdout else ""
        # Truncate to keep the model's context manageable.
        max_bytes = 20_000
        if len(text) > max_bytes:
            text = text[:max_bytes] + f"\n... [truncated {len(text) - max_bytes} chars]"
        rc = proc.returncode
        status = "ok" if rc == 0 else f"exit {rc}"
        return ToolResult(f"[{status}, {elapsed:.1f}s]\n{text}" if text else f"[{status}, {elapsed:.1f}s]")
