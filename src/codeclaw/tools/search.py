"""Search tool: ripgrep if available, otherwise a slow Python fallback."""
from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path

from .base import Tool, ToolContext, ToolResult


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search for a regex pattern across files in the project. Returns "
        "matching lines with file:line:content format. Defaults to recursive "
        "search of the project root, skipping .git, node_modules, .venv, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Python regex pattern to search for."},
            "path": {"type": "string", "description": "Directory to search. Default '.'."},
            "glob": {"type": "string", "description": "Optional filename glob filter, e.g. '*.py'."},
            "max_results": {"type": "integer", "description": "Cap on matches. Default 200."},
            "ignore_case": {"type": "boolean", "description": "Case-insensitive search. Default false."},
        },
        "required": ["pattern"],
    }

    IGNORE_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache", ".ruff_cache"}

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        pattern = args["pattern"]
        base = Path(args.get("path") or ctx.cwd)
        if not base.is_absolute():
            base = Path(ctx.cwd) / base
        base = base.resolve()
        glob = args.get("glob")
        cap = int(args.get("max_results", 200))
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            rx = re.compile(pattern, flags=flags)
        except re.error as exc:
            return ToolResult(f"invalid regex: {exc}", is_error=True)

        # Prefer ripgrep if installed — much faster on large repos.
        if shutil.which("rg") and not args.get("_force_python"):
            cmd = ["rg", "--no-heading", "--line-number", "--color=never", "--max-count", str(cap)]
            if args.get("ignore_case"):
                cmd.append("-i")
            if glob:
                cmd += ["--glob", glob]
            cmd += [pattern, str(base)]
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            out, err = await proc.communicate()
            text = (out or b"").decode("utf-8", errors="replace")
            err_text = (err or b"").decode("utf-8", errors="replace")
            if proc.returncode == 0:
                return ToolResult(text if text else "(no matches)")
            if proc.returncode == 1:
                return ToolResult("(no matches)")
            if proc.returncode == 2:
                # rg hit an error; fall through to Python
                pass
            else:
                return ToolResult(f"rg exit {proc.returncode}: {err_text}", is_error=True)

        # Pure-Python fallback.
        import fnmatch
        import os

        results: list[str] = []
        for root, dirs, files in os.walk(base, followlinks=False):
            dirs[:] = [d for d in dirs if d not in self.IGNORE_DIRS and not d.startswith(".")]
            for f in files:
                if glob and not fnmatch.fnmatch(f, glob):
                    continue
                full = Path(root) / f
                try:
                    with full.open("r", encoding="utf-8", errors="replace") as fh:
                        for i, line in enumerate(fh, 1):
                            if rx.search(line):
                                results.append(f"{full}:{i}:{line.rstrip()}")
                                if len(results) >= cap:
                                    return ToolResult("\n".join(results))
                except OSError:
                    continue
        return ToolResult("\n".join(results) if results else "(no matches)")


def _glob_match(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatch(name, pattern)
