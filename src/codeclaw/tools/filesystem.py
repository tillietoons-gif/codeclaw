"""Filesystem tools: read, write, edit, list."""
from __future__ import annotations

import os
from pathlib import Path

from .base import Tool, ToolContext, ToolResult


def _resolve(cwd: str, path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = Path(cwd) / p
    return p.resolve()


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a text file. Returns up to `max_lines` lines "
        "starting at `start_line` (0-indexed). Use this to inspect source code, "
        "config files, READMEs, etc. before making changes."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root, or absolute."},
            "start_line": {"type": "integer", "description": "0-indexed line offset. Default 0."},
            "max_lines": {"type": "integer", "description": "Cap on lines returned. Default 400."},
        },
        "required": ["path"],
    }

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        path = _resolve(ctx.cwd, args["path"])
        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)
        if not path.is_file():
            return ToolResult(f"Not a file: {path}", is_error=True)
        start = int(args.get("start_line", 0))
        limit = int(args.get("max_lines", 400))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(str(exc), is_error=True)
        lines = text.splitlines()
        end = min(start + limit, len(lines))
        body = "\n".join(f"{i + start:6d}\t{line}" for i, line in enumerate(lines[start:end], start=0))
        header = f"--- {path} ({len(lines)} lines, showing {start}-{end}) ---"
        return ToolResult(f"{header}\n{body}")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create or overwrite a file with the given content. Prefer `edit_file` "
        "for small in-place changes. This tool requires explicit human approval "
        "before it runs, because it can clobber existing files."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root, or absolute."},
            "content": {"type": "string", "description": "Full file content to write."},
        },
        "required": ["path", "content"],
    }
    requires_approval = True

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        path = _resolve(ctx.cwd, args["path"])
        content = args.get("content", "")
        existed = path.exists()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        except OSError as exc:
            return ToolResult(str(exc), is_error=True)
        action = "overwrote" if existed else "created"
        return ToolResult(f"{action} {path} ({len(content)} bytes)")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Apply a single targeted replacement to a file. `old_text` must match "
        "exactly one region. Use this for small, surgical edits; it is much "
        "safer than `write_file` because it cannot accidentally clobber "
        "unrelated changes. Requires approval before running."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path relative to the project root, or absolute."},
            "old_text": {"type": "string", "description": "The exact substring to replace. Must be unique."},
            "new_text": {"type": "string", "description": "The replacement text."},
        },
        "required": ["path", "old_text", "new_text"],
    }
    requires_approval = True

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        path = _resolve(ctx.cwd, args["path"])
        old = args["old_text"]
        new = args["new_text"]
        if not path.exists():
            return ToolResult(f"File not found: {path}", is_error=True)
        text = path.read_text(encoding="utf-8", errors="replace")
        count = text.count(old)
        if count == 0:
            return ToolResult(
                "old_text not found in file. Re-read the file and try again with the exact text.",
                is_error=True,
            )
        if count > 1:
            return ToolResult(
                f"old_text matched {count} locations. Add more surrounding context so it matches exactly one.",
                is_error=True,
            )
        new_text = text.replace(old, new, 1)
        path.write_text(new_text, encoding="utf-8")
        return ToolResult(f"edited {path} ({len(new) - len(old):+d} bytes)")


class ListDirTool(Tool):
    name = "list_dir"
    description = (
        "List the contents of a directory. Returns file names plus a one-line "
        "type/size summary. Pass `path='.'` for the project root."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Default '.'."},
            "max_depth": {"type": "integer", "description": "How deep to recurse. Default 1, max 4."},
            "ignore": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional glob patterns to ignore (beyond the default ignore list).",
            },
        },
    }

    DEFAULT_IGNORE = {
        ".git", "__pycache__", ".venv", "venv", "node_modules",
        ".pytest_cache", ".ruff_cache", ".mypy_cache", ".egg-info", "dist", "build",
    }

    async def run(self, args, ctx: ToolContext) -> ToolResult:
        path = _resolve(ctx.cwd, args.get("path", "."))
        if not path.exists():
            return ToolResult(f"Directory not found: {path}", is_error=True)
        if not path.is_dir():
            return ToolResult(f"Not a directory: {path}", is_error=True)
        max_depth = max(1, min(int(args.get("max_depth", 1)), 4))
        ignore = self.DEFAULT_IGNORE | set(args.get("ignore") or [])

        lines: list[str] = []
        base = path
        for root, dirs, files in os.walk(path):
            depth = Path(root).relative_to(base).parts if root != str(base) else ()
            if len(depth) >= max_depth:
                dirs[:] = []
                continue
            dirs[:] = sorted(d for d in dirs if d not in ignore and not d.startswith("."))
            indent = "  " * len(depth)
            if root != str(base):
                lines.append(f"{indent}{Path(root).name}/")
                indent += "  "
            for f in sorted(files):
                if f.startswith(".") or f in ignore:
                    continue
                full = Path(root) / f
                try:
                    size = full.stat().st_size
                except OSError:
                    size = -1
                lines.append(f"{indent}{f}  ({size}B)")
        return ToolResult("\n".join(lines) if lines else "(empty)")
