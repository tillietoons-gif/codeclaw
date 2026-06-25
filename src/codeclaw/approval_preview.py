"""Generate diff previews shown in the approval gate."""
from __future__ import annotations

import asyncio
import difflib
from pathlib import Path


def _resolve(cwd: str, path: str) -> Path:
    root = Path(cwd).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    resolved = p.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes project directory: {path}")
    return resolved


def preview_write_file(cwd: str, path: str, content: str) -> str:
    try:
        target = _resolve(cwd, path)
    except ValueError as exc:
        return str(exc)
    old = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
    label_old = str(target) if target.exists() else "/dev/null"
    diff = difflib.unified_diff(
        old.splitlines(keepends=True),
        content.splitlines(keepends=True),
        fromfile=label_old,
        tofile=str(target),
        lineterm="",
    )
    text = "".join(diff)
    return text or f"(new file {target}, {len(content)} bytes)"


def preview_edit_file(cwd: str, path: str, old_text: str, new_text: str) -> str:
    try:
        target = _resolve(cwd, path)
    except ValueError as exc:
        return str(exc)
    if not target.exists():
        return f"File not found: {target}"
    current = target.read_text(encoding="utf-8", errors="replace")
    if current.count(old_text) != 1:
        return f"old_text matches {current.count(old_text)} locations in {target}"
    updated = current.replace(old_text, new_text, 1)
    diff = difflib.unified_diff(
        current.splitlines(keepends=True),
        updated.splitlines(keepends=True),
        fromfile=str(target),
        tofile=str(target),
        lineterm="",
    )
    return "".join(diff) or "(no visible diff)"


def preview_apply_patch(cwd: str, patch: str) -> str:
    from .tools.patch import preview_patch

    return preview_patch(cwd, patch)


async def preview_git_commit(cwd: str) -> str:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        "--stat",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    stat_out, _ = await proc.communicate()
    stat = (stat_out or b"").decode("utf-8", errors="replace").strip()

    proc = await asyncio.create_subprocess_exec(
        "git",
        "diff",
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    diff_out, _ = await proc.communicate()
    diff = (diff_out or b"").decode("utf-8", errors="replace")
    lines = diff.splitlines()
    if len(lines) > 120:
        diff = "\n".join(lines[:120]) + f"\n... [{len(lines) - 120} more lines]"
    parts = []
    if stat:
        parts.append(stat)
    if diff.strip():
        parts.append(diff)
    return "\n\n".join(parts) if parts else "(no staged or unstaged changes to commit)"


async def build_approval_preview(tool_name: str, args: dict, cwd: str) -> str | None:
    if tool_name == "write_file":
        return preview_write_file(cwd, args.get("path", ""), args.get("content", ""))
    if tool_name == "edit_file":
        return preview_edit_file(
            cwd,
            args.get("path", ""),
            args.get("old_text", ""),
            args.get("new_text", ""),
        )
    if tool_name == "apply_patch":
        return preview_apply_patch(cwd, args.get("patch", ""))
    if tool_name == "git_commit":
        return await preview_git_commit(cwd)
    return None
