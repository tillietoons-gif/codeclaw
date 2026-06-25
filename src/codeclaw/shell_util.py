"""Cross-platform subprocess spawning (Windows-safe)."""
from __future__ import annotations

import asyncio
import os
import shlex
import sys
from collections.abc import Mapping
from typing import Any


def _needs_shell(command: str) -> bool:
    return any(ch in command for ch in "|&<>^")


def _argv_from_command(command: str) -> list[str]:
    """Parse a shell command into argv (POSIX shells only)."""
    return shlex.split(command, posix=True)


async def spawn_command(
    command: str,
    *,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    stdin: bytes | None = None,
) -> asyncio.subprocess.Process:
    """Spawn a shell command.

    Windows always runs via ``cmd.exe /d /c`` because ``shlex.split`` does not
    parse ``list2cmdline`` output correctly and leaves quote characters in argv.
    Unix uses direct exec when the command has no shell metacharacters.
    """
    merged_env = {**os.environ, **(dict(env) if env else {})}
    kwargs: dict[str, Any] = {
        "cwd": cwd,
        "env": merged_env,
        "stdin": asyncio.subprocess.PIPE if stdin is not None else None,
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.STDOUT,
    }
    if sys.platform == "win32" or _needs_shell(command):
        if sys.platform == "win32":
            comspec = os.environ.get("COMSPEC", "cmd.exe")
            return await asyncio.create_subprocess_exec(
                comspec,
                "/d",
                "/c",
                command,
                **kwargs,
            )
        return await asyncio.create_subprocess_shell(command, **kwargs)
    argv = _argv_from_command(command)
    if not argv:
        raise ValueError("empty command")
    return await asyncio.create_subprocess_exec(*argv, **kwargs)
