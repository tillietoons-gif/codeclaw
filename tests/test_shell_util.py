"""Tests for cross-platform shell spawning."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from codeclaw.shell_util import _argv_from_command, spawn_command
from shell_helpers import python_c_cmd, python_script_cmd


def test_argv_from_command_parses_quoted_python_on_unix():
    if sys.platform == "win32":
        pytest.skip("POSIX argv parsing is not used on Windows")
    script = "/usr/bin/python3"
    path = "/tmp/hook_echo.py"
    command = f"{script} {path}"
    assert _argv_from_command(command) == [script, path]


@pytest.mark.asyncio
async def test_spawn_command_runs_python_script(tmp_path):
    script = tmp_path / "say.py"
    script.write_text("print('spawn-ok')\n", encoding="utf-8")
    proc = await spawn_command(python_script_cmd(script), cwd=str(tmp_path))
    out, _ = await proc.communicate()
    assert proc.returncode == 0
    assert b"spawn-ok" in out


@pytest.mark.asyncio
async def test_spawn_command_runs_python_c(tmp_path):
    proc = await spawn_command(python_c_cmd("print('c-ok')"), cwd=str(tmp_path))
    out, _ = await proc.communicate()
    assert proc.returncode == 0
    assert b"c-ok" in out
