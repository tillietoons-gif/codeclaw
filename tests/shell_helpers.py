"""Cross-platform shell command helpers for tests (Windows-safe quoting)."""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path


def python_script_cmd(script: Path) -> str:
    args = [sys.executable, str(script)]
    if sys.platform == "win32":
        return subprocess.list2cmdline(args)
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(script))}"


def python_c_cmd(code: str) -> str:
    args = [sys.executable, "-c", code]
    if sys.platform == "win32":
        return subprocess.list2cmdline(args)
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"


def python_path_literal(path: Path) -> str:
    """Filesystem path safe to embed in Python source strings on Windows."""
    return str(path.resolve()).replace("\\", "/")
