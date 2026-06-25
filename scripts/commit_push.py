#!/usr/bin/env python3
"""Stage, commit, and push CodeClaw changes. Run from repo root."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EXCLUDE = {
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    ".sessions",
    ".codeclaw",
    "terminals",
    "mcps",
    ".env",
    "calculator.py",
    "scripts",
}


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    run("git", "status", "--short")
    run("git", "add", "-A")
    for name in sorted(EXCLUDE):
        run("git", "reset", "HEAD", "--", name)
    run("git", "status", "--short")
    msg = """Console UI overhaul, multi-provider LLM support, and Windows fixes

- Add console_ui.py with themed Rich output and REPL layout
- Add providers.py for multi-source LLM profiles
- Fix Windows shell spawning and re-enable tests
- Fix duplicate streamed output in agent/CLI
- Fix REPL prompt_toolkit toolbar and style issues
- Add tests for console UI and providers"""
    try:
        run("git", "commit", "-m", msg)
    except subprocess.CalledProcessError as exc:
        if exc.returncode == 1:
            print("Nothing to commit.", flush=True)
            return 0
        raise
    run("git", "log", "-1", "--oneline")
    run("git", "push", "origin", "master")
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
