"""Shared helpers for filesystem-bound tools."""
from __future__ import annotations

from pathlib import Path

IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules",
    ".pytest_cache", ".ruff_cache", ".mypy_cache", ".egg-info", "dist", "build",
}


def resolve_under_root(cwd: str, path: str) -> Path:
    root = Path(cwd).resolve()
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    resolved = p.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"path escapes project directory: {path}")
    return resolved
