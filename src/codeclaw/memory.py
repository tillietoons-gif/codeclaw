"""Auto-load `AGENTS.md` and `MEMORY.md` from the project directory.

These are surfaced to the model as part of the system context so it has the
same awareness a human developer would: project conventions, prior decisions,
long-running notes. Files are optional and silently skipped when missing.
"""
from __future__ import annotations

from pathlib import Path

MAX_FILE_BYTES = 32_000  # hard cap so a runaway memory file can't blow context


def _safe_read(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) > MAX_FILE_BYTES:
        text = text[:MAX_FILE_BYTES] + f"\n... [truncated, {MAX_FILE_BYTES} byte cap]"
    return text


def load_project_context(project_dir: str) -> str:
    """Return a single context block with AGENTS.md and MEMORY.md contents."""
    root = Path(project_dir).resolve()
    sections: list[str] = []
    for name, header in (
        ("AGENTS.md", "Project agent guidelines (AGENTS.md)"),
        ("MEMORY.md", "Project memory (MEMORY.md)"),
    ):
        body = _safe_read(root / name)
        if body.strip():
            sections.append(f"## {header}\n{body.strip()}")
    if not sections:
        return ""
    return "The following project-local files were found. Follow them when relevant.\n\n" + "\n\n".join(sections)
