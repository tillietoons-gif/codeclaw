"""Structured JSONL run logs for debugging and evaluation."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _runs_dir(project_dir: str | Path) -> Path:
    return Path(project_dir).resolve() / ".codeclaw" / "runs"


class RunLogger:
    """Append-only JSONL logger for a single agent run."""

    def __init__(self, project_dir: str | Path, session_id: str, run_id: str | None = None):
        self.session_id = session_id
        self.run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        directory = _runs_dir(project_dir) / session_id
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{self.run_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def log(self, event: str, **payload: Any) -> None:
        record = {
            "ts": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "event": event,
            "session_id": self.session_id,
            "run_id": self.run_id,
            **payload,
        }
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def list_run_logs(project_dir: str | Path, session_id: str | None = None, *, limit: int = 20) -> list[dict]:
    root = _runs_dir(project_dir)
    if not root.exists():
        return []
    entries: list[dict] = []
    if session_id:
        dirs = [root / session_id] if (root / session_id).is_dir() else []
    else:
        dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.stat().st_mtime, reverse=True)
    for directory in dirs:
        for path in sorted(directory.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
            entries.append(
                {
                    "session_id": directory.name,
                    "run_id": path.stem,
                    "path": str(path),
                    "size": path.stat().st_size,
                }
            )
            if len(entries) >= limit:
                return entries
    return entries
