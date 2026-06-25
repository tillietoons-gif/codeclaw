"""Project policy: exec command allow/deny rules."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExecPolicy:
    deny_patterns: tuple[str, ...] = ()
    allow_without_prompt: tuple[str, ...] = ()

    def denies(self, command: str) -> str | None:
        for pattern in self.deny_patterns:
            if _matches_pattern(command, pattern):
                return pattern
        return None

    def allows_without_prompt(self, command: str) -> bool:
        stripped = command.strip()
        for prefix in self.allow_without_prompt:
            if stripped == prefix or stripped.startswith(prefix + " "):
                return True
        return False


def _matches_pattern(command: str, pattern: str) -> bool:
    pattern = pattern.strip()
    if not pattern:
        return False
    if pattern.startswith("re:"):
        try:
            return re.search(pattern[3:], command, re.IGNORECASE) is not None
        except re.error:
            return pattern[3:].lower() in command.lower()
    return pattern.lower() in command.lower()


def _settings_path(project_dir: str | Path) -> Path:
    return Path(project_dir).resolve() / ".codeclaw" / "settings.json"


def load_exec_policy(project_dir: str | Path) -> ExecPolicy:
    path = _settings_path(project_dir)
    if not path.exists():
        return ExecPolicy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ExecPolicy()
    exec_cfg = data.get("exec") if isinstance(data, dict) else None
    if not isinstance(exec_cfg, dict):
        return ExecPolicy()
    deny = tuple(str(p) for p in (exec_cfg.get("deny_patterns") or []) if str(p).strip())
    allow = tuple(str(p) for p in (exec_cfg.get("allow_without_prompt") or []) if str(p).strip())
    return ExecPolicy(deny_patterns=deny, allow_without_prompt=allow)


def default_exec_policy_dict() -> dict:
    return {
        "deny_patterns": ["rm -rf", "sudo", "curl | sh", "wget | sh", "mkfs", "dd if="],
        "allow_without_prompt": ["pytest", "python -m pytest", "ruff check", "ruff format --check"],
    }
