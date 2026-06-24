"""Project hook loading and execution.

Hooks are intentionally small and local: a project may define command hooks in
`.codeclaw/settings.json`, and CodeClaw passes each hook a JSON payload on
stdin. Non-zero exits are surfaced to the caller; PreToolUse and
UserPromptSubmit can use that to block an action.
"""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HOOK_EVENTS: tuple[str, ...] = (
    "SessionStart",
    "SessionEnd",
    "UserPromptSubmit",
    "RunStart",
    "PreToolUse",
    "PostToolUse",
    "RunComplete",
)


@dataclass(frozen=True)
class HookResult:
    event: str
    command: str
    returncode: int
    output: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def hook_settings_path(project_dir: str | Path) -> Path:
    return Path(project_dir).resolve() / ".codeclaw" / "settings.json"


def load_hook_config(project_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    path = hook_settings_path(project_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    hooks = data.get("hooks")
    if not isinstance(hooks, dict):
        return {}

    normalized: dict[str, list[dict[str, Any]]] = {}
    for event in HOOK_EVENTS:
        entries = hooks.get(event)
        if not isinstance(entries, list):
            continue
        normalized[event] = []
        for entry in entries:
            if isinstance(entry, str):
                normalized[event].append({"type": "command", "command": entry})
            elif isinstance(entry, dict) and entry.get("command"):
                normalized[event].append(
                    {
                        "type": str(entry.get("type") or "command"),
                        "command": str(entry["command"]),
                        "timeout_s": entry.get("timeout_s"),
                    }
                )
    return {event: entries for event, entries in normalized.items() if entries}


def hook_counts(project_dir: str | Path) -> dict[str, int]:
    return {event: len(entries) for event, entries in load_hook_config(project_dir).items()}


async def run_hooks(
    project_dir: str | Path,
    event: str,
    payload: dict[str, Any],
    *,
    default_timeout_s: float = 30.0,
) -> list[HookResult]:
    config = load_hook_config(project_dir)
    results: list[HookResult] = []
    if event not in HOOK_EVENTS:
        return results

    root = Path(project_dir).resolve()
    event_payload = {"event": event, **payload}
    stdin = json.dumps(event_payload).encode("utf-8")
    for hook in config.get(event, []):
        if hook.get("type", "command") != "command":
            continue
        command = str(hook.get("command") or "").strip()
        if not command:
            continue
        timeout_s = _timeout_value(hook.get("timeout_s"), default_timeout_s)
        results.append(await _run_command_hook(root, event, command, stdin, timeout_s))
    return results


async def _run_command_hook(
    cwd: Path,
    event: str,
    command: str,
    stdin: bytes,
    timeout_s: float,
) -> HookResult:
    env = {**os.environ, "CODECLAW_HOOK_EVENT": event}
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(stdin), timeout=timeout_s)
        except TimeoutError:
            proc.kill()
            with suppress(ProcessLookupError):
                await proc.wait()
            return HookResult(event, command, 124, f"Hook timed out after {timeout_s:g}s")
    except OSError as exc:
        return HookResult(event, command, 127, f"{type(exc).__name__}: {exc}")

    output = (out or b"").decode("utf-8", errors="replace").strip()
    return HookResult(event, command, proc.returncode or 0, output[:8000])


def _timeout_value(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default

