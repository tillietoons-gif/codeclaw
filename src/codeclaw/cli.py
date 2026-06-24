"""Command-line interface: one-shot mode, REPL, and a few utility commands.

Usage:
    codeclaw                         # interactive console UI
    codeclaw "add a Makefile that runs pytest"
    codeclaw --model llama3.1:8b "..."
    codeclaw check                   # verify Ollama is reachable + model exists
    codeclaw --tools                 # print available tool schemas
    codeclaw repl                    # interactive console UI
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
import sys
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import DESTRUCTIVE_TOOLS, CodeClawAgent
from .config import load_settings
from .hooks import HOOK_EVENTS, HookResult, hook_counts, hook_settings_path, run_hooks
from .memory import load_project_context
from .ollama import OllamaClient, OllamaError
from .tools import build_default_registry
from .tools.base import ApprovalDecision

logger = logging.getLogger("codeclaw")

CONSOLE = Console()

SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "Show available slash commands."),
    ("/status", "Show current model, project, approval mode, and git state."),
    ("/plan", "Toggle read-only planning mode for future prompts."),
    ("/sessions", "List saved sessions for this project."),
    ("/current", "Show the current session details."),
    ("/resume ID", "Resume a saved session."),
    ("/memory", "Show loaded AGENTS.md and MEMORY.md context."),
    ("/hooks", "Show configured project lifecycle hooks."),
    ("/checkpoint NAME", "Save a local project snapshot."),
    ("/checkpoints", "List saved local snapshots."),
    ("/restore ID", "Restore a saved local snapshot."),
    ("/changes", "Show git status and diff summary."),
    ("/tools", "List available CodeClaw tools."),
    ("/permissions", "Show which tools require approval in this session."),
    ("/diff", "Show the current git diff summary."),
    ("/models", "Choose from installed Ollama models."),
    ("/model NAME", "Switch directly to a model."),
    ("/reset", "Clear the current prompt flow."),
    ("/quit", "Exit CodeClaw."),
)

SNAPSHOT_EXCLUDE_DIRS = {
    ".git", ".codeclaw", "__pycache__", ".pytest_cache", ".ruff_cache",
    ".mypy_cache", ".venv", "venv", "node_modules", "build", "dist",
}
SNAPSHOT_EXCLUDE_FILES = {".env", ".DS_Store"}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codeclaw",
        description="Autonomous coding agent powered by local Ollama.",
    )
    p.add_argument("objective", nargs="?", default=None, help="What to do. If omitted, enters REPL.")
    p.add_argument("--model", help="Override CODECLAW_MODEL")
    p.add_argument("--select-model", action="store_true", help="Choose from installed Ollama models before running.")
    p.add_argument("--project-dir", help="Override CODECLAW_PROJECT_DIR")
    p.add_argument("--max-steps", type=int, help="Override CODECLAW_MAX_STEPS")
    p.add_argument("--temperature", type=float, help="Override CODECLAW_TEMPERATURE")
    p.add_argument("--auto-approve", action="store_true", help="Approve all destructive actions without prompting (use with care).")
    p.add_argument("--non-interactive", action="store_true", help="Disable interactive prompts. Combine with --auto-approve for CI use.")
    p.add_argument("--check", action="store_true", help="Verify Ollama is reachable and the model is loaded, then exit.")
    p.add_argument("--tools", action="store_true", help="Print the tool registry as JSON and exit.")
    p.add_argument("--version", action="store_true", help="Print the CodeClaw version and exit.")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose logging.")
    return p


async def _async_main(args: argparse.Namespace) -> int:
    settings = load_settings()
    # Apply CLI overrides on top of env-derived settings.
    overrides: dict[str, object] = {}
    if args.model:
        overrides["model"] = args.model
    if args.project_dir:
        overrides["project_dir"] = args.project_dir
    if args.max_steps:
        overrides["max_steps"] = args.max_steps
    if args.temperature is not None:
        overrides["temperature"] = args.temperature
    if overrides:
        settings = replace(settings, **overrides)

    if args.version:
        print(f"codeclaw {__version__}")
        return 0

    if args.tools:
        reg = build_default_registry()
        print(json.dumps(reg.schemas(), indent=2))
        return 0

    client = OllamaClient(settings.ollama_host, timeout_s=settings.request_timeout_s)
    try:
        opens_interactive_ui = args.objective in ("repl", "models", "continue") or (not args.objective and sys.stdin.isatty())
        should_select_model = args.select_model or (opens_interactive_ui and not args.model)
        if should_select_model:
            if args.non_interactive:
                CONSOLE.print("[bold red]error:[/bold red] model selection cannot be used with --non-interactive")
                return 2
            selected = await _select_model(client, settings)
            if selected is None:
                return 1
            settings = selected

        if args.check or args.objective == "check":
            return await _do_check(client, settings)

        if args.objective in ("repl", "models", "continue"):
            resume_latest = args.objective == "continue"
            return await _run_repl(settings, client, args, resume_latest=resume_latest)
        if not args.objective and sys.stdin.isatty():
            return await _run_repl(settings, client, args)
        elif not args.objective:
            objective = sys.stdin.read().strip()
            if not objective:
                print("error: no objective provided (stdin empty)", file=sys.stderr)
                return 2
            return await _run_one_shot(settings, client, args, objective)
        else:
            return await _run_one_shot(settings, client, args, args.objective)
    finally:
        await client.close()


async def _do_check(client: OllamaClient, settings) -> int:
    try:
        models = await client.list_models()
    except OllamaError as exc:
        CONSOLE.print(f"[bold red]FAIL[/bold red] cannot reach Ollama at [cyan]{settings.ollama_host}[/cyan]: {exc}")
        return 1
    if not models:
        CONSOLE.print("[bold red]FAIL[/bold red] Ollama reachable but no models installed.")
        return 1
    table = Table(title=f"Ollama at {settings.ollama_host}", show_header=True, header_style="bold cyan")
    table.add_column("Model", overflow="fold")
    table.add_column("Context", justify="right")
    table.add_column("Capabilities", overflow="fold")
    for m in models:
        name = str(m.get("name") or m.get("model") or "?")
        details = dict(m.get("details") or {})
        caps = list(m.get("capabilities") or [])
        if name != "?":
            try:
                shown = await client.show_model(name)
            except OllamaError:
                shown = {}
            caps = list(shown.get("capabilities") or caps)
            details = {**details, **(shown.get("details") or {})}
            model_info = shown.get("model_info") or {}
            context_length = (
                model_info.get("qwen2.context_length")
                or model_info.get("qwen3.context_length")
                or details.get("context_length")
                or "?"
            )
        else:
            context_length = details.get("context_length", "?")
        table.add_row(
            name,
            str(context_length),
            ", ".join(caps) or "none",
        )
    CONSOLE.print("[bold green]OK[/bold green]")
    CONSOLE.print(table)
    if settings.model not in {m.get("name") for m in models} | {m.get("model") for m in models}:
        CONSOLE.print(
            f"[bold yellow]WARN[/bold yellow] configured model [cyan]{settings.model!r}[/cyan] is not installed. "
            f"Run [bold]ollama pull {settings.model}[/bold]."
        )
        return 3
    return 0


def _model_name(model: dict) -> str | None:
    name = model.get("name") or model.get("model")
    return str(name) if name else None


async def _model_display_details(client: OllamaClient, model: dict) -> tuple[str, str]:
    name = _model_name(model)
    details = dict(model.get("details") or {})
    caps = list(model.get("capabilities") or [])
    model_info = {}
    if name:
        try:
            shown = await client.show_model(name)
        except OllamaError:
            shown = {}
        caps = list(shown.get("capabilities") or caps)
        details = {**details, **(shown.get("details") or {})}
        model_info = shown.get("model_info") or {}
    context_length = (
        model_info.get("qwen2.context_length")
        or model_info.get("qwen3.context_length")
        or details.get("context_length")
        or "?"
    )
    return str(context_length), ", ".join(caps) or "none"


async def _select_model(client: OllamaClient, settings, *, console: Console = CONSOLE):
    from rich.prompt import IntPrompt

    try:
        models = await client.list_models()
    except OllamaError as exc:
        console.print(f"[bold red]FAIL[/bold red] cannot reach Ollama at [cyan]{settings.ollama_host}[/cyan]: {exc}")
        return None

    names = [name for model in models if (name := _model_name(model))]
    if not names:
        console.print("[bold red]FAIL[/bold red] Ollama reachable but no models installed.")
        return None

    table = Table(title="Select Model", show_header=True, header_style="bold cyan")
    table.add_column("#", justify="right")
    table.add_column("Model", overflow="fold")
    table.add_column("Context", justify="right")
    table.add_column("Capabilities", overflow="fold")
    for idx, model in enumerate(models, 1):
        name = _model_name(model)
        if not name:
            continue
        context_length, capabilities = await _model_display_details(client, model)
        style = "bold green" if name == settings.model else ""
        table.add_row(str(idx), name, context_length, capabilities, style=style)
    console.print(table)

    default_idx = names.index(settings.model) + 1 if settings.model in names else 1
    choice = IntPrompt.ask(
        "Model",
        choices=[str(i) for i in range(1, len(names) + 1)],
        default=default_idx,
        console=console,
    )
    selected = names[choice - 1]
    console.print(f"[dim]model ->[/dim] [cyan]{selected}[/cyan]")
    return replace(settings, model=selected)


def _print_session_header(settings, objective: str | None = None, *, console: Console = CONSOLE) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("model", f"[cyan]{settings.model}[/cyan]")
    table.add_row("project", f"[cyan]{settings.project_dir}[/cyan]")
    table.add_row("steps", str(settings.max_steps))
    if objective:
        table.add_row("objective", Text(objective, overflow="fold"))
    console.print(
        Panel(
            table,
            title=f"[bold]CodeClaw {__version__}[/bold]",
            subtitle="local coding agent",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _print_repl_header(settings, *, console: Console = CONSOLE) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("model", f"[cyan]{settings.model}[/cyan]")
    table.add_row("project", f"[cyan]{settings.project_dir}[/cyan]")
    table.add_row(
        "commands",
        "[bold]/help[/bold] commands   [bold]/plan[/bold] plan mode   [bold]/models[/bold] choose   [bold]/status[/bold] inspect",
    )
    console.print(
        Panel(
            table,
            title=f"[bold]CodeClaw {__version__}[/bold]",
            subtitle="interactive mode",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _print_command_palette(filter_text: str = "", *, console: Console = CONSOLE) -> None:
    needle = filter_text.lower().strip().removeprefix("/")
    table = Table(title="Slash Commands", show_header=True, header_style="bold cyan")
    table.add_column("Command", style="bold green", no_wrap=True)
    table.add_column("Description", overflow="fold")
    shown = 0
    for command, description in SLASH_COMMANDS:
        searchable = f"{command} {description}".lower()
        if needle and needle not in searchable:
            continue
        table.add_row(command, description)
        shown += 1
    if shown:
        console.print(table)
    else:
        console.print(f"[yellow]No slash commands match[/yellow] [bold]/{needle}[/bold]. Try [bold]/help[/bold].")


def _print_tools_table(*, console: Console = CONSOLE) -> None:
    table = Table(title="Tools", show_header=True, header_style="bold cyan")
    table.add_column("Tool", style="bold")
    table.add_column("Approval", justify="center")
    table.add_column("Description", overflow="fold")
    for tool in build_default_registry()._tools.values():
        approval = "yes" if tool.name in DESTRUCTIVE_TOOLS else "no"
        table.add_row(tool.name, approval, tool.description)
    console.print(table)


def _print_permissions(args, *, console: Console = CONSOLE) -> None:
    mode = "auto-approve" if args.auto_approve else "non-interactive deny" if args.non_interactive else "ask"
    table = Table(title="Permissions", show_header=True, header_style="bold cyan")
    table.add_column("Tool", style="bold")
    table.add_column("Mode", justify="center")
    table.add_column("Reason", overflow="fold")
    for tool in build_default_registry().names():
        if tool in DESTRUCTIVE_TOOLS:
            table.add_row(tool, mode, "Mutates files, runs shell commands, or writes git history.")
        else:
            table.add_row(tool, "allow", "Read-only project inspection.")
    console.print(table)


def _print_status(settings, args, *, plan_mode: bool = False, session_id: str = "", console: Console = CONSOLE) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("model", f"[cyan]{settings.model}[/cyan]")
    table.add_row("project", f"[cyan]{settings.project_dir}[/cyan]")
    table.add_row("host", f"[cyan]{settings.ollama_host}[/cyan]")
    table.add_row("steps", str(settings.max_steps))
    table.add_row("temperature", str(settings.temperature))
    table.add_row("approval", "auto-approve" if args.auto_approve else "non-interactive deny" if args.non_interactive else "ask")
    table.add_row("mode", "plan" if plan_mode else "act")
    if session_id:
        table.add_row("session", session_id)
    table.add_row("cwd", os.getcwd())
    console.print(Panel(table, title="Status", border_style="cyan", padding=(1, 2)))


def _session_dir(settings) -> Path:
    return Path(settings.project_dir).resolve() / ".codeclaw" / "sessions"


def _new_session(settings) -> dict:
    created_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    session_id = created_at.replace(":", "").replace("-", "").replace("Z", "")
    return {
        "id": session_id,
        "created_at": created_at,
        "updated_at": created_at,
        "project_dir": str(Path(settings.project_dir).resolve()),
        "model": settings.model,
        "turns": [],
    }


def _save_session(settings, session: dict) -> Path:
    session["updated_at"] = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    directory = _session_dir(settings)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{session['id']}.json"
    path.write_text(json.dumps(session, indent=2), encoding="utf-8")
    return path


def _append_session_turn(settings, session: dict, objective: str, result, *, plan_mode: bool) -> None:
    session["model"] = settings.model
    session["turns"].append(
        {
            "objective": objective,
            "final_message": result.final_message,
            "completed": result.completed,
            "reason": result.reason,
            "steps": len(result.steps),
            "tokens": result.total_tokens,
            "plan_mode": plan_mode,
            "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        }
    )
    _save_session(settings, session)


def _load_sessions(settings) -> list[dict]:
    directory = _session_dir(settings)
    sessions: list[dict] = []
    if not directory.exists():
        return sessions
    for path in sorted(directory.glob("*.json"), reverse=True):
        try:
            sessions.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return sessions


def _find_session(settings, session_id: str) -> dict | None:
    for session in _load_sessions(settings):
        sid = str(session.get("id", ""))
        if sid == session_id or sid.startswith(session_id):
            return session
    return None


def _latest_session(settings) -> dict | None:
    sessions = _load_sessions(settings)
    return sessions[0] if sessions else None


def _session_context(session: dict, objective: str) -> str:
    turns = session.get("turns") or []
    if not turns:
        return objective
    lines = [
        "RESUMED SESSION CONTEXT:",
        "Use the prior session turns below as context. Continue naturally from them, but follow the latest user objective.",
        "",
    ]
    for idx, turn in enumerate(turns[-8:], 1):
        lines.append(f"Turn {idx} objective: {turn.get('objective', '')}")
        final = str(turn.get("final_message", "")).strip()
        if final:
            lines.append(f"Turn {idx} result: {final[:2000]}")
        lines.append("")
    lines.append(f"Latest objective: {objective}")
    return "\n".join(lines)


def _print_current_session(session: dict, *, console: Console = CONSOLE) -> None:
    turns = session.get("turns") or []
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("id", str(session.get("id", "?")))
    table.add_row("model", str(session.get("model", "?")))
    table.add_row("created", str(session.get("created_at", "?")))
    table.add_row("updated", str(session.get("updated_at", "?")))
    table.add_row("turns", str(len(turns)))
    if turns:
        table.add_row("last", str(turns[-1].get("objective", "")))
    console.print(Panel(table, title="current session", border_style="cyan", padding=(1, 2)))


def _print_sessions(settings, *, console: Console = CONSOLE) -> None:
    sessions = _load_sessions(settings)
    if not sessions:
        console.print(Panel("No saved sessions yet.", title="sessions", border_style="yellow"))
        return
    table = Table(title="Sessions", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="bold")
    table.add_column("Updated")
    table.add_column("Model", overflow="fold")
    table.add_column("Turns", justify="right")
    table.add_column("Last Objective", overflow="fold")
    for session in sessions[:20]:
        turns = session.get("turns") or []
        last = turns[-1].get("objective", "") if turns else ""
        table.add_row(
            str(session.get("id", "?")),
            str(session.get("updated_at", "?")),
            str(session.get("model", "?")),
            str(len(turns)),
            last,
        )
    console.print(table)


def _print_memory(settings, *, console: Console = CONSOLE) -> None:
    context = load_project_context(settings.project_dir)
    if not context:
        console.print(Panel("No AGENTS.md or MEMORY.md found for this project.", title="memory", border_style="yellow"))
        return
    console.print(Panel(Markdown(context), title="memory", border_style="cyan", padding=(1, 2)))


def _print_hooks(settings, *, console: Console = CONSOLE) -> None:
    counts = hook_counts(settings.project_dir)
    path = hook_settings_path(settings.project_dir)
    if not counts:
        console.print(
            Panel(
                f"No hooks configured.\n[dim]Create {path} with a hooks object to add them.[/dim]",
                title="hooks",
                border_style="yellow",
            )
        )
        return
    table = Table(title=f"Hooks ({path})", show_header=True, header_style="bold cyan")
    table.add_column("Event", style="bold")
    table.add_column("Commands", justify="right")
    for event in HOOK_EVENTS:
        if event in counts:
            table.add_row(event, str(counts[event]))
    console.print(table)


def _checkpoint_dir(settings) -> Path:
    return Path(settings.project_dir).resolve() / ".codeclaw" / "checkpoints"


def _checkpoint_id(name: str = "") -> str:
    stamp = datetime.now(UTC).replace(microsecond=0).strftime("%Y%m%dT%H%M%SZ")
    slug = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in name.strip().lower()).strip("-")
    return f"{stamp}-{slug}" if slug else stamp


def _iter_snapshot_files(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in SNAPSHOT_EXCLUDE_DIRS]
        current_path = Path(current)
        for name in files:
            if name in SNAPSHOT_EXCLUDE_FILES:
                continue
            path = current_path / name
            if path.is_file():
                yield path


def _create_checkpoint(settings, name: str = "") -> dict:
    root = Path(settings.project_dir).resolve()
    checkpoint_id = _checkpoint_id(name)
    directory = _checkpoint_dir(settings) / checkpoint_id
    files_dir = directory / "files"
    files: list[str] = []
    files_dir.mkdir(parents=True, exist_ok=True)
    for path in _iter_snapshot_files(root):
        rel = path.relative_to(root)
        dest = files_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        files.append(rel.as_posix())
    metadata = {
        "id": checkpoint_id,
        "name": name,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "project_dir": str(root),
        "files": sorted(files),
    }
    (directory / "checkpoint.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _load_checkpoints(settings) -> list[dict]:
    directory = _checkpoint_dir(settings)
    checkpoints: list[dict] = []
    if not directory.exists():
        return checkpoints
    for path in sorted(directory.glob("*/checkpoint.json"), reverse=True):
        try:
            checkpoints.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return checkpoints


def _find_checkpoint(settings, checkpoint_id: str) -> dict | None:
    for checkpoint in _load_checkpoints(settings):
        cid = str(checkpoint.get("id", ""))
        if cid == checkpoint_id or cid.startswith(checkpoint_id):
            return checkpoint
    return None


def _restore_checkpoint(settings, checkpoint_id: str) -> tuple[bool, str]:
    checkpoint = _find_checkpoint(settings, checkpoint_id)
    if not checkpoint:
        return False, f"Checkpoint not found: {checkpoint_id}"
    root = Path(settings.project_dir).resolve()
    directory = _checkpoint_dir(settings) / checkpoint["id"]
    files_dir = directory / "files"
    manifest = set(checkpoint.get("files") or [])

    for path in list(_iter_snapshot_files(root)):
        rel = path.relative_to(root).as_posix()
        if rel not in manifest:
            path.unlink()

    for rel in manifest:
        src = files_dir / rel
        dest = root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    return True, str(checkpoint["id"])


def _print_checkpoints(settings, *, console: Console = CONSOLE) -> None:
    checkpoints = _load_checkpoints(settings)
    if not checkpoints:
        console.print(Panel("No checkpoints yet.", title="checkpoints", border_style="yellow"))
        return
    table = Table(title="Checkpoints", show_header=True, header_style="bold cyan")
    table.add_column("ID", style="bold")
    table.add_column("Created")
    table.add_column("Name", overflow="fold")
    table.add_column("Files", justify="right")
    for checkpoint in checkpoints[:20]:
        table.add_row(
            str(checkpoint.get("id", "?")),
            str(checkpoint.get("created_at", "?")),
            str(checkpoint.get("name", "")),
            str(len(checkpoint.get("files") or [])),
        )
    console.print(table)


def _checkpoint_name_from_command(line: str) -> str:
    prefix = "/checkpoint"
    return line[len(prefix):].strip() if line.startswith(prefix) else ""


def _restore_id_from_command(line: str) -> str | None:
    prefix = "/restore "
    if line.startswith(prefix):
        return line[len(prefix):].strip()
    return None


async def _git_output(project_dir: str, *args: str) -> tuple[int, str]:
    proc = await asyncio.create_subprocess_exec(
        "git",
        *args,
        cwd=project_dir,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    out, _ = await proc.communicate()
    return proc.returncode or 0, (out or b"").decode("utf-8", errors="replace")


async def _print_diff(settings, *, console: Console = CONSOLE) -> None:
    rc, out = await _git_output(settings.project_dir, "diff", "--stat")
    if rc != 0:
        console.print(Panel(out.strip() or "git diff failed", title="diff", border_style="red"))
        return
    if not out.strip():
        console.print(Panel("No working-tree diff.", title="diff", border_style="green"))
        return
    console.print(Panel(out.rstrip(), title="diff --stat", border_style="yellow"))


async def _print_changes(settings, *, console: Console = CONSOLE) -> None:
    status_rc, status = await _git_output(settings.project_dir, "status", "--short", "--branch")
    diff_rc, diff = await _git_output(settings.project_dir, "diff", "--stat")
    body = []
    if status_rc == 0:
        body.append(status.strip() or "(clean)")
    else:
        body.append(status.strip() or "git status failed")
    if diff_rc == 0 and diff.strip():
        body.append("\n" + diff.rstrip())
    console.print(Panel("\n".join(body), title="changes", border_style="cyan"))


def _log_stream(console: Console = CONSOLE) -> Callable[[str], None]:
    """Return a structured Rich logger for the agent loop."""
    def log(msg: str) -> None:
        if msg.startswith("[error]") or msg.startswith("[denied]"):
            console.print(f"[bold red]{msg}[/bold red]")
        elif msg.startswith("[approved]"):
            console.print(f"[bold green]{msg}[/bold green]")
        elif msg.startswith("[step"):
            console.rule(f"[bold cyan]{msg.strip()}[/bold cyan]", style="dim")
        elif msg.startswith("[hook]"):
            console.print(f"[dim]{msg}[/dim]")
        elif msg.startswith("  [tool_calls"):
            console.print(f"[bold magenta]{msg.strip()}[/bold magenta]")
        elif msg.startswith("  [thinking"):
            console.print(f"[bold yellow]{msg.strip()}[/bold yellow]")
        elif msg.startswith(("  [tokens", "  [hook]")):
            console.print(f"[dim]{msg.strip()}[/dim]")
        elif msg.startswith("  ?"):
            console.print(Text(msg[3:], style="dim yellow"))
        elif msg.startswith("  >"):
            console.print(Text(msg[3:], style="green"))
        elif msg.startswith("$ "):
            console.print(Panel(Text(msg, overflow="fold"), title="shell", border_style="blue"))
        else:
            console.print(msg)

    return log


def _interactive_approval(args) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    """Build the approval callback used by the agent."""
    auto = bool(args.auto_approve)
    non_interactive = bool(args.non_interactive)
    cache: dict[str, bool] = {}  # "tool_name:summary" -> approved once

    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if auto:
            return ApprovalDecision(ApprovalDecision.APPROVE, reason="--auto-approve")
        if non_interactive:
            return ApprovalDecision(
                ApprovalDecision.REJECT,
                reason="destructive action in non-interactive mode without --auto-approve",
            )
        key = f"{tool_name}:{summary}"
        if key in cache and cache[key]:
            return ApprovalDecision(ApprovalDecision.APPROVE, reason="cached")
        from rich.prompt import Confirm

        body = Table.grid(padding=(0, 1))
        body.add_column(style="bold")
        body.add_column()
        body.add_row("tool", f"[cyan]{tool_name}[/cyan]")
        body.add_row("action", Text(summary, overflow="fold"))
        CONSOLE.print(
            Panel(
                body,
                title="[bold yellow]Approve Action[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )
        try:
            ok = Confirm.ask("Allow this action?", default=False, console=CONSOLE)
        except (EOFError, KeyboardInterrupt):
            ok = False
        if ok:
            cache[key] = True
            return ApprovalDecision(ApprovalDecision.APPROVE)
        return ApprovalDecision(ApprovalDecision.REJECT, reason="user said no")

    return approve


def _plan_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="plan mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _plan_mode_objective(objective: str) -> str:
    return (
        "PLAN MODE: Research and propose a concrete implementation plan. "
        "Do not edit files, run shell commands, commit changes, or perform other side effects. "
        "You may use read-only inspection tools. End with clear steps and risks.\n\n"
        f"User objective: {objective}"
    )


def _ask_repl_line(console: Console) -> str:
    from rich.prompt import Prompt

    console.print(
        Panel(
            "[dim]Type a prompt, or use [bold]/help[/bold], [bold]/plan[/bold], "
            "[bold]/models[/bold], [bold]/status[/bold], [bold]/quit[/bold].[/dim]",
            title="prompt",
            border_style="green",
            padding=(0, 1),
        )
    )
    return Prompt.ask("[bold green]›[/bold green]", console=console).strip()


def _is_quit_command(line: str) -> bool:
    return line in ("/q", "/quit", "/exit")


def _is_reset_command(line: str) -> bool:
    return line == "/reset"


def _is_model_picker_command(line: str) -> bool:
    return line in ("/model", "/models")


def _is_help_command(line: str) -> bool:
    return line in ("/", "/help", "/?")


def _is_plan_command(line: str) -> bool:
    return line in ("/plan", "/plan on", "/plan off")


def _slash_filter(line: str) -> str | None:
    if not line.startswith("/") or " " in line:
        return None
    known = {
        "/q", "/quit", "/exit", "/reset", "/model", "/models",
        "/help", "/?", "/", "/status", "/tools", "/permissions", "/diff",
        "/plan", "/sessions", "/current", "/memory", "/hooks", "/checkpoint", "/checkpoints", "/changes",
    }
    if line.startswith("/restore ") or line.startswith("/checkpoint ") or line.startswith("/resume "):
        return None
    return None if line in known else line


def _model_name_from_command(line: str) -> str | None:
    prefix = "/model "
    if line.startswith(prefix):
        return line[len(prefix):].strip()
    return None


def _resume_id_from_command(line: str) -> str | None:
    prefix = "/resume "
    if line.startswith(prefix):
        return line[len(prefix):].strip()
    return None


async def _run_one_shot(settings, client, args, objective: str) -> int:
    with suppress(OllamaError):
        await client.show_model(settings.model)
    _print_session_header(settings, objective)
    approval = _interactive_approval(args)
    agent = CodeClawAgent(
        settings=settings,
        client=client,
        approval=approval,
        log=_log_stream(CONSOLE),
    )
    try:
        result = await agent.run(objective)
    except OllamaError as exc:
        CONSOLE.print(f"[bold red]error:[/bold red] {exc}")
        return 1
    _print_final_report(result)
    return 0 if result.completed else 2


async def _run_cli_hooks(settings, event: str, payload: dict, *, console: Console = CONSOLE) -> list[HookResult]:
    results = await run_hooks(settings.project_dir, event, payload)
    for result in results:
        status = "ok" if result.ok else f"exit {result.returncode}"
        console.print(f"[dim][hook] {event}: {status}[/dim]")
        if result.output:
            console.print(Panel(Text(result.output, overflow="fold"), title=f"hook {event}", border_style="yellow"))
    return results


def _first_failed_hook(results: list[HookResult]) -> HookResult | None:
    return next((result for result in results if not result.ok), None)


async def _run_repl(settings, client, args, *, resume_latest: bool = False) -> int:
    console = CONSOLE
    _print_repl_header(settings, console=console)
    approval = _interactive_approval(args)
    plan_mode = False
    session = _latest_session(settings) if resume_latest else None
    if session:
        settings = replace(settings, model=session.get("model") or settings.model)
        console.print(f"[dim]resumed session:[/dim] [cyan]{session['id']}[/cyan]")
    else:
        session = _new_session(settings)
        _save_session(settings, session)
    await _run_cli_hooks(
        settings,
        "SessionStart",
        {
            "session_id": session["id"],
            "model": settings.model,
            "project_dir": str(Path(settings.project_dir).resolve()),
            "resumed": bool(resume_latest),
        },
        console=console,
    )
    while True:
        try:
            line = _ask_repl_line(console)
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            await _run_cli_hooks(settings, "SessionEnd", {"session_id": session["id"], "reason": "interrupted"}, console=console)
            return 0
        if not line:
            continue
        if _is_help_command(line):
            _print_command_palette(console=console)
            continue
        filter_text = _slash_filter(line)
        if filter_text:
            _print_command_palette(filter_text, console=console)
            continue
        if _is_quit_command(line):
            await _run_cli_hooks(settings, "SessionEnd", {"session_id": session["id"], "reason": "quit"}, console=console)
            return 0
        if _is_reset_command(line):
            session = _new_session(settings)
            _save_session(settings, session)
            console.print(f"[dim]Started a fresh session:[/dim] [cyan]{session['id']}[/cyan]")
            continue
        if _is_plan_command(line):
            if line == "/plan on":
                plan_mode = True
            elif line == "/plan off":
                plan_mode = False
            else:
                plan_mode = not plan_mode
            state = "on" if plan_mode else "off"
            console.print(Panel(f"Plan mode is now [bold]{state}[/bold].", title="plan", border_style="yellow"))
            continue
        if line == "/status":
            _print_status(settings, args, plan_mode=plan_mode, session_id=session["id"], console=console)
            continue
        if line == "/sessions":
            _print_sessions(settings, console=console)
            continue
        if line == "/current":
            _print_current_session(session, console=console)
            continue
        resume_id = _resume_id_from_command(line)
        if resume_id:
            found = _find_session(settings, resume_id)
            if found:
                session = found
                settings = replace(settings, model=session.get("model") or settings.model)
                console.print(f"[dim]resumed session:[/dim] [cyan]{session['id']}[/cyan]")
            else:
                console.print(Panel(f"Session not found: {resume_id}", title="resume failed", border_style="red"))
            continue
        if line == "/memory":
            _print_memory(settings, console=console)
            continue
        if line == "/hooks":
            _print_hooks(settings, console=console)
            continue
        if line == "/checkpoints":
            _print_checkpoints(settings, console=console)
            continue
        if line == "/changes":
            await _print_changes(settings, console=console)
            continue
        if line == "/checkpoint" or line.startswith("/checkpoint "):
            checkpoint = _create_checkpoint(settings, _checkpoint_name_from_command(line))
            console.print(
                Panel(
                    f"Saved [bold]{len(checkpoint['files'])}[/bold] files.",
                    title=f"checkpoint {checkpoint['id']}",
                    border_style="green",
                )
            )
            continue
        restore_id = _restore_id_from_command(line)
        if restore_id:
            ok, message = _restore_checkpoint(settings, restore_id)
            border = "green" if ok else "red"
            title = "restore" if ok else "restore failed"
            console.print(Panel(message, title=title, border_style=border))
            continue
        if line == "/tools":
            _print_tools_table(console=console)
            continue
        if line == "/permissions":
            _print_permissions(args, console=console)
            continue
        if line == "/diff":
            await _print_diff(settings, console=console)
            continue
        if _is_model_picker_command(line):
            selected = await _select_model(client, settings, console=console)
            if selected is not None:
                settings = selected
            continue
        new_model = _model_name_from_command(line)
        if new_model:
            settings = replace(settings, model=new_model)
            with suppress(OllamaError):
                await client.show_model(settings.model)
            console.print(f"[dim]model ->[/dim] [cyan]{new_model}[/cyan]")
            continue
        title = "plan objective" if plan_mode else "objective"
        border = "yellow" if plan_mode else "green"
        console.print(Panel(Text(line, overflow="fold"), title=title, border_style=border))
        prompt_hooks = await _run_cli_hooks(
            settings,
            "UserPromptSubmit",
            {
                "session_id": session["id"],
                "prompt": line,
                "plan_mode": plan_mode,
                "model": settings.model,
            },
            console=console,
        )
        failed_hook = _first_failed_hook(prompt_hooks)
        if failed_hook:
            console.print(
                Panel(
                    f"Prompt blocked by UserPromptSubmit hook with exit code {failed_hook.returncode}.",
                    title="prompt blocked",
                    border_style="red",
                )
            )
            continue
        approval_for_run = _plan_mode_approval(approval) if plan_mode else approval
        objective = _plan_mode_objective(line) if plan_mode else _session_context(session, line)
        agent = CodeClawAgent(
            settings=settings,
            client=client,
            approval=approval_for_run,
            log=_log_stream(console),
        )
        try:
            result = await agent.run(objective)
        except OllamaError as exc:
            console.print(f"[red]error: {exc}[/red]")
            continue
        _append_session_turn(settings, session, line, result, plan_mode=plan_mode)
        _print_final_report(result, console=console)


def _print_final_report(result, console=None) -> None:
    c = console or CONSOLE
    status = "[bold green]done[/bold green]" if result.completed else f"[bold yellow]stopped: {result.reason}[/bold yellow]"
    meta = f"steps: {len(result.steps)}    tokens: {result.total_tokens}"
    border = "green" if result.completed else "yellow"
    c.print()
    c.print(Panel(f"{status}\n[dim]{meta}[/dim]", title="run", border_style=border))
    if result.final_message:
        c.print(Panel(Markdown(result.final_message), title="final report", border_style=border, padding=(1, 2)))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
