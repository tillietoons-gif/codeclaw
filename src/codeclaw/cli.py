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
from .providers import (
    PROVIDER_TEMPLATES,
    add_provider_from_template,
    apply_provider,
    load_providers,
    resolve_active_provider,
    save_active_provider,
)
from .tools import build_default_registry
from .tools.base import ApprovalDecision

logger = logging.getLogger("codeclaw")

CONSOLE = Console()

SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/help", "Show available slash commands."),
    ("/status", "Show current model, project, approval mode, and git state."),
    ("/init", "Create AGENTS.md and .codeclaw/settings.json defaults."),
    ("/config", "Show project configuration defaults."),
    ("/set KEY VALUE", "Set project defaults such as model or host."),
    ("/plan", "Toggle read-only planning mode for future prompts."),
    ("/planner", "Toggle planner mode: convert architecture into an execution plan."),
    ("/executor", "Toggle executor mode: implement the approved task or phase exactly."),
    ("/reviewer", "Toggle reviewer mode: inspect executor changes and report issues without editing code."),
    ("/verifier", "Toggle verifier mode: verify implementation against the original request, approved spec, and approved plan."),
    ("/memory-agent", "Toggle memory agent mode: maintain project memory across tasks."),
    ("/context-agent", "Toggle context agent mode: retrieve only files and snippets needed for the current task."),
    ("/command-agent", "Toggle command agent mode: suggest safe terminal commands for the current phase."),
    ("/final-report-agent", "Toggle final report agent mode: summarize completed work and verification results."),
    ("/fixer", "Toggle fixer mode: fix only issues reported by reviewer or verifier."),
    ("/compact", "Compact the current saved session context."),
    ("/todo", "Show the current session task list."),
    ("/sessions", "List saved sessions for this project."),
    ("/current", "Show the current session details."),
    ("/resume ID", "Resume a saved session."),
    ("/memory", "Show loaded AGENTS.md and MEMORY.md context."),
    ("/hooks", "Show configured project lifecycle hooks."),
    ("/hook-example", "Write example hook templates for this project."),
    ("/checkpoint NAME", "Save a local project snapshot."),
    ("/checkpoints", "List saved local snapshots."),
    ("/restore ID", "Restore a saved local snapshot."),
    ("/changes", "Show git status and diff summary."),
    ("/tools", "List available CodeClaw tools."),
    ("/permissions", "Show which tools require approval in this session."),
    ("/architect", "Toggle architect analysis mode: analyze the repo and define the implementation without writing code."),
    ("/providers", "Show provider picker and switch active provider."),
    ("/provider", "Show provider picker and switch active provider."),
    ("/diff", "Show the current git diff summary."),
    ("/models", "Choose from installed Ollama models."),
    ("/model NAME", "Switch directly to a model."),
    ("/reset", "Clear the current prompt flow."),
    ("/quit", "Exit CodeClaw."),
)

CONFIG_KEYS = {
    "host": "ollama_host",
    "ollama_host": "ollama_host",
    "model": "model",
    "provider": "provider",
    "max_steps": "max_steps",
    "context_tokens": "context_tokens",
    "temperature": "temperature",
    "request_timeout": "request_timeout_s",
    "request_timeout_s": "request_timeout_s",
}

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
    p.add_argument("--architect", action="store_true", help="Run in architect analysis mode: analyze the repo and define the implementation without writing code.")
    p.add_argument("--planner", action="store_true", help="Run in planner mode: convert an architecture specification into a concrete execution plan.")
    p.add_argument("--executor", action="store_true", help="Run in executor mode: implement only the approved task or phase.")
    p.add_argument("--reviewer", action="store_true", help="Run in reviewer mode: inspect executor agent changes without editing code.")
    p.add_argument("--verifier", action="store_true", help="Run in verifier mode: verify implementation against the original request, approved spec, and approved plan.")
    p.add_argument("--fixer", action="store_true", help="Run in fixer mode: fix only issues reported by reviewer or verifier.")
    p.add_argument("--memory-agent", action="store_true", help="Run in memory agent mode: maintain project memory across tasks.")
    p.add_argument("--context-agent", action="store_true", help="Run in context agent mode: retrieve only the files and snippets needed for the current task.")
    p.add_argument("--command-agent", action="store_true", help="Run in command agent mode: suggest safe terminal commands for the current phase.")
    p.add_argument("--final-report-agent", action="store_true", help="Run in final report agent mode: summarize completed work and verification results.")
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

    if args.objective == "install":
        return _install_codeclaw_shim()

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
        "[bold]/help[/bold] commands   [bold]/plan[/bold] plan mode   [bold]/planner[/bold] planner mode   [bold]/executor[/bold] executor mode   [bold]/reviewer[/bold] reviewer mode   [bold]/verifier[/bold] verifier mode   [bold]/memory-agent[/bold] memory agent mode   [bold]/context-agent[/bold] context agent mode   [bold]/fixer[/bold] fixer mode   [bold]/architect[/bold] architect mode   [bold]/models[/bold] choose   [bold]/status[/bold] inspect",
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


def _print_provider_picker(settings, *, console: Console = CONSOLE) -> None:
    providers = load_providers(settings.project_dir, settings=settings)
    table = Table(title="Providers", show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="bold")
    table.add_column("Backend")
    table.add_column("Default model", overflow="fold")
    table.add_column("Active", justify="center")
    active = settings.provider or ""
    for provider_id, provider in sorted(providers.items()):
        is_active = "yes" if provider_id == active else ""
        table.add_row(provider_id, provider.backend, provider.default_model or "", is_active)
    console.print(table)
    console.print(
        Panel(
            "Use /provider <name> to switch, or /provider add <template> to add a new provider.",
            title="providers",
            border_style="cyan",
        )
    )


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


def _print_status(settings, args, *, plan_mode: bool = False, planner_mode: bool = False, executor_mode: bool = False, reviewer_mode: bool = False, verifier_mode: bool = False, fixer_mode: bool = False, memory_mode: bool = False, context_mode: bool = False, command_mode: bool = False, final_report_mode: bool = False, architect_mode: bool = False, session_id: str = "", console: Console = CONSOLE) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    table.add_row("model", f"[cyan]{settings.model}[/cyan]")
    table.add_row("project", f"[cyan]{settings.project_dir}[/cyan]")
    table.add_row("host", f"[cyan]{settings.ollama_host}[/cyan]")
    table.add_row("steps", str(settings.max_steps))
    table.add_row("temperature", str(settings.temperature))
    table.add_row("approval", "auto-approve" if args.auto_approve else "non-interactive deny" if args.non_interactive else "ask")
    if executor_mode:
        table.add_row("mode", "executor")
    elif fixer_mode:
        table.add_row("mode", "fixer")
    elif memory_mode:
        table.add_row("mode", "memory")
    elif context_mode:
        table.add_row("mode", "context")
    elif command_mode:
        table.add_row("mode", "command")
    elif final_report_mode:
        table.add_row("mode", "final-report")
    elif verifier_mode:
        table.add_row("mode", "verifier")
    elif reviewer_mode:
        table.add_row("mode", "reviewer")
    elif planner_mode:
        table.add_row("mode", "planner")
    elif architect_mode:
        table.add_row("mode", "architect")
    else:
        table.add_row("mode", "plan" if plan_mode else "act")
    if session_id:
        table.add_row("session", session_id)
    table.add_row("cwd", os.getcwd())
    console.print(Panel(table, title="Status", border_style="cyan", padding=(1, 2)))


def _project_settings_path(settings) -> Path:
    return Path(settings.project_dir).resolve() / ".codeclaw" / "settings.json"


def _read_project_settings(settings) -> dict:
    path = _project_settings_path(settings)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_project_settings(settings, data: dict) -> Path:
    path = _project_settings_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def _set_project_default(settings, key: str, value: str) -> tuple[bool, str, object | None]:
    field_name = CONFIG_KEYS.get(key.strip().lower())
    if not field_name:
        return False, f"Unknown config key: {key}", None
    current = getattr(settings, field_name)
    try:
        parsed: object = type(current)(value)
    except (TypeError, ValueError):
        return False, f"Invalid value for {field_name}: {value}", None
    data = _read_project_settings(settings)
    defaults = data.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        data["defaults"] = defaults = {}
    defaults[field_name] = parsed
    _write_project_settings(settings, data)
    return True, field_name, parsed


def _print_config(settings, *, console: Console = CONSOLE) -> None:
    data = _read_project_settings(settings)
    defaults = data.get("defaults") if isinstance(data.get("defaults"), dict) else {}
    table = Table(title=f"Project Config ({_project_settings_path(settings)})", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Current")
    table.add_column("Project Default")
    for key in ("ollama_host", "model", "max_steps", "context_tokens", "temperature", "request_timeout_s"):
        table.add_row(key, str(getattr(settings, key)), str(defaults.get(key, "")))
    console.print(table)


def _init_project(settings) -> list[Path]:
    root = Path(settings.project_dir).resolve()
    created: list[Path] = []
    codeclaw_dir = root / ".codeclaw"
    codeclaw_dir.mkdir(exist_ok=True)
    settings_path = codeclaw_dir / "settings.json"
    if not settings_path.exists():
        settings_path.write_text(
            json.dumps(
                {
                    "defaults": {
                        "ollama_host": settings.ollama_host,
                        "model": settings.model,
                        "max_steps": settings.max_steps,
                    },
                    "hooks": {},
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        created.append(settings_path)
    agents_path = root / "AGENTS.md"
    if not agents_path.exists():
        agents_path.write_text(
            "# CodeClaw Project Notes\n\n"
            "- Describe build, test, and lint commands here.\n"
            "- Add project conventions and safety notes here.\n",
            encoding="utf-8",
        )
        created.append(agents_path)
    return created


def _write_hook_examples(settings) -> list[Path]:
    root = Path(settings.project_dir).resolve()
    hook_dir = root / ".codeclaw" / "hooks"
    hook_dir.mkdir(parents=True, exist_ok=True)
    log_hook = hook_dir / "log_event.py"
    if not log_hook.exists():
        log_hook.write_text(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "payload = json.load(sys.stdin)\n"
            "log = Path('.codeclaw/hook-events.log')\n"
            "log.parent.mkdir(exist_ok=True)\n"
            "with log.open('a', encoding='utf-8') as fh:\n"
            "    fh.write(json.dumps(payload, sort_keys=True) + '\\n')\n",
            encoding="utf-8",
        )
    example_settings = root / ".codeclaw" / "settings.example.json"
    example_settings.write_text(
        json.dumps(
            {
                "hooks": {
                    "SessionStart": [{"type": "command", "command": "python .codeclaw/hooks/log_event.py"}],
                    "UserPromptSubmit": [{"type": "command", "command": "python .codeclaw/hooks/log_event.py"}],
                    "PreToolUse": [{"type": "command", "command": "python .codeclaw/hooks/log_event.py"}],
                    "PostToolUse": [{"type": "command", "command": "python .codeclaw/hooks/log_event.py"}],
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return [log_hook, example_settings]


def _install_codeclaw_shim() -> int:
    target_dir = Path.home() / ".local" / "bin"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "codeclaw"
    target.write_text(
        "#!/usr/bin/env sh\n"
        f"exec {sys.executable!s} -m codeclaw.cli \"$@\"\n",
        encoding="utf-8",
    )
    target.chmod(0o755)
    CONSOLE.print(Panel(f"Installed launcher at [cyan]{target}[/cyan]", title="install", border_style="green"))
    if str(target_dir) not in os.environ.get("PATH", "").split(os.pathsep):
        CONSOLE.print(f"[yellow]Add {target_dir} to PATH to run codeclaw from any directory.[/yellow]")
    return 0


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


def _append_session_turn(settings, session: dict, objective: str, result, *, plan_mode: bool, planner_mode: bool = False, executor_mode: bool = False, reviewer_mode: bool = False, verifier_mode: bool = False, fixer_mode: bool = False, memory_mode: bool = False, context_mode: bool = False, command_mode: bool = False, final_report_mode: bool = False, architect_mode: bool = False) -> None:
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
            "planner_mode": planner_mode,
            "executor_mode": executor_mode,
            "reviewer_mode": reviewer_mode,
            "verifier_mode": verifier_mode,
            "fixer_mode": fixer_mode,
            "memory_mode": memory_mode,
            "context_mode": context_mode,
            "command_mode": command_mode,
            "final_report_mode": final_report_mode,
            "architect_mode": architect_mode,
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
    summary = str(session.get("compact_summary") or "").strip()
    if not turns and not summary:
        return objective
    lines = [
        "RESUMED SESSION CONTEXT:",
        "Use the prior session turns below as context. Continue naturally from them, but follow the latest user objective.",
        "",
    ]
    if summary:
        lines.extend(["Compacted session summary:", summary[:4000], ""])
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


def _compact_session(settings, session: dict) -> str:
    turns = session.get("turns") or []
    lines = []
    existing = str(session.get("compact_summary") or "").strip()
    if existing:
        lines.extend([existing, ""])
    for idx, turn in enumerate(turns, 1):
        final = str(turn.get("final_message", "")).strip().replace("\n", " ")
        lines.append(
            f"{idx}. {turn.get('objective', '')} -> {turn.get('reason', '')}; "
            f"{final[:280]}"
        )
    summary = "\n".join(line for line in lines if line).strip()
    session["compact_summary"] = summary[-6000:]
    session["turns"] = turns[-3:]
    _save_session(settings, session)
    return session["compact_summary"]


def _session_todos(session: dict) -> list[tuple[str, str]]:
    turns = session.get("turns") or []
    todos: list[tuple[str, str]] = []
    for idx, turn in enumerate(turns[-12:], 1):
        status = "done" if turn.get("completed") else "open"
        todos.append((status, f"{idx}. {turn.get('objective', '')}"))
    return todos


def _print_todos(session: dict, *, console: Console = CONSOLE) -> None:
    todos = _session_todos(session)
    if not todos:
        console.print(Panel("No session tasks yet.", title="todo", border_style="yellow"))
        return
    table = Table(title="Session Todo", show_header=True, header_style="bold cyan")
    table.add_column("Status", style="bold")
    table.add_column("Task", overflow="fold")
    for status, task in todos:
        table.add_row(status, task)
    console.print(table)


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


def _set_args_from_command(line: str) -> tuple[str, str] | None:
    prefix = "/set "
    if not line.startswith(prefix):
        return None
    parts = line[len(prefix):].strip().split(maxsplit=1)
    if len(parts) != 2:
        return None
    return parts[0], parts[1]


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
    stat_rc, stat = await _git_output(settings.project_dir, "diff", "--stat")
    patch_rc, patch = await _git_output(settings.project_dir, "diff", "--", ".")
    if stat_rc != 0 or patch_rc != 0:
        console.print(Panel((stat or patch).strip() or "git diff failed", title="diff", border_style="red"))
        return
    if not stat.strip() and not patch.strip():
        console.print(Panel("No working-tree diff.", title="diff", border_style="green"))
        return
    console.print(Panel(stat.rstrip(), title="diff --stat", border_style="yellow"))
    lines = patch.splitlines()
    shown = "\n".join(lines[:220])
    if len(lines) > 220:
        shown += f"\n\n... truncated {len(lines) - 220} diff lines ..."
    console.print(Panel(Text(shown, overflow="fold"), title="diff preview", border_style="cyan"))


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
    always_tools: set[str] = set()

    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if auto:
            return ApprovalDecision(ApprovalDecision.APPROVE, reason="--auto-approve")
        if non_interactive:
            return ApprovalDecision(
                ApprovalDecision.REJECT,
                reason="destructive action in non-interactive mode without --auto-approve",
            )
        if tool_name in always_tools:
            return ApprovalDecision(ApprovalDecision.APPROVE_ALWAYS, reason="approved always")
        key = f"{tool_name}:{summary}"
        if key in cache and cache[key]:
            return ApprovalDecision(ApprovalDecision.APPROVE, reason="cached")
        from rich.prompt import Prompt

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
            choice = Prompt.ask(
                "Allow action",
                choices=["y", "a", "n"],
                default="n",
                console=CONSOLE,
                show_choices=True,
                show_default=True,
            )
        except (EOFError, KeyboardInterrupt):
            choice = "n"
        if choice == "a":
            always_tools.add(tool_name)
            return ApprovalDecision(ApprovalDecision.APPROVE_ALWAYS)
        if choice == "y":
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


def _planner_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="planner mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _planner_mode_objective(objective: str) -> str:
    return (
        "PLANNER MODE: Convert the architect specification into a detailed execution plan. "
        "Break work into phases, tasks, dependencies, files to modify, and verification steps. "
        "Do not write code. Focus on a sequence of concrete implementation actions. "
        "For each task, include the exact file names, intended content goals, and how success will be verified.\n\n"
        f"User objective: {objective}"
    )


def _architect_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="architect mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _executor_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _reviewer_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="reviewer mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _reviewer_mode_objective(objective: str) -> str:
    return (
        "REVIEWER MODE: Review the executor agent's proposed or implemented changes. "
        "Identify bugs, missing requirements, architecture flaws, security issues, style problems, and risks. "
        "Do not edit files, write code, run shell commands, or commit changes. "
        "Use read-only inspection tools and explain any issues clearly.\n\n"
        f"User objective: {objective}"
    )


def _verifier_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="verifier mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _verifier_mode_objective(objective: str) -> str:
    return (
        "VERIFIER MODE: Verify the implementation against the original user request, approved spec, and approved plan. "
        "Confirm the requested behavior is satisfied and highlight any mismatches or missing requirements. "
        "Do not edit files, write code, run shell commands, or commit changes. "
        "Use read-only inspection tools and explain any discrepancies clearly.\n\n"
        f"User objective: {objective}"
    )


def _fixer_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="fixer mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _fixer_mode_objective(objective: str) -> str:
    return (
        "FIXER MODE: Fix only issues reported by the Reviewer Agent or Verifier Agent. "
        "Do not add new features, refactor unrelated code, or deviate from the approved plan. "
        "Use read-only inspection tools to identify the precise issue and apply minimal safe corrections.\n\n"
        f"User objective: {objective}"
    )


def _memory_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="memory agent mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _memory_mode_objective(objective: str) -> str:
    return (
        "MEMORY AGENT MODE: Maintain and update project memory across tasks. "
        "Store repository structure, frameworks, conventions, important files, architecture decisions, and known problems. "
        "Do not add new features or refactor unrelated code. "
        "Use read-only inspection tools and summarize key context for future tasks.\n\n"
        f"User objective: {objective}"
    )


def _context_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="context agent mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _context_mode_objective(objective: str) -> str:
    return (
        "CONTEXT AGENT MODE: Retrieve only the files and snippets needed for the current task. "
        "Do not edit files, run shell commands, commit changes, or add new features. "
        "Focus on loading and summarizing relevant project context, file contents, and code references.\n\n"
        f"User objective: {objective}"
    )


def _command_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="command agent mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _command_mode_objective(objective: str) -> str:
    return (
        "COMMAND AGENT MODE: Suggest safe terminal commands for the current phase. "
        "Prefer read-only commands first, avoid destructive operations, and explain why each command is needed. "
        "Do not edit files, run shell commands, or commit changes automatically.\n\n"
        f"User objective: {objective}"
    )


def _final_report_mode_approval(base_approval) -> Callable[[str, str], Awaitable[ApprovalDecision]]:
    async def approve(tool_name: str, summary: str) -> ApprovalDecision:
        if tool_name in DESTRUCTIVE_TOOLS:
            return ApprovalDecision(ApprovalDecision.REJECT, reason="final report agent mode is read-only")
        result = base_approval(tool_name, summary)
        if asyncio.iscoroutine(result):
            return await result
        return result

    return approve


def _final_report_mode_objective(objective: str) -> str:
    return (
        "FINAL REPORT MODE: Summarize completed work, verification results, changed files, and remaining issues. "
        "Do not edit files, run shell commands, or propose new implementation changes. "
        "Focus on producing a clear final summary for the user.\n\n"
        f"User objective: {objective}"
    )


def _executor_mode_objective(objective: str) -> str:
    return (
        "EXECUTOR MODE: Implement only the approved task or phase exactly. "
        "Use the current repository state and the approved plan. "
        "Do not propose new plans or change the objective. "
        "Execute the approved task with minimal safe changes. "
        "If the plan requires creating or updating a landing page, write complete HTML/CSS and polished SEO copy in the target files.\n\n"
        f"User objective: {objective}"
    )


def _architect_mode_objective(objective: str) -> str:
    return (
        "ARCHITECT MODE: Analyze the repository and design an implementation strategy. "
        "Do not edit files, run shell commands, commit changes, or perform other side effects. "
        "Provide architecture, specification, and implementation steps without producing working code.\n\n"
        f"User objective: {objective}"
    )


async def _ask_repl_line(console: Console) -> str:
    console.print(
        Panel(
            "[dim]Type a prompt, or use [bold]/help[/bold], [bold]/plan[/bold], "
            "[bold]/models[/bold], [bold]/status[/bold], [bold]/quit[/bold].[/dim]",
            title="prompt",
            border_style="green",
            padding=(0, 1),
        )
    )
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import InMemoryHistory

        session = getattr(_ask_repl_line, "_session", None)
        if session is None:
            session = PromptSession(history=InMemoryHistory(), multiline=False)
            _ask_repl_line._session = session
        try:
            asyncio.get_running_loop()
            return (await session.prompt_async("› ", multiline=False)).strip()
        except RuntimeError:
            return session.prompt("› ", multiline=False).strip()
    except ImportError:
        from rich.prompt import Prompt

        return Prompt.ask("[bold green]›[/bold green]", console=console).strip()


def _is_quit_command(line: str) -> bool:
    return line in ("/q", "/quit", "/exit")


def _is_reset_command(line: str) -> bool:
    return line == "/reset"


def _is_model_picker_command(line: str) -> bool:
    return line in ("/model", "/models")


def _is_provider_picker_command(line: str) -> bool:
    return line in ("/providers", "/provider")


def _provider_command_args(line: str) -> tuple[str, str] | None:
    if not line.startswith("/provider "):
        return None
    parts = line.strip().split(maxsplit=2)
    if len(parts) == 3:
        return parts[1], parts[2]
    if len(parts) == 2:
        return "switch", parts[1]
    return None


def _is_help_command(line: str) -> bool:
    return line in ("/", "/help", "/?")


def _is_planner_command(line: str) -> bool:
    return line in ("/planner on", "/planner off")


def _is_executor_command(line: str) -> bool:
    return line in ("/executor on", "/executor off")


def _is_reviewer_command(line: str) -> bool:
    return line in ("/reviewer on", "/reviewer off")


def _is_verifier_command(line: str) -> bool:
    return line in ("/verifier on", "/verifier off")


def _is_fixer_command(line: str) -> bool:
    return line in ("/fixer on", "/fixer off")


def _is_memory_agent_command(line: str) -> bool:
    return line in ("/memory-agent on", "/memory-agent off")


def _is_context_agent_command(line: str) -> bool:
    return line in ("/context-agent on", "/context-agent off")


def _is_command_agent_command(line: str) -> bool:
    return line in ("/command-agent on", "/command-agent off")


def _is_final_report_agent_command(line: str) -> bool:
    return line in ("/final-report-agent on", "/final-report-agent off")


def _is_architect_command(line: str) -> bool:
    return line in ("/architect on", "/architect off")


def _is_plan_command(line: str) -> bool:
    return line in ("/plan", "/plan on", "/plan off")


def _slash_filter(line: str) -> str | None:
    if not line.startswith("/") or " " in line:
        return None
    known = {
        "/q", "/quit", "/exit", "/reset", "/model", "/models",
        "/help", "/?", "/", "/status", "/tools", "/permissions", "/diff",
        "/init", "/config", "/compact", "/todo", "/plan", "/planner", "/executor", "/reviewer", "/verifier", "/memory-agent", "/fixer", "/architect", "/sessions", "/current", "/memory",
        "/hooks", "/hook-example", "/checkpoint", "/checkpoints", "/changes",
        "/command-agent",
        "/final-report-agent",
    }
    if line.startswith(("/restore ", "/checkpoint ", "/resume ", "/set ")):
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
    if _has_explicit_mode(args):
        return await _run_one_shot_direct(settings, client, args, objective)
    return await _run_orchestrated_flow(settings, client, args, objective)


async def _run_one_shot_direct(settings, client, args, objective: str) -> int:
    with suppress(OllamaError):
        await client.show_model(settings.model)
    _print_session_header(settings, objective)
    approval = _interactive_approval(args)
    if args.planner:
        approval = _planner_mode_approval(approval)
        objective = _planner_mode_objective(objective)
    elif args.executor:
        approval = _executor_mode_approval(approval)
        objective = _executor_mode_objective(objective)
    elif args.reviewer:
        approval = _reviewer_mode_approval(approval)
        objective = _reviewer_mode_objective(objective)
    elif args.verifier:
        approval = _verifier_mode_approval(approval)
        objective = _verifier_mode_objective(objective)
    elif args.fixer:
        approval = _fixer_mode_approval(approval)
        objective = _fixer_mode_objective(objective)
    elif args.context_agent:
        approval = _context_mode_approval(approval)
        objective = _context_mode_objective(objective)
    elif args.command_agent:
        approval = _command_mode_approval(approval)
        objective = _command_mode_objective(objective)
    elif args.final_report_agent:
        approval = _final_report_mode_approval(approval)
        objective = _final_report_mode_objective(objective)
    elif args.memory_agent:
        approval = _memory_mode_approval(approval)
        objective = _memory_mode_objective(objective)
    elif args.architect:
        approval = _architect_mode_approval(approval)
        objective = _architect_mode_objective(objective)
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


def _has_explicit_mode(args) -> bool:
    return any(
        bool(getattr(args, attr, False))
        for attr in (
            "architect",
            "planner",
            "executor",
            "reviewer",
            "verifier",
            "fixer",
            "memory_agent",
            "context_agent",
            "command_agent",
            "final_report_agent",
        )
    )


def _determine_task_type(objective: str) -> str:
    lower = objective.lower()
    dangerous = ["delete", "remove", "destroy", "rm -rf", "shutdown", "wipe", "format disk", "drop table", "kill ", "destroy", "uninstall"]
    if any(keyword in lower for keyword in dangerous):
        return "dangerous"
    complex_phrases = ["landing page", "website", "web page"]
    complex_keywords = ["design", "architecture", "specification", "plan", "refactor", "optimize", "migration", "migrate", "performance", "integrate", "security", "secure", "seo", "create", "build", "copy", "content"]
    if any(phrase in lower for phrase in complex_phrases) or any(keyword in lower for keyword in complex_keywords):
        return "complex"
    if len(objective.split()) < 12 and any(keyword in lower for keyword in ("add", "fix", "update", "implement", "correct", "patch")):
        return "simple"
    if len(objective) < 80:
        return "simple"
    return "complex"


def _new_task_state(objective: str, args) -> dict:
    return {
        "user_request": objective,
        "mode": "yolo" if bool(args.auto_approve) else "normal",
        "repo_context": {},
        "spec": {},
        "plan": {},
        "current_phase": "",
        "completed_phases": [],
        "changed_files": [],
        "commands_run": [],
        "review_results": [],
        "verification_results": [],
        "fix_attempts": 0,
        "final_status": "pending",
    }


def _normalize_list(items: list[str]) -> list[str]:
    seen = set()
    normalized = []
    for item in items:
        if not item:
            continue
        if item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def _collect_run_metrics(result) -> tuple[list[str], list[str]]:
    changed_files: list[str] = []
    commands_run: list[str] = []
    for step in getattr(result, "steps", []):
        for tc in getattr(step, "tool_calls", []) or []:
            if tc.name in ("write_file", "edit_file"):
                path = tc.arguments.get("path")
                if isinstance(path, str):
                    changed_files.append(path)
            elif tc.name == "git_commit":
                changed_files.append("git_commit")
            elif tc.name == "exec":
                command = tc.arguments.get("command")
                if isinstance(command, str):
                    commands_run.append(command)
    return _normalize_list(changed_files), _normalize_list(commands_run)


def _verification_passed(result) -> bool:
    if not result.completed:
        return False
    text = (result.final_message or "").lower()
    negative_markers = ["fail", "failed", "error", "mismatch", "issue", "problem", "bug", "incorrect", "incomplete", "missing"]
    if any(marker in text for marker in negative_markers):
        if "no issue" in text or "no problems" in text or "all good" in text or "verified successfully" in text:
            return True
        return False
    return True


async def _ask_user_approval(args, plan_summary: str) -> bool:
    if args.auto_approve:
        return True
    if args.non_interactive:
        return False
    from rich.prompt import Prompt

    CONSOLE.print(Panel(Text(plan_summary, overflow="fold"), title="proposed plan", border_style="yellow", padding=(1, 2)))
    choice = Prompt.ask(
        "Approve this plan and continue with execution?",
        choices=["y", "n"],
        default="n",
        console=CONSOLE,
        show_choices=True,
        show_default=True,
    )
    return choice == "y"


async def _run_named_agent(settings, client, args, objective: str, mode_name: str) -> object:
    approval = _interactive_approval(args)
    if mode_name == "planner":
        approval = _planner_mode_approval(approval)
        objective = _planner_mode_objective(objective)
    elif mode_name == "architect":
        approval = _architect_mode_approval(approval)
        objective = _architect_mode_objective(objective)
    elif mode_name == "executor":
        approval = _executor_mode_approval(approval)
        objective = _executor_mode_objective(objective)
    elif mode_name == "reviewer":
        approval = _reviewer_mode_approval(approval)
        objective = _reviewer_mode_objective(objective)
    elif mode_name == "verifier":
        approval = _verifier_mode_approval(approval)
        objective = _verifier_mode_objective(objective)
    elif mode_name == "fixer":
        approval = _fixer_mode_approval(approval)
        objective = _fixer_mode_objective(objective)
    elif mode_name == "context_agent":
        approval = _context_mode_approval(approval)
        objective = _context_mode_objective(objective)
    elif mode_name == "command_agent":
        approval = _command_mode_approval(approval)
        objective = _command_mode_objective(objective)
    elif mode_name == "final_report_agent":
        approval = _final_report_mode_approval(approval)
        objective = _final_report_mode_objective(objective)
    elif mode_name == "memory_agent":
        approval = _memory_mode_approval(approval)
        objective = _memory_mode_objective(objective)
    else:
        approval = _interactive_approval(args)
    agent = CodeClawAgent(
        settings=settings,
        client=client,
        approval=approval,
        log=_log_stream(CONSOLE),
    )
    return await agent.run(objective)


async def _run_orchestrated_flow(settings, client, args, objective: str) -> int:
    with suppress(OllamaError):
        await client.show_model(settings.model)
    _print_session_header(settings, objective)
    state = _new_task_state(objective, args)
    task_type = _determine_task_type(objective)

    if task_type == "dangerous":
        prompt = (
            "This request appears to be dangerous. Confirm before any execution. "
            "If you do not want to continue, answer no."
        )
        if not await _ask_user_approval(args, prompt):
            state["final_status"] = "failed"
            _print_orchestrator_summary(state)
            return 2

    state["current_phase"] = "context"
    context_input = (
        "Use repository inspection tools to gather only the files and snippets needed for the current task. "
        f"User objective: {objective}"
    )
    context_result = await _run_named_agent(settings, client, args, context_input, "context_agent")
    changed_files, commands_run = _collect_run_metrics(context_result)
    state["changed_files"].extend(changed_files)
    state["commands_run"].extend(commands_run)
    state["completed_phases"].append("context")
    state["repo_context"] = {"summary": context_result.final_message}

    if task_type == "simple":
        state["current_phase"] = "executor"
        executor_input = (
            "Implement the requested change using the approved task and the current repository state. "
            f"User objective: {objective}\n\nContext:\n{context_result.final_message}"
        )
        executor_result = await _run_named_agent(settings, client, args, executor_input, "executor")
        changed_files, commands_run = _collect_run_metrics(executor_result)
        state["changed_files"].extend(changed_files)
        state["commands_run"].extend(commands_run)
        state["completed_phases"].append("executor")

        state["current_phase"] = "reviewer"
        reviewer_input = (
            "Review the executor agent's implementation against the user request and the gathered context. "
            f"User objective: {objective}\n\nExecutor output:\n{executor_result.final_message}"
        )
        reviewer_result = await _run_named_agent(settings, client, args, reviewer_input, "reviewer")
        state["review_results"].append(reviewer_result.final_message)
        state["completed_phases"].append("reviewer")

        state["current_phase"] = "verifier"
        verifier_input = (
            "Verify the implementation against the original user request and the gathered context. "
            f"User objective: {objective}\n\nExecutor output:\n{executor_result.final_message}"
        )
        verifier_result = await _run_named_agent(settings, client, args, verifier_input, "verifier")
        state["verification_results"].append({
            "phase": "initial",
            "completed": verifier_result.completed,
            "summary": verifier_result.final_message,
        })
        state["completed_phases"].append("verifier")

        if not _verification_passed(verifier_result):
            state["current_phase"] = "fixer"
            fixer_input = (
                "Fix the issues identified by the verifier. "
                f"User objective: {objective}\n\nVerifier report:\n{verifier_result.final_message}"
            )
            fixer_result = await _run_named_agent(settings, client, args, fixer_input, "fixer")
            state["fix_attempts"] += 1
            changed_files, commands_run = _collect_run_metrics(fixer_result)
            state["changed_files"].extend(changed_files)
            state["commands_run"].extend(commands_run)
            state["completed_phases"].append("fixer")

            state["current_phase"] = "verifier"
            verifier_result = await _run_named_agent(settings, client, args, verifier_input, "verifier")
            state["verification_results"].append({
                "phase": "fixer_retry",
                "completed": verifier_result.completed,
                "summary": verifier_result.final_message,
            })

        state["current_phase"] = "final_report"
        final_report_input = (
            "Summarize what was requested, what was implemented, files changed, commands run, tests or verification results, and any remaining issues. "
            f"User objective: {objective}\n\nFinal state: {json.dumps(state, indent=2)[:4000]}"
        )
        final_report_result = await _run_named_agent(settings, client, args, final_report_input, "final_report_agent")
        state["final_status"] = "success" if _verification_passed(verifier_result) else "failed"
        state["current_phase"] = "completed"
        _print_orchestrator_summary(state)
        _print_final_report(final_report_result)
        return 0 if state["final_status"] == "success" else 2

    state["current_phase"] = "architect"
    architect_input = (
        "Analyze the repository and design a detailed architecture specification for the requested work. "
        f"User objective: {objective}\n\nContext:\n{context_result.final_message}"
    )
    architect_result = await _run_named_agent(settings, client, args, architect_input, "architect")
    state["spec"] = {"summary": architect_result.final_message}
    changed_files, commands_run = _collect_run_metrics(architect_result)
    state["changed_files"].extend(changed_files)
    state["commands_run"].extend(commands_run)
    state["completed_phases"].append("architect")

    state["current_phase"] = "planner"
    planner_input = (
        "Convert the approved architecture specification into a concrete execution plan with phases, tasks, and verification checks. "
        f"User objective: {objective}\n\nSpecification:\n{architect_result.final_message}"
    )
    planner_result = await _run_named_agent(settings, client, args, planner_input, "planner")
    state["plan"] = {"summary": planner_result.final_message}
    changed_files, commands_run = _collect_run_metrics(planner_result)
    state["changed_files"].extend(changed_files)
    state["commands_run"].extend(commands_run)
    state["completed_phases"].append("planner")

    if not await _ask_user_approval(args, planner_result.final_message):
        state["final_status"] = "failed"
        _print_orchestrator_summary(state)
        return 2

    state["current_phase"] = "executor"
    executor_input = (
        "Implement the approved plan exactly, making minimal safe changes and avoiding unrelated work. "
        f"User objective: {objective}\n\nPlan:\n{planner_result.final_message}\n\nSpecification:\n{architect_result.final_message}"
    )
    executor_result = await _run_named_agent(settings, client, args, executor_input, "executor")
    changed_files, commands_run = _collect_run_metrics(executor_result)
    state["changed_files"].extend(changed_files)
    state["commands_run"].extend(commands_run)
    state["completed_phases"].append("executor")

    state["current_phase"] = "reviewer"
    reviewer_input = (
        "Review the executor agent's implementation against the user request, approved spec, and approved plan. "
        f"User objective: {objective}\n\nExecutor output:\n{executor_result.final_message}"
    )
    reviewer_result = await _run_named_agent(settings, client, args, reviewer_input, "reviewer")
    state["review_results"].append(reviewer_result.final_message)
    state["completed_phases"].append("reviewer")

    state["current_phase"] = "verifier"
    verifier_input = (
        "Verify the implementation against the original user request, approved specification, and approved plan. "
        f"User objective: {objective}\n\nExecutor output:\n{executor_result.final_message}"
    )
    verifier_result = await _run_named_agent(settings, client, args, verifier_input, "verifier")
    state["verification_results"].append({
        "phase": "initial",
        "completed": verifier_result.completed,
        "summary": verifier_result.final_message,
    })
    state["completed_phases"].append("verifier")

    if not _verification_passed(verifier_result):
        while state["fix_attempts"] < 3 and not _verification_passed(verifier_result):
            state["current_phase"] = "fixer"
            fixer_input = (
                "Fix only the issues reported by the verifier. Do not change unrelated code. "
                f"User objective: {objective}\n\nVerifier report:\n{verifier_result.final_message}"
            )
            fixer_result = await _run_named_agent(settings, client, args, fixer_input, "fixer")
            state["fix_attempts"] += 1
            changed_files, commands_run = _collect_run_metrics(fixer_result)
            state["changed_files"].extend(changed_files)
            state["commands_run"].extend(commands_run)
            state["completed_phases"].append(f"fixer-{state['fix_attempts']}")

            state["current_phase"] = "verifier"
            verifier_result = await _run_named_agent(settings, client, args, verifier_input, "verifier")
            state["verification_results"].append({
                "phase": f"fixer_retry_{state['fix_attempts']}",
                "completed": verifier_result.completed,
                "summary": verifier_result.final_message,
            })
            if not _verification_passed(verifier_result) and state["fix_attempts"] >= 3:
                break

    state["current_phase"] = "final_report"
    final_report_input = (
        "Summarize what was requested, what was planned, what was implemented, "
        "which files and commands changed, verification results, and any remaining issues. "
        f"User objective: {objective}\n\nTask state:\n{json.dumps(state, indent=2)[:4000]}"
    )
    final_report_result = await _run_named_agent(settings, client, args, final_report_input, "final_report_agent")
    state["final_status"] = "success" if _verification_passed(verifier_result) else "failed"
    state["current_phase"] = "completed"
    _print_orchestrator_summary(state)
    _print_final_report(final_report_result)
    return 0 if state["final_status"] == "success" else 2


def _print_orchestrator_summary(state: dict) -> None:
    lines = [
        "Final orchestration summary:",
        f"What was requested: {state['user_request']}",
        f"Task type: {state.get('task_type', 'unknown')}",
        f"Final status: {state['final_status']}",
        "",
        "Planned phases:",
        *[f"- {phase}" for phase in state['completed_phases']],
        "",
        "Files changed:",
        *[f"- {path}" for path in _normalize_list(state['changed_files'])],
        "",
        "Commands run:",
        *[f"- {cmd}" for cmd in _normalize_list(state['commands_run'])],
        "",
        "Verification results:",
        *[f"- {item['phase']}: {'passed' if item['completed'] else 'failed'}" for item in state['verification_results']],
        "",
        "Remaining issues:",
        *[f"- {item['summary']}" for item in state['verification_results'] if not item['completed']],
    ]
    panel = Panel(Text("\n".join(lines), overflow="fold"), title="orchestrator summary", border_style="cyan", padding=(1, 2))
    CONSOLE.print(panel)


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
    planner_mode = bool(args.planner)
    executor_mode = bool(args.executor)
    reviewer_mode = bool(args.reviewer)
    verifier_mode = bool(args.verifier)
    fixer_mode = bool(args.fixer)
    memory_mode = bool(args.memory_agent)
    context_mode = bool(args.context_agent)
    command_mode = bool(args.command_agent)
    final_report_mode = bool(args.final_report_agent)
    architect_mode = bool(args.architect)
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
            line = await _ask_repl_line(console)
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
        if _is_planner_command(line):
            if line == "/planner on":
                planner_mode = True
            elif line == "/planner off":
                planner_mode = False
            else:
                planner_mode = not planner_mode
            state = "on" if planner_mode else "off"
            console.print(Panel(f"Planner mode is now [bold]{state}[/bold].", title="planner", border_style="blue"))
            continue
        if _is_executor_command(line):
            if line == "/executor on":
                executor_mode = True
            elif line == "/executor off":
                executor_mode = False
            else:
                executor_mode = not executor_mode
            state = "on" if executor_mode else "off"
            console.print(Panel(f"Executor mode is now [bold]{state}[/bold].", title="executor", border_style="red"))
            continue
        if _is_reviewer_command(line):
            if line == "/reviewer on":
                reviewer_mode = True
            elif line == "/reviewer off":
                reviewer_mode = False
            else:
                reviewer_mode = not reviewer_mode
            state = "on" if reviewer_mode else "off"
            console.print(Panel(f"Reviewer mode is now [bold]{state}[/bold].", title="reviewer", border_style="magenta"))
            continue
        if _is_verifier_command(line):
            if line == "/verifier on":
                verifier_mode = True
            elif line == "/verifier off":
                verifier_mode = False
            else:
                verifier_mode = not verifier_mode
            state = "on" if verifier_mode else "off"
            console.print(Panel(f"Verifier mode is now [bold]{state}[/bold].", title="verifier", border_style="purple"))
            continue
        if _is_memory_agent_command(line):
            if line == "/memory-agent on":
                memory_mode = True
            elif line == "/memory-agent off":
                memory_mode = False
            else:
                memory_mode = not memory_mode
            state = "on" if memory_mode else "off"
            console.print(Panel(f"Memory agent mode is now [bold]{state}[/bold].", title="memory agent", border_style="cyan"))
            continue
        if _is_context_agent_command(line):
            if line == "/context-agent on":
                context_mode = True
            elif line == "/context-agent off":
                context_mode = False
            else:
                context_mode = not context_mode
            state = "on" if context_mode else "off"
            console.print(Panel(f"Context agent mode is now [bold]{state}[/bold].", title="context agent", border_style="blue"))
            continue
        if _is_command_agent_command(line):
            if line == "/command-agent on":
                command_mode = True
            elif line == "/command-agent off":
                command_mode = False
            else:
                command_mode = not command_mode
            state = "on" if command_mode else "off"
            console.print(Panel(f"Command agent mode is now [bold]{state}[/bold].", title="command agent", border_style="green"))
            continue
        if _is_final_report_agent_command(line):
            if line == "/final-report-agent on":
                final_report_mode = True
            elif line == "/final-report-agent off":
                final_report_mode = False
            else:
                final_report_mode = not final_report_mode
            state = "on" if final_report_mode else "off"
            console.print(Panel(f"Final report agent mode is now [bold]{state}[/bold].", title="final report agent", border_style="green"))
            continue
        if _is_fixer_command(line):
            if line == "/fixer on":
                fixer_mode = True
            elif line == "/fixer off":
                fixer_mode = False
            else:
                fixer_mode = not fixer_mode
            state = "on" if fixer_mode else "off"
            console.print(Panel(f"Fixer mode is now [bold]{state}[/bold].", title="fixer", border_style="blue"))
            continue
        if _is_architect_command(line):
            if line == "/architect on":
                architect_mode = True
            elif line == "/architect off":
                architect_mode = False
            else:
                architect_mode = not architect_mode
            state = "on" if architect_mode else "off"
            console.print(Panel(f"Architect mode is now [bold]{state}[/bold].", title="architect", border_style="yellow"))
            continue
        if line == "/init":
            created = _init_project(settings)
            body = "\n".join(str(path) for path in created) if created else "Project already has CodeClaw files."
            console.print(Panel(body, title="init", border_style="green"))
            continue
        if line == "/config":
            _print_config(settings, console=console)
            continue
        set_args = _set_args_from_command(line)
        if set_args:
            ok, key, value = _set_project_default(settings, set_args[0], set_args[1])
            if ok:
                settings = replace(settings, **{key: value})
                console.print(Panel(f"{key} = {value}", title="config saved", border_style="green"))
            else:
                console.print(Panel(key, title="config failed", border_style="red"))
            continue
        if line == "/compact":
            summary = _compact_session(settings, session)
            console.print(Panel(summary or "Nothing to compact yet.", title="compact", border_style="green"))
            continue
        if line == "/todo":
            _print_todos(session, console=console)
            continue
        if line == "/status":
            _print_status(
                settings,
                args,
                plan_mode=plan_mode,
                planner_mode=planner_mode,
                executor_mode=executor_mode,
                reviewer_mode=reviewer_mode,
                verifier_mode=verifier_mode,
                fixer_mode=fixer_mode,
                memory_mode=memory_mode,
                context_mode=context_mode,
                command_mode=command_mode,
                final_report_mode=final_report_mode,
                architect_mode=architect_mode,
                session_id=session["id"],
                console=console,
            )
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
        if line == "/hook-example":
            files = _write_hook_examples(settings)
            console.print(Panel("\n".join(str(path) for path in files), title="hook examples", border_style="green"))
            continue
        if line == "/checkpoints":
            _print_checkpoints(settings, console=console)
            continue
        if line == "/changes":
            await _print_changes(settings, console=console)
            continue
        if line == "/planner":
            _print_planner_help(console=console)
            continue
        if line == "/executor":
            _print_executor_help(console=console)
            continue
        if line == "/reviewer":
            _print_reviewer_help(console=console)
            continue
        if line == "/verifier":
            _print_verifier_help(console=console)
            continue
        if line == "/memory-agent":
            _print_memory_agent_help(console=console)
            continue
        if line == "/context-agent":
            _print_context_agent_help(console=console)
            continue
        if line == "/command-agent":
            _print_command_agent_help(console=console)
            continue
        if line == "/final-report-agent":
            _print_final_report_agent_help(console=console)
            continue
        if line == "/fixer":
            _print_fixer_help(console=console)
            continue
        if line == "/architect":
            _print_architect_help(console=console)
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
        if line == "/providers":
            _print_provider_picker(settings, console=console)
            continue
        if line == "/provider":
            _print_provider_picker(settings, console=console)
            continue
        provider_args = _provider_command_args(line)
        if provider_args is not None:
            action, provider_id = provider_args
            if action == "add":
                ok, message, provider = add_provider_from_template(settings, provider_id)
                if ok and provider is not None:
                    console.print(Panel(message, title="provider added", border_style="green"))
                else:
                    console.print(Panel(message, title="provider add failed", border_style="red"))
                continue
            if action == "switch":
                providers = load_providers(settings.project_dir, settings=settings)
                if provider_id not in providers:
                    console.print(Panel(f"Provider not found: {provider_id}", title="provider switch failed", border_style="red"))
                    continue
                save_active_provider(settings, provider_id)
                settings = resolve_active_provider(settings)
                console.print(Panel(f"Switched active provider to {provider_id}.", title="provider switched", border_style="green"))
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
        if executor_mode:
            title = "executor objective"
            border = "red"
        elif fixer_mode:
            title = "fixer objective"
            border = "blue"
        elif context_mode:
            title = "context objective"
            border = "blue"
        elif command_mode:
            title = "command objective"
            border = "green"
        elif memory_mode:
            title = "memory objective"
            border = "cyan"
        elif reviewer_mode:
            title = "reviewer objective"
            border = "magenta"
        elif verifier_mode:
            title = "verifier objective"
            border = "purple"
        elif architect_mode:
            title = "architect objective"
            border = "magenta"
        elif plan_mode:
            title = "plan objective"
            border = "yellow"
        else:
            title = "objective"
            border = "green"
        console.print(Panel(Text(line, overflow="fold"), title=title, border_style=border))
        prompt_hooks = await _run_cli_hooks(
            settings,
            "UserPromptSubmit",
            {
                "session_id": session["id"],
                "prompt": line,
                "plan_mode": plan_mode,
                "planner_mode": planner_mode,
                "executor_mode": executor_mode,
                "reviewer_mode": reviewer_mode,
                "verifier_mode": verifier_mode,
                "fixer_mode": fixer_mode,
                "memory_mode": memory_mode,
                "architect_mode": architect_mode,
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
        approval_for_run = (
            _plan_mode_approval(approval)
            if plan_mode
            else _planner_mode_approval(approval)
            if planner_mode
            else _executor_mode_approval(approval)
            if executor_mode
            else _fixer_mode_approval(approval)
            if fixer_mode
            else _context_mode_approval(approval)
            if context_mode
            else _command_mode_approval(approval)
            if command_mode
            else _final_report_mode_approval(approval)
            if final_report_mode
            else _memory_mode_approval(approval)
            if memory_mode
            else _reviewer_mode_approval(approval)
            if reviewer_mode
            else _verifier_mode_approval(approval)
            if verifier_mode
            else _architect_mode_approval(approval)
            if architect_mode
            else approval
        )
        objective = (
            _plan_mode_objective(line)
            if plan_mode
            else _planner_mode_objective(line)
            if planner_mode
            else _executor_mode_objective(line)
            if executor_mode
            else _fixer_mode_objective(line)
            if fixer_mode
            else _context_mode_objective(line)
            if context_mode
            else _memory_mode_objective(line)
            if memory_mode
            else _command_mode_objective(line)
            if command_mode
            else _final_report_mode_objective(line)
            if final_report_mode
            else _reviewer_mode_objective(line)
            if reviewer_mode
            else _verifier_mode_objective(line)
            if verifier_mode
            else _architect_mode_objective(line)
            if architect_mode
            else _session_context(session, line)
        )
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
        _append_session_turn(
            settings,
            session,
            line,
            result,
            plan_mode=plan_mode,
            planner_mode=planner_mode,
            executor_mode=executor_mode,
            reviewer_mode=reviewer_mode,
            verifier_mode=verifier_mode,
            fixer_mode=fixer_mode,
            memory_mode=memory_mode,
            context_mode=context_mode,
            command_mode=command_mode,
            final_report_mode=final_report_mode,
            architect_mode=architect_mode,
        )
        _print_final_report(result, console=console)


def _print_planner_help(*, console: Console = CONSOLE) -> None:
    console.print(
        Panel(
            (
                "Planner mode is a read-only planning mode. "
                "Convert the architecture or specification into a concrete execution plan. "
                "Do not write code, edit files, run shell commands, or commit changes. "
                "Focus on breaking the work down into tasks, phases, dependencies, and verification steps."
            ),
            title="planner mode",
            border_style="blue",
            padding=(1, 2),
        )
    )


def _print_executor_help(*, console: Console = CONSOLE) -> None:
    console.print(
        Panel(
            (
                "Executor mode is an implementation-only mode. "
                "Execute the approved task or phase exactly using the current repository state. "
                "Do not propose new plans, change the objective, or make broad speculative changes. "
                "Apply minimal safe edits and verify the requested result."
            ),
            title="executor mode",
            border_style="red",
            padding=(1, 2),
        )
    )


def _print_reviewer_help(*, console: Console = CONSOLE) -> None:
    console.print(
        Panel(
            (
                "Reviewer mode is review-only. "
                "Inspect executor agent changes, report correctness, style, security, and architectural issues. "
                "Do not edit files, write code, run shell commands, or commit changes. "
                "Use repository inspection tools and discuss any problems clearly."
            ),
            title="reviewer mode",
            border_style="magenta",
            padding=(1, 2),
        )
    )


def _print_verifier_help(*, console: Console = CONSOLE) -> None:
    console.print(
        Panel(
            (
                "Verifier mode is a read-only verification mode. "
                "Verify the implementation against the original user request, approved spec, and approved plan. "
                "Do not edit files, write code, run shell commands, or commit changes. "
                "Use read-only inspection tools and explain any mismatches clearly."
            ),
            title="verifier mode",
            border_style="purple",
            padding=(1, 2),
        )
    )


def _print_memory_agent_help(*, console: Console = CONSOLE) -> None:
    console.print(
        Panel(
            (
                "Memory Agent mode maintains project memory across tasks. "
                "Capture repository structure, frameworks, conventions, important files, architecture decisions, and known problems. "
                "Do not add new features, refactor unrelated code, or change the approved plan. "
                "Use read-only inspection tools and summarize context for future tasks."
            ),
            title="memory agent mode",
            border_style="cyan",
            padding=(1, 2),
        )
    )


def _print_context_agent_help(*, console: Console = CONSOLE) -> None:
    console.print(
        Panel(
            (
                "Context Agent mode retrieves only the files and snippets needed for the current task. "
                "Do not edit files, run shell commands, commit changes, or add new features. "
                "Use repository inspection tools to gather the smallest relevant context and summarize it clearly."
            ),
            title="context agent mode",
            border_style="blue",
            padding=(1, 2),
        )
    )


def _print_architect_help(*, console: Console = CONSOLE) -> None:
    console.print(
        Panel(
            (
                "Architect mode is an analysis-only mode. "
                "Analyze the repository and design an implementation strategy. "
                "Do not write code, edit files, run shell commands, or commit changes. "
                "Use available inspection tools to answer the objective with architecture, specification, and clear next steps."
            ),
            title="architect mode",
            border_style="magenta",
            padding=(1, 2),
        )
    )


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
