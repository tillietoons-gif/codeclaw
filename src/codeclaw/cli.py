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
import sys
from collections.abc import Awaitable, Callable
from dataclasses import replace

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import __version__
from .agent import CodeClawAgent
from .config import load_settings
from .ollama import OllamaClient, OllamaError
from .tools import build_default_registry
from .tools.base import ApprovalDecision

logger = logging.getLogger("codeclaw")

CONSOLE = Console()


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
        opens_interactive_ui = args.objective in ("repl", "models") or (not args.objective and sys.stdin.isatty())
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

        if args.objective in ("repl", "models"):
            return await _run_repl(settings, client, args)
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
        "[bold]:q[/bold] quit   [bold]:reset[/bold] clear   [bold]:model[/bold] choose   [bold]:model NAME[/bold] switch",
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


def _log_stream(console: Console = CONSOLE) -> Callable[[str], None]:
    """Return a structured Rich logger for the agent loop."""
    def log(msg: str) -> None:
        if msg.startswith("[error]") or msg.startswith("[denied]"):
            console.print(f"[bold red]{msg}[/bold red]")
        elif msg.startswith("[approved]"):
            console.print(f"[bold green]{msg}[/bold green]")
        elif msg.startswith("[step"):
            console.rule(f"[bold cyan]{msg.strip()}[/bold cyan]", style="dim")
        elif msg.startswith("  [tool_calls"):
            console.print(f"[bold magenta]{msg.strip()}[/bold magenta]")
        elif msg.startswith("  [tokens"):
            console.print(f"[dim]{msg.strip()}[/dim]")
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


async def _run_one_shot(settings, client, args, objective: str) -> int:
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


async def _run_repl(settings, client, args) -> int:
    from rich.prompt import Prompt

    console = CONSOLE
    _print_repl_header(settings, console=console)
    approval = _interactive_approval(args)
    while True:
        try:
            line = Prompt.ask("[bold green]you[/bold green]", console=console).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye.[/dim]")
            return 0
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            return 0
        if line == ":reset":
            console.print("[dim]No persistent conversation state yet; each request starts fresh.[/dim]")
            continue
        if line in (":model", ":models"):
            selected = await _select_model(client, settings, console=console)
            if selected is not None:
                settings = selected
            continue
        if line.startswith(":model "):
            new_model = line.split(" ", 1)[1].strip()

            settings = replace(settings, model=new_model)
            console.print(f"[dim]model ->[/dim] [cyan]{new_model}[/cyan]")
            continue
        console.print(Panel(Text(line, overflow="fold"), title="objective", border_style="green"))
        agent = CodeClawAgent(
            settings=settings,
            client=client,
            approval=approval,
            log=_log_stream(console),
        )
        try:
            result = await agent.run(line)
        except OllamaError as exc:
            console.print(f"[red]error: {exc}[/red]")
            continue
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
