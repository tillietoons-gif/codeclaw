"""Command-line interface: one-shot mode, REPL, and a few utility commands.

Usage:
    codeclaw "add a Makefile that runs pytest"
    codeclaw --model llama3.1:8b "..."
    codeclaw --check           # verify Ollama is reachable + model exists
    codeclaw --tools           # print available tool schemas
    codeclaw repl              # interactive REPL
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections.abc import Awaitable, Callable

from . import __version__
from .agent import CodeClawAgent
from .config import load_settings
from .ollama import OllamaClient, OllamaError
from .tools import build_default_registry
from .tools.base import ApprovalDecision

logger = logging.getLogger("codeclaw")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="codeclaw",
        description="Autonomous coding agent powered by local Ollama.",
    )
    p.add_argument("objective", nargs="?", default=None, help="What to do. If omitted, enters REPL.")
    p.add_argument("--model", help="Override CODECLAW_MODEL")
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
        from dataclasses import replace

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
        if args.check:
            return await _do_check(client, settings)

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
        print(f"FAIL: cannot reach Ollama at {settings.ollama_host}: {exc}", file=sys.stderr)
        return 1
    if not models:
        print("FAIL: Ollama reachable but no models installed.", file=sys.stderr)
        return 1
    print(f"OK: Ollama at {settings.ollama_host}, {len(models)} model(s) installed.")
    for m in models:
        caps = m.get("capabilities") or []
        print(f"  - {m.get('name'):30s}  ctx={m.get('details',{}).get('context_length','?'):>6}  caps={','.join(caps) or 'none'}")
    if settings.model not in {m.get("name") for m in models} | {m.get("model") for m in models}:
        print(
            f"WARN: configured model {settings.model!r} not in installed set. "
            f"Run: ollama pull {settings.model}",
            file=sys.stderr,
        )
        return 3
    return 0


def _log_stream(prefix: str = "codeclaw") -> Callable[[str], None]:
    """Return a simple line-prefixed logger."""
    from rich.console import Console

    console = Console()

    def log(msg: str) -> None:
        # Let the agent print step markers, tool calls, and model output
        # in a single color stream. Errors come from the agent itself.
        if msg.startswith("[error]") or msg.startswith("[denied]"):
            console.print(f"[red]{msg}[/red]")
        elif msg.startswith("[approved]"):
            console.print(f"[yellow]{msg}[/yellow]")
        elif msg.startswith("[step"):
            console.print(f"[bold cyan]{msg}[/bold cyan]")
        elif msg.startswith("  [tool_calls"):
            console.print(f"[magenta]{msg}[/magenta]")
        elif msg.startswith("  [tokens"):
            console.print(f"[dim]{msg}[/dim]")
        elif msg.startswith("  >"):
            console.print(f"[green]{msg}[/green]")
        elif msg.startswith("$ "):
            console.print(f"[blue]{msg}[/blue]")
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
        from rich.console import Console
        from rich.prompt import Confirm

        console = Console()
        console.print(
            f"\n[bold yellow]CodeClaw wants to:[/bold yellow] [white]{summary}[/white]  "
            f"(tool: [cyan]{tool_name}[/cyan])"
        )
        try:
            ok = Confirm.ask("    Allow?", default=False)
        except (EOFError, KeyboardInterrupt):
            ok = False
        if ok:
            cache[key] = True
            return ApprovalDecision(ApprovalDecision.APPROVE)
        return ApprovalDecision(ApprovalDecision.REJECT, reason="user said no")

    return approve


async def _run_one_shot(settings, client, args, objective: str) -> int:
    approval = _interactive_approval(args)
    agent = CodeClawAgent(
        settings=settings,
        client=client,
        approval=approval,
        log=_log_stream(),
    )
    try:
        result = await agent.run(objective)
    except OllamaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _print_final_report(result)
    return 0 if result.completed else 2


async def _run_repl(settings, client, args) -> int:
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    console.print(
        f"[bold]CodeClaw {__version__}[/bold]  "
        f"model=[cyan]{settings.model}[/cyan]  "
        f"project=[cyan]{settings.project_dir}[/cyan]\n"
        "Type an objective and press Enter. Ctrl-D or `:q` to exit. "
        "`:reset` clears the conversation. `:model X` switches model."
    )
    approval = _interactive_approval(args)
    while True:
        try:
            line = Prompt.ask("[bold green]you[/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nbye.")
            return 0
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            return 0
        if line == ":reset":
            console.print("(no persistent state yet — each turn is independent in this build)")
            continue
        if line.startswith(":model "):
            new_model = line.split(" ", 1)[1].strip()
            from dataclasses import replace

            settings = replace(settings, model=new_model)
            console.print(f"model -> {new_model}")
            continue
        agent = CodeClawAgent(
            settings=settings,
            client=client,
            approval=approval,
            log=_log_stream(),
        )
        try:
            result = await agent.run(line)
        except OllamaError as exc:
            console.print(f"[red]error: {exc}[/red]")
            continue
        _print_final_report(result, console=console)


def _print_final_report(result, console=None) -> None:
    from rich.console import Console

    c = console or Console()
    if result.completed:
        c.print("\n[bold green]✓ done[/bold green]")
    else:
        c.print(f"\n[bold yellow]⚠ stopped: {result.reason}[/bold yellow]")
    c.print(
        f"  steps: {len(result.steps)}    tokens: {result.total_tokens}"
    )
    if result.final_message:
        c.rule("final report")
        c.print(result.final_message)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = _build_parser().parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        print("\ninterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
