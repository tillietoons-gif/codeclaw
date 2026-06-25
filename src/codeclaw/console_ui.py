"""Console UI helpers used by tests and the CLI.

This module provides lightweight wrappers for printing status panels,
command palettes, prompts, and run summaries.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

CONSOLE = Console()


def toast(message: str, kind: str = "ok") -> None:
    CONSOLE.print(Text(str(message), style="bold green" if kind == "ok" else "bold yellow"))


def print_command_palette(commands: Sequence[tuple[str, str]]) -> None:
    categories: dict[str, list[tuple[str, str]]] = {
        "AI & models": [],
        "Session & project": [],
        "Session control": [],
    }
    for command, description in commands:
        if command in ("/models", "/model", "/help", "/", "/?", "/status", "/tools", "/permissions", "/diff", "/changes"):
            categories["AI & models"].append((command, description))
        elif command in ("/sessions", "/current", "/todo", "/memory", "/hooks", "/hook-example", "/checkpoints", "/checkpoint", "/restore", "/resume", "/set", "/config", "/init"):
            categories["Session & project"].append((command, description))
        else:
            categories["Session control"].append((command, description))

    for title, entries in categories.items():
        if not entries:
            continue
        table = Table(title=title, show_header=True, header_style="bold cyan")
        table.add_column("Command", style="bold green", no_wrap=True)
        table.add_column("Description", overflow="fold")
        for command, description in entries:
            table.add_row(command, description)
        CONSOLE.print(table)


def print_status_panel(rows: Iterable[tuple[str, str]]) -> None:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold")
    table.add_column()
    for name, value in rows:
        table.add_row(str(name), str(value))
    CONSOLE.print(Panel(table, title="Status", border_style="cyan", padding=(1, 2)))


def print_final_report(result) -> None:
    if getattr(result, "streamed_final", False):
        CONSOLE.print(Panel("Run summary", title="Run summary", border_style="green", padding=(1, 2)))
        return
    final_message = str(getattr(result, "final_message", ""))
    if final_message.strip():
        CONSOLE.print(Panel(Markdown(final_message), title="Final report", border_style="green", padding=(1, 2)))
    else:
        CONSOLE.print(Panel("No final report.", title="Final report", border_style="green", padding=(1, 2)))


def repl_bottom_toolbar() -> str:
    return "Press enter to submit. Use /help for slash commands."


def repl_prompt_style() -> dict[str, str]:
    return {"prompt": "bold cyan"}


class _LogStream:
    def __init__(self) -> None:
        self._saw_response = False

    def __call__(self, message: str) -> None:
        if message.startswith("\n[step"):
            try:
                label = message.strip().lstrip("[").rstrip("]")
                parts = label.split()[1].split("/")
                self._print(f"Step {parts[0]} / {parts[1]}")
                return
            except Exception:
                pass
        if "[tool_calls" in message:
            self._print("Tools")
            return
        if message.startswith("  >"):
            if not self._saw_response:
                self._print("Response")
                self._saw_response = True
            self._print(message[3:])
            return
        self._print(message)

    def end_stream(self) -> None:
        pass

    def _print(self, text: str) -> None:
        CONSOLE.print(Text(text, overflow="fold"))


def make_log_stream() -> _LogStream:
    return _LogStream()
