"""Tests for console UI components."""
from __future__ import annotations

from types import SimpleNamespace

from codeclaw.console_ui import (
    make_log_stream,
    print_command_palette,
    print_final_report,
    print_status_panel,
    repl_bottom_toolbar,
    repl_prompt_style,
    toast,
)
from codeclaw import cli


def test_toast_renders(capsys):
    toast("hello", kind="ok")
    out = capsys.readouterr().out
    assert "hello" in out


def test_command_palette_groups_commands(capsys):
    print_command_palette(cli.SLASH_COMMANDS)
    out = capsys.readouterr().out
    assert "AI & models" in out
    assert "/models" in out
    assert "Session" in out


def test_status_panel_renders(capsys):
    print_status_panel([("model", "qwen"), ("project", "/tmp")])
    out = capsys.readouterr().out
    assert "Status" in out
    assert "qwen" in out


def test_final_report_skips_streamed_duplicate(capsys):
    result = SimpleNamespace(objective="x", final_message="Already streamed.", steps=[], total_tokens=42, completed=True, reason="done", streamed_final=True)
    print_final_report(result)
    out = capsys.readouterr().out
    assert "Run summary" in out
    assert "Already streamed." not in out


def test_final_report_shows_markdown_when_not_streamed(capsys):
    result = SimpleNamespace(objective="x", final_message="**Done**", steps=[object()], total_tokens=10, completed=True, reason="done", streamed_final=False)
    print_final_report(result)
    out = capsys.readouterr().out
    assert "Final report" in out
    assert "Done" in out


def test_repl_prompt_style_uses_prompt_toolkit_colors():
    from prompt_toolkit.styles import Style
    Style.from_dict(repl_prompt_style())


def test_repl_bottom_toolbar_is_plain_text():
    text = repl_bottom_toolbar()
    assert isinstance(text, str)
    assert "enter" in text
    assert "/help" in text


def test_log_stream_handles_step_and_tools(capsys):
    log = make_log_stream()
    log("\n[step 1/5]")
    log("  [tool_calls: read_file]")
    log("  >Hello")
    log.end_stream()
    out = capsys.readouterr().out
    assert "Step 1 / 5" in out
    assert "Tools" in out
    assert "Response" in out
    assert "Hello" in out
