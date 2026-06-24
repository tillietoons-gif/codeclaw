"""Tests for CLI argument routing."""
from __future__ import annotations

from dataclasses import replace

import pytest

from codeclaw import cli


class FakeClient:
    def __init__(self, *args, **kwargs):
        self.closed = False

    async def close(self):
        self.closed = True


class FakeCheckClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def list_models(self):
        return [{"name": "m", "details": {}}]

    async def show_model(self, model):
        assert model == "m"
        return {
            "capabilities": ["completion", "tools"],
            "model_info": {"qwen2.context_length": 32768},
        }


class FakeSelectClient:
    def __init__(self, *args, **kwargs):
        pass

    async def close(self):
        pass

    async def list_models(self):
        return [{"name": "m1"}, {"name": "m2"}]

    async def show_model(self, model):
        return {
            "capabilities": ["completion", "tools"],
            "model_info": {"qwen2.context_length": 32768 if model == "m1" else 40960},
        }


@pytest.mark.asyncio
async def test_repl_argument_starts_repl(monkeypatch):
    called = {}

    async def fake_repl(settings, client, args):
        called["objective"] = args.objective
        return 0

    monkeypatch.setattr("rich.prompt.IntPrompt.ask", lambda *args, **kwargs: 1)
    monkeypatch.setattr(cli, "OllamaClient", FakeSelectClient)
    monkeypatch.setattr(cli, "_run_repl", fake_repl)

    args = cli._build_parser().parse_args(["repl"])
    assert await cli._async_main(args) == 0
    assert called == {"objective": "repl"}


@pytest.mark.asyncio
async def test_check_uses_show_model_for_capabilities(capsys):
    settings = cli.load_settings()
    result = await cli._do_check(FakeCheckClient(), settings)

    out = capsys.readouterr().out
    assert result == 3
    assert "completion, tools" in out
    assert "32768" in out


@pytest.mark.asyncio
async def test_select_model_updates_settings(monkeypatch):
    monkeypatch.setattr("rich.prompt.IntPrompt.ask", lambda *args, **kwargs: 2)
    settings = replace(cli.load_settings(), model="m1")

    selected = await cli._select_model(FakeSelectClient(), settings)

    assert selected.model == "m2"


@pytest.mark.asyncio
async def test_select_model_rejected_in_non_interactive(monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", FakeClient)

    args = cli._build_parser().parse_args(["--select-model", "--non-interactive", "do thing"])

    assert await cli._async_main(args) == 2


@pytest.mark.asyncio
async def test_plain_codeclaw_starts_picker_then_repl(monkeypatch):
    called = {}

    async def fake_repl(settings, client, args):
        called["model"] = settings.model
        return 0

    monkeypatch.setattr("rich.prompt.IntPrompt.ask", lambda *args, **kwargs: 2)
    monkeypatch.setattr(cli, "OllamaClient", FakeSelectClient)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli, "_run_repl", fake_repl)

    args = cli._build_parser().parse_args([])

    assert await cli._async_main(args) == 0
    assert called == {"model": "m2"}


@pytest.mark.asyncio
async def test_check_alias_runs_health_check(monkeypatch):
    monkeypatch.setattr(cli, "OllamaClient", FakeCheckClient)

    args = cli._build_parser().parse_args(["check"])

    assert await cli._async_main(args) == 3
