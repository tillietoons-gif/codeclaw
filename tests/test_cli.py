"""Tests for CLI argument routing."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

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

    async def fake_repl(settings, client, args, *, resume_latest=False):
        called["objective"] = args.objective
        called["resume_latest"] = resume_latest
        return 0

    monkeypatch.setattr("rich.prompt.IntPrompt.ask", lambda *args, **kwargs: 1)
    monkeypatch.setattr(cli, "OllamaClient", FakeSelectClient)
    monkeypatch.setattr(cli, "_run_repl", fake_repl)

    args = cli._build_parser().parse_args(["repl"])
    assert await cli._async_main(args) == 0
    assert called == {"objective": "repl", "resume_latest": False}


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


@pytest.mark.asyncio
async def test_continue_alias_resumes_latest(monkeypatch):
    called = {}

    async def fake_repl(settings, client, args, *, resume_latest=False):
        called["resume_latest"] = resume_latest
        return 0

    monkeypatch.setattr("rich.prompt.IntPrompt.ask", lambda *args, **kwargs: 1)
    monkeypatch.setattr(cli, "OllamaClient", FakeSelectClient)
    monkeypatch.setattr(cli, "_run_repl", fake_repl)

    args = cli._build_parser().parse_args(["continue"])

    assert await cli._async_main(args) == 0
    assert called == {"resume_latest": True}


def test_slash_command_helpers():
    assert cli._is_quit_command("/quit")
    assert cli._is_quit_command("/q")
    assert cli._is_reset_command("/reset")
    assert cli._is_model_picker_command("/models")
    assert cli._is_model_picker_command("/model")
    assert cli._is_help_command("/help")
    assert cli._is_help_command("/")
    assert cli._is_plan_command("/plan")
    assert cli._is_plan_command("/plan on")
    assert cli._model_name_from_command("/model qwen3:14b") == "qwen3:14b"
    assert cli._slash_filter("/sta") == "/sta"
    assert cli._slash_filter("/status") is None
    assert cli._slash_filter("/hooks") is None
    assert cli._slash_filter("/checkpoint before-change") is None
    assert cli._slash_filter("/restore abc") is None
    assert cli._slash_filter("normal prompt") is None
    assert cli._checkpoint_name_from_command("/checkpoint before change") == "before change"
    assert cli._restore_id_from_command("/restore abc123") == "abc123"
    assert cli._resume_id_from_command("/resume abc123") == "abc123"
    assert cli._set_args_from_command("/set model qwen3:14b") == ("model", "qwen3:14b")


def test_command_palette_renders(capsys):
    cli._print_command_palette()

    out = capsys.readouterr().out
    assert "/models" in out
    assert "/permissions" in out
    assert "/plan" in out
    assert "/sessions" in out
    assert "/hooks" in out
    assert "/init" in out
    assert "/compact" in out
    assert "/todo" in out


def test_tools_and_permissions_render(capsys):
    args = cli._build_parser().parse_args([])

    cli._print_tools_table()
    cli._print_permissions(args)

    out = capsys.readouterr().out
    assert "read_file" in out
    assert "exec" in out
    assert "Approval" in out


def test_status_renders(capsys):
    args = cli._build_parser().parse_args([])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args)

    out = capsys.readouterr().out
    assert "Status" in out
    assert "model" in out
    assert "m" in out


def test_session_helpers_roundtrip(tmp_path):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")
    session = cli._new_session(settings)
    result = SimpleNamespace(
        final_message="done",
        completed=True,
        reason="done",
        steps=[object(), object()],
        total_tokens=12,
    )

    cli._append_session_turn(settings, session, "do thing", result, plan_mode=True)
    sessions = cli._load_sessions(settings)

    assert len(sessions) == 1
    assert sessions[0]["turns"][0]["objective"] == "do thing"
    assert sessions[0]["turns"][0]["plan_mode"] is True
    assert cli._find_session(settings, sessions[0]["id"][:8])["id"] == sessions[0]["id"]
    assert cli._latest_session(settings)["id"] == sessions[0]["id"]


def test_sessions_and_memory_render(tmp_path, capsys):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")
    session = cli._new_session(settings)
    result = SimpleNamespace(final_message="done", completed=True, reason="done", steps=[], total_tokens=0)
    cli._append_session_turn(settings, session, "do thing", result, plan_mode=False)
    (tmp_path / "MEMORY.md").write_text("remember this")

    cli._print_sessions(settings)
    cli._print_current_session(session)
    cli._print_memory(settings)

    out = capsys.readouterr().out
    assert "Sessions" in out
    assert "current session" in out
    assert "do thing" in out
    assert "remember this" in out


def test_hooks_render(tmp_path, capsys):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")
    settings_dir = tmp_path / ".codeclaw"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(
        '{"hooks": {"SessionStart": ["echo hi"], "PreToolUse": ["echo check"]}}',
        encoding="utf-8",
    )

    cli._print_hooks(settings)

    out = capsys.readouterr().out
    assert "Hooks" in out
    assert "SessionStart" in out
    assert "PreToolUse" in out


def test_init_config_and_hook_examples(tmp_path, capsys):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")

    created = cli._init_project(settings)
    ok, key, value = cli._set_project_default(settings, "host", "http://x:11434")
    hook_files = cli._write_hook_examples(settings)
    cli._print_config(settings)

    out = capsys.readouterr().out
    assert tmp_path / "AGENTS.md" in created
    assert ok
    assert key == "ollama_host"
    assert value == "http://x:11434"
    assert (tmp_path / ".codeclaw" / "settings.json").exists()
    assert all(path.exists() for path in hook_files)
    assert "http://x:11434" in out


def test_compact_session_and_todos(tmp_path, capsys):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")
    session = cli._new_session(settings)
    result = SimpleNamespace(final_message="finished work", completed=True, reason="done", steps=[], total_tokens=0)
    cli._append_session_turn(settings, session, "do thing", result, plan_mode=False)

    summary = cli._compact_session(settings, session)
    cli._print_todos(session)

    out = capsys.readouterr().out
    assert "do thing" in summary
    assert "Session Todo" in out
    assert len(session["turns"]) == 1


def test_session_context_includes_prior_turns(tmp_path):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")
    session = cli._new_session(settings)
    result = SimpleNamespace(final_message="finished previous work", completed=True, reason="done", steps=[], total_tokens=0)
    cli._append_session_turn(settings, session, "previous objective", result, plan_mode=False)

    context = cli._session_context(session, "next objective")

    assert "RESUMED SESSION CONTEXT" in context
    assert "previous objective" in context
    assert "finished previous work" in context
    assert "next objective" in context


@pytest.mark.asyncio
async def test_plan_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    plan_approval = cli._plan_mode_approval(approve)

    assert not (await plan_approval("exec", "run shell")).approved
    assert (await plan_approval("read_file", "read")).approved


@pytest.mark.asyncio
async def test_architect_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    architect_approval = cli._architect_mode_approval(approve)

    assert not (await architect_approval("git_commit", "commit changes")).approved
    assert (await architect_approval("read_file", "inspect")).approved


def test_plan_mode_objective_is_read_only():
    out = cli._plan_mode_objective("change the app")
    assert "PLAN MODE" in out
    assert "Do not edit files" in out
    assert "change the app" in out


def test_architect_mode_objective_is_analysis_only():
    out = cli._architect_mode_objective("design the architecture")
    assert "ARCHITECT MODE" in out
    assert "Do not edit files" in out
    assert "design the architecture" in out


@pytest.mark.asyncio
async def test_planner_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    planner_approval = cli._planner_mode_approval(approve)

    assert not (await planner_approval("write_file", "modify a file")).approved
    assert (await planner_approval("read_file", "inspect")).approved


def test_planner_mode_objective_is_plan_focused():
    out = cli._planner_mode_objective("convert architecture to plan")
    assert "PLANNER MODE" in out
    assert "Convert the architect specification" in out
    assert "convert architecture to plan" in out

@pytest.mark.asyncio
async def test_executor_mode_approval_passes_through():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    executor_approval = cli._executor_mode_approval(approve)

    assert (await executor_approval("write_file", "modify a file")).approved
    assert (await executor_approval("read_file", "inspect")).approved


def test_executor_mode_objective_is_executor_only():
    out = cli._executor_mode_objective("apply approved patch")
    assert "EXECUTOR MODE" in out
    assert "approved task" in out.lower()
    assert "apply approved patch" in out

@pytest.mark.asyncio
async def test_reviewer_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    reviewer_approval = cli._reviewer_mode_approval(approve)

    assert not (await reviewer_approval("write_file", "modify a file")).approved
    assert (await reviewer_approval("read_file", "inspect")).approved


def test_reviewer_mode_objective_is_review_only():
    out = cli._reviewer_mode_objective("review executor changes")
    assert "REVIEWER MODE" in out
    assert "review the executor agent" in out.lower()
    assert "review executor changes" in out

@pytest.mark.asyncio
async def test_verifier_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    verifier_approval = cli._verifier_mode_approval(approve)

    assert not (await verifier_approval("write_file", "modify a file")).approved
    assert (await verifier_approval("read_file", "inspect")).approved


def test_verifier_mode_objective_is_verification_only():
    out = cli._verifier_mode_objective("verify implementation")
    assert "VERIFIER MODE" in out
    assert "verify the implementation" in out.lower()
    assert "verify implementation" in out

@pytest.mark.asyncio
async def test_run_one_shot_reviewer_mode(monkeypatch):
    class FakeClientWithShow:
        async def show_model(self, model):
            return {}

    class FakeAgent:
        def __init__(self, settings, client, approval, log):
            self.settings = settings
            self.client = client
            self.approval = approval
            self.log = log
            self.objective = None

        async def run(self, objective):
            self.objective = objective
            return SimpleNamespace(final_message="ok", completed=True, reason="done", steps=[], total_tokens=0)

    captured = {}

    def fake_agent_factory(settings, client, approval, log):
        captured["agent"] = FakeAgent(settings, client, approval, log)
        return captured["agent"]

    monkeypatch.setattr(cli, "CodeClawAgent", fake_agent_factory)
    settings = cli.load_settings()
    args = cli._build_parser().parse_args(["--reviewer", "review executor changes"])
    result = await cli._run_one_shot(settings, FakeClientWithShow(), args, "review executor changes")

    assert result == 0
    assert "REVIEWER MODE" in captured["agent"].objective
    assert "review the executor" in captured["agent"].objective.lower()


def test_status_renders_reviewer_mode(capsys):
    args = cli._build_parser().parse_args(["--reviewer"])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args, reviewer_mode=True)

    out = capsys.readouterr().out
    assert "mode" in out
    assert "reviewer" in out


def test_status_renders_verifier_mode(capsys):
    args = cli._build_parser().parse_args(["--verifier"])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args, verifier_mode=True)

    out = capsys.readouterr().out
    assert "mode" in out
    assert "verifier" in out


def test_status_renders_fixer_mode(capsys):
    args = cli._build_parser().parse_args(["--fixer"])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args, fixer_mode=True)

    out = capsys.readouterr().out
    assert "mode" in out
    assert "fixer" in out


def test_status_renders_memory_agent_mode(capsys):
    args = cli._build_parser().parse_args(["--memory-agent"])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args, memory_mode=True)

    out = capsys.readouterr().out
    assert "mode" in out
    assert "memory" in out


def test_status_renders_context_agent_mode(capsys):
    args = cli._build_parser().parse_args(["--context-agent"])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args, context_mode=True)

    out = capsys.readouterr().out
    assert "mode" in out
    assert "context" in out

@pytest.mark.asyncio
async def test_command_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    command_approval = cli._command_mode_approval(approve)

    assert not (await command_approval("exec", "run shell")).approved
    assert (await command_approval("read_file", "read")).approved


def test_command_mode_objective_is_command_focused():
    out = cli._command_mode_objective("suggest safe commands")
    assert "COMMAND AGENT MODE" in out
    assert "safe terminal commands" in out.lower()
    assert "suggest safe commands" in out

@pytest.mark.asyncio
async def test_final_report_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    final_report_approval = cli._final_report_mode_approval(approve)

    assert not (await final_report_approval("write_file", "modify a file")).approved
    assert (await final_report_approval("read_file", "inspect")).approved


def test_final_report_mode_objective_is_summary_focused():
    out = cli._final_report_mode_objective("summarize results")
    assert "FINAL REPORT MODE" in out
    assert "summarize completed work" in out.lower()
    assert "summarize results" in out


def test_status_renders_command_agent_mode(capsys):
    args = cli._build_parser().parse_args(["--command-agent"])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args, command_mode=True)

    out = capsys.readouterr().out
    assert "mode" in out
    assert "command" in out


def test_status_renders_final_report_agent_mode(capsys):
    args = cli._build_parser().parse_args(["--final-report-agent"])
    settings = replace(cli.load_settings(), model="m")

    cli._print_status(settings, args, final_report_mode=True)

    out = capsys.readouterr().out
    assert "mode" in out
    assert "final-report" in out

@pytest.mark.asyncio
async def test_orchestrated_flow_runs_simple_task_sequence(monkeypatch):
    calls = []
    results = [
        SimpleNamespace(final_message="context gathered", completed=True, reason="done", steps=[], total_tokens=0),
        SimpleNamespace(final_message="executor result", completed=True, reason="done", steps=[], total_tokens=0),
        SimpleNamespace(final_message="reviewer report", completed=True, reason="done", steps=[], total_tokens=0),
        SimpleNamespace(final_message="verification passed", completed=True, reason="done", steps=[], total_tokens=0),
        SimpleNamespace(final_message="final summary", completed=True, reason="done", steps=[], total_tokens=0),
    ]

    async def fake_run_named_agent(settings, client, args, objective, mode_name):
        calls.append(mode_name)
        return results.pop(0)

    async def fake_ask_user_approval(args, plan_summary):
        return True

    monkeypatch.setattr(cli, "_run_named_agent", fake_run_named_agent)
    monkeypatch.setattr(cli, "_ask_user_approval", fake_ask_user_approval)
    monkeypatch.setattr(cli, "_print_orchestrator_summary", lambda state: None)
    monkeypatch.setattr(cli, "_print_final_report", lambda result, console=None: None)
    monkeypatch.setattr(cli, "_print_session_header", lambda settings, objective: None)

    class FakeClientWithShow:
        async def show_model(self, model):
            return {}
        async def close(self):
            pass

    args = cli._build_parser().parse_args(["fix typo in README"])
    settings = replace(cli.load_settings(), model="m")

    result = await cli._run_one_shot(settings, FakeClientWithShow(), args, "fix typo in README")

    assert result == 0
    assert calls == ["context_agent", "executor", "reviewer", "verifier", "final_report_agent"]


def test_has_explicit_mode_detects_cli_flags():
    args = cli._build_parser().parse_args(["--executor", "do thing"])
    assert cli._has_explicit_mode(args)

    args = cli._build_parser().parse_args(["do thing"])
    assert not cli._has_explicit_mode(args)

@pytest.mark.asyncio
async def test_verifier_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    verifier_approval = cli._verifier_mode_approval(approve)

    assert not (await verifier_approval("write_file", "modify a file")).approved
    assert (await verifier_approval("read_file", "inspect")).approved


def test_verifier_mode_objective_is_verification_only():
    out = cli._verifier_mode_objective("verify implementation")
    assert "VERIFIER MODE" in out
    assert "verify the implementation" in out.lower()
    assert "verify implementation" in out

@pytest.mark.asyncio
async def test_fixer_mode_approval_rejects_destructive_tools():
    async def approve(name, summary):
        return cli.ApprovalDecision(cli.ApprovalDecision.APPROVE)

    fixer_approval = cli._fixer_mode_approval(approve)

    assert not (await fixer_approval("write_file", "modify a file")).approved
    assert (await fixer_approval("read_file", "inspect")).approved


def test_fixer_mode_objective_is_fix_only():
    out = cli._fixer_mode_objective("fix reported issue")
    assert "FIXER MODE" in out
    assert "fix only issues" in out.lower()
    assert "fix reported issue" in out

@pytest.mark.asyncio
async def test_run_one_shot_executor_mode(monkeypatch):
    class FakeClientWithShow:
        async def show_model(self, model):
            return {}

    class FakeAgent:
        def __init__(self, settings, client, approval, log):
            self.settings = settings
            self.client = client
            self.approval = approval
            self.log = log
            self.objective = None

        async def run(self, objective):
            self.objective = objective
            return SimpleNamespace(final_message="ok", completed=True, reason="done", steps=[], total_tokens=0)

    monkeypatch.setattr(cli, "CodeClawAgent", FakeAgent)
    settings = cli.load_settings()
    args = cli._build_parser().parse_args(["--executor", "do thing"])
    result = await cli._run_one_shot(settings, FakeClientWithShow(), args, "do thing")

    assert result == 0
    assert hasattr(cli.CodeClawAgent, "run")


def test_checkpoint_helpers_roundtrip(tmp_path):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")
    (tmp_path / "a.txt").write_text("one")
    (tmp_path / ".env").write_text("secret")

    checkpoint = cli._create_checkpoint(settings, "before")
    (tmp_path / "a.txt").write_text("two")
    (tmp_path / "new.txt").write_text("new")

    ok, restored = cli._restore_checkpoint(settings, checkpoint["id"])

    assert ok
    assert restored == checkpoint["id"]
    assert (tmp_path / "a.txt").read_text() == "one"
    assert not (tmp_path / "new.txt").exists()
    assert (tmp_path / ".env").read_text() == "secret"


def test_checkpoints_render(tmp_path, capsys):
    settings = replace(cli.load_settings(), project_dir=str(tmp_path), model="m")
    (tmp_path / "a.txt").write_text("one")
    cli._create_checkpoint(settings, "before")

    cli._print_checkpoints(settings)

    out = capsys.readouterr().out
    assert "Checkpoints" in out
    assert "before" in out
