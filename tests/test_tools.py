"""Tests for the tool registry and the base layer.

These tests don't touch the network or the filesystem outside a tempdir.
"""
from __future__ import annotations

import pytest

from codeclaw.tools import ToolContext, build_default_registry, parse_args
from codeclaw.tools.base import ApprovalDecision, ToolRegistry


def _ctx(tmp_path) -> ToolContext:
    return ToolContext(
        cwd=str(tmp_path),
        approval=lambda n, s: ApprovalDecision(ApprovalDecision.APPROVE),
        log=lambda m: None,
    )


def test_default_registry_has_expected_tools():
    reg = build_default_registry()
    expected = {
        "read_file", "write_file", "edit_file", "list_dir",
        "grep", "exec", "git_status", "git_diff", "git_log", "git_commit",
    }
    assert expected.issubset(set(reg.names()))


def test_schemas_are_well_formed():
    reg = build_default_registry()
    for schema in reg.schemas():
        assert schema["type"] == "function"
        fn = schema["function"]
        assert fn["name"]
        assert fn["description"]
        params = fn["parameters"]
        assert params["type"] == "object"
        assert isinstance(params.get("properties"), dict)


def test_parse_args_handles_dict_and_json_string():
    assert parse_args({"a": 1}) == {"a": 1}
    assert parse_args('{"a": 1}') == {"a": 1}
    assert parse_args("not-json") == {"_raw": "not-json"}
    assert parse_args(None) == {}


@pytest.mark.asyncio
async def test_read_write_edit_roundtrip(tmp_path):
    reg = build_default_registry()
    ctx = _ctx(tmp_path)

    write_tool = reg.get("write_file")
    res = await write_tool.run({"path": "hello.txt", "content": "alpha\nbeta\ngamma\n"}, ctx)
    assert not res.is_error, res.output
    assert (tmp_path / "hello.txt").read_text() == "alpha\nbeta\ngamma\n"

    read_tool = reg.get("read_file")
    res = await read_tool.run({"path": "hello.txt"}, ctx)
    assert not res.is_error
    assert "alpha" in res.output
    assert "beta" in res.output

    edit_tool = reg.get("edit_file")
    res = await edit_tool.run(
        {"path": "hello.txt", "old_text": "beta", "new_text": "BETA"}, ctx,
    )
    assert not res.is_error
    assert (tmp_path / "hello.txt").read_text() == "alpha\nBETA\ngamma\n"

    # Edit with a non-unique match should fail.
    (tmp_path / "dups.txt").write_text("x\nx\n")
    res = await edit_tool.run({"path": "dups.txt", "old_text": "x", "new_text": "y"}, ctx)
    assert res.is_error
    assert "matched 2" in res.output


@pytest.mark.asyncio
async def test_filesystem_tools_reject_paths_outside_project(tmp_path):
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret")
    reg = build_default_registry()
    ctx = _ctx(tmp_path)

    read_res = await reg.get("read_file").run({"path": str(outside)}, ctx)
    write_res = await reg.get("write_file").run({"path": "../outside.txt", "content": "nope"}, ctx)

    assert read_res.is_error
    assert write_res.is_error
    assert "escapes project directory" in read_res.output
    assert outside.read_text() == "secret"


@pytest.mark.asyncio
async def test_list_dir_skips_dotfiles_and_venv(tmp_path):
    reg = build_default_registry()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.txt").write_text("shh")
    (tmp_path / ".venv").mkdir()
    (tmp_path / ".venv" / "lib").write_text("x")
    res = await reg.get("list_dir").run({"path": ".", "max_depth": 3}, _ctx(tmp_path))
    assert not res.is_error
    assert "main.py" in res.output
    assert ".hidden" not in res.output
    assert ".venv" not in res.output


@pytest.mark.asyncio
async def test_exec_runs_command_and_returns_output(tmp_path):
    reg = build_default_registry()
    (tmp_path / "marker.txt").write_text("ok")
    res = await reg.get("exec").run({"command": "cat marker.txt", "timeout_s": 10}, _ctx(tmp_path))
    assert not res.is_error, res.output
    assert "ok" in res.output


@pytest.mark.asyncio
async def test_exec_timeout_kills_process(tmp_path):
    reg = build_default_registry()
    res = await reg.get("exec").run({"command": "sleep 5", "timeout_s": 1}, _ctx(tmp_path))
    assert res.is_error
    assert "timed out" in res.output


@pytest.mark.asyncio
async def test_exec_rejects_cwd_outside_project(tmp_path):
    reg = build_default_registry()
    res = await reg.get("exec").run({"command": "pwd", "cwd": ".."}, _ctx(tmp_path))
    assert res.is_error
    assert "escapes project directory" in res.output


@pytest.mark.asyncio
async def test_grep_rejects_path_outside_project(tmp_path):
    reg = build_default_registry()
    res = await reg.get("grep").run({"pattern": "anything", "path": ".."}, _ctx(tmp_path))
    assert res.is_error
    assert "escapes project directory" in res.output


def test_destructive_tools_marked_correctly():
    reg = build_default_registry()
    # write_file, edit_file, exec, git_commit require approval; the rest don't.
    assert reg.get("write_file").requires_approval is True
    assert reg.get("edit_file").requires_approval is True
    assert reg.get("exec").requires_approval is True
    assert reg.get("git_commit").requires_approval is True
    assert reg.get("read_file").requires_approval is False
    assert reg.get("list_dir").requires_approval is False
    assert reg.get("grep").requires_approval is False


def test_duplicate_registration_rejected():
    reg = ToolRegistry()
    reg.register(build_default_registry().get("read_file"))
    with pytest.raises(ValueError):
        reg.register(build_default_registry().get("read_file"))


def test_unknown_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        reg.get("nope")
