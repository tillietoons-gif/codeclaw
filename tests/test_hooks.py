from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from codeclaw.hooks import hook_counts, load_hook_config, run_hooks


def _write_settings(root: Path, data: dict) -> None:
    settings_dir = root / ".codeclaw"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.asyncio
async def test_run_command_hook_receives_event_payload(tmp_path):
    command = (
        f"{sys.executable} -c "
        "\"import json,sys; data=json.load(sys.stdin); print(data['event'] + ':' + data['value'])\""
    )
    _write_settings(
        tmp_path,
        {"hooks": {"RunStart": [{"type": "command", "command": command}]}},
    )

    results = await run_hooks(tmp_path, "RunStart", {"value": "ok"})

    assert len(results) == 1
    assert results[0].ok
    assert results[0].output == "RunStart:ok"


@pytest.mark.asyncio
async def test_run_command_hook_reports_nonzero_exit(tmp_path):
    command = f"{sys.executable} -c \"import sys; print('nope'); sys.exit(7)\""
    _write_settings(tmp_path, {"hooks": {"UserPromptSubmit": [command]}})

    results = await run_hooks(tmp_path, "UserPromptSubmit", {"prompt": "hello"})

    assert len(results) == 1
    assert not results[0].ok
    assert results[0].returncode == 7
    assert results[0].output == "nope"


def test_load_hook_config_normalizes_strings_and_counts(tmp_path):
    _write_settings(
        tmp_path,
        {
            "hooks": {
                "SessionStart": ["echo hi", {"command": "echo there"}],
                "Unknown": ["ignored"],
            }
        },
    )

    config = load_hook_config(tmp_path)

    assert [entry["command"] for entry in config["SessionStart"]] == ["echo hi", "echo there"]
    assert hook_counts(tmp_path) == {"SessionStart": 2}
