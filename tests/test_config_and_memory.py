"""Tests for config loading and project memory."""
from __future__ import annotations

import pytest

from codeclaw import memory
from codeclaw.config import Settings, load_settings


def test_defaults_when_no_env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for k in (
        "OLLAMA_HOST", "CODECLAW_MODEL", "CODECLAW_MAX_STEPS",
        "CODECLAW_CONTEXT_TOKENS", "CODECLAW_TEMPERATURE",
        "CODECLAW_DANGEROUS_PATTERNS", "CODECLAW_PROJECT_DIR",
    ):
        monkeypatch.delenv(k, raising=False)
    s = load_settings()
    assert s.ollama_host == "http://127.0.0.1:11434"
    assert s.model == "qwen2.5-coder:32b"
    assert s.max_steps == 40
    assert s.temperature == 0.2
    assert "rm -rf" in s.dangerous_patterns


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OLLAMA_HOST", "http://other:9999")
    monkeypatch.setenv("CODECLAW_MODEL", "llama3.1:8b")
    monkeypatch.setenv("CODECLAW_MAX_STEPS", "5")
    s = load_settings()
    assert s.ollama_host == "http://other:9999"
    assert s.model == "llama3.1:8b"
    assert s.max_steps == 5


def test_memory_loader_includes_present_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("# agents\n- be careful")
    (tmp_path / "MEMORY.md").write_text("# memory\n- the project is python")
    out = memory.load_project_context(str(tmp_path))
    assert "be careful" in out
    assert "the project is python" in out
    assert "AGENTS.md" in out
    assert "MEMORY.md" in out


def test_memory_loader_handles_missing_files(tmp_path):
    assert memory.load_project_context(str(tmp_path)) == ""


def test_memory_loader_caps_huge_files(tmp_path):
    (tmp_path / "AGENTS.md").write_text("x" * 100_000)
    out = memory.load_project_context(str(tmp_path))
    assert "truncated" in out


def test_settings_is_frozen():
    s = Settings()
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        s.model = "x"
