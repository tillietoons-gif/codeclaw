"""Tests for named LLM provider profiles."""
from __future__ import annotations

import json
from dataclasses import replace

from codeclaw.config import Settings
from codeclaw.providers import (
    PROVIDER_TEMPLATES,
    add_provider_from_template,
    apply_provider,
    load_providers,
    normalize_provider_id,
    provider_api_key_env,
    resolve_active_provider,
    save_active_provider,
)
from codeclaw import cli


def test_normalize_provider_id():
    assert normalize_provider_id("Groq") == "groq"
    assert normalize_provider_id("my-api") == "my-api"
    assert normalize_provider_id("bad id!") is None


def test_load_providers_includes_ollama(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path))
    providers = load_providers(tmp_path, settings=settings)
    assert "ollama" in providers
    assert providers["ollama"].backend == "ollama"


def test_add_provider_from_template_writes_settings(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path))
    ok, message, provider = add_provider_from_template(settings, "groq")
    assert ok
    assert provider is not None
    assert provider.backend == "openai"
    assert "groq.com" in provider.openai_base_url
    path = tmp_path / ".codeclaw" / "settings.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "groq" in data["providers"]


def test_apply_provider_switches_openai_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("CODECLAW_BACKEND", raising=False)
    monkeypatch.delenv("CODECLAW_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CODECLAW_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CODECLAW_MODEL", raising=False)
    settings = replace(Settings(), project_dir=str(tmp_path))
    add_provider_from_template(settings, "openrouter")
    applied = apply_provider(settings, "openrouter")
    assert applied is not None
    assert applied.provider == "openrouter"
    assert applied.backend == "openai"
    assert "openrouter.ai" in applied.openai_base_url
    assert applied.model == PROVIDER_TEMPLATES["openrouter"]["default_model"]


def test_provider_api_key_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("CODECLAW_OPENAI_API_KEY", raising=False)
    settings = replace(Settings(), project_dir=str(tmp_path))
    add_provider_from_template(settings, "groq")
    env_name = provider_api_key_env("groq")
    monkeypatch.setenv(env_name, "gsk-test-key")
    applied = apply_provider(settings, "groq")
    assert applied is not None
    assert applied.openai_api_key == "gsk-test-key"


def test_resolve_active_provider_from_project_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("CODECLAW_PROVIDER", raising=False)
    monkeypatch.delenv("CODECLAW_BACKEND", raising=False)
    monkeypatch.delenv("CODECLAW_OPENAI_BASE_URL", raising=False)
    settings_dir = tmp_path / ".codeclaw"
    settings_dir.mkdir()
    (settings_dir / "settings.json").write_text(json.dumps({"defaults": {"provider": "together"}, "providers": {"together": PROVIDER_TEMPLATES["together"]}}), encoding="utf-8")
    settings = replace(Settings(), project_dir=str(tmp_path))
    resolved = resolve_active_provider(settings)
    assert resolved.provider == "together"
    assert resolved.backend == "openai"
    assert "together.xyz" in resolved.openai_base_url


def test_save_active_provider_persists_default(tmp_path):
    settings = replace(Settings(), project_dir=str(tmp_path))
    save_active_provider(settings, "groq")
    data = json.loads((tmp_path / ".codeclaw" / "settings.json").read_text(encoding="utf-8"))
    assert data["defaults"]["provider"] == "groq"


def test_provider_command_args_parser():
    assert cli._provider_command_args("/provider add groq") == ("add", "groq")
    assert cli._provider_command_args("/provider groq") == ("switch", "groq")
    assert cli._provider_command_args("/providers") is None
    assert cli._provider_command_args("/provider") is None


def test_cli_is_provider_picker_command():
    assert cli._is_provider_picker_command("/providers")
    assert cli._is_provider_picker_command("/provider")
    assert not cli._is_provider_picker_command("/provider groq")
