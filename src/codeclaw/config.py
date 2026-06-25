"""Runtime configuration loaded from environment variables.

All settings can be overridden via `.env` or the shell. The defaults assume
Ollama is running locally on the default port.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=False)


def _env(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None and val != "" else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    provider: str = field(default_factory=lambda: _env("CODECLAW_PROVIDER", ""))
    backend: str = field(default_factory=lambda: _env("CODECLAW_BACKEND", "ollama"))
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    openai_base_url: str = field(default_factory=lambda: _env("CODECLAW_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1"))
    openai_api_key: str = field(default_factory=lambda: _env("CODECLAW_OPENAI_API_KEY", ""))
    model: str = field(default_factory=lambda: _env("CODECLAW_MODEL", "qwen2.5-coder:32b"))
    max_steps: int = field(default_factory=lambda: _env_int("CODECLAW_MAX_STEPS", 40))
    context_tokens: int = field(default_factory=lambda: _env_int("CODECLAW_CONTEXT_TOKENS", 24000))
    temperature: float = field(default_factory=lambda: _env_float("CODECLAW_TEMPERATURE", 0.2))
    project_dir: str = field(default_factory=lambda: _env("CODECLAW_PROJECT_DIR", "."))
    request_timeout_s: float = field(default_factory=lambda: _env_float("CODECLAW_REQUEST_TIMEOUT", 300.0))


def load_settings() -> Settings:
    """Build a fresh Settings from the current environment."""
    settings = Settings()
    settings = _apply_project_defaults(settings)
    settings = _resolve_project_dir(settings)
    from .providers import resolve_active_provider
    return resolve_active_provider(settings)


def _resolve_project_dir(settings: Settings) -> Settings:
    from dataclasses import replace
    raw = settings.project_dir.strip() or "."
    resolved = str(Path(raw).resolve()) if raw != "." else str(Path.cwd().resolve())
    if resolved == settings.project_dir:
        return settings
    return replace(settings, project_dir=resolved)


def _apply_project_defaults(settings: Settings) -> Settings:
    path = Path(settings.project_dir).resolve() / ".codeclaw" / "settings.json"
    if not path.exists():
        return settings
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return settings
    defaults = data.get("defaults") if isinstance(data, dict) else None
    if not isinstance(defaults, dict):
        return settings
    overrides = {}
    env_map = {
        "provider": "CODECLAW_PROVIDER",
        "backend": "CODECLAW_BACKEND",
        "ollama_host": "OLLAMA_HOST",
        "openai_base_url": "CODECLAW_OPENAI_BASE_URL",
        "openai_api_key": "CODECLAW_OPENAI_API_KEY",
        "model": "CODECLAW_MODEL",
        "max_steps": "CODECLAW_MAX_STEPS",
        "context_tokens": "CODECLAW_CONTEXT_TOKENS",
        "temperature": "CODECLAW_TEMPERATURE",
        "request_timeout_s": "CODECLAW_REQUEST_TIMEOUT",
    }
    for field_name, env_name in env_map.items():
        if os.getenv(env_name) not in (None, "") or field_name not in defaults:
            continue
        current = getattr(settings, field_name)
        raw = defaults[field_name]
        try:
            overrides[field_name] = type(current)(raw)
        except (TypeError, ValueError):
            continue
    if not overrides:
        return settings
    from dataclasses import replace
    return replace(settings, **overrides)
