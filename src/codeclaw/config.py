"""Runtime configuration loaded from environment variables.

All settings can be overridden via `.env` or the shell. The defaults assume
Ollama is running locally on the default port.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the current working directory if present, then from
# the project root used at install time. This lets a user `cd` into a
# project and have its `.env` take effect.
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
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model: str = field(default_factory=lambda: _env("CODECLAW_MODEL", "qwen2.5-coder:32b"))
    max_steps: int = field(default_factory=lambda: _env_int("CODECLAW_MAX_STEPS", 40))
    context_tokens: int = field(default_factory=lambda: _env_int("CODECLAW_CONTEXT_TOKENS", 24000))
    temperature: float = field(default_factory=lambda: _env_float("CODECLAW_TEMPERATURE", 0.2))
    project_dir: str = field(default_factory=lambda: _env("CODECLAW_PROJECT_DIR", "."))
    request_timeout_s: float = field(default_factory=lambda: _env_float("CODECLAW_REQUEST_TIMEOUT", 300.0))


def load_settings() -> Settings:
    """Build a fresh Settings from the current environment."""
    return Settings()
