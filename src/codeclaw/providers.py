"""Named LLM provider profiles (Ollama, OpenAI-compatible APIs, etc.)."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import Settings

_PROVIDER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

PROVIDER_TEMPLATES: dict[str, dict[str, Any]] = {
    "ollama": {"label": "Local Ollama", "backend": "ollama", "ollama_host": "http://127.0.0.1:11434", "default_model": "qwen2.5-coder:32b"},
    "openai": {"label": "OpenAI", "backend": "openai", "openai_base_url": "https://api.openai.com/v1", "default_model": "gpt-4o"},
    "openrouter": {"label": "OpenRouter", "backend": "openai", "openai_base_url": "https://openrouter.ai/api/v1", "default_model": "anthropic/claude-sonnet-4"},
    "groq": {"label": "Groq", "backend": "openai", "openai_base_url": "https://api.groq.com/openai/v1", "default_model": "llama-3.3-70b-versatile"},
    "together": {"label": "Together AI", "backend": "openai", "openai_base_url": "https://api.together.xyz/v1", "default_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "fireworks": {"label": "Fireworks AI", "backend": "openai", "openai_base_url": "https://api.fireworks.ai/inference/v1", "default_model": "accounts/fireworks/models/llama-v3p3-70b-instruct"},
    "lmstudio": {"label": "LM Studio (local)", "backend": "openai", "openai_base_url": "http://127.0.0.1:1234/v1", "default_model": "local-model"},
    "vllm": {"label": "vLLM (local)", "backend": "openai", "openai_base_url": "http://127.0.0.1:8000/v1", "default_model": "local-model"},
}


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    backend: str
    ollama_host: str = ""
    openai_base_url: str = ""
    openai_api_key: str = ""
    default_model: str = ""

    @classmethod
    def from_dict(cls, provider_id: str, data: dict[str, Any]) -> Provider | None:
        if not isinstance(data, dict):
            return None
        backend = str(data.get("backend") or "ollama").strip().lower()
        if backend not in ("ollama", "openai"):
            backend = "openai" if backend in ("openai_compat", "openai-compatible") else "ollama"
        return cls(
            id=provider_id,
            label=str(data.get("label") or provider_id),
            backend=backend,
            ollama_host=str(data.get("ollama_host") or ""),
            openai_base_url=str(data.get("openai_base_url") or ""),
            openai_api_key=str(data.get("openai_api_key") or ""),
            default_model=str(data.get("default_model") or data.get("model") or ""),
        )

    def to_dict(self) -> dict[str, str]:
        out: dict[str, str] = {"label": self.label, "backend": self.backend}
        if self.ollama_host:
            out["ollama_host"] = self.ollama_host
        if self.openai_base_url:
            out["openai_base_url"] = self.openai_base_url
        if self.openai_api_key:
            out["openai_api_key"] = self.openai_api_key
        if self.default_model:
            out["default_model"] = self.default_model
        return out


def normalize_provider_id(raw: str) -> str | None:
    slug = raw.strip().lower().replace(" ", "-")
    if not slug or not _PROVIDER_ID_RE.match(slug):
        return None
    return slug


def _project_settings_path(project_dir: str | Path) -> Path:
    return Path(project_dir).resolve() / ".codeclaw" / "settings.json"


def _read_project_data(project_dir: str | Path) -> dict[str, Any]:
    path = _project_settings_path(project_dir)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _builtin_ollama_provider(settings: Settings) -> Provider:
    return Provider(id="ollama", label="Local Ollama", backend="ollama", ollama_host=settings.ollama_host, default_model=settings.model)


def load_providers(project_dir: str | Path, *, settings: Settings | None = None) -> dict[str, Provider]:
    data = _read_project_data(project_dir)
    raw = data.get("providers")
    providers: dict[str, Provider] = {}
    if isinstance(raw, dict):
        for provider_id, entry in raw.items():
            norm = normalize_provider_id(str(provider_id))
            if not norm:
                continue
            provider = Provider.from_dict(norm, entry if isinstance(entry, dict) else {})
            if provider is not None:
                providers[norm] = provider
    if settings is not None:
        providers.setdefault("ollama", _builtin_ollama_provider(settings))
    elif "ollama" not in providers:
        providers["ollama"] = Provider.from_dict("ollama", PROVIDER_TEMPLATES["ollama"]) or Provider(id="ollama", label="Local Ollama", backend="ollama", ollama_host="http://127.0.0.1:11434")
    return providers


def provider_api_key_env(provider_id: str) -> str:
    return f"CODECLAW_PROVIDER_{provider_id.upper().replace('-', '_')}_API_KEY"


def _env_set(name: str) -> bool:
    return os.getenv(name) not in (None, "")


def _resolve_api_key(provider: Provider) -> str:
    env_name = provider_api_key_env(provider.id)
    if _env_set(env_name):
        return os.getenv(env_name, "")
    if _env_set("CODECLAW_OPENAI_API_KEY"):
        return os.getenv("CODECLAW_OPENAI_API_KEY", "")
    return provider.openai_api_key


def apply_provider(settings: Settings, provider_id: str) -> Settings | None:
    providers = load_providers(settings.project_dir, settings=settings)
    provider = providers.get(provider_id)
    if provider is None:
        return None
    updates: dict[str, object] = {"provider": provider_id}
    if not _env_set("CODECLAW_BACKEND"):
        updates["backend"] = provider.backend
    if provider.backend == "ollama":
        if provider.ollama_host and not _env_set("OLLAMA_HOST"):
            updates["ollama_host"] = provider.ollama_host
    else:
        if provider.openai_base_url and not _env_set("CODECLAW_OPENAI_BASE_URL"):
            updates["openai_base_url"] = provider.openai_base_url
        if not _env_set("CODECLAW_OPENAI_API_KEY"):
            updates["openai_api_key"] = _resolve_api_key(provider)
    if provider.default_model and not _env_set("CODECLAW_MODEL"):
        updates["model"] = provider.default_model
    return replace(settings, **updates)


def resolve_active_provider(settings: Settings) -> Settings:
    provider_id = (settings.provider or "").strip()
    if not provider_id:
        defaults = _read_project_data(settings.project_dir).get("defaults")
        if isinstance(defaults, dict):
            provider_id = str(defaults.get("provider") or "").strip()
    if not provider_id:
        return settings
    applied = apply_provider(replace(settings, provider=provider_id), provider_id)
    return applied if applied is not None else settings


def provider_endpoint_label(settings: Settings) -> str:
    if (settings.backend or "ollama").strip().lower() == "ollama":
        return f"Ollama at {settings.ollama_host}"
    return f"OpenAI-compatible API at {settings.openai_base_url}"


def active_provider_label(settings: Settings) -> str:
    provider_id = (settings.provider or "").strip()
    if not provider_id:
        return settings.backend
    providers = load_providers(settings.project_dir, settings=settings)
    provider = providers.get(provider_id)
    if provider is None:
        return provider_id
    return f"{provider.label} ({provider_id})"


def save_active_provider(settings: Settings, provider_id: str) -> Path:
    path = _project_settings_path(settings.project_dir)
    data = _read_project_data(settings.project_dir)
    defaults = data.setdefault("defaults", {})
    if not isinstance(defaults, dict):
        data["defaults"] = defaults = {}
    defaults["provider"] = provider_id
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def add_provider_from_template(settings: Settings, template_id: str, *, provider_id: str | None = None) -> tuple[bool, str, Provider | None]:
    norm_template = normalize_provider_id(template_id)
    if not norm_template or norm_template not in PROVIDER_TEMPLATES:
        known = ", ".join(sorted(PROVIDER_TEMPLATES))
        return False, f"Unknown provider template. Choose one of: {known}", None
    norm_id = normalize_provider_id(provider_id or norm_template)
    if not norm_id:
        return False, "Invalid provider id (use lowercase letters, numbers, - or _)", None
    template = dict(PROVIDER_TEMPLATES[norm_template])
    provider = Provider.from_dict(norm_id, template)
    if provider is None:
        return False, "Failed to build provider from template", None
    data = _read_project_data(settings.project_dir)
    providers_raw = data.setdefault("providers", {})
    if not isinstance(providers_raw, dict):
        data["providers"] = providers_raw = {}
    providers_raw[norm_id] = provider.to_dict()
    path = _project_settings_path(settings.project_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return True, f"Added provider [cyan]{norm_id}[/cyan] ({provider.label})", provider


def list_provider_templates() -> list[tuple[str, str, str]]:
    return [(pid, str(meta.get("label") or pid), str(meta.get("backend") or "openai")) for pid, meta in sorted(PROVIDER_TEMPLATES.items())]
