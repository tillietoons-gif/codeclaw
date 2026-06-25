"""Factory for LLM backend clients."""
from __future__ import annotations

from typing import Any

from .config import Settings
from .ollama import OllamaClient
from .openai_compat import OpenAICompatClient


def create_llm_client(settings: Settings) -> Any:
    backend = (settings.backend or "ollama").strip().lower()
    if backend in ("openai", "openai_compat", "openai-compatible"):
        return OpenAICompatClient(
            base_url=settings.openai_base_url or "http://127.0.0.1:8000",
            api_key=settings.openai_api_key,
            timeout_s=settings.request_timeout_s,
            model=settings.model,
        )
    return OllamaClient(settings.ollama_host, timeout_s=settings.request_timeout_s)
