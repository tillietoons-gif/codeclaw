"""Model capability helpers."""
from __future__ import annotations


async def model_supports_tools(client, model: str) -> bool:
    if hasattr(client, "model_supports_tools"):
        return await client.model_supports_tools(model)
    caps = await client.model_capabilities(model)
    return "tools" in caps


async def model_context_length(client, model: str) -> int | None:
    try:
        shown = await client.show_model(model)
    except Exception:
        return None
    details = dict(shown.get("details") or {})
    model_info = shown.get("model_info") or {}
    for key in ("qwen2.context_length", "qwen3.context_length", "context_length"):
        if key in model_info:
            try:
                return int(model_info[key])
            except (TypeError, ValueError):
                continue
        if key in details:
            try:
                return int(details[key])
            except (TypeError, ValueError):
                continue
    return None
