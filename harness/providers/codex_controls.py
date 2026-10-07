"""Local, credential-free validation of Codex model selection and controls."""
from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_MODELS = {
    "scripted": "claude-opus-4-8",
    "claude": "claude-opus-4-8",
    "codex": "gpt-6-luna",
    "openai": "gpt-5.6-luna",
    "deepseek": "deepseek-v4-flash",
    "opencode": "opencode-go/gpt-5.6-luna",
}
REASONING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra")
_STANDARD = ("low", "medium", "high", "xhigh")
_DOCUMENTED = {
    "gpt-6-luna": _STANDARD + ("max",),
    "gpt-5.6-luna": _STANDARD + ("max",),
    "gpt-6-sol": _STANDARD + ("max", "ultra"),
    "gpt-6.1-sol": _STANDARD + ("max", "ultra"),
    "gpt-6-astra": _STANDARD + ("max", "ultra"),
    "gpt-5.5": _STANDARD,
}


class ProviderConfigurationError(ValueError):
    """An invalid selection must fail before a request or destructive reset."""


def model_defaults(provider: str) -> str:
    try:
        return DEFAULT_MODELS[provider]
    except KeyError as exc:
        raise ProviderConfigurationError(f"unknown provider {provider!r}") from exc


def switch_defaults(values: dict, previous: str, explicit=()) -> list[str]:
    """Replace inherited defaults on a provider switch, never explicit overrides.

    Callers must surface the returned notices to make this change visible.
    """
    if previous == values["provider"]:
        return []
    notices = []
    old, new = model_defaults(previous), model_defaults(values["provider"])
    if "model" not in explicit and values.get("model") == old and new != old:
        values["model"] = new
        notices.append(f"provider default: model={new}")
    return notices


def _catalog() -> dict[str, tuple[str, ...]]:
    """Read public model metadata only; never read config.toml or auth.json.

    The installed client's cache is preferred over documented fallback values.
    Cache presence establishes capabilities, not account entitlement or login.
    """
    root = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    try:
        data = json.loads((root / "models_cache.json").read_text(encoding="utf-8"))
        models = data.get("models", [])
        result = {}
        for item in models:
            if not isinstance(item, dict) or not isinstance(item.get("slug"), str):
                continue
            levels = item.get("supported_reasoning_levels", [])
            supported = tuple(row["effort"] for row in levels
                              if isinstance(row, dict) and row.get("effort") in REASONING_LEVELS)
            if supported:
                result[item["slug"]] = supported
        return result
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def reasoning_levels(model: str) -> tuple[str, ...]:
    return _catalog().get(model, _DOCUMENTED.get(model, ()))


def validate_model(model: str | None) -> str:
    if not isinstance(model, str) or not model.strip():
        raise ProviderConfigurationError("Codex requires an explicit model id; client-default fallback is disabled")
    model = model.strip()
    if not model.startswith(("gpt", "chatgpt", "o1", "o3", "o4", "codex")) and model not in _catalog():
        raise ProviderConfigurationError(
            f"model {model!r} is incompatible with provider='codex'; select a Codex model explicitly")
    return model


def validate_controls(model: str | None, effort: str | None) -> str:
    selected = validate_model(model)
    if effort is not None:
        levels = reasoning_levels(selected)
        if effort not in levels:
            choices = ", ".join(levels) if levels else "Client default (capabilities unknown)"
            raise ProviderConfigurationError(
                f"Codex model {selected!r} does not advertise reasoning_effort={effort!r}; choose {choices}")
    return selected


def validate_config(config) -> None:
    from ..factor.protocol import require_contract
    require_contract(config.math_contract)
    model_defaults(config.provider)
    if config.provider == "opencode":
        parts = config.model.split("/", 1) if isinstance(config.model, str) else []
        if len(parts) != 2 or not all(part.strip() for part in parts):
            raise ProviderConfigurationError("OpenCode requires a provider/model id")
    if config.provider != "codex":
        return
    validate_controls(config.model, config.reasoning_effort)
