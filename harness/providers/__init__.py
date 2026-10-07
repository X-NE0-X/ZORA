"""LLM provider adapters. The core never imports a vendor SDK directly.

Use :func:`get_provider` to obtain a provider by name; vendor SDKs / CLIs are
imported (or located) lazily inside their adapter so the harness runs offline
with the scripted one.

Available providers:
  scripted  --- offline canned responses (tests, demos; no network)
  claude    --- Anthropic API           (needs anthropic + ANTHROPIC_API_KEY)
  openai    --- OpenAI API              (needs openai + OPENAI_API_KEY)
  deepseek  --- DeepSeek API            (needs openai + DEEPSEEK_API_KEY)
  codex     --- Codex CLI, ChatGPT login (needs `codex` on PATH, `codex login`)
  opencode  --- opencode CLI            (needs `opencode` on PATH)
"""
from __future__ import annotations

from .base import (LLMProvider, ProviderResponse, close_provider,
                   provider_lifecycle)
from .scripted import ScriptedProvider

__all__ = [
    "LLMProvider", "ProviderResponse", "ScriptedProvider",
    "get_provider", "PROVIDERS", "close_provider", "provider_lifecycle",
]

PROVIDERS = ("scripted", "claude", "openai", "deepseek", "codex", "opencode")


def get_provider(name: str, **kwargs) -> LLMProvider:
    if name == "scripted":
        return ScriptedProvider(**kwargs)
    # Real providers read their API keys from os.environ; load this
    # installation's .env first so keys stored there are available (shell env
    # wins). Which file that is depends on the install --- env.resolve_env_path()
    # picks the vendored ENV_MGMT/.env for a source/editable checkout and a
    # per-user config path for a wheel, which must not write into site-packages.
    from ..env import load_provider_env
    load_provider_env()
    if name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(**kwargs)
    if name == "openai":
        from .openai_api import OpenAIProvider
        return OpenAIProvider(**kwargs)
    if name == "deepseek":
        from .openai_api import DeepSeekProvider
        return DeepSeekProvider(**kwargs)
    if name == "codex":
        from .codex import CodexProvider
        return CodexProvider(**kwargs)
    if name == "opencode":
        from .opencode import OpenCodeProvider
        return OpenCodeProvider(**kwargs)
    raise ValueError(f"unknown provider: {name!r}")
