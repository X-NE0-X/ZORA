"""Provider protocol --- the single seam between the harness and any LLM.

The harness only ever calls ``complete(...)`` and reads ``.text`` off the
response. The LLM appears exclusively in the propose/iterate step; it never
touches backtesting, statistics, or the Pass/Fail decision.
"""
from __future__ import annotations

import hashlib
import inspect
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field


def hash_prompt(system: str, messages: list[dict]) -> str:
    """Stable content hash of a prompt (system + all message contents)."""
    payload = system + "\n" + "\n".join(m.get("content", "") for m in messages)
    return hashlib.blake2b(payload.encode(), digest_size=16).hexdigest()


@dataclass
class ProviderResponse:
    text: str
    model: str
    provider: str
    prompt_hash: str = ""
    meta: dict = field(default_factory=dict)


class LLMProvider(ABC):
    name: str = "base"
    supports_response_format: bool = False
    supports_structured_output: bool = False

    @abstractmethod
    def complete(
        self,
        system: str,
        messages: list[dict],
        *,
        model: str,
        seed: int = 17,
        max_tokens: int = 10000,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        response_format: dict | None = None,
        structured_schema: dict | None = None,
        structured_retry_count: int = 2,
    ) -> ProviderResponse:
        """Return a completion. ``messages`` is a list of {role, content} dicts."""
        raise NotImplementedError


def close_provider(provider) -> None:
    """Close a provider if it owns a network client or child process."""
    close = getattr(provider, "close", None)
    if callable(close):
        close()


@contextmanager
def provider_lifecycle(provider):
    """Bound provider resources to one run/call lifecycle."""
    try:
        yield provider
    finally:
        close_provider(provider)


def complete_provider(
    provider: LLMProvider,
    system: str,
    messages: list[dict],
    *,
    model: str,
    seed: int = 17,
    max_tokens: int = 10000,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    response_format: dict | None = None,
    structured_schema: dict | None = None,
    structured_retry_count: int = 2,
) -> ProviderResponse:
    """Call a provider while preserving compatibility with small test adapters.

    Third-party and test providers may implement the historic ``complete``
     signature without newer optional controls. Only pass a control when the
     adapter advertises the argument; native schema output additionally requires
     the explicit ``supports_structured_output`` capability.
    """
    from ..cancellation import check_cancelled
    check_cancelled()
    fn = provider.complete
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual callable
        params = {}
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )

    kwargs = {
        "model": model,
        "seed": seed,
        "max_tokens": max_tokens,
    }
    if temperature is not None and ("temperature" in params or accepts_kwargs):
        kwargs["temperature"] = temperature
    if reasoning_effort is not None and ("reasoning_effort" in params or accepts_kwargs):
        kwargs["reasoning_effort"] = reasoning_effort
    if (
        response_format is not None
        and getattr(provider, "supports_response_format", False)
        and ("response_format" in params or accepts_kwargs)
    ):
        kwargs["response_format"] = response_format
    if (
        structured_schema is not None
        and getattr(provider, "supports_structured_output", False)
        and ("structured_schema" in params or accepts_kwargs)
    ):
        kwargs["structured_schema"] = structured_schema
        if "structured_retry_count" in params or accepts_kwargs:
            kwargs["structured_retry_count"] = structured_retry_count
    response = fn(system, messages, **kwargs)
    check_cancelled()
    return response
