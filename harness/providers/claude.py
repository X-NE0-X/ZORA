"""Claude provider adapter (lazy ``anthropic`` import).

Only imported when ``provider='claude'`` is selected, so the harness has no hard
dependency on the SDK. Reads the API key from ``ANTHROPIC_API_KEY``.
"""
from __future__ import annotations

import os

from .base import LLMProvider, ProviderResponse, hash_prompt
from .structured import api_schema


class ClaudeProvider(LLMProvider):
    name = "claude"
    supports_structured_output = True

    def __init__(self, api_key: str | None = None, timeout: float = 120.0):
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "provider='claude' requires the anthropic SDK "
                "(pip install anthropic), or use provider='scripted'"
            ) from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        self._client = anthropic.Anthropic(api_key=key, timeout=float(timeout))

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def complete(
        self,
        system: str,
        messages: list[dict],
        *,
        model: str,
        seed: int = 17,
         max_tokens: int = 10000,
        temperature: float | None = None,
        response_format: dict | None = None,
        structured_schema: dict | None = None,
        structured_retry_count: int = 2,
    ) -> ProviderResponse:
        # temperature=0 for maximally deterministic sampling. Note the Anthropic
        # Messages API has NO seed parameter, so ``seed`` cannot be honoured here;
        # bit-exact reproducibility of a Claude run comes from the recorded replay
        # trace, not from re-calling the model (see providers/replay.py).
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": 0 if temperature is None else temperature,
            "system": system,
            "messages": [
                {"role": m["role"], "content": m["content"]} for m in messages
            ],
        }
        if structured_schema is not None:
            wire_schema, hint = api_schema(structured_schema, "claude")
            kwargs["system"] = system + hint
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "schema": wire_schema,
                }
            }
        resp = self._client.messages.create(**kwargs)
        text = "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )
        prompt_hash = hash_prompt(system, messages)
        return ProviderResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_hash=prompt_hash,
            meta={
                "stop_reason": getattr(resp, "stop_reason", None),
                # Kept so a run can be costed; runner.extract_usage reads it.
                # Without it every run records usage=None ("not costed").
                "usage": (
                    None if getattr(resp, "usage", None) is None
                    else {k: getattr(resp.usage, k, None)
                          for k in ("input_tokens", "output_tokens")}
                ),
            },
        )
