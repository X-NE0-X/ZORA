"""OpenAI-compatible API providers (lazy ``openai`` SDK import).

``OpenAIProvider`` targets the OpenAI API; ``DeepSeekProvider`` is the same
Chat Completions call pointed at DeepSeek's OpenAI-compatible endpoint. Neither
is imported unless selected, so the harness keeps no hard SDK dependency.
"""
from __future__ import annotations

import os

from .base import LLMProvider, ProviderResponse, hash_prompt
from .codex_controls import DEFAULT_MODELS
from .structured import api_schema


class OpenAIProvider(LLMProvider):
    name = "openai"
    supports_response_format = True
    supports_structured_output = True
    base_url: str | None = None
    default_model = DEFAULT_MODELS["openai"]
    api_key_env = "OPENAI_API_KEY"

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 default_model: str | None = None, timeout: float = 120.0):
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                f"provider='{self.name}' requires the openai SDK "
                "(pip install openai), or use provider='scripted'"
            ) from exc
        key = api_key or os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        if timeout <= 0:
            raise ValueError(f"timeout must be > 0, got {timeout}")
        kwargs = {"api_key": key, "timeout": float(timeout)}
        bu = base_url or self.base_url
        if bu:
            kwargs["base_url"] = bu
        self._client = openai.OpenAI(**kwargs)
        self._default_model = default_model or self.default_model

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if callable(close):
            close()

    def _resolve_model(self, model: str | None) -> str:
        """Honour the caller's model; fall back to the default only if unset.

        The provider is chosen explicitly (``provider="openai"`` vs
        ``"deepseek"``), so the model string is trusted as-is rather than being
        silently swapped for the default whenever it fails a name-prefix guess.
        """
        return model or self._default_model

    def _chat(self, model: str, msgs: list[dict], max_tokens: int,
              seed: int, temperature: float | None = 0,
              response_format: dict | None = None,
              extra_body: dict | None = None) -> tuple[str, object | None]:
        """Chat completion pinned for determinism, degrading gracefully.

        Returns ``(text, usage)``. The usage object is handed back rather than
        discarded so :meth:`complete` can put it in ``meta`` --- without it a run
        cannot be costed, and ``runner.extract_usage`` records the run as "not
        costed" rather than as "cost nothing".

        ``temperature=0`` + ``seed`` give best-effort reproducibility (OpenAI is
        deterministic only up to ``system_fingerprint``). Some reasoning models
        (o-series / gpt-5.x) reject ``temperature`` and/or ``seed``, and the token
        limit param was renamed to ``max_completion_tokens``; so if the API
        complains about a specific optional param, drop just that one and retry
        rather than failing the whole call.
        """
        token_key = "max_tokens"
        optional: dict[str, object] = {"seed": seed}
        if temperature is not None:
            optional["temperature"] = temperature
        while True:
            kwargs = {"model": model, "messages": msgs,
                      token_key: max_tokens, **optional}
            if response_format is not None:
                kwargs["response_format"] = response_format
            if extra_body is not None:
                kwargs["extra_body"] = extra_body
            try:
                comp = self._client.chat.completions.create(**kwargs)
                return (comp.choices[0].message.content or "",
                        getattr(comp, "usage", None))
            except Exception as exc:  # noqa: BLE001 - strip one rejected param, retry
                msg = str(exc)
                if token_key == "max_tokens" and (
                    "max_completion_tokens" in msg or "max_tokens" in msg
                ):
                    token_key = "max_completion_tokens"
                    continue
                dropped = next(
                    (p for p in ("temperature", "seed") if p in optional and p in msg),
                    None,
                )
                if dropped is not None:
                    optional.pop(dropped)
                    continue
                raise

    def complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                 temperature: float | None = None,
                 response_format: dict | None = None,
                 structured_schema: dict | None = None,
                 structured_retry_count: int = 2):
        return self._complete(
            system, messages, model=model, seed=seed, max_tokens=max_tokens,
            temperature=temperature, response_format=response_format,
            structured_schema=structured_schema,
        )

    def _complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                  temperature: float | None = None,
                  response_format: dict | None = None,
                  structured_schema: dict | None = None,
                  extra_body: dict | None = None):
        use_model = self._resolve_model(model)
        msgs = [{"role": "system", "content": system}]
        msgs += [{"role": m["role"], "content": m["content"]} for m in messages]
        # Direct API callers default to deterministic sampling, while still
        # allowing a caller to choose another temperature explicitly.
        temp = 0 if temperature is None else temperature
        if structured_schema is not None:
            wire_schema, hint = api_schema(structured_schema, "openai")
            msgs[0] = {**msgs[0], "content": system + hint}
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "zora_harness",
                    "strict": True,
                    "schema": wire_schema,
                },
            }
        text, usage = self._chat(
            use_model, msgs, max_tokens, seed, temp, response_format,
            extra_body,
        )
        meta = {}
        if usage is not None:
            # runner.extract_usage understands both SDK dialects; hand it the
            # OpenAI spelling verbatim rather than translating here.
            meta["usage"] = {
                k: getattr(usage, k, None)
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        return ProviderResponse(
            text=text, model=use_model, provider=self.name,
            prompt_hash=hash_prompt(system, messages), meta=meta,
        )


class DeepSeekProvider(OpenAIProvider):
    name = "deepseek"
    supports_structured_output = False
    base_url = "https://api.deepseek.com"
    default_model = DEFAULT_MODELS["deepseek"]
    api_key_env = "DEEPSEEK_API_KEY"

    def _complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                  temperature: float | None = None,
                  response_format: dict | None = None,
                  structured_schema: dict | None = None,
                  extra_body: dict | None = None):
        if structured_schema is not None and response_format is None:
            response_format = {"type": "json_object"}
        if (
            isinstance(response_format, dict)
            and response_format.get("type") == "json_object"
            and "json" not in system.lower()
        ):
            # DeepSeek JSON Output requires the prompt to mention JSON. This is
            # an internal transport hint; the original prompt hash stays stable.
            system = system + "\nReturn the final answer as valid JSON."
        return super()._complete(
            system, messages, model=model, seed=seed, max_tokens=max_tokens,
            temperature=temperature, response_format=response_format,
            # DeepSeek documents JSON-object mode, not strict JSON Schema.
            structured_schema=None,
            extra_body={"thinking": {"type": "enabled"}},
        )
