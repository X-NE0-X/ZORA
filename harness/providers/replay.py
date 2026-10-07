"""Record + replay the LLM layer so a run reproduces without calling the model.

The harness's execution layer (formula -> backtest -> split -> verdict) is fully
deterministic given the provider's outputs, the seed, and the data version. So
if we capture every provider completion during a real run, we can re-drive the
exact same pipeline later from those cached outputs --- no model call, no LLM
cost --- and get bit-identical formulas / metrics / verdicts. This is what makes
a run auditable and provider-independent.

(Replay still reloads the market data: for ``data_source='synthetic'`` that is a
seeded, offline, byte-stable regenerate; for ``yfinance``/``ctx`` it re-reads the
source --- which may touch the network/disk and, if the upstream data changed,
yield a different data version. Replay compares that version against the recorded
one so a data drift surfaces instead of silently changing results.)

  * :class:`RecordingProvider` wraps any provider and appends each completion
    (or the error it raised) to a trace, in call order.
  * :class:`ReplayProvider` replays such a trace: it returns the cached text for
    each call in order (re-raising recorded provider errors). The answer is fixed
    by call order, not the prompt --- but it still recomputes each prompt's hash
    and flags any that no longer match the recorded one, so a trace recorded under
    a different config/prompt version doesn't reproduce silently.
"""
from __future__ import annotations

from .base import (LLMProvider, ProviderResponse, complete_provider,
                   hash_prompt)
from .cli_base import redact_secrets


class ReplayDesyncError(RuntimeError):
    """The replay trace no longer lines up with the calls being replayed.

    A hard reproducibility failure (running past the recorded completions), so it
    must NOT be swallowed as a transient provider error the way a *recorded*
    provider error is --- the caller re-raises it to abort the replay.
    """


class TracePersistenceError(RuntimeError):
    """A completion could not be durably recorded and the run must stop."""


class RecordingProvider(LLMProvider):
    """Transparent proxy that records every completion of ``inner`` in order.

    ``on_record`` (if given) is called with each trace entry as it is produced,
    so a caller can persist the trace live (survives a mid-run kill).
    """

    def __init__(self, inner: LLMProvider, on_record=None):
        self.inner = inner
        self.name = inner.name
        self.supports_response_format = getattr(
            inner, "supports_response_format", False
        )
        self.supports_structured_output = getattr(
            inner, "supports_structured_output", False
        )
        self.trace: list[dict] = []
        self._on_record = on_record

    def _record(self, entry: dict) -> None:
        clean = {
            key: redact_secrets(value) if isinstance(value, str) else value
            for key, value in entry.items()
        }
        if self._on_record is not None:
            try:
                self._on_record(clean)
            except Exception as exc:  # noqa: BLE001 - persistence is fail-stop
                raise TracePersistenceError(
                    f"LLM trace persistence failed: {type(exc).__name__}: {exc}"
                ) from exc
        self.trace.append(clean)

    def complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                 temperature: float | None = None,
                 reasoning_effort: str | None = None,
                 response_format: dict | None = None,
                 structured_schema: dict | None = None,
                 structured_retry_count: int = 2):
        try:
            resp = complete_provider(
                self.inner, system, messages, model=model, seed=seed,
                max_tokens=max_tokens, temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format=response_format, structured_schema=structured_schema,
                structured_retry_count=structured_retry_count,
            )
        except Exception as exc:  # noqa: BLE001 - faithfully record the failure
            self._record({
                "error": f"{type(exc).__name__}: {exc}",
                "prompt_hash": hash_prompt(system, messages),
                "reasoning_effort": reasoning_effort,
            })
            raise
        entry = {
            "text": resp.text,
            "model": resp.model,
            "provider": resp.provider,
            # Some custom/local providers omit the response hash. The recorder
            # owns the replay contract, so it computes the authoritative hash
            # from the exact call instead of persisting a hashless trace.
            "prompt_hash": resp.prompt_hash or hash_prompt(system, messages),
            "reasoning_effort": reasoning_effort,
        }
        if resp.meta.get("configuration"):
            entry["configuration"] = resp.meta["configuration"]
        self._record(entry)
        return resp


class ReplayProvider(LLMProvider):
    """Replay a recorded trace: return cached completions in order, no model call.

    The cached text is returned by call ORDER, not by matching the prompt, so a
    replay stays defined by the call sequence. But we DO recompute each call's
    prompt hash and compare it to the recorded one (when present): a faithful
    reproduction regenerates byte-identical prompts, so a mismatch flags that the
    trace was recorded under a different config/prompt version. Mismatches are
    collected in ``prompt_mismatches`` for the caller to surface (a warning, not a
    hard failure --- the returned completions are still the recorded ones).
    """

    name = "replay"

    def __init__(self, trace: list[dict]):
        self.trace = list(trace)
        self._i = 0
        self.prompt_mismatches: list[int] = []

    def complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                 temperature: float | None = None,
                 reasoning_effort: str | None = None,
                 response_format: dict | None = None,
                 structured_schema: dict | None = None,
                 structured_retry_count: int = 2):
        if self._i >= len(self.trace):
            raise ReplayDesyncError(
                "replay trace exhausted: more provider calls than were recorded "
                f"({len(self.trace)}); the run is not reproducible from this trace"
            )
        entry = self.trace[self._i]
        selection = entry.get("configuration", {})
        if selection.get("requested_model") is not None and selection["requested_model"] != model:
            raise ReplayDesyncError("replay requested model differs from recorded call")
        if entry.get("reasoning_effort") != reasoning_effort:
            raise ReplayDesyncError("replay reasoning_effort differs from recorded call")
        recorded_hash = entry.get("prompt_hash")
        if not isinstance(recorded_hash, str) or not recorded_hash.strip():
            raise ReplayDesyncError(
                f"replay trace entry {self._i} has no prompt_hash; prompt drift "
                "cannot be checked, so bit-identical replay is unprovable"
            )
        if hash_prompt(system, messages) != recorded_hash:
            self.prompt_mismatches.append(self._i)
        self._i += 1
        if "error" in entry:
            # reproduce the exact failure the original run saw at this call
            raise RuntimeError(f"[replayed provider error] {entry['error']}")
        return ProviderResponse(
            text=entry.get("text", ""),
            model=entry.get("model", "replay"),
            provider=entry.get("provider", self.name),
            prompt_hash=entry.get("prompt_hash", ""),
            meta={"configuration": entry["configuration"]} if "configuration" in entry else {},
        )
