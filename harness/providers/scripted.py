"""Offline scripted provider --- deterministic canned responses, no network.

Lets the full loop (propose -> backtest -> iterate -> OOS -> Pass/Fail) run and
be tested without an API key. Feed it a list of response strings (typically the
JSON the proposer expects); each ``complete`` call returns the next one, holding
on the last once exhausted --- and saying so (see :meth:`ScriptedProvider.complete`).
"""
from __future__ import annotations

import json
import warnings

from .base import LLMProvider, ProviderResponse, hash_prompt

# A default script of factor proposals (JSON, exactly what proposer expects).
# Each "refines" the previous idea --- a plausible research trajectory.
#
# It is EXACTLY as long as the default ``RunConfig.max_iters`` (8) on purpose:
# this is the only thing a first-time user can run without a key, and a shorter
# script made the demo re-print the same formula for the last five rounds (the
# provider holds on its final response), which reads like the search hung. Keep
# the two in step if either changes.
DEFAULT_FORMULAS = [
    "-rank(delta(close,w))", "rank(ts_mean(returns,w))",
    "-rank(close-ts_mean(close,w))", "rank(ts_std(returns,w))",
    "rank(ts_rank(returns,w))", "-rank(delta(volume,w))",
    "rank(decay_linear(returns,w))", "-rank(ts_mean(returns,w)/ts_std(returns,w))",
]


def _candidate(index):
    return {"formula": DEFAULT_FORMULAS[index % len(DEFAULT_FORMULAS)],
            "parameters": {"w": {"type": "Window", "value": 20}},
            "rationale": "Recent movement provides a bounded testable predictive hypothesis about the next return.",
            "mechanism": "Temporary price pressure can persist or unwind; this is an unverified economic hypothesis.",
            "expected_sign": 1}


DEFAULT_SCRIPT = [json.dumps(_candidate(i)) for i in range(len(DEFAULT_FORMULAS))]


class ScriptedProvider(LLMProvider):
    name = "scripted"
    supports_structured_output = True

    def __init__(self, script: list[str] | None = None):
        self.script = list(script) if script is not None else list(DEFAULT_SCRIPT)
        self._default_script = script is None
        self._i = 0
        self._warned_exhausted = False

    def complete(
        self,
        system: str,
        messages: list[dict],
        *,
        model: str = "scripted",
        seed: int = 17,
         max_tokens: int = 10000,
        temperature: float | None = None,
        response_format: dict | None = None,
        structured_schema: dict | None = None,
        structured_retry_count: int = 2,
    ) -> ProviderResponse:
        """Return the next scripted response; hold on the last one, but say so.

        Running past the end of the script is legitimate (a short script driving
        a longer search is how several tests are written), so it must not raise
        --- but it used to be SILENT, and a silently repeated response is
        indistinguishable from a search that stopped making progress. It is now
        flagged twice: once as a ``RuntimeWarning`` the first time it happens, and
        on every such response via ``meta['exhausted']``.
        """
        if not self.script:
            raise RuntimeError("ScriptedProvider has an empty script")
        n = len(self.script)
        exhausted = self._i >= n
        idx = min(self._i, n - 1)
        text = self.script[idx]
        if self._default_script:
            count = (structured_schema or {}).get("properties", {}).get("candidates", {}).get("minItems", 1)
            items = [_candidate(idx+i) for i in range(count)]
            text = json.dumps(items[0] if count == 1 else {"candidates": items})
        self._i += 1

        if exhausted and not self._warned_exhausted:
            self._warned_exhausted = True
            warnings.warn(
                f"scripted provider exhausted: the script holds {n} response(s) "
                f"but call {self._i} was made, so this round and every later one "
                f"replays response #{n} verbatim and adds nothing to the search. "
                f"Set max_iters={n} (or lengthen the script) to make the run "
                "mean what it prints.",
                RuntimeWarning, stacklevel=2,
            )

        prompt_hash = hash_prompt(system, messages)
        return ProviderResponse(
            text=text,
            model=model,
            provider=self.name,
            prompt_hash=prompt_hash,
            meta={"script_index": idx, "exhausted": exhausted},
        )
