"""R01/F06: retain rejected bindings through real optimise/fold/prompt paths."""
from copy import deepcopy
import json

import pytest

from harness import proposer, runner
from harness.config import RunConfig
from harness.data import make_synthetic
from harness.factor.contract import compile_factor
from harness.factor.errors import FactorEvalError
from harness.providers.base import ProviderResponse


@pytest.mark.parametrize("kind", ["uncomputable", "structure"])
@pytest.mark.parametrize("parallel", [False, True])
def test_rejected_bindings_distinguish_actual_next_round_prompts(monkeypatch, kind, parallel):
    panel = make_synthetic(["A", "B", "C"], "2019-01-01", "2022-12-31")
    metrics = {"sortino": 1.5, "cagr": 0.1, "avg_gross": 1.0,
               "n": 252, "n_active": 252}
    rejected_formula = ("rank(ts_mean(vwap,w))" if kind == "structure"
                        else "rank(ts_mean(returns,w))")
    failure = "controlled offline IS scoring failure"
    scored = []

    def is_metrics(factor, _panel, _cfg):
        window = factor.bindings()["w"]["value"]
        scored.append(window)
        if window != 20:
            raise FactorEvalError(failure)
        return deepcopy(metrics)

    monkeypatch.setattr(runner, "_is_metrics", is_metrics)

    class Pool:
        def metrics(self, factors, _panel, _cfg):
            results = []
            for factor in factors:
                try:
                    results.append((True, is_metrics(factor, _panel, _cfg)))
                except FactorEvalError as exc:
                    results.append((False, str(exc)))
            return results

    def attempt(rejected_window):
        class Offline:
            name = "offline-rejected-binding"
            supports_structured_output = True

            def __init__(self):
                self.requests = []

            def complete(self, _system, messages, **_kwargs):
                self.requests.append(deepcopy(messages))
                first = len(self.requests) == 1
                payload = {"formula": rejected_formula if first else "rank(ts_mean(returns,w))",
                           "parameters": {"w": {"type": "Window", "value": rejected_window if first else 20}},
                           "rationale": "offline hypothesis", "mechanism": "offline mechanism",
                           "expected_sign": 1}
                return ProviderResponse(text=json.dumps(payload), model="offline", provider=self.name)

        cfg = RunConfig(symbols=panel.symbols, max_iters=2, tri_align=False,
                        research_date="2021-01-01", is_years=1, memory=False)
        provider = Offline()
        result = runner.optimize(provider, cfg, panel, sleep=lambda _: None,
                                 pool=Pool() if parallel else None)
        assert len(provider.requests) == 2
        assert result["best"]["parameters"] == {"w": {"type": "Window", "value": 20}}
        assert len(result["history"]) == 1
        prompt = provider.requests[1][0]["content"]
        line = next(line for line in prompt.splitlines() if "parameters:" in line)
        bindings = json.loads(line.split("parameters:", 1)[1])
        assert bindings == {"w": {"type": "Window", "value": rejected_window}}
        assert f"formula: {rejected_formula}" in prompt
        assert f"reason: {kind}:" in prompt
        return provider.requests[1]

    assert attempt(5) != attempt(120)
    hashes = [compile_factor(rejected_formula, {"w": {"type": "Window", "value": w}}).record()["expression_hash"]
              for w in (5, 120)]
    assert hashes[0] != hashes[1]
    if kind == "structure":
        # The real synthetic-VWAP provenance gate rejects before IS scoring.
        assert scored == [20, 20]
    else:
        assert scored == [5, 20, 120, 20]


def test_dead_end_feedback_retains_full_binding_types_and_values():
    bindings = {"w": {"type": "Window", "value": 120},
                "c": {"type": "Coefficient", "value": 0.25}}
    negative = {"formula": "rank(ts_mean(returns,w)*c)",
                "parameters": bindings, "reason": "uncomputable: controlled boundary"}
    original = deepcopy(negative)
    text = proposer._dead_ends_block([negative])
    rendered = next(line for line in text.splitlines() if "parameters:" in line)
    assert json.loads(rendered.split("parameters:", 1)[1]) == bindings
    assert negative == original


def test_dead_end_without_parameters_does_not_invent_binding_evidence():
    text = proposer._dead_ends_block([{"formula": None, "reason": "invalid candidate"}])
    assert "unparseable candidate" in text and "invalid candidate" in text
    assert "parameters:" not in text
