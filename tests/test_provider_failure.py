"""A transient provider failure mid-search must not abort the whole run.

The optimiser calls the provider once per iteration; a network / rate-limit /
CLI crash on one iteration is logged and skipped, and the search continues with
the iterations that did succeed.
"""
import json

from harness.config import RunConfig
from harness.data import make_synthetic
from harness.providers.base import LLMProvider, ProviderResponse
from harness.runner import optimize

SYMS = ["A", "B", "C", "D", "E", "F"]
GOOD = json.dumps({"formula": "-delta(close,w)", "parameters": {"w": {"type": "Window", "value": 5}}, "rationale": "reversal",
                   "mechanism": "overreaction reverts", "expected_sign": -1})


class _FlakyProvider(LLMProvider):
    name = "flaky"

    def __init__(self):
        self.calls = 0

    def complete(self, system, messages, *, model="m", seed=17, max_tokens=2048):
        self.calls += 1
        if self.calls % 2 == 1:            # fail every odd call
            raise RuntimeError("simulated provider 500")
        return ProviderResponse(text=GOOD, model=model, provider=self.name,
                                prompt_hash="h")


class _AlwaysFails(LLMProvider):
    name = "dead"

    def complete(self, system, messages, *, model="m", seed=17, max_tokens=2048):
        raise RuntimeError("provider is completely down")


class _AuthFails(LLMProvider):
    name = "authless"

    def complete(self, system, messages, *, model="m", seed=17, max_tokens=2048):
        raise RuntimeError("Error code: 401 - authentication failed")


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=252,
        max_iters=4, run_name="test_flaky", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def test_optimize_survives_intermittent_failures():
    cfg = _cfg()
    panel = make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)
    res = optimize(_FlakyProvider(), cfg, panel)
    assert res["best"]["formula"] == "-delta(close,w)"
    # only the even (successful) calls became history entries
    assert 1 <= len(res["history"]) <= cfg.max_iters


def test_optimize_raises_when_provider_always_fails():
    cfg = _cfg()
    panel = make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)
    try:
        optimize(_AlwaysFails(), cfg, panel)
    except RuntimeError:
        return
    raise AssertionError("optimize must raise when no proposal ever succeeds")


def test_optimize_surfaces_auth_error_not_generic():
    # An auth failure (bad key / lapsed CLI login) is not transient: it must
    # surface as ITSELF so the onboarding layer (preflight.explain_failure) can
    # turn it into a setup guide -- NOT be retried to max_iters and replaced by
    # the generic "no valid active factor proposal" error (which has no auth
    # hint and would leave the user with a misleading message).
    from harness import preflight
    cfg = _cfg()
    panel = make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)
    try:
        optimize(_AuthFails(), cfg, panel)
    except Exception as exc:  # noqa: BLE001
        assert preflight.looks_like_auth_error(str(exc))
        assert "no valid active factor" not in str(exc)
        return
    raise AssertionError("optimize must surface the auth error for onboarding")
