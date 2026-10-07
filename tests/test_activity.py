"""Activity floor: a degenerate empty book can never win or pass.

Before this guard, objectives that reward *inactivity* (minimise turnover, or
minimise volatility) would happily crown the empty book: a factor whose weights
are identically zero has zero turnover and zero volatility, i.e. a "perfect"
score. The fix measures average gross exposure and rejects any ~zero-exposure
book both in the optimiser (it is never selected) and in the verdict (it never
passes).
"""
import pytest


import json

from harness import objective as _obj
from harness.config import RunConfig
from harness.data import make_synthetic
from harness.providers.scripted import ScriptedProvider
from harness.runner import optimize
from harness.validate import evaluate_windows, verdict
from harness.factor.contract import compile_factor
from harness.factor.evaluate import FactorEvalError
from tests.factor_fixtures import factor_object, window_bindings

SYMS = ["A", "B", "C", "D", "E", "F"]
ZERO_FORMULA = 'ts_mean(returns,w5) - ts_mean(returns,w5)'   # identically flat


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=252,
        require_sign_consistency=False, run_name="test_activity", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def test_empty_book_fails_turnover_and_vol_objectives():
    for objective in ("avg_turnover", "ann_vol"):
        cfg = _cfg(objective=objective, pass_line=0.05)
        empty = {"n": 252, "n_active": 0, "avg_gross": 0.0,
                 "avg_turnover": 0.0, "ann_vol": 0.0, "ann_return": 0.0}
        result = verdict(empty, empty, cfg)
        assert not result["passed"]
        assert any("inactive" in reason for reason in result["reasons"])



def test_optimizer_refuses_all_empty_candidates():
    zero = json.dumps(factor_object("close - close"))
    cfg = _cfg(objective="avg_turnover", pass_line=0.05, max_iters=3)
    prov = ScriptedProvider(script=[zero] * cfg.max_iters)
    panel = make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)
    try:
        optimize(prov, cfg, panel)
    except RuntimeError as exc:
        assert "active" in str(exc)
    else:
        raise AssertionError("optimize must refuse an all-empty-book provider")


def test_real_factor_stays_active():
    """A genuine factor is active, so the floor never touches it."""
    cfg = _cfg(objective="sortino", pass_line=1.0)
    panel = make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)
    w = evaluate_windows(compile_factor("-delta(close,w5)", window_bindings("-delta(close,w5)")), panel, cfg)
    assert w["is_metrics"]["avg_gross"] > 0.0
    assert w["oos_metrics"]["avg_gross"] > 0.0


def test_activity_gate_reads_the_executed_book_at_any_gross():
    """avg_gross is the executed budget, so the gate stays honest off the default.

    Two bugs used to break the correspondence: the budget was applied twice (a
    gross=0.25 config executed a 0.0625 book) and the engine's ceiling was pinned
    at 1.0 (a gross=2.0 config executed a 1.0 book while reporting 2.0). Since
    avg_gross is the ONLY input to the gate, both made it describe a book that
    was never traded.
    """
    panel = make_synthetic(SYMS, "2015-01-01", "2023-12-31", 42)
    for g in (0.25, 2.0):
        cfg = _cfg(objective="sortino", pass_line=1.0, gross=g, seed=42)
        w = evaluate_windows(compile_factor("-delta(close,w5)", window_bindings("-delta(close,w5)")), panel, cfg)
        for tag in ("is_metrics", "oos_metrics"):
            m = w[tag]
            assert _obj.is_active(m), f"gross={g}: {tag} read as inactive"
            # Realized weights include slippage and mark-to-market drift between
            # target-percent rebalances, so they should track the budget closely
            # without being forced back to the requested target in the report.
            assert abs(m["avg_gross"] - g) < 1e-3 * max(1.0, g), \
                f"gross={g}: {tag} {m['avg_gross']}"
            # every bar of a full IS/OOS window is live -- nothing to discount
            assert m["n_active"] == m["n"]

        with pytest.raises(FactorEvalError, match="DEGENERATE"):
            evaluate_windows(compile_factor(ZERO_FORMULA, window_bindings(ZERO_FORMULA)), panel, cfg)
