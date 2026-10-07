"""Metrics: the BacktestEngine metric map is faithfully adapted to the objective keys.

The engine numbers themselves come from the user's BacktestEngine (excess returns over
a 2% risk-free rate, annualisation inferred from the datetime index), so we do
NOT re-derive Sharpe/Sortino with a naive numpy formula. Instead we check the
adapter is a faithful, correctly-scaled translation of the raw performance map,
and that the two metrics BacktestEngine doesn't emit (hit-rate, turnover) are filled.
"""
import math
import logging

import numpy as np
import pandas as pd

from harness import backtest as backtest_module
from harness import infra_engine
from harness import objective as o
from harness.backtest import metrics, run_backtest
from harness.data import make_synthetic
from harness.factor.evaluate import evaluate

R = [0.010, -0.020, 0.030, -0.010, 0.020, 0.005, -0.015, 0.025]
IDX = pd.bdate_range("2020-01-01", periods=len(R))

SYMS = ["A", "B", "C", "D", "E", "F"]
_PANEL_CACHE: dict = {}


def _bt(returns, index=None):
    s = pd.Series(returns, index=index if index is not None
                  else pd.bdate_range("2020-01-01", periods=len(returns)))
    return {"strategy_returns": s, "turnover": pd.Series(0.0, index=s.index)}


def _panel():
    """One shared synthetic panel (the backtests below are the slow part)."""
    if "p" not in _PANEL_CACHE:
        _PANEL_CACHE["p"] = make_synthetic(SYMS, "2019-01-01", "2021-12-31", 17)
    return _PANEL_CACHE["p"]


def test_adapter_matches_engine():
    r = pd.Series(R, index=IDX)
    m = metrics(_bt(R, IDX))

    # rebuild the exact NAV metrics(...) scores, straight from BacktestEngine:
    # the SAME prepended-baseline NAV (so the first return isn't dropped) and the
    # SAME 252 trading-day annualisation that metrics(...) pins by default.
    nav = infra_engine.nav_from_returns(r)
    raw = infra_engine.performance(nav, periods_per_year=252)

    # ratios pass through unchanged
    assert abs(m["sortino"] - raw["sortino_ratio"]) < 1e-9
    assert abs(m["sharpe"] - raw["sharpe_ratio"]) < 1e-9
    assert abs(m["calmar"] - raw["calmar_ratio"]) < 1e-9
    # returns/vol: percent -> fraction; cagr mirrors ann_return
    assert abs(m["cagr"] - raw["annualized_return_rate"] / 100.0) < 1e-9
    assert m["cagr"] == m["ann_return"]
    assert abs(m["ann_vol"] - raw["yield_volatility"]) < 1e-9
    # drawdown: positive-magnitude percent -> negative fraction
    assert abs(m["maxdd"] - (-abs(raw["maxdd_rate"]) / 100.0)) < 1e-9
    assert m["maxdd"] <= 0.0
    # filled by the adapter, not BacktestEngine
    assert abs(m["hit_rate"] - (r > 0).mean()) < 1e-12
    assert m["avg_turnover"] == 0.0
    assert m["n"] == len(R)
    # sanity: this basket has a positive mean return
    assert m["sharpe"] > 0 and m["sortino"] > 0


def test_degenerate_series():
    one = metrics(_bt([0.01]))               # n < 2 -> nothing is defined
    assert one["n"] == 1
    assert math.isnan(one["sharpe"]) and math.isnan(one["sortino"])

    flat = metrics(_bt([0.0, 0.0, 0.0, 0.0]))  # zero variance
    # Sharpe: excess std is 0 -> undefined.
    assert math.isnan(flat["sharpe"])
    # Sortino is NOT nan here: against a 2% risk-free, a flat book earns a
    # negative excess with non-zero downside deviation -> a finite, negative
    # Sortino. The contract is only that a flat book is never rewarded.
    assert not (flat["sortino"] > 0)
    assert flat["n"] == 4


def test_first_return_is_not_dropped():
    """#9: nav_from_returns preserves ALL returns (no silent loss of r[0])."""
    r = pd.Series(R, index=IDX)
    nav = infra_engine.nav_from_returns(r)
    # one baseline point is prepended, so pct_change recovers every return
    assert len(nav) == len(r) + 1
    recovered = nav.pct_change().dropna().to_numpy()
    assert np.allclose(recovered, r.to_numpy(), atol=1e-12)


def test_annualization_is_trading_days():
    """#2: annualisation is pinned to 252, not the 365-calendar-day default."""
    r = pd.Series(R, index=IDX)
    nav = infra_engine.nav_from_returns(r)
    ppy252 = infra_engine.performance(nav, periods_per_year=252)
    ppy365 = infra_engine.performance(nav, periods_per_year=365)
    # a smaller periods-per-year lowers the annualised Sharpe (sqrt scaling)
    assert ppy252["sharpe_ratio"] < ppy365["sharpe_ratio"]
    # and metrics(...) uses the 252 clock by default
    m = metrics(_bt(R, IDX))
    assert abs(m["sharpe"] - ppy252["sharpe_ratio"]) < 1e-9


GROSS_SETTINGS = (0.25, 0.5, 1.0, 2.0)


def test_gross_is_applied_exactly_once():
    """correctness-01: the executed book carries sum|w| == gross, not gross**2.

    The harness used to scale the cross-section to ``gross`` AND hand the same
    ``gross`` to Position's ``gross_target``, which multiplies again -- a
    gross-squared book. It was invisible at the default (1**2 == 1) and the
    engine's row cap only ever scales an over-budget row DOWN, so gross < 1 was
    never corrected (gross=0.5 executed a 0.25 book).
    """
    panel = _panel()
    signal = evaluate("-delta(close,w)", panel, {"w": {"type": "Window", "value": 5}})
    for g in GROSS_SETTINGS:
        bt = run_backtest(signal, panel, cost_bps=3.0, gross=g)
        row_gross = bt["target_positions"].abs().sum(axis=1)
        live = row_gross[row_gross > 1e-12]
        assert len(live) > 0, f"gross={g}: book is empty"
        assert np.allclose(live.to_numpy(), g, atol=1e-12), (
            f"gross={g}: executed sum|w| in "
            f"[{live.min():.10f}, {live.max():.10f}], expected {g}"
        )


def test_run_backtest_forwards_portfolio_selection_controls():
    panel = _panel()
    signal = evaluate("close", panel)
    expected = {
        "selection_mode": "top_q",
        "top_q": 0.2,
        "hold_every": 2,
        "rebalance_every": 2,
    }
    seen = {}
    original = backtest_module.infra_engine.compute_weight_panel

    def wrapped(test_data, signal_strength, *, gross=1.0, weight_config=None):
        seen["weight_config"] = dict(weight_config or {})
        return original(test_data, signal_strength, gross=gross,
                        weight_config=weight_config)

    backtest_module.infra_engine.compute_weight_panel = wrapped
    try:
        run_backtest(signal, panel, gross=2.0, weight_config=expected)
    finally:
        backtest_module.infra_engine.compute_weight_panel = original

    assert seen["weight_config"] == expected


def test_top_k_selects_at_most_k_names_per_signed_side():
    panel = _panel()
    signal = evaluate("close", panel)
    bt = run_backtest(
        signal, panel, gross=2.0,
        weight_config={"selection_mode": "top_k", "top_k": 1},
    )
    active_names = bt["target_positions"].abs().gt(1e-12).sum(axis=1)
    assert active_names.max() <= 2


def test_portfolio_valuation_warning_filter_is_narrow():
    filt = infra_engine._PortfolioValuationWarningFilter()
    noisy = logging.LogRecord(
        "root", logging.WARNING, __file__, 1,
        "[WARNING] Portfolio valuation price panel contains %s NaN values",
        (123,), None,
    )
    other = logging.LogRecord("root", logging.WARNING, __file__, 1,
                             "other warning", (), None)
    assert filt.filter(noisy) is False
    assert filt.filter(other) is True


def test_run_backtest_forwards_close_delisted_policy():
    panel = _panel()
    signal = evaluate("close", panel)
    seen = {}
    original = backtest_module.infra_engine.run_vbt

    def wrapped(*args, **kwargs):
        seen["close_delisted_at_last"] = kwargs.get("close_delisted_at_last")
        return original(*args, **kwargs)

    backtest_module.infra_engine.run_vbt = wrapped
    try:
        run_backtest(signal, panel, close_delisted_at_last=True)
    finally:
        backtest_module.infra_engine.run_vbt = original

    assert seen["close_delisted_at_last"] is True


def test_gross_reaches_the_portfolio_and_is_reported_honestly():
    """correctness-02: the reported book is the book the engine actually ran.

    ``max_gross_exposure`` was pinned to 1.0, so the engine clamped any
    over-budget book back to 1.0 while ``avg_gross`` / ``avg_turnover`` were
    still computed from the un-clamped panel: gross=1/2/5 reported avg_gross
    0.99/1.98/4.96 with sortino and ann_vol identical to six decimals. Since
    avg_gross is the only input to the activity gate and avg_turnover is a
    first-class objective, that let a knob that moves no trade flip a verdict.
    """
    infra_engine._ensure()
    from BacktestEngine.PortfolioEngine import PortfolioEngine

    panel = _panel()
    signal = evaluate("-delta(close,w)", panel, {"w": {"type": "Window", "value": 5}})
    seen = {}
    for g in GROSS_SETTINGS:
        bt = run_backtest(signal, panel, cost_bps=3.0, gross=g)
        m = metrics(bt)
        seen[g] = m

        # Target construction stays within the requested budget, while the
        # reported gross is calculated from vectorbt's realized fill state.
        inp = PortfolioEngine.portfolio_inputs(
            panel.test_data, bt["target_positions"], cal_column="Close")
        targets = inp["weight_panel"].abs().sum(axis=1)
        assert targets.max() <= g + 1e-9, (
            f"gross={g}: target exceeds cap (max row gross {targets.max():.10f})")
        realized = bt["positions"].abs().sum(axis=1)
        assert abs(m["avg_gross"] - float(realized.mean())) < 1e-9

    # gross now actually reaches the portfolio: risk scales with the budget
    vols = [seen[g]["ann_vol"] for g in GROSS_SETTINGS]
    assert all(a < b for a, b in zip(vols, vols[1:])), f"ann_vol did not move: {vols}"
    # ... and roughly proportionally (a linear book, up to costs/rounding), so a
    # reported avg_gross / avg_turnover of 2x really is a 2x book
    assert abs(seen[2.0]["ann_vol"] / seen[1.0]["ann_vol"] - 2.0) < 0.05
    # Filled-order turnover divides realized notional by realized NAV, so costs
    # and rejected/partial fills make the ratio close to, not algebraically, 2x.
    assert abs(seen[2.0]["avg_turnover"] / seen[1.0]["avg_turnover"] - 2.0) < 0.02


def _window(returns, gross):
    """A windowed metrics dict from a hand-made return / per-bar-gross pair."""
    idx = pd.bdate_range("2021-01-01", periods=len(returns))
    return infra_engine.window_metrics(
        pd.Series(returns, index=idx),
        pd.Series(0.0, index=idx),
        pd.Series(gross, index=idx),
        periods_per_year=252,
    )


def test_no_losing_day_is_not_the_worst_score():
    """correctness-03: an ACTIVE window with no downside scores +inf, not -inf.

    Sortino's downside deviation and Calmar's max drawdown both vanish when no
    bar loses, so BacktestEngine reports NaN -- and objective.goodness maps NaN
    to -inf, which ranked the best possible window below every losing one and
    auto-FAILed it at the Pass Line.
    """
    m = _window([0.004] * 40, 1.0)
    assert m["sortino"] == math.inf and m["calmar"] == math.inf
    assert o.goodness(m["sortino"], "sortino") == math.inf
    assert o.better(m["sortino"], 3.0, "sortino")
    assert o.passes(m["sortino"], 1.0, "sortino")
    assert o.is_good(m["sortino"], "sortino")
    # the engine really did leave them NaN -- the rescue is ours, not its
    raw = infra_engine.performance(
        infra_engine.nav_from_returns(pd.Series(
            [0.004] * 40, index=pd.bdate_range("2021-01-01", periods=40))),
        periods_per_year=252)
    assert math.isnan(raw["sortino_ratio"]) and math.isnan(raw["calmar_ratio"])


def test_empty_book_keeps_the_worst_score():
    """correctness-03, other side: an INACTIVE book is never promoted to +inf.

    Same no-downside return stream, but a book that held nothing. The activity
    gate is what separates "undefined because there was no downside" from
    "uncomputable because the book was empty"; only the first is rescued.
    """
    m = _window([0.004] * 40, 0.0)
    assert math.isnan(m["sortino"]) and math.isnan(m["calmar"])
    assert o.goodness(m["sortino"], "sortino") == -math.inf
    assert not o.passes(m["sortino"], 1.0, "sortino")
    assert not o.is_active(m)

    # a genuinely flat book stays exactly as it was: against the 2% risk-free it
    # has real downside, so it was never a NaN case to begin with.
    flat = _window([0.0] * 40, 0.0)
    assert not (flat["sortino"] > 0)

    # and a metrics dict built without any gross series stays conservative:
    # with no activity measure there is nothing to justify a promotion.
    no_gross = metrics(_bt([0.004] * 40))
    assert math.isnan(no_gross["sortino"])


def test_n_active_counts_live_bars_not_calendar_bars():
    """correctness-08: `n` counts bars in the window, `n_active` bars with exposure.

    The engine forces an undefined bar's return to 0.0, so a strategy that held
    nothing for most of the window still reports a full-length `n`. `n_active`
    is the honest length -- the bars the book was actually alive for.
    """
    m = _window([0.001] * 40, [1.0] * 10 + [0.0] * 30)
    assert m["n"] == 40 and m["n_active"] == 10
    assert abs(m["avg_gross"] - 0.25) < 1e-12

    full = _window([0.001] * 40, 1.0)
    assert full["n"] == 40 and full["n_active"] == 40

    dead = _window([0.0] * 40, 0.0)
    assert dead["n"] == 40 and dead["n_active"] == 0
    # no gross series at all -> unknown, reported as 0 (and read as inactive)
    assert metrics(_bt([0.01, 0.02, 0.03]))["n_active"] == 0


def test_delisting_settlement_is_causal_and_prefix_invariant():
    panel = make_synthetic(["A", "B"], "2020-01-01", "2020-01-10", 17)
    test_data = panel.test_data.iloc[:6].copy()
    test_data.loc[test_data.index[4]:, "A_Open"] = np.nan
    targets = pd.DataFrame({"A": 0.5, "B": -0.5}, index=test_data.index)

    short = infra_engine.run_vbt(
        test_data.iloc[:5], targets.iloc[:5], cost_bps=0.0,
        close_delisted_at_last=True,
    )
    long = infra_engine.run_vbt(
        test_data, targets, cost_bps=0.0, close_delisted_at_last=True,
    )
    a = short["executed_weight_panel"]
    b = long["executed_weight_panel"].iloc[:5]
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-12)
    # The last valid-open row is untouched; settlement happens only after the
    # missing open is actually observed.
    assert abs(a.iloc[3]["A"]) > 0.49
    assert abs(a.iloc[4]["A"]) < 1e-12


def test_report_uses_realized_positions_and_filled_order_turnover():
    panel = make_synthetic(["A", "B"], "2020-01-01", "2020-01-10", 19)
    dates = panel.dates
    target = pd.DataFrame({"A": 0.5, "B": -0.5}, index=dates)
    realized = pd.DataFrame(0.0, index=dates, columns=panel.symbols)
    realized.iloc[3, 0] = 0.5
    filled_turnover = pd.Series(0.0, index=dates)
    filled_turnover.iloc[3] = 0.5
    original_weights = backtest_module.infra_engine.compute_weight_panel
    original_run = backtest_module.infra_engine.run_vbt

    def fake_weights(*_args, **_kwargs):
        return target

    def fake_run(*_args, **_kwargs):
        return {
            "returns": pd.Series(0.0, index=dates),
            "nav": pd.Series(1_000_000.0, index=dates),
            "executed_weight_panel": realized,
            "turnover": filled_turnover,
            "metric_map": {},
        }

    backtest_module.infra_engine.compute_weight_panel = fake_weights
    backtest_module.infra_engine.run_vbt = fake_run
    try:
        bt = run_backtest(evaluate("close", panel), panel)
    finally:
        backtest_module.infra_engine.compute_weight_panel = original_weights
        backtest_module.infra_engine.run_vbt = original_run

    assert bt["positions"].equals(realized)
    assert bt["target_positions"].abs().sum(axis=1).gt(0).sum() > 1
    assert metrics(bt)["n_active"] == 1
    assert bt["turnover"].equals(filled_turnover)


def test_constant_target_still_reports_real_rebalance_orders():
    panel = make_synthetic(["A", "B"], "2020-01-01", "2020-01-15", 23)
    targets = pd.DataFrame({"A": 0.5, "B": -0.5}, index=panel.dates)
    res = infra_engine.run_vbt(panel.test_data, targets, cost_bps=0.0)
    target_delta = targets.diff().abs().sum(axis=1).fillna(0.0)
    assert target_delta.iloc[1:].sum() == 0.0
    assert res["turnover"].iloc[2:].sum() > 0.0
    assert len(res["portfolio"].orders.records) > 2
