"""No-look-ahead: structurally in the operators, and via the trade lag."""
import pytest


import numpy as np
import pandas as pd

from harness.backtest import metrics, run_backtest, signal_to_positions
from harness.data import make_synthetic
from harness.factor.evaluate import evaluate
from harness.factor.contract import compile_factor, ContractError

SYMS = ["A", "B", "C", "D", "E", "F"]


def _panel():
    return make_synthetic(SYMS, "2016-01-01", "2018-12-31", seed=17)


def _bt(returns: pd.Series) -> dict:
    """Wrap a bare return series as a backtest result for ``metrics``."""
    return {"strategy_returns": returns,
            "turnover": pd.Series(0.0, index=returns.index)}


def test_signal_uses_only_past():
    """Signal at date t is identical whether or not future rows exist."""
    panel = _panel()
    formula = compile_factor("ts_mean(returns,w1)/ts_std(returns,w2)", {"w1": {"type": "Window", "value": 5}, "w2": {"type": "Window", "value": 10}})
    full = evaluate(formula, panel)

    t_idx = 200
    cut = panel.slice(end=panel.dates[t_idx])
    partial = evaluate(formula, cut)

    a = full.iloc[t_idx].to_numpy()
    b = partial.iloc[-1].to_numpy()
    assert np.allclose(a, b, equal_nan=True), "signal at t changed when future was hidden"


def test_negative_lag_rejected():
    for trap in ("delay(close,w)", "delta(close,w)", "ts_mean(returns,w)"):
        with pytest.raises(ContractError):
            evaluate(trap, _panel(), {"w": {"type": "Window", "value": -1}})



def test_trade_lag_prevents_same_day_pnl():
    """Using today's return as the signal must NOT earn today's return."""
    panel = _panel()
    signal = evaluate("returns", panel)          # signal = same-day return (cheat bait)
    asset_ret = panel.fields["returns"]

    # harness: positions are lagged one bar (executed by BacktestEngine at OPEN[t])
    bt = run_backtest(signal, panel, cost_bps=0.0)
    honest = metrics(bt)

    # cheating: no lag -> buy today's winners, earn today's return
    pos = signal_to_positions(signal).reindex(
        index=asset_ret.index, columns=asset_ret.columns)
    cheat_ret = (pos * asset_ret).sum(axis=1, min_count=1)
    cheat = metrics(_bt(cheat_ret))

    assert cheat["sharpe"] > 5.0, "cheat should look spectacular"
    assert honest["sharpe"] < cheat["sharpe"] / 3.0, "lag failed to remove look-ahead"
