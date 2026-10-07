"""The Pass branch fires: a factor with genuine, stationary edge passes OOS.

Guards against the gate silently degrading into an always-FAIL check. We plant a
stationary reversal (negative AR(1)) into the returns, so the reversal factor
``-returns`` has real predictive power both in- and out-of-sample.
"""
import numpy as np
import pandas as pd

from harness.config import RunConfig
from harness.data import _panel_from_ohlcv
from harness.validate import evaluate_windows, verdict

SYMS = ["A", "B", "C", "D", "E", "F"]


def _planted_panel():
    dates = pd.bdate_range("2015-01-01", "2023-12-31")
    t, n = len(dates), len(SYMS)
    rng = np.random.default_rng(7)
    noise = rng.normal(0.0, 0.01, (t, n))
    ret = np.zeros((t, n))
    for i in range(1, t):
        ret[i] = -0.5 * ret[i - 1] + noise[i]      # stationary reversal
    close = pd.DataFrame(100 * np.cumprod(1 + ret, axis=0), index=dates, columns=SYMS)
    prev = close.shift(1)
    prev.iloc[0] = close.iloc[0]
    high = np.maximum(prev, close)
    low = np.minimum(prev, close)
    vol = pd.DataFrame(1e6, index=dates, columns=SYMS)
    return _panel_from_ohlcv(prev, high, low, close, vol, SYMS, "planted")


def test_edge_factor_passes():
    cfg = RunConfig(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2022-01-01", is_years=5, oos_days=252,
        objective="sortino", pass_line=1.0,
        require_sign_consistency=True, min_oos_days=20,
        run_name="test_pass", logging=False,
    )
    panel = _planted_panel()
    w = evaluate_windows("-returns", panel, cfg)
    v = verdict(w["is_metrics"], w["oos_metrics"], cfg)

    assert v["passed"], f"a factor with real edge must PASS, got {v}"
    assert v["label"] == "PASS"
    assert w["is_metrics"]["sortino"] > 0 and w["oos_metrics"]["sortino"] >= 1.0
