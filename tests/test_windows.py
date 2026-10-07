"""IS/OOS windowing: no bar is ever in both, and optimisation never sees OOS.

Two guarantees the whole Pass/Fail claim rests on:
  * the in-sample and out-of-sample masks are disjoint and correctly ordered,
    even when the *configured* windows overlap (validate forces OOS to start
    strictly after IS ends);
  * the refine loop optimises on IS only --- changing the OOS horizon cannot
    change which factor is chosen.
"""
import numpy as np
import pandas as pd

from harness.config import RunConfig
from harness.data import make_synthetic
from harness.factor.contract import compile_factor
from harness.runner import optimize
from harness.validate import evaluate_windows

SYMS = ["A", "B", "C", "D", "E", "F"]


def _panel(cfg):
    return make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)


def _assert_disjoint(w):
    """Assert on the ACTUAL bars evaluate_windows scored (validate exposes them),
    so removing the clamp in validate.py WOULD break this test."""
    is_idx = w["is_index"]
    oos_idx = w["oos_index"]
    assert len(is_idx) and len(oos_idx), "both windows must be populated"
    assert not set(is_idx) & set(oos_idx), "a bar is in BOTH windows"
    assert oos_idx.min() > is_idx.max(), "OOS must start strictly after IS"


def test_clock_windows_disjoint_and_ordered():
    cfg = RunConfig(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=252,
        run_name="test_windows", logging=False,
    )
    factor = compile_factor("-delta(close, w)", {"w": {"type": "Window", "value": 5}})
    w = evaluate_windows(factor, _panel(cfg), cfg)
    assert w["is_metrics"]["n"] > 50 and w["oos_metrics"]["n"] > 50
    _assert_disjoint(w)


def test_overlapping_config_still_never_leaks():
    """Even if the operator *asks* for an OOS window that overlaps IS, the
    effective OOS is clipped to start strictly after IS ends (no leakage).

    This asserts on validate's real ``oos_index``: if the ``(idx > is_end)``
    clamp were removed, oos_index would include the 2019-2020 overlap and
    ``oos_idx.min() > is_idx.max()`` would fail."""
    cfg = RunConfig(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_start="2017-01-01", is_end="2020-12-31",
        oos_start="2019-01-01", oos_end="2022-12-31",   # deliberately overlaps IS
        run_name="test_windows", logging=False,
    )
    factor = compile_factor("-delta(close, w)", {"w": {"type": "Window", "value": 5}})
    w = evaluate_windows(factor, _panel(cfg), cfg)
    _assert_disjoint(w)
    # the clamp really fired: the requested oos_start (2019) is inside IS, yet no
    # scored OOS bar is on or before is_end (2020-12-31)
    assert w["oos_index"].min() > pd.Timestamp("2020-12-31")


def test_optimize_ignores_oos_horizon():
    """The refine loop uses IS only: changing oos_days must not change the pick."""
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5,
        provider="scripted", max_iters=4, seed=17,
        run_name="test_windows", logging=False,
    )
    from harness.providers import get_provider

    cfg_a = RunConfig(oos_days=252, **base)
    cfg_b = RunConfig(oos_days=20, **base)
    panel = _panel(cfg_a)
    a = optimize(get_provider("scripted"), cfg_a, panel)
    b = optimize(get_provider("scripted"), cfg_b, panel)
    assert a["best"]["formula"] == b["best"]["formula"]
    assert np.isclose(a["best"]["is_metrics"]["sortino"],
                      b["best"]["is_metrics"]["sortino"], equal_nan=True)
