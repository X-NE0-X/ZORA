"""A factor with no cross-sectional information must FAIL (null check)."""
import pytest


from harness.config import RunConfig
from harness.data import make_synthetic
from harness.validate import evaluate_windows, verdict

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg():
    return RunConfig(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=252,
        provider="scripted", seed=17, run_name="test_no_signal", logging=False,
    )


def test_zero_dispersion_factor_fails():
    cfg = _cfg()
    panel = make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)
    # identically zero cross-section -> no positions -> no pnl
    # Valid time-varying signal, identical across names: numerical coverage
    # passes, yet it cannot create cross-sectional portfolio exposure.
    panel.fields["close"].iloc[:, :] = panel.fields["close"].iloc[:, 0].to_numpy()[:, None]
    w = evaluate_windows("close", panel, cfg)
    v = verdict(w["is_metrics"], w["oos_metrics"], cfg)
    assert not v["passed"], "a no-signal factor must not PASS"
    assert v["label"] == "FAIL"
    # A flat book holds no cross-sectional edge, so the objective must not land
    # on the profitable side: OOS Sortino is never positive (against a 2%
    # risk-free it is a finite negative, not nan).
    assert not (w["oos_metrics"]["sortino"] > 0)
