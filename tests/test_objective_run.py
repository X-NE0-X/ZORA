"""End-to-end runs with different objectives + interpretability on scripted data."""
from harness.config import RunConfig
from harness.runner import run_once

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=252,
        provider="scripted", seed=17, logging=False, run_name="test_obj",
    )
    base.update(kw)
    return RunConfig(**base)


def test_objective_can_be_switched():
    cases = [("sharpe", 0.5), ("maxdd", -0.2), ("ann_vol", 0.1),
             ("cagr", 0.05), ("calmar", 0.5)]
    for obj, pl in cases:
        rec = run_once(_cfg(objective=obj, pass_line=pl))
        assert rec["verdict"]["objective"] == obj
        assert rec["verdict"]["label"] in ("PASS", "FAIL")
        assert rec["objective"] == obj and rec["pass_line"] == pl


def test_interpretability_run_records_mechanism():
    rec = run_once(_cfg(require_interpretability=True))
    assert rec["require_interpretability"] is True
    assert rec["mechanism"]                   # scripted responses carry a mechanism
    assert len(rec["mechanism"]) >= 40


def test_record_distinguishes_configured_and_actual_scored_windows():
    rec = run_once(_cfg(
        is_start="2020-01-01", is_end="2021-01-01",
        oos_start="2020-12-01", oos_end="2021-06-30",
        min_is_days=20, min_oos_days=20,
    ))
    assert rec["configured_oos_start"] == "2020-12-01"
    assert rec["configured_oos_end"] == "2021-06-30"
    assert rec["oos_start"] > rec["is_end"]
    assert rec["oos_start"] != rec["configured_oos_start"]
    assert rec["is_start"] >= rec["configured_is_start"]
    assert rec["oos_end"] <= rec["configured_oos_end"]
