"""correctness-08: the min_oos_days floor counts bars with LIVE EXPOSURE.

The vbt engine scores a bar the book was flat on as a ``0.0`` return, not NaN,
so the plain observation count ``n`` counts *calendar* bars. A strategy that
holds exposure on three days of a sixty-bar out-of-sample window therefore used
to clear a twenty-day floor on fifty-seven vacuous zeros --- and then be judged,
Pass or Fail, on three days of evidence.

The metric itself (``n_active``) is produced by infra_engine.adapt_metrics and
tested in tests/test_metrics.py; what is tested here is the GATE in
harness/validate.py that the Pass/Fail verdict hangs off.
"""
import json

from harness.config import RunConfig
from harness.data import make_synthetic
from harness.providers.scripted import ScriptedProvider
from harness.runner import optimize
import harness.runner as _runner
from harness.validate import verdict


def _cfg(**kw):
    base = dict(min_oos_days=20, objective="sortino", pass_line=1.0,
                require_sign_consistency=False, logging=False)
    base.update(kw)
    return RunConfig(**base)


def _m(**kw):
    """A metrics dict shaped exactly like infra_engine.adapt_metrics builds one.

    Defaults describe a healthy, fully-invested window that clears every other
    gate, so each test can move ONE field and know what caused the outcome.
    """
    m = {"sortino": 2.0, "sharpe": 1.5, "calmar": 1.0, "cagr": 0.2,
         "ann_return": 0.2, "ann_vol": 0.1, "maxdd": -0.05, "hit_rate": 0.6,
         "avg_turnover": 0.2, "avg_gross": 1.0, "n": 60, "n_active": 60}
    m.update(kw)
    return m


def test_healthy_window_clears_the_floor():
    # the control: everything else in these dicts must already pass, otherwise
    # the tests below would prove nothing.
    v = verdict(_m(), _m(), _cfg())
    assert v["passed"] is True, v["reasons"]


def test_mostly_flat_book_cannot_clear_the_floor_on_zero_return_bars():
    # 60 calendar bars, exposure on 3 of them: the old check compared n (60)
    # against the floor (20) and let this through.
    v = verdict(_m(), _m(n=60, n_active=3), _cfg(min_oos_days=20))
    assert v["passed"] is False
    reason = "; ".join(v["reasons"])
    assert "live exposure" in reason
    assert "3 < 20" in reason          # what was counted, against what was needed
    assert "60 bar(s) scored" in reason  # ...and the calendar count, for context


def test_floor_is_met_exactly_at_min_oos_days_of_exposure():
    cfg = _cfg(min_oos_days=20)
    assert verdict(_m(), _m(n=60, n_active=20), cfg)["passed"] is True
    v = verdict(_m(), _m(n=60, n_active=19), cfg)
    assert v["passed"] is False
    assert any("live exposure" in r for r in v["reasons"])


def test_the_floor_reads_the_oos_window_not_the_in_sample_one():
    # a long in-sample history must never substitute for out-of-sample evidence
    v = verdict(_m(n=1260, n_active=1260), _m(n=60, n_active=5), _cfg())
    assert v["passed"] is False
    assert any("live exposure" in r for r in v["reasons"])


def test_one_live_is_bar_cannot_pass_against_a_healthy_oos_window():
    v = verdict(_m(n=1260, n_active=1), _m(n=20, n_active=20),
                _cfg(min_is_days=20, min_oos_days=20))
    assert v["passed"] is False
    reason = "; ".join(v["reasons"])
    assert "insufficient IS" in reason and "1 < 20" in reason


def test_optimizer_refuses_candidates_below_the_is_live_bar_floor():
    cfg = _cfg(min_is_days=20, max_iters=1, tri_align=False,
               symbols=["A", "B", "C", "D"], data_source="synthetic",
               data_start="2015-01-01", data_end="2023-12-31",
               research_date="2021-01-01")
    panel = make_synthetic(cfg.symbols, cfg.data_start, cfg.data_end, seed=17)
    response = json.dumps({
        "formula": "rank(close)",
        "parameters": {},
        "rationale": "cross sectional price ranking for a deterministic probe",
        "mechanism": "relative prices form the candidate signal for this probe",
        "expected_sign": 1,
    })
    real = _runner._is_metrics
    _runner._is_metrics = lambda *_args, **_kwargs: _m(n=1260, n_active=1)
    try:
        try:
            optimize(ScriptedProvider([response]), cfg, panel)
        except RuntimeError as exc:
            assert "min_is_days=20" in str(exc)
        else:
            raise AssertionError("one-live-bar IS candidate must not win optimisation")
    finally:
        _runner._is_metrics = real


def test_losing_strategy_cannot_pass_a_pure_risk_objective():
    for objective, pass_line in (("maxdd", -0.20), ("ann_vol", 0.20),
                                 ("avg_turnover", 0.50)):
        losing = _m(cagr=-0.50, ann_return=-0.50)
        v = verdict(losing, losing,
                    _cfg(objective=objective, pass_line=pass_line,
                         require_sign_consistency=True))
        assert v["passed"] is False, (objective, v)
        assert v["profitability_metric"] == "cagr"
        assert any("CAGR" in reason for reason in v["reasons"])


def test_metrics_without_n_active_prove_no_exposure():
    # a dict that cannot evidence exposure must not be given the benefit of the
    # doubt --- the same stance objective.is_active takes on a missing avg_gross.
    m = _m()
    m.pop("n_active")
    v = verdict(_m(), m, _cfg())
    assert v["passed"] is False
    assert any("live exposure" in r for r in v["reasons"])


def test_floor_failure_does_not_hide_the_other_verdict_fields():
    # the verdict is still a complete, recordable object when the floor fails
    v = verdict(_m(), _m(n_active=1), _cfg())
    assert v["label"] == "FAIL"
    assert v["objective"] == "sortino"
    assert v["oos_value"] == 2.0        # still reported, just not trusted
