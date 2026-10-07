"""The objective metric registry: direction, Pass Line, goodness, sign check."""
import math

from harness import objective as o


def test_direction():
    assert o.higher_is_better("sortino")
    assert o.higher_is_better("maxdd")          # stored negative; less DD is better
    assert not o.higher_is_better("ann_vol")
    assert not o.higher_is_better("avg_turnover")


def test_passes_higher_and_lower():
    assert o.passes(1.5, 1.0, "sortino")
    assert not o.passes(0.5, 1.0, "sortino")
    assert o.passes(0.10, 0.20, "ann_vol")      # lower is better
    assert not o.passes(0.30, 0.20, "ann_vol")
    assert not o.passes(float("nan"), 1.0, "sortino")


def test_goodness_and_better():
    assert o.better(2.0, 1.0, "sortino")
    assert o.better(0.10, 0.20, "ann_vol")      # lower vol is "better"
    assert not o.better(float("nan"), -5.0, "sortino")   # NaN is the worst
    assert o.goodness(float("nan"), "sortino") == -math.inf


def test_is_good_uses_neutral():
    assert o.is_good(0.5, "sortino")
    assert not o.is_good(-0.5, "sortino")
    assert o.is_good(0.6, "hit_rate")           # neutral 0.5
    assert not o.is_good(0.4, "hit_rate")


def test_pure_risk_metrics_do_not_claim_profit_from_a_finite_value():
    for name, value in (("maxdd", -0.05), ("ann_vol", 0.10),
                        ("avg_turnover", 0.20)):
        assert not o.is_good(value, name)
        assert o.profitability_metric(name) == "cagr"
        assert not o.is_profitable({name: value, "cagr": -0.50}, name)
        assert o.is_profitable({name: value, "cagr": 0.01}, name)


def test_value_missing_is_nan():
    assert math.isnan(o.value({}, "sortino"))
    assert math.isnan(o.value({"sortino": None}, "sortino"))


def test_infinite_is_the_best_score():
    """A window with no losing day scores +inf, and +inf must behave as "best".

    ``infra_engine._resolve_no_downside`` promotes the NaN that BacktestEngine
    reports for a vanished Sortino/Calmar denominator (only for an ACTIVE book),
    so the registry has to rank that above every finite score and clear any
    finite Pass Line -- otherwise the fix would just move the auto-FAIL.
    """
    assert o.goodness(math.inf, "sortino") == math.inf
    assert o.better(math.inf, 99.0, "sortino")
    assert not o.better(99.0, math.inf, "sortino")
    assert o.better(math.inf, float("nan"), "sortino")   # NaN is still the worst
    assert o.passes(math.inf, 1e9, "sortino")
    assert o.is_good(math.inf, "sortino")


def test_infinite_avg_gross_is_not_active():
    """The activity gate demands a FINITE average gross: inf/NaN is a broken book."""
    assert o.is_active({"avg_gross": 0.5})
    assert not o.is_active({"avg_gross": math.inf})
    assert not o.is_active({"avg_gross": float("nan")})
    assert not o.is_active({"avg_gross": 0.0})
    assert not o.is_active({})
