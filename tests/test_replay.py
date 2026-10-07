"""Reproducibility: same seed + config => bit-identical data and results."""
import math

from harness.config import RunConfig
from harness.data import make_synthetic
from harness.runner import run_once

SYMS = ["A", "B", "C", "D", "E", "F"]


def _num_eq(a, b) -> bool:
    """Equality that treats NaN == NaN as True (plain ``==`` says False)."""
    fa = isinstance(a, float) and math.isnan(a)
    fb = isinstance(b, float) and math.isnan(b)
    if fa or fb:
        return fa and fb
    return a == b


def _metrics_equal(d1: dict, d2: dict) -> bool:
    return set(d1) == set(d2) and all(_num_eq(d1[k], d2[k]) for k in d1)


def _cfg():
    return RunConfig(
        symbols=SYMS,
        data_source="synthetic",
        data_start="2015-01-01",
        data_end="2023-12-31",
        research_date="2021-01-01",
        is_years=5,
        oos_days=252,
        provider="scripted",
        max_iters=4,
        seed=17,
        run_name="test_replay",
        logging=False,
    )


def test_data_version_stable():
    a = make_synthetic(SYMS, "2015-01-01", "2020-12-31", seed=17)
    b = make_synthetic(SYMS, "2015-01-01", "2020-12-31", seed=17)
    assert a.version == b.version
    c = make_synthetic(SYMS, "2015-01-01", "2020-12-31", seed=18)
    assert a.version != c.version


def test_run_is_reproducible():
    r1 = run_once(_cfg())
    r2 = run_once(_cfg())
    for key in ("formula", "expected_sign", "data_version", "n_iterations"):
        assert r1[key] == r2[key], f"{key} differs across identical runs"
    assert _metrics_equal(r1["is_metrics"], r2["is_metrics"])
    assert _metrics_equal(r1["oos_metrics"], r2["oos_metrics"])
    assert r1["verdict"] == r2["verdict"]
