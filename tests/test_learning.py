"""Learning-by-doing: earlier PASS factors become proposer context for later
research dates, and the accumulation is faithfully reflected in the ledger."""
import tempfile

from harness.config import RunConfig
from harness.proposer import build_messages
from harness.runner import run_walk_forward

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2020-01-01", t_p="2022-01-01", frequency="YS",
        provider="scripted", seed=17, max_iters=2, run_name="test_learning", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


PRIOR = [
    {"research_date": "2020-01-01", "formula": "-delta(close, 5)",
     "objective": "sortino", "oos_value": 1.23, "rationale": "reversal"},
]


def test_prior_block_injected_only_when_present():
    cfg = _cfg()
    with_prior = build_messages(cfg, history=None, prior_factors=PRIOR)
    without = build_messages(cfg, history=None, prior_factors=None)

    text_with = with_prior[0]["content"]
    text_without = without[0]["content"]
    # the prior factor and its "already PASSED" framing must appear...
    assert "already PASSED" in text_with
    assert "-delta(close, 5)" in text_with
    assert "2020-01-01" in text_with
    # ...and must NOT appear when there is nothing learned yet
    assert "already PASSED" not in text_without


def test_prior_block_survives_nan_oos_value():
    # a prior with a NaN OOS value must render without crashing ("n/a")
    cfg = _cfg()
    bad = [{"research_date": "2020-01-01", "formula": "close",
            "objective": "sortino", "oos_value": float("nan")}]
    msg = build_messages(cfg, history=None, prior_factors=bad)
    assert "n/a" in msg[0]["content"]


def test_walk_forward_accumulates_prior_factors():
    with tempfile.TemporaryDirectory() as root:
        out = run_walk_forward(_cfg(), artifacts_root=root)
        recs = out["records"]
        assert len(recs) == 3

        # date k must have been shown exactly the PASS factors from dates < k
        cum_pass = 0
        for r in recs:
            assert r["n_prior_factors"] == cum_pass, (
                f"{r['research_date']}: saw {r['n_prior_factors']} priors, "
                f"expected {cum_pass}"
            )
            if r["verdict"]["passed"]:
                cum_pass += 1

        # the very first date can never have priors
        assert recs[0]["n_prior_factors"] == 0
