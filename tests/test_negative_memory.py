"""Causal negative-memory context, independent of obsolete LLM screening."""
import tempfile
from harness.config import RunConfig
from harness.proposer import build_messages
from harness.runner import run_walk_forward

SYMS = ["A", "B", "C", "D", "E", "F"]

def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=126, min_oos_days=20,
        provider="scripted", seed=17, max_iters=3,
        run_name="test_negative", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def _wf_cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2021-01-01", t_p="2022-01-01", frequency="YS",
        provider="scripted", seed=17, max_iters=2,
        run_name="test_negative_wf", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def test_dead_ends_block_injected_when_present():
    cfg = _cfg()
    negs = [{"formula": "rank(close)", "reason": "uncomputable: near-constant"},
            {"formula": None, "reason": "invalid factor formula: unknown name"}]
    with_neg = build_messages(cfg, history=None, negatives=negs)[0]["content"]
    without = build_messages(cfg, history=None, negatives=None)[0]["content"]
    assert "Do NOT repeat these dead ends" in with_neg
    assert "rank(close)" in with_neg                     # formula dead end shown
    assert "near-constant" in with_neg                   # its reason shown
    assert "unparseable candidate" in with_neg           # formula-less rejection
    assert "Do NOT repeat these dead ends" not in without


def test_prior_failures_block_injected_and_handles_nan():
    cfg = _cfg()
    fails = [{"research_date": "2020-01-01", "formula": "ts_mean(close,w250)",
              "objective": "sortino", "oos_value": -0.7,
              "reasons": ["OOS Sortino -0.7000 fails Pass Line (>= 1.0000)"]}]
    text = build_messages(cfg, history=None, prior_failures=fails)[0]["content"]
    assert "FAILED out-of-sample" in text
    assert "ts_mean(close,w250)" in text
    assert "fails Pass Line" in text
    # a NaN OOS value renders as n/a, never crashes
    nan_fail = [{"research_date": "2020-01-01", "formula": "close",
                 "objective": "sortino", "oos_value": float("nan"), "reasons": []}]
    nan_text = build_messages(cfg, history=None, prior_failures=nan_fail)[0]["content"]
    assert "n/a" in nan_text
    # absent when there are no prior failures
    assert "FAILED out-of-sample" not in \
        build_messages(cfg, history=None)[0]["content"]


def test_cross_date_failures_are_point_in_time_gated():
    # a FAILED factor becomes negative memory for LATER dates --- but only once
    # its OOS window has CLOSED, exactly like the PASS learning gate. A verdict
    # that still depends on not-yet-observable OOS bars must not leak backwards.
    with tempfile.TemporaryDirectory() as root:
        # short OOS: 2021's window closes long before 2022 -> 2022 may see it
        out = run_walk_forward(
            _wf_cfg(oos_days=126, run_name="neg_pit_shown"),
            artifacts_root=root)
        recs = out["records"]
        r21 = next(r for r in recs if r["research_date"] == "2021-01-01")
        r22 = next(r for r in recs if r["research_date"] == "2022-01-01")
        assert r21["n_prior_failures"] == 0            # first date: nothing before
        assert r22["n_prior_failures"] == (0 if r21["verdict"]["passed"] else 1)

    with tempfile.TemporaryDirectory() as root:
        # long OOS: 2021's window is still open at 2022 -> withheld (no leak)
        out = run_walk_forward(
            _wf_cfg(oos_days=300, run_name="neg_pit_hidden"),
            artifacts_root=root)
        r22 = next(r for r in out["records"]
                   if r["research_date"] == "2022-01-01")
        assert r22["n_prior_failures"] == 0, \
            "a failure whose OOS window is still open at T_n must be withheld"


