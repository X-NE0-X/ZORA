"""Causal cross-date learning and replay-integrity regressions.
Numeric operator semantics and unsafe syntax are owned by test_strict_math."""
import pytest

import math
import tempfile
import warnings

import pandas as pd

from harness.config import RunConfig
from harness.runner import _prior_oos_closed, replay, run_walk_forward
from harness.store import Store

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2021-01-01", t_p="2022-01-01", frequency="YS",
        provider="scripted", seed=17, max_iters=2,
        run_name="test_review2", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


# --- M4: point-in-time learning gate ---------------------------------------
def test_prior_oos_closed_gate():
    prior = {"research_date": "2021-01-01"}
    tn = pd.Timestamp("2022-01-01")
    # short OOS closed ~2021-06 -> available; long OOS still open ~2022-02 -> not
    assert _prior_oos_closed(prior, _cfg(oos_days=126), tn) is True
    assert _prior_oos_closed(prior, _cfg(oos_days=300), tn) is False


def test_learning_withholds_still_open_oos_prior():
    # oos_days=300: date-1 (2021) OOS window closes 2022-02-25, AFTER date-2
    # (2022-01-01) -- so 2022 must see 0 priors even if 2021 passed (no leak).
    with tempfile.TemporaryDirectory() as root:
        out = run_walk_forward(
            _cfg(oos_days=300, run_name="rev2_leak"), artifacts_root=root)
        rec_2022 = next(r for r in out["records"]
                        if r["research_date"] == "2022-01-01")
        assert rec_2022["n_prior_factors"] == 0, \
            "a prior whose OOS window is still open at T_n must be withheld"


def test_learning_shows_closed_oos_prior():
    # the same schedule with the default short OOS: 2021's window closes long
    # before 2022, so if 2021 passed it IS shown (gate is a no-op here).
    with tempfile.TemporaryDirectory() as root:
        out = run_walk_forward(
            _cfg(oos_days=126, run_name="rev2_noleak"), artifacts_root=root)
        recs = out["records"]
        rec_2021 = next(r for r in recs if r["research_date"] == "2021-01-01")
        rec_2022 = next(r for r in recs if r["research_date"] == "2022-01-01")
        expected = 1 if rec_2021["verdict"]["passed"] else 0
        assert rec_2022["n_prior_factors"] == expected


# --- replay length / desync hardening --------------------------------------
def test_replay_rejects_wrong_length_trace():
    with tempfile.TemporaryDirectory() as root:
        run_walk_forward(_cfg(run_name="rev2_len"), artifacts_root=root)
        store = Store(root, "rev2_len")
        manifest = store.read_manifest()
        trace = store.read_trace()
        for bad in (trace[:-1], trace + trace[:1]):   # one short, one long
            try:
                replay(manifest, bad)
            except ValueError as exc:
                assert "trace" in str(exc)
            else:
                raise AssertionError("replay must reject a mismatched trace length")
        # the exact-length trace still replays fine
        assert len(replay(manifest, trace)) == manifest["n_factors"]


def test_replay_warns_on_prompt_drift():
    # a recorded run replays cleanly; corrupting one entry's prompt_hash makes the
    # regenerated prompt no longer match -> a prompt-drift warning (not a failure).
    with tempfile.TemporaryDirectory() as root:
        run_walk_forward(_cfg(run_name="rev2_phash"), artifacts_root=root)
        store = Store(root, "rev2_phash")
        manifest = store.read_manifest()
        trace = store.read_trace()
        with warnings.catch_warnings(record=True) as clean:
            warnings.simplefilter("always")
            replay(manifest, trace)
        assert not any("prompt drift" in str(w.message) for w in clean), \
            "a faithful replay must NOT warn about prompt drift"

        trace[0] = {**trace[0], "prompt_hash": "DEADBEEF-not-a-real-hash"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            replay(manifest, trace)
        assert any("prompt drift" in str(w.message) for w in caught), \
            "replay must warn when a recorded prompt hash no longer matches"
