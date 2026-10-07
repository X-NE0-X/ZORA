"""Current compiler admission and optional field/sign annotations.
Static rejects and their feedback replay exactly. Annotations never disable
the mandatory compiler or certify the proposed economic mechanism."""
import pytest


import tempfile
import json
from tests.factor_fixtures import window_bindings
import warnings

import numpy as np
import pandas as pd

from harness import alignment
from harness.config import RunConfig
from harness.data import FIELD_NAMES, Panel, make_synthetic
from harness.providers.base import LLMProvider, ProviderResponse, hash_prompt
from harness.providers.replay import RecordingProvider, ReplayProvider
from harness.providers.scripted import ScriptedProvider
from harness.proposer import FactorProposal
from harness.runner import _deep_eq, _replay_core, optimize, replay, run_walk_forward
from harness.store import Store

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=126, min_oos_days=20,
        provider="scripted", seed=17, max_iters=2,
        run_name="test_align", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def _panel(cfg):
    return make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)


class _CapturingProvider(LLMProvider):
    """Position-scripted provider that records what each call was asked (so a
    dead-end fed back into the next prompt can be asserted)."""

    name = "scripted"

    def __init__(self, script):
        self.script = list(script)
        self._i = 0
        self.calls: list[dict] = []

    def complete(self, system, messages, *, model, seed=17, max_tokens=2048):
        content = "\n".join(m.get("content", "") for m in messages)
        self.calls.append({"system": system, "content": content, "model": model})
        idx = min(self._i, len(self.script) - 1)
        text = self.script[idx]
        self._i += 1
        return ProviderResponse(text=text, model=model, provider=self.name,
                                prompt_hash=hash_prompt(system, messages))


def _prop(formula, sign=1, rationale="r" * 50, mechanism="m" * 50):
    return FactorProposal(formula=formula, rationale=rationale,
                          mechanism=mechanism, expected_sign=sign, raw_text="", parameters=window_bindings(formula))


# =====================================================================
# D1 --- single-monomial structure (hard reject)
# =====================================================================


# =====================================================================
# D2 --- every formula field must be named in the story (warn)
# =====================================================================
def test_d2_flags_field_absent_from_story():
    warns = alignment.check_field_mentions(
        "rank(volume)",
        rationale="A cross-sectional signal ranking names by a raw input.",
        mechanism="It sorts the universe and takes the extremes.",
    )
    assert len(warns) == 1 and "volume" in warns[0]


def test_d2_passes_when_every_field_mentioned():
    # close via "price", volume via "liquidity" (generous stem match)
    warns = alignment.check_field_mentions(
        'rank(delta(close,w5)) * rank(volume)',
        rationale="Recent price moves reverse, scaled by trading liquidity.",
        mechanism="Crowded price rallies unwind faster in liquid names.",
    )
    assert warns == []


def test_d2_word_boundary_avoids_false_match():
    # "low" must not be considered mentioned merely because "flow" appears
    warns = alignment.check_field_mentions(
        "rank(low)",
        rationale="Order flow drives the signal across the universe of names.",
        mechanism="Flow imbalance pushes the cross-section around over the week.",
    )
    assert len(warns) == 1 and "low" in warns[0]


def _vwap_panel(vwap_source: str) -> Panel:
    """A minimal panel whose ``vwap`` is declared real or proxy."""
    dates = pd.bdate_range("2020-01-01", periods=5)
    frame = pd.DataFrame(1.0, index=dates, columns=SYMS)
    return Panel(fields={name: frame.copy() for name in FIELD_NAMES},
                 symbols=list(SYMS), source="test", vwap_source=vwap_source)


def test_d2_refuses_to_bless_a_volume_story_told_about_the_hlc3_proxy():
    """The gate must not stamp "consistent" on a story about an absent input.

    When no source VWAP exists, ``vwap`` holds (high+low+close)/3 -- no volume
    term at all -- so an "institutional flow / volume-weighted execution" story
    describes a mechanism that is not in the data.
    """
    warns = alignment.check_field_mentions(
        "close - vwap",
        rationale="Price snaps back to the volume-weighted execution price.",
        mechanism="Institutional flow anchors the day's fair value.",
        panel=_vwap_panel("hlc3"),
    )
    assert any("no volume term" in w for w in warns), warns


def test_d2_accepts_the_same_volume_story_on_a_real_vwap_panel():
    # identical story, but this panel's vwap really IS volume-weighted
    warns = alignment.check_field_mentions(
        "close - vwap",
        rationale="Price snaps back to the volume-weighted execution price.",
        mechanism="Institutional flow anchors the day's fair value.",
        panel=_vwap_panel("source"),
    )
    assert warns == [], warns


def test_d2_accepts_a_typical_price_story_on_a_proxy_panel():
    warns = alignment.check_field_mentions(
        "close - vwap",
        rationale="Price reverts to the bar's typical price over the session.",
        mechanism="The HLC midpoint is a cheap fair-value anchor for the day.",
        panel=_vwap_panel("hlc3"),
    )
    assert warns == [], warns


def test_d2_defaults_to_the_proxy_when_no_panel_is_supplied():
    # the harness only ships a real VWAP when a source supplies one, so the
    # cautious reading is the default for a caller that passes no panel
    warns = alignment.check_field_mentions(
        "close - vwap",
        rationale="Deviation from the volume-weighted average price mean-reverts.",
        mechanism="Volume-weighted anchoring by execution desks.",
    )
    assert any("no volume term" in w for w in warns), warns


# =====================================================================
# D3 --- realised in-sample IC sign vs expected_sign (warn)
# =====================================================================
def _engineered_panel(mode: str) -> Panel:
    """Panel whose forward return is a known function of close's cross-sectional
    rank, so the factor ``close`` has a CONTROLLED in-sample IC.

      "aligned"      : returns[t] = rank(close[t-1]) -> IC(close, fwd) = +1
      "alternating"  : rank sign flips each date       -> mean IC ~ 0 (< eps)
    """
    dates = pd.bdate_range("2019-01-01", periods=80)
    rng = np.random.default_rng(1)
    close = pd.DataFrame(rng.normal(0.0, 1.0, (len(dates), len(SYMS))),
                         index=dates, columns=SYMS)
    ranks = close.rank(axis=1)
    if mode == "aligned":
        returns = ranks.shift(1)                       # fwd[t] = rank(close[t])
    elif mode == "alternating":
        flip = pd.Series([1 if i % 2 == 0 else -1 for i in range(len(dates))],
                         index=dates)
        returns = ranks.mul(flip, axis=0).shift(1)     # sign alternates -> mean~0
    else:
        raise ValueError(mode)
    fields = {name: close.copy() for name in FIELD_NAMES}
    fields["close"] = close
    fields["returns"] = returns
    return Panel(fields=fields, symbols=list(SYMS), source="engineered")


def _d3_cfg():
    return RunConfig(symbols=SYMS, data_source="synthetic",
                     research_date="2019-12-31", is_years=5,
                     run_name="d3", logging=False)


def test_d3_ic_recovers_engineered_sign():
    panel, cfg = _engineered_panel("aligned"), _d3_cfg()
    ic = alignment.in_sample_ic("close", panel, cfg)
    assert ic is not None and ic > 0.9            # engineered +1 correlation


def test_d3_warns_only_on_sign_contradiction():
    panel, cfg = _engineered_panel("aligned"), _d3_cfg()
    assert alignment.check_sign("close", panel, cfg, +1) == []          # agrees
    warns = alignment.check_sign("close", panel, cfg, -1)               # contradicts
    assert len(warns) == 1 and "contradicts" in warns[0]


def test_d3_silent_when_no_reliable_signal():
    # mean IC ~ 0 (< eps): can't tell, so never warn either way
    panel, cfg = _engineered_panel("alternating"), _d3_cfg()
    ic = alignment.in_sample_ic("close", panel, cfg)
    assert ic is not None and abs(ic) < alignment._IC_EPS
    assert alignment.check_sign("close", panel, cfg, +1) == []
    assert alignment.check_sign("close", panel, cfg, -1) == []


def test_d3_none_ic_for_no_cross_sectional_dispersion():
    panel, cfg = _engineered_panel("aligned"), _d3_cfg()
    panel.fields["close"] = pd.DataFrame(
        np.repeat(np.arange(len(panel.dates))[:, None], len(SYMS), axis=1),
        index=panel.dates, columns=SYMS, dtype=float)
    assert alignment.in_sample_ic("close", panel, cfg) is None
    assert alignment.check_sign("close", panel, cfg, +1) == []


def test_d3_is_strictly_point_in_time_at_the_boundary():
    # A panel that EXTENDS past is_end (has OOS bars), engineered so the forward
    # return flips sign at the boundary: in-sample targets are +rank(close), OOS
    # targets are -rank(close). A strictly point-in-time IC uses ONLY the pair
    # whose forward bar is <= is_end -> IC = +1. If it leaked the terminal pair
    # (whose forward return is the first OOS bar) or any OOS date, the -1 targets
    # would drag the mean to 0 or negative. So IC > 0.9 proves the filter works.
    dates = pd.bdate_range("2020-01-01", periods=8)
    rng = np.random.default_rng(3)
    close = pd.DataFrame(rng.normal(0.0, 1.0, (len(dates), len(SYMS))),
                         index=dates, columns=SYMS)
    ranks = close.rank(axis=1)
    is_end = dates[1]                                  # research_date == T_n
    returns = pd.DataFrame(np.nan, index=dates, columns=SYMS)
    for i in range(1, len(dates)):
        sign = 1.0 if dates[i] <= is_end else -1.0     # target bar in IS -> +, OOS -> -
        returns.iloc[i] = sign * ranks.iloc[i - 1].to_numpy()
    fields = {name: close.copy() for name in FIELD_NAMES}
    fields["close"], fields["returns"] = close, returns
    panel = Panel(fields=fields, symbols=list(SYMS), source="eng-oos")
    cfg = RunConfig(symbols=SYMS, data_source="synthetic",
                    research_date=str(is_end.date()), is_years=5,
                    run_name="d3_boundary", logging=False)
    ic = alignment.in_sample_ic("close", panel, cfg)
    assert ic is not None and ic > 0.9, \
        "D3 must exclude the terminal-date pair (first OOS return) and all OOS dates"


# =====================================================================
# runner wiring --- D1 rejects into negatives; D2/D3 attach as warnings
# =====================================================================
_NONMONO = (
    '{"formula": "rank(close) - rank(volume)", "rationale": "Buy cheap high-volume names, short expensive thin ones.", "mechanism": "Liquidity-demand premium: price level vs trading volume.", "expected_sign": 1, "parameters": {}}'
)
_MONO = (
    '{"formula": "-delta(close,w5)", "rationale": "Short-term price reversal over five days.", "mechanism": "Overreaction to recent price moves mean-reverts.", "expected_sign": -1, "parameters": {"w5": {"type": "Window", "value": 5}}}'
)
_MONO2 = (
    '{"formula": "-rank(delta(close,w5))", "rationale": "Cross-sectional five-day price reversal, ranked.", "mechanism": "Ranked overreaction to recent price moves reverts.", "expected_sign": -1, "parameters": {"w5": {"type": "Window", "value": 5}}}'
)


def test_optimize_rejects_nonmonomial_and_feeds_dead_end():
    cfg = _cfg(tri_align=True, max_iters=2)
    prov = _CapturingProvider([_NONMONO, _MONO])
    opt = optimize(prov, cfg, _panel(cfg))

    formulas = [h["formula"] for h in opt["history"]]
    assert "rank(close) - rank(volume)" not in formulas, \
        "a non-monomial must be rejected before it reaches the backtest"
    assert formulas == ['-delta(close,w5)']
    assert opt["best"]["formula"] == '-delta(close,w5)'
    # the reject is fed back to the NEXT propose as a structure dead end
    after = prov.calls[1]["content"]
    assert "Do NOT repeat these dead ends" in after
    assert "rank(close) - rank(volume)" in after
    assert "OWN_BASELINE_REQUIRED" in after


def test_annotations_off_cannot_disable_static_admission():
    cfg = _cfg(tri_align=False, max_iters=2)
    prov = _CapturingProvider([_NONMONO, _MONO])
    opt = optimize(prov, cfg, _panel(cfg))
    assert [h["formula"] for h in opt["history"]] == ["-delta(close,w5)"]


def test_optimize_attaches_d2_warning_to_history():
    # a monomial whose story never names its field is KEPT but flagged (D2 warn)
    cfg = _cfg(tri_align=True, max_iters=2)
    silent_vol = (
        '{"formula": "rank(volume)", "rationale": "Rank the raw cross-section and take the extremes.", "mechanism": "A generic sort of the universe with no story for the input.", "expected_sign": 1, "parameters": {}}'
    )
    prov = _CapturingProvider([silent_vol, _MONO])
    opt = optimize(prov, cfg, _panel(cfg))
    first = next(h for h in opt["history"] if h["formula"] == "rank(volume)")
    assert any("volume" in note for note in first["alignment"]), \
        "D2 should flag the unmentioned field on the kept candidate"
    # and the warning is fed back into the next refine prompt
    assert "alignment:" in prov.calls[1]["content"]


# =====================================================================
# D1 rejection through record -> replay (the crown-jewel invariant end-to-end)
# =====================================================================
def test_d1_reject_records_and_replays_bit_for_bit():
    # judge OFF: a D1-rejected iteration must still consume EXACTLY one completion
    # (the propose call), so the trace length stays pinned, and the whole run ---
    # rejection included --- must replay bit-for-bit from the recorded trace.
    cfg = _cfg(tri_align=True, max_iters=3)
    panel = _panel(cfg)
    rec = RecordingProvider(ScriptedProvider([_NONMONO, _MONO, _MONO2]))
    opt1 = optimize(rec, cfg, panel)
    assert len(rec.trace) == cfg.max_iters, \
        "a D1 reject must consume exactly the propose call (judge off) --- no more, no less"
    f1 = [h["formula"] for h in opt1["history"]]
    assert "rank(close) - rank(volume)" not in f1

    rep = ReplayProvider(rec.trace)
    opt2 = optimize(rep, cfg, panel)
    assert rep._i == len(rec.trace), "replay must consume every recorded completion, exactly"
    assert not rep.prompt_mismatches, "prompts (incl. the structure dead end) must regenerate identically"
    assert f1 == [h["formula"] for h in opt2["history"]]
    assert opt1["best"]["formula"] == opt2["best"]["formula"]


# =====================================================================
# record -> replay stays bit-identical with tri_align on (crown jewel)
# =====================================================================
def test_tri_align_walk_forward_replays_bit_for_bit():
    with tempfile.TemporaryDirectory() as root:
        cfg = RunConfig(
            symbols=SYMS, data_source="synthetic",
            data_start="2015-01-01", data_end="2023-12-31",
            is_years=5, oos_days=126, min_oos_days=20,
            t_0="2021-01-01", t_p="2022-01-01", frequency="YS",
            provider="scripted", seed=17, max_iters=2,
            tri_align=True, run_name="test_align_wf", logging=False,
        )
        out = run_walk_forward(cfg, artifacts_root=root)
        recorded = out["records"]
        store = Store(root, cfg.run_name)
        trace = store.read_trace()
        manifest = store.read_manifest()

        # tri_align makes NO provider call, so judge-off length is still pinned
        assert len(trace) == len(recorded) * cfg.max_iters

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            replayed = replay(manifest, trace)
        assert not any("prompt drift" in str(w.message) for w in caught)
        assert len(replayed) == len(recorded)
        for rec, rep in zip(recorded, replayed):
            assert _deep_eq(_replay_core(rec), _replay_core(rep)), \
                f"tri_align replay diverged at {rec['research_date']}"


def test_record_carries_alignment_field():
    cfg = _cfg(tri_align=True, max_iters=2, run_name="align_rec")
    from harness.runner import run_once
    with tempfile.TemporaryDirectory() as root:
        rec = run_once(cfg, store=Store(root, cfg.run_name), register=False)
    assert "alignment" in rec and isinstance(rec["alignment"], list)


# =====================================================================
# Tier-3 soft alignment folded into the Judge prompt
# =====================================================================


# =====================================================================
# config knob
# =====================================================================
def test_tri_align_defaults_on_and_roundtrips():
    assert RunConfig(symbols=SYMS).tri_align is True
    d = RunConfig(symbols=SYMS, tri_align=False).to_dict()
    assert d["tri_align"] is False
    assert RunConfig.from_dict(d).tri_align is False
