"""Current single-object and exact-count multi-candidate JSON workflows.
One completion serves a whole round. Individual static rejects preserve valid
siblings; ordered scoring, record shape and replay remain reproducible."""
import pytest


import json
import tempfile
import warnings

from harness import promptlib
from tests.factor_fixtures import factor_object
from harness.config import RunConfig
from harness.data import make_synthetic
from harness.proposer import (ProposalError, _extract_candidate_dicts,
                              build_messages, parse_proposal,
                              propose_many)
from harness.providers.replay import RecordingProvider, ReplayProvider
from harness.providers.scripted import ScriptedProvider
from harness.runner import (_deep_eq, _evolution_dates, _replay_core, optimize, replay,
                            run_walk_forward)
from harness.store import Store
import harness.runner as _runner

SYMS = ["A", "B", "C", "D", "E"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=126, min_oos_days=20,
        provider="scripted", seed=17, max_iters=2, run_name="test_mc",
        logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def _panel(cfg):
    return make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)


def _obj(formula, sign=1):
    return factor_object(formula, sign, "reason for " + formula, "the mechanism behind " + formula)


def _arr(formulas):
    return json.dumps({"candidates": [_obj(f) for f in formulas]})


def _with_scripted(script, fn):
    """Run ``fn`` with runner.get_provider('scripted') swapped for a scripted
    provider driven by ``script`` (so a walk-forward can be fed JSON arrays)."""
    orig = _runner.get_provider
    _runner.get_provider = (
        lambda name, **kw: ScriptedProvider(list(script)) if name == "scripted"
        else orig(name, **kw))
    try:
        return fn()
    finally:
        _runner.get_provider = orig


# --- config knob -----------------------------------------------------------
def test_candidates_per_round_default_and_validation():
    assert RunConfig().candidates_per_round == 1          # back-compat default
    RunConfig(candidates_per_round=5)                     # valid
    for bad in (0, -3):
        try:
            RunConfig(candidates_per_round=bad)
            raise AssertionError(f"candidates_per_round={bad} must be rejected")
        except ValueError:
            pass


# --- parsing ---------------------------------------------------------------


# --- prompt ----------------------------------------------------------------
def test_build_messages_k1_byte_identical_k_gt1_uses_wrapper():
    assert "multi_candidate_block" in promptlib.list_prompts()
    base = build_messages(_cfg(), None)[0]["content"]
    explicit1 = build_messages(_cfg(candidates_per_round=1), None,
                               n_candidates=1)[0]["content"]
    assert base == explicit1                               # K=1 unchanged
    assert "ARRAY" not in base
    multi = build_messages(_cfg(candidates_per_round=3), None,
                           n_candidates=3)[0]["content"]
    assert "JSON ARRAY of exactly 3" in multi
    assert '"candidates"' in multi
    assert "NOT a single object" not in multi
    assert base != multi                                    # schema is conditional


# --- propose_many ----------------------------------------------------------
def test_propose_many_single_object_schema():
    prov = ScriptedProvider([json.dumps(_obj("rank(close)"))])
    props, errors, resp = propose_many(prov, _cfg(), None)
    assert [p.formula for p in props] == ["rank(close)"] and errors == []


@pytest.mark.parametrize("breadth", [1, 2, 4])
def test_default_scripted_schema_breadth_survives_record_replay(breadth):
    config = _cfg(candidates_per_round=breadth)
    inner = ScriptedProvider()
    recorded = RecordingProvider(inner)
    proposals, errors, response = propose_many(recorded, config, None)
    assert len(proposals) == breadth and errors == []
    assert inner._i == len(recorded.trace) == 1
    for proposal in proposals:
        proposal.checked()
    replayed = ReplayProvider(recorded.trace)
    again, replay_errors, replay_response = propose_many(replayed, config, None)
    assert again == proposals and replay_errors == []
    assert replay_response.text == response.text
    assert replayed._i == 1 and replayed.prompt_mismatches == []


def test_propose_many_k_candidates_one_completion():
    cfg = _cfg(candidates_per_round=3)
    prov = RecordingProvider(ScriptedProvider(
        [_arr(["rank(close)", 'delta(close,w3)', 'ts_mean(returns,w10)'])]))
    props, errors, resp = propose_many(prov, cfg, None)
    assert [p.formula for p in props] == ["rank(close)", 'delta(close,w3)',
                                          'ts_mean(returns,w10)']
    assert errors == [] and len(prov.trace) == 1          # ONE call for K candidates


def test_propose_many_keeps_valid_and_reports_invalid():
    cfg = _cfg(candidates_per_round=3)
    payload = json.dumps({"candidates": [_obj("rank(close)"),
                          {"formula": 'delta(close,w3)'},          # missing keys
                          _obj('ts_mean(returns,w10)')]})
    props, errors, resp = propose_many(ScriptedProvider([payload]), cfg, None)
    assert [p.formula for p in props] == ["rank(close)", 'ts_mean(returns,w10)']
    assert len(errors) == 1 and "proposal keys" in errors[0]


def test_propose_many_raises_on_no_json():
    try:
        propose_many(ScriptedProvider(["sorry, no json here"]), _cfg(), None)
        raise AssertionError("must raise when nothing decodes")
    except ProposalError:
        pass


def test_propose_many_k1_byte_identical_to_single_path():
    # THE K=1 invariant, asserted by construction: for any response, propose_many at
    # candidates_per_round=1 must return the SAME proposal (or raise the SAME error)
    # as the historic propose -> parse_proposal path. Includes the adversarial nested
    # shapes where a structural parser would diverge from _extract_json: a wrapper
    # whose sibling key holds a formula object before the list, and objects that
    # carry both a formula and a candidates/factors key.
    cfg = _cfg()                                          # candidates_per_round=1
    texts = [
        json.dumps(_obj("rank(close)")),
        "```json\n" + json.dumps(_obj('ts_mean(returns,w10)')) + "\n```",
        '[{"a": 1}, ' + json.dumps(_obj('delta(close,w3)')) + ']',
        json.dumps({"example": _obj("close - open"),
                    "candidates": [_obj("rank(volume)")]}),   # sibling formula first
        json.dumps({**_obj('rank(delta(close,w3))'),
                    "factors": [{"name": "close"}]}),          # formula + wrapper key
        json.dumps({**_obj('rank(delta(close,w3))'),
                    "candidates": [_obj('ts_mean(volume,w5)')]}),
        '{"formula": ',                                        # truncated -> raises
        "sorry, no json here",                                 # no brace -> raises
        json.dumps({"candidates": []}),                        # empty wrapper -> raises
    ]
    for text in texts:
        try:
            expect = parse_proposal(text).formula
            expect_err = None
        except ProposalError as exc:
            expect, expect_err = None, str(exc)
        try:
            props, errors, _ = propose_many(ScriptedProvider([text]), cfg, None)
            got = props[0].formula if props else None
            got_err = None
            assert errors == [], (text, errors)
        except ProposalError as exc:
            got, got_err = None, str(exc)
        assert got == expect and got_err == expect_err, (text, expect, expect_err,
                                                         got, got_err)


# --- optimize breadth ------------------------------------------------------
def test_optimize_k3_scores_all_candidates_one_call_per_round():
    cfg = _cfg(candidates_per_round=3, max_iters=2)
    panel = _panel(cfg)
    script = [_arr(['-delta(close,w5)', "rank(returns)", '-rank(ts_mean(returns,w20))']),
              _arr(['ts_std(returns,w20)', "rank(volume)", '-rank(delta(close,w10))'])]
    rec = RecordingProvider(ScriptedProvider(script))
    opt = optimize(rec, cfg, panel)
    assert len(rec.trace) == cfg.max_iters                # ONE completion per round
    assert len(opt["history"]) == cfg.max_iters * 3       # every candidate scored
    assert {(h["iteration"], h["candidate"]) for h in opt["history"]} == {
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)}


def test_optimize_k1_default_unchanged():
    cfg = _cfg()                                          # candidates_per_round=1
    rec = RecordingProvider(ScriptedProvider())          # DEFAULT_SCRIPT: single objects
    opt = optimize(rec, cfg, _panel(cfg))
    assert len(rec.trace) == cfg.max_iters
    assert all(h["candidate"] == 0 for h in opt["history"])
    assert len(opt["history"]) <= cfg.max_iters


def test_optimize_k3_d1_rejects_one_candidate_others_survive():
    cfg = _cfg(candidates_per_round=3, max_iters=1, tri_align=True)
    # the middle candidate is a two-term rank spread -> D1 hard reject; the two
    # monomials survive and are scored, all from the SAME single completion.
    script = [_arr(['-delta(close,w5)', "rank(high) - rank(low)", "rank(returns)"])]
    rec = RecordingProvider(ScriptedProvider(script))
    opt = optimize(rec, cfg, _panel(cfg))
    forms = [h["formula"] for h in opt["history"]]
    assert "rank(high) - rank(low)" not in forms          # D1-rejected
    assert '-delta(close,w5)' in forms and "rank(returns)" in forms
    assert len(rec.trace) == 1                            # still one propose call


def test_optimize_k3_replays_bit_for_bit():
    cfg = _cfg(candidates_per_round=3, max_iters=2)
    panel = _panel(cfg)
    script = [_arr(['-delta(close,w5)', "rank(returns)", 'ts_std(returns,w20)'])] * cfg.max_iters
    rec = RecordingProvider(ScriptedProvider(script))
    opt1 = optimize(rec, cfg, panel)
    rep = ReplayProvider(rec.trace)
    opt2 = optimize(rep, cfg, panel)
    assert rep._i == len(rec.trace) and not rep.prompt_mismatches
    assert opt1["best"]["formula"] == opt2["best"]["formula"]


def test_optimize_k3_propose_failure_consumes_one_completion():
    # invariant A under failure: a round whose response is unparseable still
    # consumes EXACTLY one completion (the propose call fired before parsing) and
    # the search continues, so judge-off trace length stays n_dates*max_iters for
    # K>1 even when a round fails to parse.
    cfg = _cfg(candidates_per_round=3, max_iters=2)
    panel = _panel(cfg)
    script = ["sorry, no json this round",                       # round 0: fails
              _arr(['-delta(close,w5)', "rank(returns)", 'ts_std(returns,w20)'])]
    rec = RecordingProvider(ScriptedProvider(script))
    opt = optimize(rec, cfg, panel)
    assert len(rec.trace) == cfg.max_iters      # failing round still cost 1 call
    assert {h["iteration"] for h in opt["history"]} == {1}   # only round 1 scored
    assert opt["best"] is not None


# --- walk-forward record -> replay (crown jewel with breadth) ---------------
def _wf_cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2021-01-01", t_p="2022-01-01", frequency="YS",
        provider="scripted", seed=17, max_iters=2,
        run_name="mc_wf", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def _walk_script(cfg, response):
    calls = len(_evolution_dates(cfg)) * cfg.max_iters
    return [response] * calls


def test_walk_forward_k3_records_and_replays():
    with tempfile.TemporaryDirectory() as root:
        cfg = _wf_cfg(candidates_per_round=3, tri_align=True)
        script = _walk_script(cfg, _arr(['-delta(close,w5)', "rank(returns)", 'ts_std(returns,w20)']))
        out = _with_scripted(script,
                             lambda: run_walk_forward(cfg, artifacts_root=root))
        recorded = out["records"]
        store = Store(root, cfg.run_name)
        trace, manifest = store.read_trace(), store.read_manifest()
        # judge off + one call per round -> trace pinned to base regardless of K
        assert len(trace) == len(recorded) * cfg.max_iters
        replayed = replay(manifest, trace)
        assert len(replayed) == len(recorded)
        for a, b in zip(recorded, replayed):
            assert _deep_eq(_replay_core(a), _replay_core(b)), a["research_date"]


def test_k1_refine_prompt_and_record_carry_no_candidate_key():
    # invariant B, the two surfaces it explicitly names: the internal "candidate"
    # key optimize stamps on each history entry must NOT leak into (a) a refine
    # prompt rebuilt from history, nor (b) the persisted factors.jsonl record.
    cfg = _cfg()                                          # K=1
    m = {"sortino": 1.0, "sharpe": 0.5, "calmar": 0.2, "n": 100}
    with_cand = [{"iteration": 0, "candidate": 0, "formula": "rank(close)",
                  "is_metrics": m, "alignment": []}]
    without = [{"iteration": 0, "formula": "rank(close)",
                "is_metrics": m, "alignment": []}]
    # (a) the refine prompt must be byte-identical with or without the extra key
    prompt_with = build_messages(cfg, with_cand)[0]["content"]
    assert prompt_with == build_messages(cfg, without)[0]["content"]
    assert "ARRAY" not in prompt_with                    # no multi-candidate block at K=1
    # (b) the persisted record has no top-level "candidate" and the historic
    # iterations shape (one entry per history row, three fixed keys).
    with tempfile.TemporaryDirectory() as root:
        wf = _wf_cfg(candidates_per_round=1, run_name="mc_k1_rec")
        out = _with_scripted(_walk_script(wf, json.dumps(_obj('-delta(close,w5)'))),
                             lambda: run_walk_forward(wf, artifacts_root=root))
        for r in out["records"]:
            assert "candidate" not in r
            for it in r["iterations"]:
                assert set(it) == {"iteration", "formula", "is_value", "parameters", "math_contract"}


# --- CLI exposure ----------------------------------------------------------
def test_cli_flags_override_and_validate():
    from harness.cli import _load_config, build_parser
    with tempfile.TemporaryDirectory() as root:
        p = f"{root}/cfg.json"
        RunConfig(run_name="x", max_iters=8, candidates_per_round=1).to_json(p)
        args = build_parser().parse_args(
            ["run", "--config", p, "--max-iters", "5", "--candidates-per-round", "4"])
        cfg = _load_config(args)
        assert cfg.max_iters == 5 and cfg.candidates_per_round == 4
        bad = build_parser().parse_args(
            ["run", "--config", p, "--candidates-per-round", "0"])
        try:
            _load_config(bad)
            raise AssertionError("--candidates-per-round 0 must be rejected")
        except ValueError:
            pass
