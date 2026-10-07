"""Strict v2 acceptance, numerical semantics and execution boundaries."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from harness.config import RunConfig
from harness.data import make_synthetic
from harness.factor.contract import compile_factor, ContractError
from harness.factor.evaluate import evaluate, FactorEvalError
from harness.factor.checked_evaluate import ResearchScope
from harness.factor.protocol import STRICT
from harness.proposer import parse_proposal, ProposalError


def panel():
    return make_synthetic(["A", "B", "C"], "2020-01-01", "2022-12-31", seed=17)


def window(value=20):
    return {"w": {"type": "Window", "value": value}}


CASES = json.loads(Path(__file__).with_name("strict_math_cases.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_acceptance_spec(case):
    data = panel()
    data.vwap_source = case.get("context", {}).get("vwap_source", "hlc3")
    if case["expected"] == "REJECT":
        with pytest.raises(ContractError):
            compile_factor(case["formula"], case["parameters"], panel=data)
    else:
        checked = compile_factor(case["formula"], case["parameters"], panel=data)
        assert len(checked.parameters) == case.get("expected_parameter_count", len(case["parameters"]))
        result = evaluate(checked, data)
        assert not np.isinf(result.to_numpy()).any()


def test_no_direct_literal_bypass():
    with pytest.raises(ContractError):
        evaluate("ts_mean(close,20)", panel())


def test_overflow_cannot_be_hidden_by_rank():
    with pytest.raises(FactorEvalError, match="NONFINITE"):
        evaluate("rank(close**p)", panel(), {"p": {"type": "Exponent", "value": 200}})


def test_window_and_warmup():
    data = panel()
    result = evaluate("ts_mean(close,w)", data, window())
    assert result.iloc[:19].isna().all().all()
    pd.testing.assert_frame_equal(result, data.fields["close"].rolling(20).mean(), check_flags=False)


def test_ordinary_power_and_explicit_signed_power():
    from harness.factor.contract import SPECS
    data = panel()
    p = {"p": {"type": "Exponent", "value": 0}}
    ordinary = SPECS["pow"].kernel(data.fields["returns"], 0)
    signed = evaluate("signed_pow(returns,p)", data, p)
    assert ordinary.iloc[1:].eq(1).all().all()
    pd.testing.assert_frame_equal(signed, np.sign(data.fields["returns"]), check_flags=False)
    with pytest.raises(FactorEvalError, match="DEGENERATE"):
        evaluate("returns**p", data, p)


def test_unexpected_nan_rejected_but_declared_zero_variance_masked():
    data = panel()
    data.fields["close"].iloc[:, :] = 1
    with pytest.raises(FactorEvalError, match="VALID_COVERAGE"):
        evaluate("zscore(close)", data)
    with pytest.raises(FactorEvalError, match="DEGENERATE"):
        evaluate("rank(close)", data)


def test_checked_ir_tamper_detection():
    from dataclasses import replace
    checked = compile_factor("ts_mean(close,w)", window())
    with pytest.raises(ContractError, match="IR_MISMATCH"):
        evaluate(replace(checked, node=compile_factor("volume").node), panel())


@pytest.mark.parametrize("change", [{"expected_sign": True}, {"expected_sign": 1.9},
                                    {"rationale": {}}, {"extra": 1}])
def test_strict_json_schema(change):
    obj = dict(formula="rank(close)", parameters={}, rationale="hypothesis", mechanism="driver", expected_sign=1)
    obj.update(change)
    with pytest.raises(ProposalError):
        parse_proposal(json.dumps(obj))


def test_duplicate_keys_and_trailing_prose_rejected():
    with pytest.raises(ProposalError):
        parse_proposal('{"formula":"close","formula":"volume"}')
    with pytest.raises(ProposalError):
        parse_proposal('{"formula":"close"} trailing prose')


@pytest.mark.parametrize("version", ["legacy-v1", "strict-math-v1", "unknown", None])
def test_obsolete_or_unknown_math_contract_rejected(version):
    with pytest.raises(ValueError, match="unsupported mathematical contract"):
        RunConfig(math_contract=version)
    with pytest.raises(ValueError, match="unsupported mathematical contract"):
        RunConfig.from_dict({"math_contract": version})



def test_future_numeric_changes_do_not_affect_is_admission():
    data = panel()
    cut = pd.Timestamp("2021-12-31")
    first = evaluate("rank(close**p)", data, {"p": {"type": "Exponent", "value": 2}}, scope=ResearchScope(end=cut, phase="IS"))
    data.fields["close"].loc[cut+pd.Timedelta(days=1):] = 1e308
    second = evaluate("rank(close**p)", data, {"p": {"type": "Exponent", "value": 2}}, scope=ResearchScope(end=cut, phase="IS"))
    pd.testing.assert_frame_equal(first, second)


def test_static_reject_never_scores_even_when_alignment_off(monkeypatch):
    import harness.runner as runner
    from harness.providers.scripted import ScriptedProvider
    cfg = RunConfig(math_contract=STRICT, research_date="2021-12-31", data_start="2020-01-01", data_end="2022-12-31", is_years=1, max_iters=1, tri_align=False, memory=False)
    bad = {"formula": "rank(close)+rank(volume)", "parameters": {}, "rationale": "hypothesis", "mechanism": "driver", "expected_sign": 1}
    calls = []
    monkeypatch.setattr(runner, "_is_metrics", lambda *a: calls.append(a))
    with pytest.raises(RuntimeError, match="no eligible"):
        runner.optimize(ScriptedProvider([json.dumps(bad)]), cfg, panel(), sleep=lambda _: None)
    assert not calls


def test_exact_candidate_count():
    from harness.proposer import _extract_candidate_dicts
    with pytest.raises(ProposalError, match="exactly 2"):
        _extract_candidate_dicts('{"candidates":[]}', 2)


def test_admission_uses_only_the_proposal_completion(monkeypatch):
    from harness import runner
    from harness.providers.replay import RecordingProvider
    from harness.providers.scripted import ScriptedProvider

    cfg = RunConfig(math_contract=STRICT, max_iters=1,
                    research_date="2021-12-31", is_years=1, memory=False)
    candidate = {"formula": "rank(close)", "parameters": {}, "rationale": "hypothesis",
                 "mechanism": "driver", "expected_sign": 1}
    provider = RecordingProvider(ScriptedProvider([json.dumps(candidate)]))
    assert not hasattr(runner, "judge_screen")
    monkeypatch.setattr(runner, "_is_metrics", lambda *a, **k: {
        "sortino": 1.0, "avg_gross": 1.0, "n_active": 252, "n": 252,
    })
    result = runner.optimize(provider, cfg, panel(), sleep=lambda _: None)
    assert result["best"]["checked_factor"].formula == candidate["formula"]
    assert len(provider.trace) == 1


def test_codex_isolated_event_response_preserves_prompt_hash(monkeypatch):
    from harness.providers import codex
    from harness.providers.base import hash_prompt
    monkeypatch.setattr(codex, "ensure_binary", lambda _: "codex")
    calls = []
    def cli(cmd, **kwargs):
        calls.append(cmd)
        return '\n'.join(json.dumps(e) for e in [
            {"type": "thread.started", "thread_id": "test"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": '{"ok":true}'}},
            {"type": "turn.completed"}])
    monkeypatch.setattr(codex, "run_cli", cli)
    messages = [{"role": "user", "content": "hello"}]
    response = codex.CodexProvider().complete("system", messages, model="gpt-6-luna", reasoning_effort="max")
    assert response.prompt_hash == hash_prompt("system", messages)
    assert response.text == '{"ok":true}'
    assert "--ignore-user-config" in calls[0]
    assert 'web_search="disabled"' in calls[0]


def test_codex_tool_event_is_not_accepted(monkeypatch):
    from harness.providers import codex
    from harness.providers.codex_controls import ProviderConfigurationError
    monkeypatch.setattr(codex, "ensure_binary", lambda _: "codex")
    monkeypatch.setattr(codex, "run_cli", lambda *a, **k: json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}))
    with pytest.raises(ProviderConfigurationError, match="unexpected item"):
        codex.CodexProvider().complete("system", [], model="gpt-6-luna", reasoning_effort="max")


def test_parameter_alpha_identity_and_distinct_ids():
    left = compile_factor("ts_mean(close,w)", window()).record()
    right = compile_factor("ts_mean(close,lookback)", {"lookback": {"type": "Window", "value": 20}}).record()
    assert left["expression_hash"] == right["expression_hash"]
    shared = compile_factor("ts_mean(close,w)/ts_std(close,w)", window()).record()
    separate = compile_factor("ts_mean(close,a)/ts_std(close,b)", {
        "a": {"type": "Window", "value": 20}, "b": {"type": "Window", "value": 20}}).record()
    assert shared["parameter_count"] == 1 and separate["parameter_count"] == 2
    assert shared["expression_hash"] != separate["expression_hash"]


@pytest.mark.parametrize("transform", ["rank({})", "zscore({})", "ts_mean({},w)", "abs({})", "-({})"])
@pytest.mark.parametrize("bad", ["rank(close)*rank(volume)", "rank(volume)*rank(close)", "rank(close)+rank(returns)", "rank(close)-rank(ts_std(close,w))"])
def test_wrapping_cannot_launder_independent_signals(transform, bad):
    formula = transform.format(bad)
    with pytest.raises(ContractError):
        compile_factor(formula, window() if "w" in formula else {})


OPERATOR_FORMULAS = [
    ("abs(returns)", {}), ("sign(returns)", {}), ("signed_log1p(returns)", {}),
    ("pow(returns,p)", {"p": {"type": "Exponent", "value": 2}}),
    ("signed_pow(returns,p)", {"p": {"type": "Exponent", "value": 0.5}}),
    ("min(close,k)", {"k": {"type": "Coefficient", "value": 120}}),
    ("max(close,k)", {"k": {"type": "Coefficient", "value": 90}}),
    *[(f"{op}(returns)", {}) for op in ("rank", "zscore", "demean", "scale")],
    *[(f"{op}(returns,w)", window(5)) for op in ("delay", "delta", "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_rank", "decay_linear", "product")],
    *[(f"{op}(returns,volume,w)", window(5)) for op in ("ts_corr", "ts_cov")],
]


@pytest.mark.parametrize("formula,bindings", OPERATOR_FORMULAS)
def test_registry_all_operators_prefix_invariance(formula, bindings):
    from harness.factor.contract import SPECS
    data = panel()
    cut = data.dates[90]
    prefix = evaluate(formula, data, bindings, scope=ResearchScope(end=cut))
    full = evaluate(formula, data, bindings)
    pd.testing.assert_frame_equal(prefix, full.loc[:cut], check_flags=False)
    op = formula.split("(")[0]
    assert SPECS[op].kernel and SPECS[op].kernel_version


def test_domain_mask_zero_denominator_and_axis_mismatch():
    data = panel()
    data.fields["close"].iloc[5, 0] = 0
    result = evaluate("(high-low)/close", data)
    assert pd.isna(result.iloc[5, 0])
    assert result.attrs["zora_admission"]["node_masks"][-1]["mask_rule"] == "input_mask + zero_denominator"
    data.fields["volume"] = data.fields["volume"].iloc[:, ::-1]
    with pytest.raises(FactorEvalError, match="AXES"):
        evaluate("volume", data)


def test_admission_bridge_rejects_unchecked_or_changed_signal():
    from harness.factor.checked_evaluate import require_validated_signal
    from harness.infra_engine import register_signal
    data = panel()
    with pytest.raises(FactorEvalError, match="ADMISSION_REQUIRED"):
        register_signal(None, "unchecked", data.fields["close"])
    signal = evaluate("rank(close)", data)
    require_validated_signal(signal)
    signal.iloc[0, 0] += 1
    with pytest.raises(FactorEvalError, match="ADMISSION_REQUIRED"):
        require_validated_signal(signal)


def test_numeric_reject_never_calls_backtest(monkeypatch):
    import harness.runner as runner
    from harness.providers.scripted import ScriptedProvider
    cfg = RunConfig(math_contract=STRICT, research_date="2021-12-31", is_years=1, max_iters=1, tri_align=False, memory=False)
    idea = {"formula": "rank(close**p)", "parameters": {"p": {"type": "Exponent", "value": 200}},
            "rationale": "hypothesis", "mechanism": "driver", "expected_sign": 1}
    calls = []
    monkeypatch.setattr(runner, "run_backtest", lambda *a, **kw: calls.append(a))
    with pytest.raises(RuntimeError, match="no eligible"):
        runner.optimize(ScriptedProvider([json.dumps(idea)]), cfg, panel(), sleep=lambda _: None)
    assert not calls


def test_oos_numeric_reject_does_not_reask_or_register(monkeypatch):
    import harness.runner as runner
    from harness.providers.scripted import ScriptedProvider
    data = panel()
    data.fields["close"].loc["2022-01-01":] = 1e308
    cfg = RunConfig(math_contract=STRICT, research_date="2021-12-31", is_years=1, max_iters=1, tri_align=False, memory=False)
    idea = {"formula": "rank(close**p)", "parameters": {"p": {"type": "Exponent", "value": 2}},
            "rationale": "hypothesis", "mechanism": "driver", "expected_sign": 1}
    provider = ScriptedProvider([json.dumps(idea)])
    monkeypatch.setattr(runner, "load_panel", lambda *a, **kw: data)
    calls = []
    monkeypatch.setattr(runner, "_register_factor", lambda *a: calls.append(a))
    record = runner.run_once(cfg, provider=provider, register=True, sleep=lambda _: None)
    assert record["verdict"]["label"] == "NUMERICAL_REJECT"
    assert record["math_admission"]["numerical_status"] == "rejected"
    assert record["is_math_admission"]["numerical_status"] == "validated"
    assert provider._i == 1 and not calls


def test_strict_manifest_freezes_policy_and_rejects_drift(tmp_path):
    from harness.store import Store
    from harness.factor.contract import POLICY_HASH
    from harness.runner import replay
    store = Store(str(tmp_path), "policy")
    store.write_manifest({"config": RunConfig().to_dict()})
    manifest = store.read_manifest()
    assert manifest["math_policy_hash"] == POLICY_HASH
    manifest["math_policy_hash"] = "changed"
    with pytest.raises(ValueError, match="MATH_POLICY_MISMATCH"):
        replay(manifest, [])


def test_huge_dimensional_exponent_is_controlled_rejection():
    with pytest.raises(ContractError, match="UNIT_RANGE"):
        compile_factor("pow(pow(close,p),p)", {"p": {"type": "Exponent", "value": 1e308}})


async def test_tui_current_contract_visible_without_obsolete_controls(tmp_path):
    from harness.tui import HarnessTUI
    from textual.widgets import Label
    app = HarnessTUI(config=RunConfig(memory=False), config_path=str(tmp_path / "config.json"), artifacts_root=str(tmp_path))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._build_config().math_contract == STRICT
        assert STRICT in str(app.query_one("#math_contract_label", Label).render())
        assert not app.query("#judge_prescreen")
        assert not app.query("#judge_model")
        app._apply_config(RunConfig(tri_align=False, memory=False))
        await pilot.pause()
        assert app._build_config().math_contract == STRICT
        assert not app._build_config().tri_align



def test_oos_math_failure_is_not_generation_feedback(monkeypatch):
    import harness.runner as runner
    from harness import memory
    from harness.providers.scripted import ScriptedProvider
    cfg = RunConfig(math_contract=STRICT, walk_forward=True, t_0="2021-01-01", t_p="2022-01-01", frequency="YS", max_iters=1, memory=False)
    feedback = []
    def run(step, **kw):
        feedback.append(kw.get("prior_failures"))
        return {"research_date": step.research_date, "formula": "close",
                "verdict": {"passed": False, "label": "NUMERICAL_REJECT"}}
    monkeypatch.setattr(runner, "run_once", run)
    runner._evolve(cfg, ScriptedProvider(), register=False)
    assert feedback == [None, None]
    from harness.factor.contract import POLICY_HASH
    monkeypatch.setattr(memory, "load_entries", lambda _: [{
        "manifest": {"config": cfg.to_dict(), "math_policy_hash": POLICY_HASH},
        "factors": [{"formula": "close", "verdict": {"label": "NUMERICAL_REJECT"}}]}])
    assert memory.lessons("unused") is None


def test_oos_missingness_and_overflow_do_not_change_actual_is_selection():
    import copy
    import harness.runner as runner
    from harness.providers.scripted import ScriptedProvider
    data = panel()
    changed = copy.deepcopy(data)
    changed.fields["close"].loc["2022-01-01":] = 1e308
    changed.fields["volume"].loc["2022-01-01":] = np.nan
    cfg = RunConfig(math_contract=STRICT, research_date="2021-12-31", is_years=1, max_iters=1, candidates_per_round=2, tri_align=False, memory=False)
    candidates = [{"formula": "rank(close**p)", "parameters": {"p": {"type": "Exponent", "value": 2}},
                   "rationale": "hypothesis", "mechanism": "driver", "expected_sign": 1},
                  {"formula": "rank(volume)", "parameters": {}, "rationale": "hypothesis", "mechanism": "driver", "expected_sign": 1}]
    script = [json.dumps({"candidates": candidates})]
    original = runner.optimize(ScriptedProvider(script), cfg, data, sleep=lambda _: None)
    future_changed = runner.optimize(ScriptedProvider(script), cfg, changed, sleep=lambda _: None)
    assert original["best"]["formula"] == future_changed["best"]["formula"]
    assert runner._deep_eq(original["best"]["is_metrics"], future_changed["best"]["is_metrics"])


def test_single_date_strict_trace_persists_and_replays_with_unused_walk_bounds(tmp_path):
    from harness.store import Store
    import harness.runner as runner
    from harness.providers.scripted import ScriptedProvider
    cfg = RunConfig(math_contract=STRICT, data_source="synthetic", symbols=["A", "B", "C"],
                    research_date="2021-12-31", data_start="2020-01-01", data_end="2022-12-31",
                    t_0="2020-01-01", t_p="2022-01-01", is_years=1, max_iters=1, memory=False)
    store = Store(str(tmp_path), "single")
    store.write_manifest({"execution_mode": "single-date", "config": cfg.to_dict()})
    original = runner.run_once(cfg, provider=ScriptedProvider(), store=store, register=False)
    assert original["n_completions"] == 1 and len(store.read_trace()) == 1
    actual = runner.replay(store.read_manifest(), store.read_trace())
    assert len(actual) == 1 and runner._deep_eq(runner._replay_core(original), runner._replay_core(actual[0]))
