"""F02-F11 acceptance regressions. No credentials or paid model requests."""
from copy import deepcopy
import json
import pathlib
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pandas as pd
import pytest

from harness import memory, objective, proposer, runner, tui
from harness.calendars import session_bounds
from harness.cancellation import RunCancelled, RunControl, cancellation_scope
from harness.config import RunConfig
from harness.data import make_synthetic
from harness.factor.contract import compile_factor
from harness.infra_engine import window_metrics
from harness.providers.base import ProviderResponse
from harness.providers.claude import ClaudeProvider
from harness.providers.cli_base import CLIProviderError, run_cli, start_owned_process
from harness.providers.codex_controls import DEFAULT_MODELS, model_defaults, switch_defaults
from harness.providers.openai_api import OpenAIProvider
from harness.providers.opencode import OpenCodeProvider
from harness.providers.scripted import ScriptedProvider
from harness.providers.structured import api_schema
from harness.store import Store
from harness.validate import verdict
from tests.test_memory import _factor, _manifest
from tests.test_resume import _cfg
from tests.test_runtime_fixes import PROBE, _alive


def _snapshot(store, include_log=False):
    paths = [store.manifest_path, store.ledger_path, store.trace_path]
    if include_log:
        paths.append(str(pathlib.Path(store.run_dir) / "run.log"))
    return {p: pathlib.Path(p).read_bytes() for p in paths}


def test_f02_resume_drift_preserves_all_bytes_even_orphan_trace(tmp_path, monkeypatch):
    cfg = _cfg(t_0="2021-01-01", t_p="2021-01-01", max_iters=1)
    panel = make_synthetic(cfg.symbols, cfg.data_start, cfg.data_end, seed=17)
    monkeypatch.setattr(runner, "load_panel", lambda _: panel)
    run = runner.run_walk_forward(cfg, artifacts_root=str(tmp_path))
    store = Store(str(tmp_path), cfg.run_name)
    assert run["manifest"]["data_version"] == panel.version
    store.append_trace({"text": "orphan completion", "model": "x"})
    original = _snapshot(store)
    drift = make_synthetic(cfg.symbols, cfg.data_start, cfg.data_end, seed=29)
    monkeypatch.setattr(runner, "load_panel", lambda _: drift)
    provider = ScriptedProvider()
    with pytest.raises(ValueError, match="data version differs"):
        runner.run_walk_forward(_cfg(t_0="2021-01-01", t_p="2022-01-01", max_iters=1),
                                artifacts_root=str(tmp_path), resume=True, provider=provider)
    assert _snapshot(store) == original
    assert provider._i == 0


def test_f02_walk_pins_one_load_and_checkpoint_identity(tmp_path, monkeypatch):
    cfg = _cfg(t_0="2021-01-01", t_p="2022-01-01", max_iters=1)
    panel = make_synthetic(cfg.symbols, cfg.data_start, cfg.data_end)
    loads = []
    def load(_):
        loads.append(True)
        if len(loads) > 1:
            raise AssertionError("walk must not download a second panel")
        return panel
    monkeypatch.setattr(runner, "load_panel", load)
    checkpoints = []
    original = Store.write_manifest
    def write(store, payload):
        checkpoints.append(deepcopy(payload))
        original(store, payload)
    monkeypatch.setattr(Store, "write_manifest", write)
    out = runner.run_walk_forward(cfg, artifacts_root=str(tmp_path))
    assert len(loads) == 1 and len(out["records"]) == 2
    assert all(r["data_version"] == panel.version for r in out["records"])
    assert checkpoints[0]["data_version"] == panel.version
    assert checkpoints[0]["n_factors"] == checkpoints[0]["n_pass"] == 0


def _schema_nodes(node):
    yield node
    for child in node.get("properties", {}).values():
        yield from _schema_nodes(child)
    if "items" in node:
        yield from _schema_nodes(node["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        for child in node.get(key, []):
            yield from _schema_nodes(child)


@pytest.mark.parametrize("provider", ["openai", "claude"])
@pytest.mark.parametrize("count", [1, 2, 80])
def test_f03_real_proposal_wire_schema_supported_and_local_unchanged(provider, count):
    schema = proposer._proposal_schema(count)
    original = deepcopy(schema)
    wire, hint = api_schema(schema, provider)
    assert schema == original
    assert "parameters MUST be an array" in hint
    item = wire if count == 1 else wire["properties"]["candidates"]["items"]
    assert item["properties"]["parameters"]["type"] == "array"
    for node in _schema_nodes(wire):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            if provider == "openai":
                assert set(node["required"]) == set(node["properties"])
        if provider == "claude":
            assert "maxItems" not in node
            assert node.get("minItems", 0) in (0, 1)


@pytest.mark.parametrize("kind", ["openai", "claude"])
@pytest.mark.parametrize("count", [1, 2])
def test_f03_fake_sdk_captures_wire_conversion(kind, count):
    calls = []
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
            content=[SimpleNamespace(type="text", text="{}")],
            stop_reason="end_turn", usage=None)
    cls = OpenAIProvider if kind == "openai" else ClaudeProvider
    prov = object.__new__(cls)
    prov._default_model = DEFAULT_MODELS[kind]
    prov._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        messages=SimpleNamespace(create=create))
    prov.complete("original system", [{"role": "user", "content": "q"}],
                  model=DEFAULT_MODELS[kind], structured_schema=proposer._proposal_schema(count))
    if kind == "openai":
        wire = calls[-1]["response_format"]["json_schema"]["schema"]
        system = calls[-1]["messages"][0]["content"]
    else:
        wire = calls[-1]["output_config"]["format"]["schema"]
        system = calls[-1]["system"]
    assert wire == api_schema(proposer._proposal_schema(count), kind)[0]
    assert "API wire format override" in system


def _wire_proposal(value=20):
    return {"formula": "rank(ts_mean(returns,w))",
            "parameters": [{"name": "w", "type": "Window", "value": value}],
            "rationale": "r" * 50, "mechanism": "m" * 50, "expected_sign": 1}


def test_f03_array_binding_is_identical_to_local_map():
    proposal = proposer.parse_proposal(json.dumps(_wire_proposal()))
    expected = {"w": {"type": "Window", "value": 20}}
    assert proposal.parameters == expected
    assert proposal.checked().record()["expression_hash"] == compile_factor(proposal.formula, expected).record()["expression_hash"]


@pytest.mark.parametrize("model", ["opencode-go/gpt-5.6-luna", "opencode-go/glm-5.2"])
def test_f03_opencode_native_strict_route_also_closes_proposal_schema(model):
    prov = object.__new__(OpenCodeProvider)
    prov._timeout = 1
    prov._go_api_key = lambda: "offline-placeholder"
    calls = []
    def http(_base, _method, path, body, **_kw):
        calls.append((path, body))
        return {"output_text": "{}", "choices": [{"message": {"content": "{}"}}]}
    prov._http_json = http
    prov._native_go_complete("system", [{"role": "user", "content": "q"}],
                             use_model=model, seed=17, max_tokens=100,
                             temperature=0, schema=proposer._proposal_schema(2))
    path, body = calls[0]
    if path == "/responses":
        schema, messages = body["text"]["format"]["schema"], body["input"]
    else:
        schema, messages = body["response_format"]["json_schema"]["schema"], body["messages"]
    assert schema == api_schema(proposer._proposal_schema(2), "openai")[0]
    assert "parameters MUST be an array" in messages[0]["content"]


def test_f03_valid_two_candidate_array_keeps_exact_count():
    class Fake:
        supports_structured_output = True
        def complete(self, *_args, **_kw):
            return ProviderResponse(text=json.dumps({"candidates": [_wire_proposal(5), _wire_proposal(120)]}),
                                    model="offline", provider="fake")
    accepted, errors, _ = proposer.propose_many(Fake(), RunConfig(candidates_per_round=2))
    assert len(accepted) == 2 and errors == []
    assert [p.parameters["w"]["value"] for p in accepted] == [5, 120]


@pytest.mark.parametrize("change", ["duplicate", "bool", "fraction", "unused", "keys", "budget"])
def test_f03_api_relaxation_does_not_relax_math_admission(change):
    candidate = _wire_proposal()
    entries = candidate["parameters"]
    if change == "duplicate":
        entries.append(deepcopy(entries[0]))
    elif change == "bool":
        entries[0]["value"] = True
    elif change == "fraction":
        entries[0]["value"] = 2.5
    elif change == "unused":
        entries[0]["name"] = "unused"
    elif change == "keys":
        entries[0]["extra"] = 1
    else:
        entries.extend([{"name": "x", "type": "Window", "value": 5},
                        {"name": "y", "type": "Window", "value": 10}])
    with pytest.raises(proposer.ProposalError):
        proposer.parse_proposal(json.dumps(candidate))


@pytest.mark.parametrize("received", [0, 1, 3])
def test_f03_claude_container_count_still_enforced_locally(received):
    class Fake:
        supports_structured_output = True
        def complete(self, *_args, **_kw):
            return ProviderResponse(text=json.dumps({"candidates": [_wire_proposal()] * received}),
                                    model="offline", provider="fake")
    with pytest.raises(proposer.ProposalError, match="exactly 2"):
        proposer.propose_many(Fake(), RunConfig(candidates_per_round=2))


def _metrics(returns):
    idx = pd.bdate_range("2023-01-02", periods=len(returns))
    r = pd.Series(returns, index=idx)
    return window_metrics(r, pd.Series(0.01, index=idx), pd.Series(1.0, index=idx),
                          periods_per_year=252)


def test_f04_original_high_win_rate_loss_rejected():
    metrics = _metrics([0.001] * 60 + [-0.01] * 40)
    assert metrics["hit_rate"] == pytest.approx(0.6) and metrics["cagr"] < -0.5
    cfg = RunConfig(objective="hit_rate", pass_line=0.55)
    result = verdict(metrics, metrics, cfg)
    assert not result["passed"] and result["profitability_metric"] == "cagr"
    assert any("CAGR" in r for r in result["reasons"])
    cfg.require_sign_consistency = False
    assert verdict(metrics, metrics, cfg)["passed"]


@pytest.mark.parametrize("name", objective.names())
def test_f04_every_objective_requires_positive_net_growth(name):
    metrics = _metrics([0.5, -0.4] * 50)
    assert metrics["cagr"] < 0
    assert metrics["sharpe"] > 0 and metrics["sortino"] > 0
    cfg = RunConfig(objective=name, pass_line=metrics[name])
    result = verdict(metrics, metrics, cfg)
    assert not result["passed"]
    assert any("consistently profitable on CAGR" in r for r in result["reasons"])


@pytest.mark.parametrize("growth", [None, 0.0, -0.5, float("nan"), float("inf")])
@pytest.mark.parametrize("window", ["IS", "OOS"])
def test_f04_missing_nonfinite_or_nonpositive_growth_fail_closed(growth, window):
    metrics = {"n": 100, "n_active": 100, "avg_gross": 1.0, "hit_rate": 0.7, "cagr": 0.2}
    invalid = {**metrics, "cagr": growth}
    a, b = (invalid, metrics) if window == "IS" else (metrics, invalid)
    assert not verdict(a, b, RunConfig(objective="hit_rate", pass_line=0.55))["passed"]


@pytest.mark.parametrize("provider", list(DEFAULT_MODELS))
def test_f05_complete_default_mapping_switch_and_explicit_override(provider):
    cfg = RunConfig(provider=provider, model=None)
    assert cfg.model == model_defaults(provider)
    values = RunConfig(provider="scripted").to_dict()
    values["provider"] = provider
    switch_defaults(values, "scripted")
    assert RunConfig.from_dict(values).model == DEFAULT_MODELS[provider]
    custom = "gpt-custom-model" if provider == "codex" else "my-provider/my-custom-model"
    explicit = {**values, "model": custom}
    switch_defaults(explicit, "scripted", explicit={"model"})
    assert RunConfig.from_dict(explicit).model == custom
    if provider == "opencode":
        assert "/" in cfg.model


def test_f05_opencode_bare_namespace_rejected_before_run():
    with pytest.raises(ValueError, match="provider/model"):
        RunConfig(provider="opencode", model="claude-opus-4-8")


def _entry(window=5, objective_name="sortino", **kw):
    cfg = _manifest(objective=objective_name, **kw)
    rec = _factor("2021-01-01", passed=True, formula="rank(ts_mean(returns,w))", oos=1.5)
    rec["parameters"] = {"w": {"type": "Window", "value": window}}
    rec["objective"] = objective_name
    return {"run_id": f"j-{window}-{objective_name}", "manifest": cfg, "factors": [rec]}


@pytest.mark.parametrize("difference", ["bindings", "objective", "universe", "cost", "clock", "data", "delist", "floor"])
def test_f06_expression_and_experiment_context_do_not_merge(monkeypatch, difference):
    first, second = _entry(), _entry()
    if difference == "bindings":
        second = _entry(window=120)
    elif difference == "objective":
        second = _entry(objective_name="cagr")
        second["factors"][0]["verdict"]["oos_value"] = 0.07
    elif difference == "universe":
        second = _entry(symbols=["D", "E", "F"])
    elif difference == "cost":
        second = _entry(cost_bps=20)
    elif difference == "clock":
        second = _entry(oos_days=126)
    elif difference == "delist":
        second = _entry(close_delisted_at_last=True)
    elif difference == "floor":
        second = _entry(min_is_days=50)
    else:
        second["manifest"]["data_version"] = "different-data"
    monkeypatch.setattr(memory, "_current_entries", lambda _: [first, second])
    lessons = memory.lessons("unused")
    assert len(lessons["wins"]) == 2
    assert all(x["n_seen"] == 1 and x["parameters"] and x["experiment_context"] for x in lessons["wins"])
    if difference == "objective":
        assert {(x["objective"], x["oos_value"]) for x in lessons["wins"]} == {("sortino", 1.5), ("cagr", 0.07)}


def test_f06_repeats_merge_but_feedback_keeps_bindings_and_labels(monkeypatch):
    entry = _entry()
    entry["factors"].append({**deepcopy(entry["factors"][0]), "research_date": "2022-01-01"})
    monkeypatch.setattr(memory, "_current_entries", lambda _: [entry])
    lessons = memory.lessons("unused")
    assert len(lessons["wins"]) == 1 and lessons["wins"][0]["n_seen"] == 2
    rec = entry["factors"][0]
    rec["experiment_context"] = memory.evidence_context(entry, rec)
    for summarize in (runner._win_summary, runner._fail_summary):
        summary = summarize(rec)
        assert summary["parameters"] == rec["parameters"]
        assert summary["experiment_context"] == rec["experiment_context"]
        for block in (proposer._prior_block, proposer._prior_fail_block):
            text = block([summary], RunConfig(objective="cagr"))
            assert '"value": 5' in text and "OOS Sortino=1.5000" in text
    for text in (proposer._journal_block(lessons), memory.render_notes([entry])):
        assert '"value": 5' in text and "Sortino" in text and "context=" in text
    changed = deepcopy(entry)
    changed["factors"][0]["parameters"]["w"]["value"] = 120
    assert memory.corpus_fingerprint([entry]) != memory.corpus_fingerprint([changed])


def test_f06_alpha_renaming_merges_with_representative_bindings(monkeypatch):
    entry = _entry()
    rec = deepcopy(entry["factors"][0])
    rec.update(formula="rank(ts_mean(returns,lookback))", research_date="2022-01-01",
               parameters={"lookback": {"type": "Window", "value": 5}})
    entry["factors"].append(rec)
    monkeypatch.setattr(memory, "_current_entries", lambda _: [entry])
    lessons = memory.lessons("unused")
    assert len(lessons["wins"]) == 1 and lessons["wins"][0]["n_seen"] == 2
    winner = lessons["wins"][0]
    assert winner["parameters"] == rec["parameters"]
    assert compile_factor(winner["formula"], winner["parameters"]).record()["expression_hash"] == winner["expression_hash"]


def test_f06_synthesis_receives_complete_bound_evidence(tmp_path, monkeypatch):
    entry = _entry(window=120, objective_name="cagr")
    monkeypatch.setattr(memory, "_current_entries", lambda _: [entry])
    monkeypatch.setattr(memory, "rebuild", lambda _: None)
    messages = []
    class Fake:
        def complete(self, _system, request, **_kw):
            messages.extend(request)
            return ProviderResponse(text="offline summary", model="offline", provider="fake")
    memory.synthesize(Fake(), RunConfig(), str(tmp_path))
    text = messages[0]["content"]
    assert '"value": 120' in text and "CAGR" in text
    assert "context=" in text and "cost_bps" in text and "data_version" in text


def test_f07_ordinary_cli_timeout_kills_children_with_inherited_pipes(tmp_path):
    ready = tmp_path / "owned-tree.json"
    started = time.monotonic()
    with pytest.raises(CLIProviderError, match="timed out"):
        run_cli([sys.executable, PROBE, "tree", str(ready)], timeout=2)
    assert time.monotonic() - started < 5
    assert ready.is_file(), "the probe must actually have spawned its child"
    pids = json.loads(ready.read_text(encoding="utf-8"))
    assert all(not _alive(pid) for pid in pids)


def test_f07_opencode_close_kills_child_even_after_parent_exits(tmp_path):
    ready = tmp_path / "exited-tree.json"
    proc, job = start_owned_process([sys.executable, PROBE, "exited-parent", str(ready)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    prov = object.__new__(OpenCodeProvider)
    prov._server_proc, prov._server_job = proc, job
    prov._server_base = "http://127.0.0.1:1"
    try:
        assert proc.wait(timeout=5) == 0
        pids = json.loads(ready.read_text(encoding="utf-8"))
        assert _alive(pids[1])
        prov.close()
        assert all(not _alive(pid) for pid in pids)
        assert prov._server_proc is prov._server_job is prov._server_base is None
    finally:
        prov.close()


def test_f07_http_cancellation_does_not_wait_for_request_timeout(monkeypatch):
    control, entered, release = RunControl(), threading.Event(), threading.Event()
    def block(*_args):
        entered.set()
        assert release.wait(5)
        return {"ok": True}
    monkeypatch.setattr(OpenCodeProvider, "_blocking_http_json", staticmethod(block))
    timer = threading.Timer(0.2, control.cancel)
    started = time.monotonic()
    timer.start()
    try:
        with cancellation_scope(control), pytest.raises(RunCancelled):
            OpenCodeProvider._http_json("http://127.0.0.1", "GET", "/", timeout=600)
        assert entered.is_set() and time.monotonic() - started < 1.5
    finally:
        release.set()
        timer.join()


def test_f09_per_step_weekend_coverage_accepts_last_business_session():
    cfg = _cfg(is_years=1, t_0="2023-01-01", t_p="2023-01-01", data_end="2023-12-31", oos_mode="per_step")
    panel = make_synthetic(cfg.symbols, cfg.data_start, cfg.data_end)
    assert panel.dates[-1] == pd.Timestamp("2023-12-29")
    runner.prepare_panel(cfg, walk_forward=True, panel=panel)
    with pytest.raises(ValueError, match="before required OOS end"):
        runner.prepare_panel(cfg, walk_forward=True, panel=panel.slice(end="2023-12-28"))


def test_f09_exchange_holiday_is_not_inferred_from_observed_last_bar():
    cfg = RunConfig(data_source="ctx", parquet_daily=["data/Data_EQT_US_D.parquet"])
    start, end = session_bounds(cfg, "2023-12-20", "2023-12-25")
    assert start == pd.Timestamp("2023-12-20") and end == pd.Timestamp("2023-12-22")
    panel = make_synthetic(["A", "B"], "2023-12-20", "2023-12-22")
    runner._check_panel_data_window(panel, "2023-12-20", "2023-12-25", cfg)
    with pytest.raises(ValueError, match="before required OOS end"):
        runner._check_panel_data_window(panel.slice(end="2023-12-21"), "2023-12-20", "2023-12-25", cfg)


def test_f09_crypto_still_requires_weekend_bar_and_unknown_calendar_fails():
    cfg = RunConfig(data_source="ctx", asset_class="crypto", parquet_daily=["data/Data_SPOT_D.parquet"])
    assert session_bounds(cfg, "2023-12-29", "2023-12-31")[1] == pd.Timestamp("2023-12-31")
    panel = make_synthetic(["A", "B"], "2023-12-29", "2023-12-31")
    with pytest.raises(ValueError, match="before required OOS end"):
        runner._check_panel_data_window(panel, "2023-12-29", "2023-12-31", cfg)
    cfg.parquet_daily = ["data/unknown.parquet"]
    with pytest.raises(ValueError, match="calendar"):
        session_bounds(cfg, "2023-12-29", "2023-12-31")


def test_f09_yahoo_market_suffix_and_mixed_calendar_fail_closed():
    cfg = RunConfig(data_source="yfinance", symbols=["0700.HK", "9988.HK"])
    assert session_bounds(cfg, "2023-12-22", "2023-12-26")[1] == pd.Timestamp("2023-12-22")
    cfg.symbols.append("AAPL")
    with pytest.raises(ValueError, match="one data-source session calendar"):
        session_bounds(cfg, "2023-12-22", "2023-12-26")


def test_f10_config_explicitly_uses_authorized_share_class():
    cfg = RunConfig.from_json(str(pathlib.Path(__file__).resolve().parents[1] / "field_config.json"))
    assert "GOOGL" in cfg.symbols and "GOOG" not in cfg.symbols
    assert cfg.allow_missing_symbols is False


@pytest.mark.parametrize("walk", [False, True])
def test_f11_tui_failed_preflight_preserves_previous_log_and_artifacts(tmp_path, monkeypatch, walk):
    cfg = _cfg()
    store = Store(str(tmp_path), cfg.run_name)
    store.ensure()
    for path in (store.manifest_path, store.ledger_path, store.trace_path,
                 str(pathlib.Path(store.run_dir) / "run.log")):
        pathlib.Path(path).write_bytes(b"previous completed attempt\r\n")
    original = _snapshot(store, include_log=True)
    closed = []
    provider = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(tui, "get_provider", lambda _: provider)
    def reject(*_args, **_kw):
        raise ValueError("data coverage does not reach the final OOS session")
    monkeypatch.setattr(runner, "prepare_panel", reject)
    app = tui.HarnessTUI(cfg, artifacts_root=str(tmp_path), memory_root=str(tmp_path / "memory"))
    app._ui_closed = True
    app._execute_run(cfg, walk)
    assert _snapshot(store, include_log=True) == original
    assert closed == [True]
