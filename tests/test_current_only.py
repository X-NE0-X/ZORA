"""Unsupported contracts and stale run schemas fail without migration or mutation."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from harness import memory, runner
from harness.config import RunConfig
from harness.factor.contract import POLICY_HASH
from harness.providers.scripted import ScriptedProvider
from harness.store import Store


def stale_manifest(defect):
    manifest = {"config": RunConfig(memory=False).to_dict(), "math_policy_hash": POLICY_HASH}
    if defect == "unsupported_contract":
        manifest["config"]["math_contract"] = "legacy-v1"
    elif defect == "missing_contract":
        manifest["config"].pop("math_contract")
    elif defect == "missing_field":
        manifest["config"].pop("reasoning_effort")
    elif defect == "obsolete_field":
        manifest["config"]["judge_prescreen"] = False
    elif defect == "stale_policy":
        manifest["math_policy_hash"] = "obsolete"
    return manifest


DEFECTS = ["unsupported_contract", "missing_contract", "missing_field", "obsolete_field", "stale_policy"]


@pytest.mark.parametrize("defect", DEFECTS)
def test_stale_replay_rejected_before_market_data_or_provider(monkeypatch, defect):
    monkeypatch.setattr(runner, "load_panel", lambda *_: pytest.fail("rejected replay read market data"))
    monkeypatch.setattr(runner, "ReplayProvider", lambda *_: pytest.fail("rejected replay created a provider"))
    with pytest.raises(ValueError):
        runner.replay(stale_manifest(defect), [])


@pytest.mark.parametrize("defect", DEFECTS)
def test_stale_resume_preserves_manifest_ledger_and_trace(tmp_path, monkeypatch, defect):
    config = RunConfig(memory=False)
    store = Store(str(tmp_path), config.run_name)
    store.ensure()
    manifest = stale_manifest(defect)
    Path(store.manifest_path).write_text(json.dumps(manifest), encoding="utf-8")
    Path(store.trace_path).write_text("original-trace\n", encoding="utf-8")
    Path(store.ledger_path).write_text("original-ledger\n", encoding="utf-8")
    paths = [Path(p) for p in (store.manifest_path, store.trace_path, store.ledger_path)]
    before = [p.read_bytes() for p in paths]
    monkeypatch.setattr(runner, "check_data_coverage", lambda *a, **k: config.is_window())
    provider = ScriptedProvider()
    with pytest.raises(ValueError):
        runner.run_walk_forward(config, artifacts_root=str(tmp_path), resume=True, provider=provider)
    assert [p.read_bytes() for p in paths] == before
    assert provider._i == 0


@pytest.mark.parametrize("defect", DEFECTS)
def test_stale_journal_is_retained_but_never_generation_feedback(monkeypatch, defect):
    entry = {"manifest": stale_manifest(defect), "factors": [{
        "formula": "close", "research_date": "2020-01-01", "oos_end": "2020-12-31",
        "verdict": {"label": "PASS", "passed": True, "oos_value": 2}}]}
    original = deepcopy(entry)
    monkeypatch.setattr(memory, "load_entries", lambda *_: [entry])
    assert memory.lessons("unused") is None
    assert entry == original


def test_missing_explicit_oos_close_is_never_reconstructed_from_old_clock(monkeypatch):
    entry = {"manifest": {"config": RunConfig().to_dict(), "math_policy_hash": POLICY_HASH},
             "factors": [{"formula": "close", "research_date": "2020-01-01",
                         "verdict": {"label": "FAIL", "passed": False, "oos_value": -1}}]}
    monkeypatch.setattr(memory, "load_entries", lambda *_: [entry])
    assert memory.lessons("unused")["losses"][0]["oos_end"] is None


def test_store_cannot_restamp_stale_policy(tmp_path):
    store = Store(str(tmp_path), "policy")
    with pytest.raises(ValueError, match="MATH_POLICY_MISMATCH"):
        store.write_manifest(stale_manifest("stale_policy"))
    assert not Path(store.manifest_path).exists()


def test_old_executors_parsers_and_judge_assets_removed():
    root = Path(__file__).resolve().parents[1]
    for path in ("harness/legacy_proposer.py", "harness/factor/expr.py", "harness/factor/strict_kernels.py",
                 "harness/prompts/system_judge.json", "harness/prompts/user_judge.json"):
        assert not (root / path).exists()
