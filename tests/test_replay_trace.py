"""Record/replay: a recorded walk-forward reproduces bit-identically from its
cached LLM trace --- no model call --- and the RecordingProvider/ReplayProvider
pair behaves correctly on ordering, recorded errors, and trace exhaustion."""
import contextlib
import io
import json
import tempfile
import pytest

from harness import cli
from harness.config import RunConfig
from harness.providers.base import LLMProvider, ProviderResponse, hash_prompt
from harness.providers.replay import (RecordingProvider, ReplayDesyncError,
                                      ReplayProvider)
from harness.runner import _deep_eq, _replay_core, replay, run_walk_forward, UsageMeter
from harness.store import Store

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2021-01-01", t_p="2022-01-01", frequency="YS",
        provider="scripted", seed=17, max_iters=2, run_name="test_replay_trace", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


# --- provider-pair units ---------------------------------------------------
class _CannedProvider(LLMProvider):
    """Returns each canned text in order; raises for entries marked as errors."""

    name = "canned"

    def __init__(self, items):
        self._items = list(items)
        self._i = 0

    def complete(self, system, messages, *, model, seed=17, max_tokens=2048):
        item = self._items[self._i]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        return ProviderResponse(text=item, model=model, provider=self.name)


def test_recording_captures_order_and_errors():
    inner = _CannedProvider(["first", RuntimeError("boom"), "third"])
    seen = []
    rec = RecordingProvider(inner, on_record=seen.append)

    assert rec.name == "canned"                       # transparent identity
    assert rec.complete("s", [], model="m").text == "first"
    try:
        rec.complete("s", [], model="m")
    except RuntimeError:
        pass
    else:
        raise AssertionError("inner error must propagate through RecordingProvider")
    assert rec.complete("s", [], model="m").text == "third"

    assert [e.get("text", "ERR") for e in rec.trace] == ["first", "ERR", "third"]
    assert seen == rec.trace                           # on_record saw them live
    assert "RuntimeError: boom" in rec.trace[1]["error"]


def test_replay_returns_in_order_and_reraises_errors():
    ph = hash_prompt("ignored", [])
    trace = [{"text": "a", "prompt_hash": ph},
             {"error": "RuntimeError: boom", "prompt_hash": ph},
             {"text": "c", "prompt_hash": ph}]
    rp = ReplayProvider(trace)
    assert rp.complete("ignored", [], model="m").text == "a"
    try:
        rp.complete("ignored", [], model="m")
    except RuntimeError as exc:
        assert "boom" in str(exc)
    else:
        raise AssertionError("recorded error must be re-raised on replay")
    assert rp.complete("ignored", [], model="m").text == "c"


def test_replay_ignores_prompt():
    rp = ReplayProvider([{"text": "fixed",
                          "prompt_hash": hash_prompt("recorded", [])}])
    # totally different prompt -> same cached answer (replay is by call order)
    assert rp.complete("whatever system", [{"role": "user", "content": "x"}],
                       model="m").text == "fixed"
    # The provider records drift; runner.replay owns the aggregate warning.
    assert rp.prompt_mismatches == [0]


def test_replay_exhaustion_raises():
    rp = ReplayProvider([{"text": "only", "prompt_hash": hash_prompt("s", [])}])
    rp.complete("s", [], model="m")
    try:
        rp.complete("s", [], model="m")
    except ReplayDesyncError as exc:
        assert "exhausted" in str(exc)
    else:
        raise AssertionError("over-reading a trace must raise ReplayDesyncError")


def test_hashless_trace_is_not_declared_reproducible():
    rp = ReplayProvider([{"text": "legacy-without-hash"}])
    try:
        rp.complete("s", [], model="m")
    except ReplayDesyncError as exc:
        assert "no prompt_hash" in str(exc)
    else:
        raise AssertionError("hashless replay must fail closed")


# --- end-to-end record -> replay -------------------------------------------
def test_walk_forward_replays_bit_identically():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg()
        out = run_walk_forward(cfg, artifacts_root=root)
        recorded = out["records"]

        store = Store(root, cfg.run_name)
        trace = store.read_trace()
        manifest = store.read_manifest()
        # every LLM call was captured (max_iters per date, 2 dates)
        assert len(trace) == cfg.max_iters * len(recorded)

        replayed = replay(manifest, trace)
        assert len(replayed) == len(recorded)
        for rec, rep in zip(recorded, replayed):
            a, b = _replay_core(rec), _replay_core(rep)
            assert _deep_eq(a, b), f"replay diverged at {rec['research_date']}"
        # replay is read-only: no FactorEngine lineage attached
        assert all(rep["factor_ref"] is None for rep in replayed)


def test_cli_replay_reports_reproducible(capsys=None):
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(run_name="test_cli_replay")
        run_walk_forward(cfg, artifacts_root=root)

        prev = cli._ARTIFACTS
        cli._ARTIFACTS = root
        try:
            code = cli.main(["replay", "--run-name", cfg.run_name])
        finally:
            cli._ARTIFACTS = prev
        assert code == 0, "a freshly recorded run must replay reproducibly"


def test_cli_hashless_trace_fails_without_bit_identical_claim():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(run_name="test_cli_hashless", max_iters=1,
                   t_p="2021-01-01")
        run_walk_forward(cfg, artifacts_root=root)
        store = Store(root, cfg.run_name)
        trace = store.read_trace()
        trace[0].pop("prompt_hash", None)
        payload = "".join(json.dumps(entry) + "\n" for entry in trace).encode("utf-8")
        store._atomic_write(store.trace_path, payload)

        prev = cli._ARTIFACTS
        cli._ARTIFACTS = root
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                code = cli.main(["replay", "--run-name", cfg.run_name])
        finally:
            cli._ARTIFACTS = prev
        assert code == 1
        assert "no prompt_hash" in buf.getvalue()
        assert "reproduced bit-identically" not in buf.getvalue()


@pytest.mark.parametrize("meter_depth", [0, 1, 2])
def test_single_date_reuses_recording_inside_usage_proxies(tmp_path, monkeypatch, meter_depth):
    """A completion persists once even when the recorder has transparent wrappers."""
    from harness import runner

    store = Store(str(tmp_path), "wrapped")
    recorder = RecordingProvider(_CannedProvider(["answer"]), on_record=store.append_trace)
    provider = recorder
    meters = []
    for _ in range(meter_depth):
        provider = UsageMeter(provider)
        meters.append(provider)

    def complete_one(config, provider, **kwargs):
        provider.complete("sys", [], model=config.model)
        return {"n_completions": len(provider.trace)}

    monkeypatch.setattr(runner, "_run_once_owned", complete_one)
    result = runner.run_once(_cfg(), provider=provider, store=store)
    assert result["n_completions"] == 1
    assert store.read_trace() == recorder.trace and len(recorder.trace) == 1
    assert all(meter.usage_totals["n_calls"] == 1 for meter in meters)
