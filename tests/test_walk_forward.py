"""Walk-forward evolution: empty schedules are rejected; a real walk produces a
ledger + manifest that round-trips through the Store.

Also covers the two things a run needs to survive contact with a real provider:
a transient failure must PACE the remaining rounds instead of firing them
back-to-back, and the tokens a run burns must be counted so it can be costed.
"""
import json
import os
import tempfile

from harness.config import RunConfig
from harness.providers.base import LLMProvider, ProviderResponse, complete_provider
from harness.providers.replay import RecordingProvider, TracePersistenceError
from harness.runner import (UsageMeter, _evolution_dates, _usage_total,
                            extract_usage, optimize, run_once, run_walk_forward)
from harness.store import Store

SYMS = ["A", "B", "C", "D", "E", "F"]
GOOD = json.dumps({"formula": "-delta(close,w)", "parameters": {"w": {"type": "Window", "value": 5}}, "rationale": "reversal",
                   "mechanism": "overreaction reverts", "expected_sign": -1})


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2021-01-01", t_p="2022-01-01", frequency="YS",
        provider="scripted", seed=17, max_iters=2, run_name="test_wf", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def test_empty_schedule_raises():
    # t_0 after t_p -> no dates on the schedule -> explicit error, not a silent
    # fallback to a single research_date
    cfg = _cfg(t_0="2022-01-01", t_p="2021-01-01")
    try:
        _evolution_dates(cfg)
    except ValueError:
        return
    raise AssertionError("empty walk-forward schedule must raise ValueError")


def test_schedule_has_expected_stops():
    dates = _evolution_dates(_cfg())
    assert [d.year for d in dates] == [2021, 2022]


def test_walk_forward_writes_ledger_and_manifest():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg()
        out = run_walk_forward(cfg, artifacts_root=root)
        recs = out["records"]
        assert len(recs) == 2
        man = out["manifest"]
        assert man["n_factors"] == 2
        assert man["n_pass"] == sum(1 for r in recs if r["verdict"]["passed"])
        assert man["data_version"] == recs[0]["data_version"]

        # everything the Store persisted must round-trip
        store = Store(root, cfg.run_name)
        assert os.path.isfile(store.manifest_path)
        assert store.read_manifest()["run_name"] == cfg.run_name
        assert len(store.read_factors()) == 2
        # each research date appears once, in order
        assert [r["research_date"] for r in store.read_factors()] == \
            ["2021-01-01", "2022-01-01"]


# --- transient-failure backoff ------------------------------------------------
class _ScriptedFailures(LLMProvider):
    """Raises the given exceptions in order, then answers with a valid proposal."""

    name = "flaky"

    def __init__(self, failures: int, error: str = "429 Too Many Requests"):
        self.failures = failures
        self.error = error
        self.calls = 0

    def complete(self, system, messages, *, model="m", seed=17, max_tokens=2048):
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(self.error)
        return ProviderResponse(text=GOOD, model=model, provider=self.name,
                                prompt_hash="h")


def _panel(cfg):
    from harness.data import make_synthetic
    return make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)


def test_transient_failure_paces_the_remaining_rounds():
    """A 429 on round 1 must not let the rest of the search fire in a second.

    The pause grows with the number of CONSECUTIVE failures and is capped, and
    it resets the moment the provider answers --- so one blip mid-search does not
    leave a healthy provider throttled for the rest of the run.
    """
    cfg = _cfg(max_iters=5, retry_backoff=1.0, retry_max_delay=3.0)
    waits: list[float] = []
    res = optimize(_ScriptedFailures(3), cfg, _panel(cfg), sleep=waits.append)

    assert waits == [1.0, 2.0, 3.0], "exponential, capped at retry_max_delay"
    assert res["best"]["formula"] == "-delta(close,w)"
    # the two rounds that did answer are the ones that became history
    assert len(res["history"]) == 2


def test_backoff_skips_the_pause_after_the_last_round():
    # nothing follows the final round, so sleeping there only delays the error
    cfg = _cfg(max_iters=3, retry_backoff=1.0)
    waits: list[float] = []
    try:
        optimize(_ScriptedFailures(99), cfg, _panel(cfg), sleep=waits.append)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a provider that never answers must still raise")
    assert waits == [1.0, 2.0], "no pause is spent after the last round"


def test_backoff_can_be_switched_off():
    cfg = _cfg(max_iters=3, retry_backoff=0.0)
    waits: list[float] = []
    optimize(_ScriptedFailures(2), cfg, _panel(cfg), sleep=waits.append)
    assert waits == [], "retry_backoff=0 must not sleep at all"


def test_zero_retry_ceiling_disables_sleep():
    cfg = _cfg(max_iters=3, retry_backoff=1.0, retry_max_delay=0.0)
    waits: list[float] = []
    optimize(_ScriptedFailures(2), cfg, _panel(cfg), sleep=waits.append)
    assert waits == [], "retry_max_delay=0 must disable retry sleeping"


def test_trace_write_failure_is_not_retried_as_provider_transient():
    cfg = _cfg(max_iters=3, retry_backoff=1.0)
    waits: list[float] = []

    def fail_write(_entry):
        raise OSError("disk full")

    provider = RecordingProvider(_ScriptedFailures(0), on_record=fail_write)
    try:
        optimize(provider, cfg, _panel(cfg), sleep=waits.append)
    except TracePersistenceError as exc:
        assert "persistence failed" in str(exc)
    else:
        raise AssertionError("trace persistence failure must abort the run")
    assert waits == []


def test_auth_error_is_surfaced_before_any_pause():
    # An auth wall fails identically every round: sleeping through eight of them
    # only makes the wrong answer slower, and the onboarding layer needs the
    # error itself (see preflight.explain_failure).
    from harness import preflight
    cfg = _cfg(max_iters=4, retry_backoff=1.0)
    waits: list[float] = []
    try:
        optimize(_ScriptedFailures(99, "Error code: 401 - authentication failed"),
                 cfg, _panel(cfg), sleep=waits.append)
    except Exception as exc:  # noqa: BLE001
        assert preflight.looks_like_auth_error(str(exc))
    else:
        raise AssertionError("an auth error must surface, not be retried")
    assert waits == [], "an auth failure must not be backed off"


def test_backoff_leaves_the_trace_shape_untouched():
    """The retry IS the next round, so a paced search still records one per round.

    This is the invariant the replay length check rests on: if a retry made an
    extra provider call, a run that hit one 429 could never be replayed.
    """
    cfg = _cfg(max_iters=4, retry_backoff=0.0)
    rec = RecordingProvider(_ScriptedFailures(2))
    optimize(rec, cfg, _panel(cfg), sleep=lambda _s: None)
    assert len(rec.trace) == cfg.max_iters
    assert [("error" in e) for e in rec.trace] == [True, True, False, False]


# --- token accounting -----------------------------------------------------------
class _MeteredProvider(LLMProvider):
    """Answers with a valid proposal and reports usage the way an SDK would."""

    name = "metered"

    def __init__(self, usage: dict | None):
        self.usage = usage

    def complete(self, system, messages, *, model="m", seed=17, max_tokens=2048):
        meta = {"usage": dict(self.usage)} if self.usage is not None else {}
        return ProviderResponse(text=GOOD, model=model, provider=self.name,
                                prompt_hash="h", meta=meta)


def test_extract_usage_normalises_both_sdk_dialects():
    anthropic = ProviderResponse(text="", model="m", provider="p",
                                 meta={"usage": {"input_tokens": 10,
                                                 "output_tokens": 4}})
    assert extract_usage(anthropic) == {"input_tokens": 10, "output_tokens": 4,
                                        "total_tokens": 14}
    openai = ProviderResponse(text="", model="m", provider="p",
                              meta={"usage": {"prompt_tokens": 7,
                                              "completion_tokens": 3,
                                              "total_tokens": 10}})
    assert extract_usage(openai) == {"input_tokens": 7, "output_tokens": 3,
                                     "total_tokens": 10}
    # a provider that reports nothing is "not costed", which is NOT "cost zero"
    assert extract_usage(ProviderResponse(text="", model="m", provider="p")) is None


def test_usage_is_metered_per_date_and_summed_for_the_run():
    cfg = _cfg(max_iters=3, research_date="2023-01-01")
    meter = UsageMeter(_MeteredProvider({"input_tokens": 100, "output_tokens": 20}))
    rec = run_once(cfg, provider=meter, register=False)

    assert rec["usage"] == {"input_tokens": 300, "output_tokens": 60,
                            "total_tokens": 360, "n_calls": 3, "n_reported": 3}
    # the manifest total is summed off the LEDGER, so a resumed run costs the
    # dates it inherited as well as the ones it ran
    assert _usage_total([rec, rec]) == {"input_tokens": 600, "output_tokens": 120,
                                        "total_tokens": 720, "n_calls": 6,
                                        "n_reported": 6}
    assert meter.usage_totals["n_calls"] == 3
    assert meter.name == "metered", "the meter must stay transparent"


def test_a_provider_that_reports_no_usage_is_recorded_as_uncosted():
    cfg = _cfg(max_iters=2, research_date="2023-01-01")
    rec = run_once(cfg, provider=UsageMeter(_MeteredProvider(None)), register=False)
    assert rec["usage"] is None, "0 tokens would be a lie, not a measurement"
    assert _usage_total([rec]) is None

    # ... and so is a whole scripted walk-forward, end to end
    with tempfile.TemporaryDirectory() as root:
        out = run_walk_forward(_cfg(), artifacts_root=root)
        assert out["manifest"]["usage"] is None
        assert all(r["usage"] is None for r in out["records"])


class _StructuredProvider(LLMProvider):
    name = "structured"
    supports_response_format = True
    supports_structured_output = True

    def __init__(self):
        self.seen = []

    def complete(self, system, messages, *, model="m", seed=17, max_tokens=2048,
                 response_format=None, structured_schema=None,
                 structured_retry_count=2):
        self.seen.append((response_format, structured_schema, structured_retry_count))
        return ProviderResponse(text=GOOD, model=model, provider=self.name,
                                prompt_hash="h")


def test_usage_meter_preserves_wrapped_provider_capabilities():
    inner = _StructuredProvider()
    provider = UsageMeter(RecordingProvider(inner))
    response_format = {"type": "json_object"}
    schema = {"type": "object", "properties": {"formula": {"type": "string"}}}

    assert provider.supports_response_format is True
    assert provider.supports_structured_output is True
    complete_provider(
        provider, "system", [], model="m", response_format=response_format,
        structured_schema=schema, structured_retry_count=4,
    )

    assert inner.seen == [(response_format, schema, 4)]
