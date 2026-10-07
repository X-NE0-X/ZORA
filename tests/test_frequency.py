"""Walk-forward step (``frequency``): quarterly, validation, and the data-loss bug.

The walk steps T_n from T_0 to T_p by a pandas offset alias. Quarterly (``QS``)
always worked mechanically --- nothing but the docs said so --- but the field had
NO validation at all, and that was expensive: ``run_walk_forward`` used to clear
the run directory before it built the schedule, so one mistyped alias deleted a
finished run's ledger AND its irreplaceable LLM trace and then died in a raw
pandas traceback. Two layers now stop that: ``RunConfig`` rejects a bad alias at
construction, and the schedule is built before anything destructive happens.
"""
import json
import tempfile
import warnings
import pandas as pd

from harness.config import FREQUENCIES, RunConfig
from harness.providers.scripted import ScriptedProvider
from harness.runner import (_deep_eq, _evolution_dates, _replay_core, _step_config,
                            replay, required_data_window, run_walk_forward)
from harness.store import Store
import harness.cli as _cli
import harness.runner as _runner

SYMS = ["A", "B", "C", "D", "E"]


def _obj(formula, sign=1):
    formula = formula.replace("delta(close, 5)", "delta(close,w)")
    return {"formula": formula, "parameters": {"w": {"type": "Window", "value": 5}}, "rationale": "reason for " + formula,
            "mechanism": "the mechanism behind " + formula, "expected_sign": sign}


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2021-01-01", t_p="2021-10-01", frequency="QS",
        provider="scripted", seed=17, max_iters=1,
        run_name="freq_wf", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def _with_scripted(script, fn):
    orig = _runner.get_provider
    _runner.get_provider = (
        lambda name, **kw: ScriptedProvider(list(script)) if name == "scripted"
        else orig(name, **kw))
    try:
        return fn()
    finally:
        _runner.get_provider = orig


def _script_for(cfg):
    return [json.dumps(_obj("-delta(close, 5)"))] * (len(_evolution_dates(cfg)) * cfg.max_iters)


def _days(freq, t_0="2020-01-01", t_p="2021-01-01"):
    return [d.date().isoformat()
            for d in _evolution_dates(RunConfig(frequency=freq, t_0=t_0, t_p=t_p))]


# --- the advertised set -----------------------------------------------------
def test_every_advertised_frequency_builds_a_schedule():
    # whatever the error text / CLI help / TUI caption names must actually work.
    for freq in FREQUENCIES:
        assert len(_days(freq)) >= 1, freq


def test_quarterly_steps_four_times_a_year():
    # the answer to "is it only year/month/week?": QS is a first-class step.
    assert _days("QS") == ["2020-01-01", "2020-04-01", "2020-07-01",
                           "2020-10-01", "2021-01-01"]
    assert _days("2QS") == ["2020-01-01", "2020-07-01", "2021-01-01"]
    assert _days("QE")[:2] == ["2020-03-31", "2020-06-30"]      # quarter END
    # an anchored quarter starts on its own anchor, which may skip T_0 entirely
    # --- documented, not a bug, but the reason 'QS' (Jan/Apr/Jul/Oct) is the
    # default advice rather than 'QS-DEC' (Dec/Mar/Jun/Sep).
    assert _days("QS-DEC")[0] == "2020-03-01"


def test_multiples_and_anchors_are_accepted():
    assert _days("6MS") == ["2020-01-01", "2020-07-01", "2021-01-01"]
    assert _days("13W")[:2] == ["2020-01-05", "2020-04-05"]
    assert _days("W-MON")[0] == "2020-01-06"


def test_per_step_oos_uses_the_next_research_stop():
    cfg = _cfg(oos_mode="per_step", warmup=252, t_0="2021-01-01",
               t_p="2021-07-01", frequency="QS")
    first, second = _evolution_dates(cfg)[:2]
    step = _step_config(cfg, first, second)
    assert step.oos_window() == (first, second - pd.Timedelta(days=1))
    required_start, required_end = required_data_window(cfg, walk_forward=True)
    assert required_start == pd.Timestamp("2016-01-01") - pd.tseries.offsets.BDay(252)
    assert required_end == pd.Timestamp("2021-10-01") - pd.Timedelta(days=1)


def test_data_bounds_fail_before_resetting_an_existing_run():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(run_name="range_guard")
        store, n_factors, n_trace = _record_a_run(root, cfg)
        too_late = _cfg(run_name="range_guard", data_start="2020-01-01")
        try:
            run_walk_forward(too_late, artifacts_root=root)
            raise AssertionError("late data_start must be rejected")
        except ValueError as exc:
            assert "data_start" in str(exc) and "required" in str(exc)
        assert len(store.read_factors()) == n_factors
        assert len(store.read_trace()) == n_trace

        too_early = _cfg(run_name="range_guard", data_end="2021-06-30")
        try:
            run_walk_forward(too_early, artifacts_root=root)
            raise AssertionError("early data_end must be rejected")
        except ValueError as exc:
            assert "data_end" in str(exc) and "required" in str(exc)
        assert len(store.read_factors()) == n_factors
        assert len(store.read_trace()) == n_trace


# --- validation -------------------------------------------------------------
def test_a_typo_is_rejected_at_construction():
    # including the pandas-2 spellings pandas 3 REMOVED outright (not deprecated):
    # 'Q' was the natural guess for quarterly and is now a hard error.
    for bad in ("Q", "M", "Y", "A", "AS", "BQ", "QQ", "quarterly", "qs", "", None):
        try:
            RunConfig(frequency=bad)
            raise AssertionError(f"frequency={bad!r} must be rejected")
        except ValueError as exc:
            assert "frequency" in str(exc), (bad, exc)
    # and the message points at something that works
    try:
        RunConfig(frequency="Q")
    except ValueError as exc:
        assert "QS" in str(exc) and "pandas 3" in str(exc)


def test_sub_daily_and_non_advancing_steps_are_rejected():
    # finer than a bar: two stops share a research_date --- the ledger's key ---
    # so the second is silently dropped on resume and replay then fails.
    for bad in ("4h", "30min", "1s"):
        try:
            RunConfig(frequency=bad)
            raise AssertionError(f"frequency={bad!r} must be rejected")
        except ValueError as exc:
            assert "finer than the daily bar" in str(exc), (bad, exc)
    # a step that does not move the clock forward is not a schedule
    for bad in ("0D", "-1D", "-2QS"):
        try:
            RunConfig(frequency=bad)
            raise AssertionError(f"frequency={bad!r} must be rejected")
        except ValueError as exc:
            assert "step FORWARD" in str(exc), (bad, exc)


# --- the data-loss regression ----------------------------------------------
def _record_a_run(root, cfg):
    script = _script_for(cfg)
    out = _with_scripted(script, lambda: run_walk_forward(cfg, artifacts_root=root))
    store = Store(root, cfg.run_name)
    assert len(store.read_factors()) == len(out["records"]) >= 2
    assert len(store.read_trace()) >= 2
    return store, len(store.read_factors()), len(store.read_trace())


def test_a_broken_schedule_cannot_destroy_a_recorded_run():
    # The bug: run_walk_forward reset the run directory (ledger + LLM trace, which
    # cost real money and cannot be regenerated) BEFORE it built the schedule, so a
    # config that could never run still wiped the previous one. Both ways of
    # reaching that point are covered: a frequency __post_init__ cannot see
    # (assigned onto a built config, as --set / a hand-edited json effectively
    # does), and an empty schedule, which no field-level check can catch.
    for label, mutate in (
        ("bad frequency", lambda c: setattr(c, "frequency", "QQ")),
        ("empty schedule", lambda c: setattr(c, "t_0", "2023-06-01")),
    ):
        with tempfile.TemporaryDirectory() as root:
            cfg = _cfg()
            store, n_factors, n_trace = _record_a_run(root, cfg)
            broken = _cfg()
            mutate(broken)                       # bypasses __post_init__ on purpose
            try:
                run_walk_forward(broken, artifacts_root=root)
                raise AssertionError(f"{label}: must refuse to run")
            except ValueError:
                pass
            assert len(store.read_factors()) == n_factors, label
            assert len(store.read_trace()) == n_trace, label


# --- record -> replay on a quarterly walk -----------------------------------
def test_quarterly_walk_records_and_replays_bit_for_bit():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(frequency="QS", t_0="2021-01-01", t_p="2022-01-01",
                   run_name="freq_qs")
        script = _script_for(cfg)
        out = _with_scripted(script,
                             lambda: run_walk_forward(cfg, artifacts_root=root))
        recorded = out["records"]
        assert [r["research_date"] for r in recorded] == [
            "2021-01-01", "2021-04-01", "2021-07-01", "2021-10-01", "2022-01-01"]
        store = Store(root, cfg.run_name)
        trace, manifest = store.read_trace(), store.read_manifest()
        assert len(trace) == len(recorded) * cfg.max_iters
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)   # prompt drift = failure
            replayed = replay(manifest, trace)
        assert len(replayed) == len(recorded)
        for a, b in zip(recorded, replayed):
            assert _deep_eq(_replay_core(a), _replay_core(b)), a["research_date"]


def test_a_short_step_with_a_long_oos_starves_the_learning_context():
    # Not a bug, but the consequence of choosing QS that will surprise someone:
    # learning-by-doing is point-in-time, so a date only sees earlier factors whose
    # OOS window has CLOSED. With a quarterly step and the default oos_days=252
    # (~1 trading year) the first four stops each learn from nothing.
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(frequency="QS", t_0="2021-01-01", t_p="2022-01-01",
                   oos_days=252, run_name="freq_starve")
        script = _script_for(cfg)
        out = _with_scripted(script,
                             lambda: run_walk_forward(cfg, artifacts_root=root))
        seen = [r["n_prior_factors"] + r["n_prior_failures"] for r in out["records"]]
        assert seen[:4] == [0, 0, 0, 0] and seen[4] >= 1, seen


# --- CLI --------------------------------------------------------------------
def test_cli_frequency_flag_overrides_the_config():
    args = _cli.build_parser().parse_args(["run", "--frequency", "QS"])
    assert _cli._load_config(args).frequency == "QS"
    assert _cli._load_config(
        _cli.build_parser().parse_args(["run"])).frequency == "YS"
    # a junk alias fails at config-merge time, before any run directory is touched
    bad = _cli.build_parser().parse_args(["run", "--frequency", "Q"])
    try:
        _cli._load_config(bad)
        raise AssertionError("--frequency Q must be rejected")
    except ValueError:
        pass
