"""Kill-and-resume: the ledger is the checkpoint. A resumed walk-forward keeps
what was already recorded, skips those dates instead of re-running them, and
seeds the learning-by-doing context from the earlier PASS factors."""
import json
import contextlib
import io
import os
import tempfile

from harness.config import RunConfig
from harness.data import _harness_root
from harness.runner import (_config_conflicts, _deep_eq, _replay_core, norm_config,
                            replay, run_walk_forward)
from harness.store import Store

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        frequency="YS", provider="scripted", seed=17, max_iters=2,
        run_name="test_resume", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def test_resume_skips_done_dates_and_keeps_ledger():
    with tempfile.TemporaryDirectory() as root:
        # phase 1 --- a "killed" run that only got through the first date (2021)
        first = run_walk_forward(_cfg(t_0="2021-01-01", t_p="2021-01-01"),
                                 artifacts_root=root)
        assert [r["research_date"] for r in first["records"]] == ["2021-01-01"]

        store = Store(root, "test_resume")
        pre_ledger = store.read_factors()
        pre_rec_2021 = pre_ledger[0]
        pre_trace_len = len(store.read_trace())
        assert len(pre_ledger) == 1

        # phase 2 --- resume with the full schedule (2021 + 2022)
        out = run_walk_forward(_cfg(t_0="2021-01-01", t_p="2022-01-01"),
                               artifacts_root=root, resume=True)
        dates = [r["research_date"] for r in out["records"]]

        # ledger was NOT wiped: 2021 kept, 2022 appended, each exactly once
        assert dates == ["2021-01-01", "2022-01-01"]
        after = store.read_factors()
        assert len(after) == 2
        # the pre-existing 2021 record is untouched (not re-run)
        assert after[0] == pre_rec_2021

        # only the resumed date added trace entries (2021 was skipped, not re-called)
        assert len(store.read_trace()) == pre_trace_len + _cfg().max_iters

        # learning-by-doing seeded from the existing ledger: 2022 saw exactly the
        # PASS factors already on disk from 2021
        expected_prior = sum(1 for r in pre_ledger if r["verdict"]["passed"])
        rec_2022 = next(r for r in after if r["research_date"] == "2022-01-01")
        assert rec_2022["n_prior_factors"] == expected_prior


def test_resume_drops_orphan_trace_from_partial_kill():
    # Regression (H1): a date killed mid-run leaves orphan trace entries on disk
    # with no ledger row. Resume must cut them so the trace realigns with the
    # ledger and replay stays bit-exact --- otherwise replay consumes the orphans
    # in place of the re-run date's completions and diverges from date 2 onward.
    with tempfile.TemporaryDirectory() as root:
        mi = _cfg().max_iters
        # date 1 (2020) completes cleanly: ledger=[2020], trace=[2020 x mi]
        run_walk_forward(_cfg(t_0="2020-01-01", t_p="2020-01-01",
                              run_name="test_orphan"), artifacts_root=root)
        store = Store(root, "test_orphan")
        assert len(store.read_factors()) == 1
        assert len(store.read_trace()) == mi

        # simulate date 2 being called a few times, then the process dying BEFORE
        # its ledger row was written -> orphan completions sitting in the trace.
        # Make them valid-but-distinct proposals: if they ever leak into replay,
        # the replayed formula would differ and the assertions below would catch it.
        orphan = json.dumps({"formula": "close", "parameters": {}, "rationale": "orphan " * 12,
                             "mechanism": "orphan mechanism " * 6, "expected_sign": 1})
        k = min(3, mi - 1)  # a partial date, not a completed proposal block
        for _ in range(k):
            store.append_trace({"text": orphan, "model": "x",
                                "provider": "scripted", "prompt_hash": ""})
        assert len(store.read_trace()) == mi + k          # orphans present on disk

        # resume the full 3-date schedule
        out = run_walk_forward(_cfg(t_0="2020-01-01", t_p="2022-01-01",
                                    run_name="test_orphan"),
                               artifacts_root=root, resume=True)
        recorded = out["records"]
        assert [r["research_date"] for r in recorded] == \
            ["2020-01-01", "2021-01-01", "2022-01-01"]

        # orphans were truncated: exactly one clean completion block per date
        trace = store.read_trace()
        assert len(trace) == mi * len(recorded), \
            "orphan completions must be dropped on resume so the trace realigns"

        # and the realigned run replays bit-identically (no orphan leakage)
        replayed = replay(store.read_manifest(), trace)
        assert len(replayed) == len(recorded)
        for rec, rep in zip(recorded, replayed):
            assert _deep_eq(_replay_core(rec), _replay_core(rep)), \
                f"replay diverged at {rec['research_date']} (orphan trace leaked)"


# --- the resume guard's notion of "same data" ---------------------------------
REL_PARQUET = "artifacts/cache/Data_EQT_US_D_abc.parquet"


def _spellings_of_the_same_file() -> list[str]:
    """The same parquet written the ways a real config actually spells it."""
    native_abs = os.path.join(str(_harness_root()), "artifacts", "cache",
                              "Data_EQT_US_D_abc.parquet")
    return [
        REL_PARQUET,                                          # portable, committed
        native_abs,                                           # this machine's absolute
        os.path.join("artifacts", "cache", "Data_EQT_US_D_abc.parquet"),  # native rel
        "artifacts/cache/../cache/Data_EQT_US_D_abc.parquet",  # a redundant detour
    ]


def test_parquet_identity_is_one_identity_across_path_spellings():
    """A committed config must name the same data on Windows and on Linux.

    ``norm_config`` used to resolve to a native ABSOLUTE path, which is exactly
    the machine-dependent thing it exists to remove: two colleagues running the
    identical committed config got different ``run_id``s (duplicate journal
    entries, wrong dedup counts) and ``--resume`` refused with "config differs on
    ['parquet_daily']". The identity is now harness-root-relative with forward
    slashes, so it is the same string on every platform.
    """
    ident = {tuple(norm_config(_cfg(data_source="ctx", parquet_daily=[p]).to_dict())
                   ["parquet_daily"])
             for p in _spellings_of_the_same_file()}
    assert len(ident) == 1, f"one file, {len(ident)} identities: {ident}"
    only = next(iter(ident))
    assert only == (REL_PARQUET,), only
    assert "\\" not in only[0] and not os.path.isabs(only[0]), \
        "the identity must carry neither a native separator nor a machine path"

    # ... so the resume guard reads the two spellings as the same run
    for spelling in _spellings_of_the_same_file():
        assert _config_conflicts(
            _cfg(data_source="ctx", parquet_daily=[REL_PARQUET]).to_dict(),
            _cfg(data_source="ctx", parquet_daily=[spelling]).to_dict(),
        ) == [], f"{spelling} must not read as a different run"

    # a genuinely different FILE still conflicts --- the guard is not toothless
    assert _config_conflicts(
        _cfg(data_source="ctx", parquet_daily=[REL_PARQUET]).to_dict(),
        _cfg(data_source="ctx",
             parquet_daily=["artifacts/cache/Data_EQT_US_D_zzz.parquet"]).to_dict(),
    ) == ["parquet_daily"]


def test_backoff_knobs_do_not_block_a_resume():
    """Pacing is operational, not scientific: it may change on a resume.

    ``retry_backoff`` / ``retry_max_delay`` only decide how long a FAILING round
    waits; they change nothing computed and nothing about how many completions a
    clean round consumes. If the guard compared them, turning the pause off to
    debug a rate limit would make the run unresumable.
    """
    base = _cfg().to_dict()
    for field, value in (("retry_backoff", 0.0), ("retry_max_delay", 5.0)):
        changed = dict(base)
        changed[field] = value
        assert _config_conflicts(base, changed) == [], \
            f"{field} must not block a resume"


def test_resume_on_empty_ledger_runs_all():
    # resume with nothing recorded yet must behave like a fresh run (no crash)
    with tempfile.TemporaryDirectory() as root:
        out = run_walk_forward(_cfg(t_0="2021-01-01", t_p="2022-01-01"),
                               artifacts_root=root, resume=True)
        assert [r["research_date"] for r in out["records"]] == \
            ["2021-01-01", "2022-01-01"]


def test_single_run_resume_is_explicitly_rejected():
    from harness import cli

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(["run", "--resume", "--run-name", "single_resume"])
    assert code == 2
    assert "only supported for walk-forward" in buf.getvalue()
