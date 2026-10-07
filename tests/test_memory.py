"""Research journal: a durable, self-sufficient record that outlives artifacts/.

The journal's whole reason to exist is that ``artifacts/`` is disposable ---
git-ignored, wiped by ``Store.reset()`` on every re-use of a ``run_name``. So the
load-bearing properties are: an entry is self-contained, a re-run of the SAME
experiment updates one entry instead of forking a near-duplicate, a genuinely
different experiment gets its own, and the markdown is a pure function of
``raw/`` (so it can never drift from the records it claims to summarise).
"""
import json
import os
import shutil
import tempfile

from harness import cli, memory
from harness.config import RunConfig
from harness.factor.contract import POLICY_HASH
from tests.factor_fixtures import window_bindings
from harness.providers.base import LLMProvider, ProviderResponse
from harness.proposer import build_messages
from harness.runner import (_RESUME_IGNORE, _journal_snapshot, _replay_core,
                            _deep_eq, oos_close_date, replay, run_walk_forward)
from harness.store import Store

STAMP = "2026-01-02T03:04:05Z"


def _manifest(**kw) -> dict:
    base = dict(run_name="j", symbols=["A", "B", "C"], data_source="synthetic",
                logging=False)
    base.update(kw)
    cfg = RunConfig(**base).to_dict()
    return {"run_name": "j", "config": cfg, "math_policy_hash": POLICY_HASH, "provider": "scripted",
            "model": "m", "objective": cfg["objective"],
            "pass_line": cfg["pass_line"], "seed": 17,
            "data_version": "cafebabe0000", "n_factors": 1, "n_pass": 0}


def _factor(date: str, *, passed: bool, formula="rank(close)", oos=-1.5,
            reasons=None, oos_days=252) -> dict:
    return {
        "research_date": date, "formula": formula,
        "parameters": window_bindings(formula),
        "oos_end": oos_close_date(date, oos_days).date().isoformat(),
        "mechanism": "m", "rationale": "r", "expected_sign": -1,
        "verdict": {"passed": passed, "label": "PASS" if passed else "FAIL",
                    "objective": "sortino", "is_value": 0.5, "oos_value": oos,
                    "reasons": [] if passed else (reasons or [
                        f"OOS Sortino {oos:.4f} fails Pass Line (>= 1.0000)"])},
    }


# --- identity ----------------------------------------------------------------
def test_run_id_is_content_addressed():
    base = _manifest()
    assert memory.run_id(base) == memory.run_id(_manifest()), "must be stable"
    # a real knob is part of the experiment's identity
    assert memory.run_id(_manifest(pass_line=2.0)) != memory.run_id(base)
    assert memory.run_id(_manifest(max_iters=3)) != memory.run_id(base)
    # ... and so is the data the run was scored on
    other = _manifest()
    other["data_version"] = "0000deadbeef"
    assert memory.run_id(other) != memory.run_id(base)
    assert memory.run_id(base).startswith("j-")


def test_run_id_is_over_meaning_not_spelling():
    """The same parquet named two ways is the same experiment, not two.

    Identity goes through ``runner.norm_config``, so re-pointing a config at the
    same file with a repo-relative instead of an absolute path updates the entry
    instead of forking a near-duplicate. A different FILE still forks.
    """
    from harness.data import _harness_root

    rel = "artifacts/cache/Data_EQT_US_D_abc.parquet"
    absolute = str(_harness_root() / "artifacts" / "cache" / "Data_EQT_US_D_abc.parquet")
    assert memory.run_id(_manifest(parquet_daily=[rel])) == \
        memory.run_id(_manifest(parquet_daily=[absolute]))
    assert memory.run_id(_manifest(parquet_daily=[rel])) != \
        memory.run_id(_manifest(parquet_daily=["artifacts/cache/Data_EQT_US_D_zzz.parquet"]))


def test_resume_tolerated_fields_do_not_fork_a_new_entry():
    """Extending t_p continues a run; it must land back in the SAME entry."""
    base = _manifest()
    for field, value in (("t_p", "2030-01-01"), ("logging", True),
                         ("data_dir", "somewhere/else")):
        assert memory.run_id(_manifest(**{field: value})) == memory.run_id(base), \
            f"{field} must not change run identity"


def test_identity_ignore_mirrors_the_resume_guard():
    # The two answer the same question ("may this differ and still be the same
    # run?"). If they ever drift, a resumed run silently forks a duplicate entry.
    assert memory._IDENTITY_IGNORE is _RESUME_IGNORE


def test_run_id_rejects_an_unsafe_run_name():
    bad = _manifest()
    bad["run_name"] = "../../etc"
    try:
        memory.run_id(bad)
    except ValueError:
        return
    raise AssertionError("a traversal run_name must not become a journal filename")


# --- archive -----------------------------------------------------------------
def test_archive_is_idempotent_and_merges_dates():
    with tempfile.TemporaryDirectory() as mem:
        m = _manifest()
        a = memory.archive(m, [_factor("2021-01-01", passed=False)], mem, now=STAMP)
        b = memory.archive(m, [_factor("2021-01-01", passed=False)], mem,
                           now="2026-02-02T00:00:00Z")
        assert a["run_id"] == b["run_id"]
        assert len(b["factors"]) == 1, "re-archiving must not duplicate a date"
        assert b["first_archived"] == STAMP, "the original archive time is kept"
        assert b["last_archived"] == "2026-02-02T00:00:00Z"
        # a continued run contributes its new dates to the same entry
        c = memory.archive(m, [_factor("2021-01-01", passed=False),
                               _factor("2022-01-01", passed=True, oos=1.9)], mem)
        assert [f["research_date"] for f in c["factors"]] == ["2021-01-01", "2022-01-01"]
        assert len(os.listdir(memory.raw_dir(mem))) == 1


def test_hand_written_note_survives_a_refresh():
    with tempfile.TemporaryDirectory() as mem:
        m = _manifest()
        e = memory.archive(m, [_factor("2021-01-01", passed=False)], mem)
        path = os.path.join(memory.raw_dir(mem), e["run_id"] + ".json")
        stored = json.loads(open(path, encoding="utf-8").read())
        stored["note"] = "first pass at HK reversal"
        open(path, "w", encoding="utf-8").write(json.dumps(stored))

        again = memory.archive(m, [_factor("2021-01-01", passed=False)], mem)
        assert again["note"] == "first pass at HK reversal"
        assert "first pass at HK reversal" in memory.render_notes([again])


def test_archive_store_skips_an_empty_run():
    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        assert memory.archive_store(Store(art, "never_ran"), mem) is None
        assert memory.load_entries(mem) == []


def test_remember_never_raises_on_a_broken_journal():
    """A journal problem must not fail a run whose ledger is already on disk."""
    with tempfile.TemporaryDirectory() as art:
        store = Store(art, "r")
        store.write_manifest(_manifest())
        store.append_factor(_factor("2021-01-01", passed=False))
        # a FILE where the journal root must be a directory -> OSError inside
        with tempfile.TemporaryDirectory() as d:
            blocked = os.path.join(d, "memory")
            open(blocked, "w").close()
            try:
                memory.archive_store(store, blocked)
            except OSError:
                pass                      # confirms the path really does blow up
            else:
                raise AssertionError("expected the blocked journal root to raise")
            assert memory.remember(store, blocked) is None, "remember must swallow it"
            # ... and the same store archives fine once the root is usable
            assert memory.remember(store, os.path.join(d, "ok")) is not None


# --- the durability promise --------------------------------------------------
def test_journal_outlives_the_artifacts_tree():
    with tempfile.TemporaryDirectory() as mem:
        art = tempfile.mkdtemp()
        store = Store(art, "wf")
        store.write_manifest(_manifest())
        store.append_factor(_factor("2021-01-01", passed=True,
                                    formula='ts_mean(volume,w10)', oos=1.7))
        assert memory.remember(store, mem) is not None

        shutil.rmtree(art)                      # artifacts/ is disposable
        entries = memory.load_entries(mem)
        assert len(entries) == 1
        assert entries[0]["factors"][0]["formula"] == 'ts_mean(volume,w10)'
        assert 'ts_mean(volume,w10)' in open(memory.notes_path(mem),
                                             encoding="utf-8").read()


# --- render ------------------------------------------------------------------
def test_notes_render_is_a_pure_function():
    # no clock, no I/O: the markdown can only ever say what raw/ says.
    e = {"run_id": "j-1", "first_archived": STAMP, "last_archived": STAMP,
         "note": "", "manifest": _manifest(),
         "factors": [_factor("2021-01-01", passed=False)]}
    assert memory.render_notes([e]) == memory.render_notes([e])
    assert STAMP in memory.render_notes([e])


def test_notes_empty_state_is_explanatory():
    out = memory.render_notes([])
    assert "No runs archived yet" in out
    assert out.startswith("# ZORA Research Notes")


def test_notes_split_wins_from_losses_and_group_failure_modes():
    e = {"run_id": "j-1", "first_archived": STAMP, "last_archived": STAMP,
         "note": "", "manifest": _manifest(),
         "factors": [
             _factor("2021-01-01", passed=True, formula="rank(volume)", oos=1.8),
             _factor("2022-01-01", passed=False, formula="rank(close)", oos=-1.1),
             # same failure MODE, different numbers -> must tally as 2, not 1x2
             _factor("2023-01-01", passed=False, formula='delta(close,w5)', oos=-0.4),
         ]}
    out = memory.render_notes([e])
    assert "`rank(volume)`" in out.split("## What did not work")[0], \
        "a PASS factor belongs under 'What worked'"
    modes = out.split("### Dominant failure modes")[1].split("### Rejected")[0]
    assert "| 2 | OOS Sortino # fails Pass Line (>= #) |" in modes
    # operator usage separates winning from losing primitives
    usage = out.split("## Operator usage")[1]
    assert "| rank | 1 | 1 |" in usage
    assert "| delta | 0 | 1 |" in usage


def test_one_corrupt_entry_does_not_sink_the_journal():
    with tempfile.TemporaryDirectory() as mem:
        good = memory.archive(_manifest(), [_factor("2021-01-01", passed=False)], mem)
        # a half-written file (hard kill mid-archive) must be skipped, not fatal
        open(os.path.join(memory.raw_dir(mem), "j-truncated.json"),
             "w", encoding="utf-8").write('{"run_id": "j-trunc", "fact')
        entries = memory.load_entries(mem)
        assert [e["run_id"] for e in entries] == [good["run_id"]]
        assert os.path.exists(memory.rebuild(mem))


def test_render_survives_a_formula_it_cannot_parse():
    e = {"run_id": "j-1", "first_archived": STAMP, "last_archived": STAMP,
         "note": "", "manifest": _manifest(),
         "factors": [_factor("2021-01-01", passed=False, formula="!! not a formula")]}
    out = memory.render_notes([e])                     # must not raise
    assert "!! not a formula" in out


# --- synthesis ---------------------------------------------------------------
class _OneShotProvider(LLMProvider):
    name = "stub"

    def __init__(self, text="### Where the evidence points\nreversal keeps dying."):
        self.text = text
        self.calls = 0

    def complete(self, system, messages, *, model, seed=17, max_tokens=2048):
        self.calls += 1
        self.system, self.messages = system, messages
        return ProviderResponse(text=self.text, model=model, provider=self.name)


def test_synthesis_costs_one_call_and_is_embedded():
    with tempfile.TemporaryDirectory() as mem:
        memory.archive(_manifest(), [_factor("2021-01-01", passed=False)], mem)
        p = _OneShotProvider()
        payload = memory.synthesize(p, RunConfig(logging=False), mem, now=STAMP)
        assert p.calls == 1, "the narrative must cost exactly one completion"
        # it is shown the deterministic digest, not raw JSON
        assert "## What did not work" in p.messages[0]["content"]
        notes = open(memory.notes_path(mem), encoding="utf-8").read()
        assert "## Synthesis" in notes
        assert "reversal keeps dying." in notes
        assert "STALE" not in notes
        assert payload["fingerprint"] == memory.corpus_fingerprint(
            memory.load_entries(mem))


def test_synthesis_is_marked_stale_when_the_journal_moves_on():
    with tempfile.TemporaryDirectory() as mem:
        memory.archive(_manifest(), [_factor("2021-01-01", passed=False)], mem)
        memory.synthesize(_OneShotProvider(), RunConfig(logging=False), mem, now=STAMP)
        # re-archiving the SAME content must not invalidate it (timestamps only)
        memory.archive(_manifest(), [_factor("2021-01-01", passed=False)], mem)
        memory.rebuild(mem)
        assert "STALE" not in open(memory.notes_path(mem), encoding="utf-8").read()
        # a new result must
        memory.archive(_manifest(), [_factor("2022-01-01", passed=True, oos=2.0)], mem)
        memory.rebuild(mem)
        assert "STALE" in open(memory.notes_path(mem), encoding="utf-8").read()


def test_synthesize_on_an_empty_journal_is_a_no_op():
    with tempfile.TemporaryDirectory() as mem:
        p = _OneShotProvider()
        assert memory.synthesize(p, RunConfig(logging=False), mem) is None
        assert p.calls == 0, "an empty journal must not burn a completion"


# --- CLI ---------------------------------------------------------------------
def _with_cli_roots(art, mem, fn):
    prev = (cli._ARTIFACTS, cli._MEMORY)
    cli._ARTIFACTS, cli._MEMORY = art, mem
    try:
        return fn()
    finally:
        cli._ARTIFACTS, cli._MEMORY = prev


def test_cli_memory_archives_lists_and_rebuilds():
    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        store = Store(art, "j")
        store.write_manifest(_manifest())
        store.append_factor(_factor("2021-01-01", passed=False))

        assert _with_cli_roots(art, mem, lambda: cli.main(["memory", "--list"])) == 1
        assert _with_cli_roots(
            art, mem, lambda: cli.main(["memory", "--archive", "--run-name", "j"])) == 0
        assert _with_cli_roots(art, mem, lambda: cli.main(["memory", "--list"])) == 0
        assert os.path.exists(memory.notes_path(mem))


def test_cli_refuses_to_synthesize_with_the_offline_double():
    """The scripted provider replays factor JSON; it must not land in the notes."""
    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        memory.archive(_manifest(), [_factor("2021-01-01", passed=False)], mem)
        code = _with_cli_roots(art, mem, lambda: cli.main(
            ["memory", "--synthesize", "--provider", "scripted"]))
        assert code == 2
        assert memory.read_synthesis(mem) is None, "nothing may be cached"
        assert "Synthesis" not in open(memory.rebuild(mem), encoding="utf-8").read()


def test_tui_writes_to_the_journal_too():
    """The TUI is the other way a run finishes; it must archive as well."""
    try:
        from harness.tui import HarnessTUI
    except ImportError:
        return                              # textual is an optional extra
    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        store = Store(art, "j")
        store.write_manifest(_manifest())
        store.append_factor(_factor("2021-01-01", passed=False))
        app = HarnessTUI(artifacts_root=art, memory_root=mem)
        said: list[str] = []
        app._remember(RunConfig(logging=False, memory=True), store, said.append)
        assert [e["run_id"] for e in memory.load_entries(mem)] == [
            memory.run_id(_manifest())]
        assert said and "journal:" in said[0]
        # the switch is honoured on the write side too
        app._remember(RunConfig(logging=False, memory=False), store, said.append)
        assert len(said) == 1


# --- lessons: the slice the proposer is shown ---------------------------------
def test_lessons_dedupe_by_formula_and_rank_repeats_first():
    with tempfile.TemporaryDirectory() as mem:
        memory.archive(_manifest(), [
            _factor("2021-01-01", passed=False, formula='-rank(delta(close,w5))'),
            _factor("2022-01-01", passed=False, formula='-rank(delta(close,w5))'),
            _factor("2023-01-01", passed=False, formula='-rank(delta(close,w5))'),
            _factor("2023-06-01", passed=False, formula='ts_std(returns,w20)'),
            _factor("2024-01-01", passed=True, formula='ts_mean(volume,w10)', oos=1.7),
        ], mem)
        les = memory.lessons(mem)
        # one lesson per formula, not per row --- and the repeat carries its weight
        assert [l["formula"] for l in les["losses"]] == [
            '-rank(delta(close,w5))', 'ts_std(returns,w20)']
        assert les["losses"][0]["n_seen"] == 3
        assert les["losses"][1]["n_seen"] == 1
        assert [w["formula"] for w in les["wins"]] == ['ts_mean(volume,w10)']
        assert les["n_runs"] == 1
        # the representative row is the most RECENT occurrence
        assert les["losses"][0]["research_date"] == "2023-01-01"
        # ... and the point-in-time stamp is the LAST occurrence's close date, so
        # the repeat count cannot reveal a verdict that is not knowable yet
        assert les["losses"][0]["oos_end"] == str(
            oos_close_date("2023-01-01", 252).date())


def test_lessons_respects_caps_and_is_none_when_empty():
    with tempfile.TemporaryDirectory() as mem:
        assert memory.lessons(mem) is None
        memory.archive(_manifest(), [
            _factor(f"20{10 + i}-01-01", passed=False, formula=f"delta(close,w{i+1})")
            for i in range(20)], mem)
        assert len(memory.lessons(mem, max_losses=5)["losses"]) == 5


def test_journal_is_absent_from_the_prompt_unless_supplied():
    """The replay-safety invariant: no journal -> byte-identical historic prompt."""
    cfg = RunConfig(logging=False)
    plain = build_messages(cfg, None)[0]["content"]
    assert build_messages(cfg, None, journal=None)[0]["content"] == plain
    assert "RESEARCH JOURNAL" not in plain


def _journal(oos_end: str | None) -> dict:
    """A two-lesson journal whose OOS windows all close on ``oos_end``."""
    return {"n_runs": 2,
            "wins": [{"formula": 'ts_mean(volume,w10)', "objective": "sortino",
                      "oos_value": 1.7, "research_date": "2021-01-01",
                      "oos_end": oos_end, "n_seen": 1}],
            "losses": [{"formula": '-rank(delta(close,w5))', "objective": "sortino",
                        "oos_value": -1.2, "n_seen": 4, "oos_end": oos_end,
                        "modes": ["OOS Sortino # fails Pass Line (>= #)"]}]}


def test_journal_reaches_the_proposer_prompt():
    cfg = RunConfig(logging=False, research_date="2023-01-01")
    content = build_messages(cfg, None, journal=_journal("2022-06-30"))[0]["content"]
    assert "RESEARCH JOURNAL" in content and "across 2 previous" in content
    assert 'ts_mean(volume,w10)' in content
    assert '-rank(delta(close,w5)) x4' in content, "repeat count must be shown"
    assert "OOS Sortino # fails Pass Line" in content


# --- the point-in-time gate on cross-run memory -------------------------------
def test_journal_lesson_is_withheld_until_its_oos_window_closes():
    """The leakage regression: an OOS grade must not reach a T_n before its window.

    ``run_id`` is content-addressed, so re-running an experiment reuses its
    journal entry --- and without this gate the second run would be handed, at
    simulated T_n, the exact score a formula earned on a holdout window that had
    not happened yet. Same information, same clock: it is look-ahead, not craft.
    """
    journal = _journal("2023-06-30")

    # T_n BEFORE the lesson's OOS window closes -> nothing about it may appear
    early = build_messages(
        RunConfig(logging=False, research_date="2023-01-01"),
        None, journal=journal)[0]["content"]
    assert "RESEARCH JOURNAL" not in early, "a fully-withheld journal renders no block"
    for leak in ('ts_mean(volume,w10)', "1.7000", '-rank(delta(close,w5))', "-1.2000"):
        assert leak not in early, f"{leak!r} leaked backwards in time"

    # the day the window closes it becomes knowable, and only then
    on_close = build_messages(
        RunConfig(logging=False, research_date="2023-06-30"),
        None, journal=journal)[0]["content"]
    assert 'ts_mean(volume,w10)' in on_close and "1.7000" in on_close
    assert '-rank(delta(close,w5))' in on_close and "-1.2000" in on_close

    # ... and stays visible afterwards
    later = build_messages(
        RunConfig(logging=False, research_date="2024-01-01"),
        None, journal=journal)[0]["content"]
    assert 'ts_mean(volume,w10)' in later


def test_a_lesson_with_no_recoverable_close_date_is_withheld():
    """Fail safe: "unknown" must never render as "already knowable"."""
    for missing in (None, "", "not-a-date"):
        content = build_messages(
            RunConfig(logging=False, research_date="2030-01-01"),
            None, journal=_journal(missing))[0]["content"]
        assert "RESEARCH JOURNAL" not in content, f"oos_end={missing!r} was not withheld"


def test_lessons_carry_the_originating_runs_own_oos_clock():
    """A lesson's close date comes from ITS run's oos_days, not the reader's.

    Two runs may hold out wildly different horizons. Computing the close date
    with whatever config happens to be reading the journal would declare a
    still-open window closed (or hide a closed one) purely by coincidence.
    """
    with tempfile.TemporaryDirectory() as mem:
        memory.archive(_manifest(run_name="short", oos_days=20),
                       [_factor("2021-01-01", passed=False, formula="rank(close)", oos_days=20)],
                       mem)
        memory.archive(_manifest(run_name="long", oos_days=500),
                       [_factor("2021-01-01", passed=False, formula="rank(volume)", oos_days=500)],
                       mem)
        by_formula = {l["formula"]: l for l in memory.lessons(mem)["losses"]}
        assert by_formula["rank(close)"]["oos_end"] == str(
            oos_close_date("2021-01-01", 20).date())
        assert by_formula["rank(volume)"]["oos_end"] == str(
            oos_close_date("2021-01-01", 500).date())

        # ... and the gate honours each of them separately at one T_n
        cfg = RunConfig(logging=False, research_date="2021-06-01")
        content = build_messages(cfg, None, journal=memory.lessons(mem))[0]["content"]
        assert "rank(close)" in content, "the 20-day holdout closed long ago"
        assert "rank(volume)" not in content, "the 500-day holdout is still open"


def test_lessons_are_unfiltered_so_replay_stays_reproducible():
    """The snapshot is the record; the gate is a render-time view of it.

    If ``lessons`` filtered, the manifest would hold a different journal for every
    T_n and a recorded run could not regenerate its own prompts.
    """
    with tempfile.TemporaryDirectory() as mem:
        memory.archive(_manifest(oos_days=500),
                       [_factor("2021-01-01", passed=False)], mem)
        les = memory.lessons(mem)
        assert len(les["losses"]) == 1, "nothing is dropped at snapshot time"
        assert les["losses"][0]["oos_end"] > "2021-01-01"


def test_memory_switch_gates_the_snapshot():
    with tempfile.TemporaryDirectory() as mem:
        memory.archive(_manifest(), [_factor("2021-01-01", passed=False)], mem)
        on = RunConfig(logging=False, memory=True)
        off = RunConfig(logging=False, memory=False)
        assert _journal_snapshot(on, mem) is not None
        assert _journal_snapshot(off, mem) is None, "config.memory=False must opt out"
        assert _journal_snapshot(on, None) is None, "no root -> hermetic by default"


# --- end to end --------------------------------------------------------------
def test_a_real_walk_forward_lands_in_the_journal():
    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        cfg = RunConfig(symbols=["A", "B", "C", "D", "E", "F"],
                        data_source="synthetic", data_start="2015-01-01",
                        data_end="2023-12-31", is_years=5, oos_days=126,
                        t_0="2021-01-01", t_p="2021-01-01", max_iters=2,
                        provider="scripted", run_name="test_memory_e2e",
                        logging=False)
        out = run_walk_forward(cfg, artifacts_root=art)
        entry = memory.remember(Store(art, cfg.run_name), mem)
        assert entry is not None
        assert len(entry["factors"]) == len(out["records"]) == 1
        assert entry["factors"][0]["formula"] == out["records"][0]["formula"]
        notes = open(memory.notes_path(mem), encoding="utf-8").read()
        assert out["records"][0]["formula"] in notes
        assert cfg.run_name in notes


def _wf_cfg(name, **kw):
    base = dict(symbols=["A", "B", "C", "D", "E", "F"], data_source="synthetic",
                data_start="2015-01-01", data_end="2023-12-31", is_years=5,
                oos_days=126, t_0="2021-01-01", t_p="2022-01-01", max_iters=2,
                provider="scripted", run_name=name, logging=False)
    base.update(kw)
    return RunConfig(**base)


def test_a_later_run_is_taught_by_an_earlier_one_and_still_replays():
    """The whole loop: run 1 writes, run 2 reads it into its prompt, replay holds.

    Run 2 sits a year past run 1's last research date, so run 1's holdout windows
    (126 business days) have long closed and its lessons are legitimately
    knowable --- see the sibling test for what happens when they have not.
    """
    import warnings

    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        run_walk_forward(_wf_cfg("r1"), artifacts_root=art, memory_root=mem)
        memory.remember(Store(art, "r1"), mem)

        run_walk_forward(_wf_cfg("r2", t_0="2023-01-01", t_p="2023-01-01"),
                         artifacts_root=art, memory_root=mem)
        store = Store(art, "r2")
        manifest, recorded = store.read_manifest(), store.read_factors()
        journal = manifest["journal"]
        assert journal and journal["losses"], "run 2 must have seen run 1's failures"
        # provenance is on the ledger row, and it counts what was SHOWN --- here
        # the whole snapshot, because every lesson's window had closed by 2023.
        assert recorded[0]["n_journal_losses"] == len(journal["losses"])
        assert recorded[0]["n_journal_wins"] == len(journal["wins"])

        # the recorded snapshot is what makes a journal-informed run replayable:
        # prompts regenerate byte-identically, so no drift warning is raised.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            replayed = replay(manifest, store.read_trace())
        assert all(_deep_eq(_replay_core(a), _replay_core(b))
                   for a, b in zip(recorded, replayed))


def test_an_open_lesson_is_withheld_from_a_real_run_and_still_replays():
    """End-to-end leakage guard: run 2 sits INSIDE run 1's holdout window.

    The snapshot in the manifest still carries the lesson (it is the record), but
    no ledger row may claim to have been shown it, and the replay of that run must
    regenerate the same withheld prompt --- no drift warning.
    """
    import warnings

    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        # 300-day holdouts: run 1's 2021 window does not close until 2022-02 and
        # its 2022 one not until 2023-02, so run 2 at 2022-01-01 sits inside both
        run_walk_forward(_wf_cfg("open1", oos_days=300), artifacts_root=art,
                         memory_root=mem)
        memory.remember(Store(art, "open1"), mem)

        run_walk_forward(_wf_cfg("open2", oos_days=300, t_0="2022-01-01",
                                 t_p="2022-01-01"),
                         artifacts_root=art, memory_root=mem)
        store = Store(art, "open2")
        manifest, recorded = store.read_manifest(), store.read_factors()
        snapshot = manifest["journal"]
        assert snapshot and (snapshot["wins"] or snapshot["losses"]), \
            "the manifest must still record the unfiltered snapshot"
        assert recorded[0]["n_journal_wins"] == 0
        assert recorded[0]["n_journal_losses"] == 0, \
            "a lesson whose OOS window is still open must not reach the prompt"

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            replayed = replay(manifest, store.read_trace())
        assert all(_deep_eq(_replay_core(a), _replay_core(b))
                   for a, b in zip(recorded, replayed))


def test_resume_keeps_the_journal_the_run_started_with():
    """A journal that grew mid-session must not re-prompt the second half."""
    with tempfile.TemporaryDirectory() as art, tempfile.TemporaryDirectory() as mem:
        run_walk_forward(_wf_cfg("r1"), artifacts_root=art, memory_root=mem)
        memory.remember(Store(art, "r1"), mem)
        started_with = Store(art, "r1").read_manifest()["journal"]

        # somebody else's run lands in the shared journal between sessions
        memory.archive(_manifest(run_name="other"),
                       [_factor("2020-01-01", passed=True,
                                formula='ts_mean(volume,w99)', oos=2.5)], mem)
        assert memory.lessons(mem) != started_with, "the journal really did move"

        out = run_walk_forward(_wf_cfg("r1", t_p="2023-01-01"), artifacts_root=art,
                               memory_root=mem, resume=True)
        assert out["manifest"]["journal"] == started_with, \
            "a resumed walk must keep prompting from the snapshot it began with"
