"""Worker processes for the candidate backtests (``n_jobs``).

The only stage of a round that can be parallelised is the backtest of the K
candidates the proposer returned in one call: refine rounds read the previous
round's metrics and each walk-forward date reads the earlier dates' verdicts, so
depth and the walk are sequential by construction. All proposal completions remain in the parent.

What every test here is really defending is that ``n_jobs`` is PURE WALL CLOCK.
The dangerous failure is not a crash, it is a silent reordering: ``history`` and
``negatives`` are rendered into the NEXT round's prompt in list order, and the
incumbent keeps an exact tie, so folding results back out of order would change
the prompt (and its hash), the recorded trace, and sometimes the winner --- while
every existing test still passed. So the gate is a real walk recorded WITH
workers and replayed WITHOUT them, compared bit for bit.
"""
import json
import tempfile
import warnings
import pytest

from harness.config import RunConfig
from harness.factor.contract import compile_factor
from harness.factor.protocol import STRICT
from harness.data import make_synthetic
from harness.parallel import CandidatePool, pool_for
from harness.providers.replay import RecordingProvider, ReplayProvider
from harness.providers.scripted import ScriptedProvider
from harness.runner import (_config_conflicts, _deep_eq, _is_metrics,
                            _replay_core, optimize, replay, run_walk_forward)
from harness.store import Store
import harness.cli as _cli
import harness.runner as _runner

SYMS = ["A", "B", "C", "D", "E"]


# c0 passes static admission but overflows on IS values; c1 is
# a two-term spread (D1 hard reject), c2 and c3 are ordinary monomials. This
# exact mix is the point: c0's dead end is discovered in the backtest stage and
# c1's before it, so a naive "screen everything, then backtest everything"
# refactor records them as [structure(c1), uncomputable(c0)] where the serial
# loop records [uncomputable(c0), structure(c1)] --- and that list, in order,
# goes into the next round's prompt.
MIXED = ["ts_mean(close, 0)", "rank(high) - rank(low)", "-delta(close, 5)",
         "rank(returns)"]


def _obj(formula, sign=1):
    parameters = {}
    formulas = {
        "ts_mean(close, 0)": ("rank(close**p)", {"p": {"type": "Exponent", "value": 200}}),
        "-delta(close, 5)": ("-delta(close,w)", {"w": {"type": "Window", "value": 5}}),
        "ts_std(returns, 20)": ("ts_std(returns,w)", {"w": {"type": "Window", "value": 20}}),
        "-rank(delta(close, 10))": ("-rank(delta(close,w))", {"w": {"type": "Window", "value": 10}}),
        "ts_mean(returns, 10)": ("ts_mean(returns,w)", {"w": {"type": "Window", "value": 10}}),
    }
    formula, parameters = formulas.get(formula, (formula, {}))
    return {"formula": formula, "parameters": parameters, "rationale": "reason for " + formula,
            "mechanism": "the mechanism behind " + formula, "expected_sign": sign}


def _arr(formulas):
    candidates = [_obj(f) for f in formulas]
    return json.dumps({"candidates": candidates})


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        research_date="2021-01-01", is_years=5, oos_days=126, min_oos_days=20,
        provider="scripted", seed=17, max_iters=2, candidates_per_round=4,
        run_name="par", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


def _panel(cfg):
    return make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed)


def _with_scripted(script, fn):
    orig = _runner.get_provider
    _runner.get_provider = (
        lambda name, **kw: ScriptedProvider(list(script)) if name == "scripted"
        else orig(name, **kw))
    try:
        return fn()
    finally:
        _runner.get_provider = orig


def _shape(opt):
    """The parts of an optimize() result a reordering would disturb."""
    return [(h["iteration"], h["candidate"], h["formula"], h["is_metrics"],
             h["alignment"], h["prompt_hash"]) for h in opt["history"]]


# --- the knob ---------------------------------------------------------------
def test_n_jobs_default_and_validation():
    assert RunConfig().n_jobs == 1                    # inline, historic behaviour
    RunConfig(n_jobs=8)
    for bad in (0, -2):
        try:
            RunConfig(n_jobs=bad)
            raise AssertionError(f"n_jobs={bad} must be rejected")
        except ValueError:
            pass


def test_pool_for_only_builds_a_pool_that_can_pay():
    notes = []
    # the default: no pool at all, so optimize takes the historic inline path
    assert pool_for(_cfg(n_jobs=1, candidates_per_round=4)) is None
    # workers with no breadth to spend them on: one backtest per round, nothing
    # to overlap. Say so rather than forking idle processes.
    assert pool_for(_cfg(n_jobs=4, candidates_per_round=1),
                    log=notes.append) is None
    assert notes and "candidates_per_round=1" in notes[0]
    # more workers than candidates is capped, not honoured (each extra process
    # would pay seconds of engine import to then sit idle)
    assert pool_for(_cfg(n_jobs=9, candidates_per_round=3)).n_workers == 3
    assert pool_for(_cfg(n_jobs=2, candidates_per_round=8)).n_workers == 2


def test_n_jobs_is_exempt_from_the_resume_guard():
    # it changes wall clock and nothing else, so a run recorded on one machine
    # must resume on another with a different core count. Contrast max_iters,
    # which changes what is computed.
    base = _cfg().to_dict()
    assert _config_conflicts(base, {**base, "n_jobs": 6}) == []
    assert _config_conflicts(base, {**base, "max_iters": 3}) == ["max_iters"]


# --- the pool itself --------------------------------------------------------
def test_pool_ships_one_panel_per_data_version():
    # the panel is pickled to a temp file ONCE and reused by every task and every
    # date; only genuinely different data gets a second file.
    cfg = _cfg()
    panel = _panel(cfg)
    with CandidatePool(2) as pool:
        first = pool._panel_path(panel)
        assert pool._panel_path(panel) == first          # same panel: no rewrite
        assert pool._panel_path(_panel(cfg)) == first    # equal data, new object
        other = make_synthetic(SYMS, cfg.data_start, cfg.data_end, cfg.seed + 1)
        assert pool._panel_path(other) != first
        tmp_dir = pool._dir
    import os
    assert not os.path.exists(tmp_dir)                   # close() cleans up


def test_pool_metrics_are_ordered_and_bit_identical_to_inline():
    cfg = _cfg()
    panel = _panel(cfg)
    objects = [_obj(f) for f in MIXED[2:] + ["ts_std(returns, 20)", MIXED[0]]]
    forms = [compile_factor(obj["formula"], obj["parameters"], panel=panel) for obj in objects]
    with CandidatePool(2) as pool:
        got = pool.metrics(forms, panel, cfg)
    assert len(got) == len(forms)
    for formula, (ok, payload) in zip(forms, got):
        try:
            expect = _is_metrics(formula, panel, cfg)
        except Exception as exc:      # noqa: BLE001 - the uncomputable one
            assert ok is False and payload == str(exc), formula
            continue
        # bit-identical, not merely close: a worker that annualised differently
        # or picked up a different thread count would show up right here.
        assert ok is True and payload == expect, formula
    assert pool_for(_cfg(n_jobs=1)) is None
    with CandidatePool(2) as pool:
        assert pool.metrics([], panel, cfg) == []        # no work, no processes
        assert pool._ex is None


def test_pool_close_bounds_shutdown_and_kills_a_wedged_worker():
    class FakeProcess:
        def __init__(self):
            self.terminated = False
            self.killed = False

        def join(self, _timeout=None):
            return None

        def is_alive(self):
            return not self.killed

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    class FakeExecutor:
        def __init__(self, proc):
            self._processes = {1: proc}
            self.shutdown_args = None

        def shutdown(self, **kwargs):
            self.shutdown_args = kwargs

    proc = FakeProcess()
    executor = FakeExecutor(proc)
    pool = CandidatePool(1, shutdown_timeout=0)
    pool._ex = executor
    pool.close()
    assert executor.shutdown_args == {"wait": False, "cancel_futures": True}
    assert proc.terminated and proc.killed
    assert pool._ex is None


# --- optimize: serial vs parallel ------------------------------------------
def test_parallel_optimize_is_indistinguishable_from_serial():
    cfg = _cfg(candidates_per_round=4, max_iters=2, tri_align=True, math_contract=STRICT)
    panel = _panel(cfg)
    script = [_arr(MIXED), _arr(["ts_std(returns, 20)", "rank(volume)",
                                 "-rank(delta(close, 10))", "ts_mean(returns, 10)"])]

    rec_s = RecordingProvider(ScriptedProvider(list(script)))
    serial = optimize(rec_s, cfg, panel)

    pool = pool_for(_cfg(n_jobs=3, candidates_per_round=4, max_iters=2))
    assert pool is not None
    try:
        rec_p = RecordingProvider(ScriptedProvider(list(script)))
        par = optimize(rec_p, cfg, panel, pool=pool)
    finally:
        pool.close()

    assert _shape(serial) == _shape(par)
    assert serial["best"] == par["best"]
    # round 1's prompt embeds round 0's dead ends IN ORDER, so an identical
    # prompt hash is the proof that `negatives` was rebuilt in candidate order
    # (the uncomputable c0 before the structure-rejected c1) --- the one thing a
    # phase-split refactor silently gets wrong.
    assert [e["prompt_hash"] for e in rec_s.trace] == \
           [e["prompt_hash"] for e in rec_p.trace]
    assert len({h["formula"] for h in serial["history"]}) >= 5   # a real search


def test_a_parallel_recording_replays_serially_bit_for_bit():
    # This checks the inline optimize path; persisted replay has a separate gate.
    cfg = _cfg(candidates_per_round=4, max_iters=2, tri_align=True, math_contract=STRICT)
    panel = _panel(cfg)
    script = [_arr(MIXED)] * cfg.max_iters
    pool = pool_for(_cfg(n_jobs=2, candidates_per_round=4))
    try:
        rec = RecordingProvider(ScriptedProvider(list(script)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # scripted exhausted
            par = optimize(rec, cfg, panel, pool=pool)
    finally:
        pool.close()
    rep = ReplayProvider(rec.trace)
    back = optimize(rep, cfg, panel)                      # no pool
    assert rep._i == len(rec.trace) and not rep.prompt_mismatches
    assert _shape(par) == _shape(back)
    assert par["best"] == back["best"]


# --- the gate: a real walk, recorded with workers, replayed without ---------
def test_walk_forward_with_workers_replays_bit_for_bit():
    with tempfile.TemporaryDirectory() as root:
        wf = dict(symbols=SYMS, data_source="synthetic",
                  data_start="2015-01-01", data_end="2023-12-31",
                  is_years=5, oos_days=126, min_oos_days=20,
                  t_0="2021-01-01", t_p="2022-01-01", frequency="QS",
                  provider="scripted", seed=17, max_iters=2,
                  candidates_per_round=4, logging=False, math_contract=STRICT)
        script = [_arr(MIXED)] * 10

        serial = _with_scripted(script, lambda: run_walk_forward(
            RunConfig(**wf, n_jobs=1, run_name="par_wf_serial"),
            artifacts_root=root))
        par = _with_scripted(script, lambda: run_walk_forward(
            RunConfig(**wf, n_jobs=3, run_name="par_wf_workers"),
            artifacts_root=root))

        assert len(par["records"]) == 5                   # quarterly walk
        for a, b in zip(serial["records"], par["records"]):
            assert _deep_eq(_replay_core(a), _replay_core(b)), a["research_date"]
            # iterations is ORDERED and is what the record shows a reader; a
            # reshuffle here would not touch _replay_core at all.
            assert a["iterations"] == b["iterations"], a["research_date"]
            assert a["n_completions"] == b["n_completions"]

        s_store = Store(root, "par_wf_serial")
        p_store = Store(root, "par_wf_workers")
        s_trace, p_trace = s_store.read_trace(), p_store.read_trace()
        assert len(p_trace) == 10                         # 5 dates x 2 rounds
        # byte-identical prompts, in the same order: the trace IS the run's
        # provenance, so this is the reproducibility contract itself.
        assert [(e["prompt_hash"], e["text"]) for e in s_trace] == \
               [(e["prompt_hash"], e["text"]) for e in p_trace]

        # and the parallel recording replays (serially) with no prompt drift
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            replayed = replay(p_store.read_manifest(), p_trace)
        for a, b in zip(par["records"], replayed):
            assert _deep_eq(_replay_core(a), _replay_core(b)), a["research_date"]


# --- CLI --------------------------------------------------------------------
def test_cli_n_jobs_flag_overrides_the_config():
    args = _cli.build_parser().parse_args(["run", "--n-jobs", "4"])
    assert _cli._load_config(args).n_jobs == 4
    assert _cli._load_config(_cli.build_parser().parse_args(["run"])).n_jobs == 1
    try:
        _cli._load_config(_cli.build_parser().parse_args(["run", "--n-jobs", "0"]))
        raise AssertionError("--n-jobs 0 must be rejected")
    except ValueError:
        pass
