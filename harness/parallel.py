"""Ordered worker processes for in-sample candidate backtests.

Breadth is independent within one proposal round. Refinement rounds and
walk-forward dates remain sequential because their feedback is causal.
The parent owns proposal completions and ordered state updates. Workers
return only deterministic metrics; they never call a model or mutate traces.
Each worker limits numerical-library threads before bootstrap."""
from __future__ import annotations

import os
import pickle
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from concurrent.futures import ProcessPoolExecutor, TimeoutError

from .cancellation import check_cancelled, current_control

# --- worker side -------------------------------------------------------------
# Panels are shipped once per pool via a temp file rather than per task: the
# panel is a few MB, a task is a formula string, and the walk reuses one panel
# for every date. Keyed by the path so a worker loads it at most once.
_PANEL_CACHE: dict = {}
_THREAD_LIMITS = None
_THREAD_ENV = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
_SPAWN_ENV_LOCK = threading.RLock()


@contextmanager
def _spawn_environment():
    """Pin child imports before Windows spawn reloads the main module.

    Initializers run too late to constrain NumPy imported by that module.
    ProcessPoolExecutor starts its spawn workers synchronously during submit.
    Restore the caller's environment even when submission fails.
    """
    with _SPAWN_ENV_LOCK:
        previous = {name: os.environ.get(name) for name in _THREAD_ENV}
        try:
            os.environ.update({name: "1" for name in _THREAD_ENV})
            yield
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


def _pin_threads() -> None:
    """One worker, one thread --- see the module docstring.

    Belt and braces on purpose. Spawn-time environment pinning constrains math
    libraries imported while reloading the main module; ``threadpoolctl`` here
    re-caps whatever is already loaded, and numba keeps its own count.
    Any of the three missing is not an error: an unpinned worker
    is slower under contention, never wrong.
    """
    global _THREAD_LIMITS
    for var in _THREAD_ENV:
        os.environ.setdefault(var, "1")
    try:
        import threadpoolctl
        # Held in a module global: the limits are restored when the controller
        # object is collected, so a local would un-pin the worker immediately.
        _THREAD_LIMITS = threadpoolctl.threadpool_limits(limits=1)
    except Exception:       # noqa: BLE001 - optional dependency / unknown backend
        pass
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:       # noqa: BLE001 - numba absent or already frozen
        pass


def _worker_init() -> None:
    """Pay the expensive one-off costs once per worker, not once per task.

    Importing the vendored engine stack takes ~4s and compiling numba kernels
    takes more; doing it inside the first task would make that task look like an
    outlier and would serialise the pool behind it. Failures are swallowed here
    and left to surface from the task itself, where they carry a real traceback.
    """
    _pin_threads()
    try:
        from . import infra_engine
        infra_engine._ensure()
    except Exception:       # noqa: BLE001 - the first task re-raises it properly
        pass


def _load_panel(path: str):
    panel = _PANEL_CACHE.get(path)
    if panel is None:
        with open(path, "rb") as fh:
            panel = pickle.load(fh)
        _PANEL_CACHE[path] = panel
    return panel


def _metrics_task(payload):
    """Score ONE formula on the in-sample window. Runs in a worker process.

    Returns ``(True, metrics)`` or ``(False, message)`` for the one failure the
    caller treats as data rather than as a crash --- an uncomputable formula,
    which the serial path also catches and feeds back as a dead end. Anything
    else propagates: a broken engine or a corrupt panel is a run-ending fault
    and must not be laundered into "this candidate didn't work out".
    """
    panel_path, formula, config_dict = payload
    from . import runner
    from .config import RunConfig
    from .factor.evaluate import FactorEvalError

    panel = _load_panel(panel_path)
    config = RunConfig.from_dict(config_dict)
    try:
        return True, runner._is_metrics(formula, panel, config)
    except FactorEvalError as exc:
        return False, str(exc)


# --- parent side -------------------------------------------------------------
class CandidatePool:
    """A process pool that evaluates candidate formulas, and nothing else.

    Created once per run and kept for the whole walk: a worker pays ~4s of
    vendored-engine import plus numba compilation before its first backtest, so
    a pool per date (let alone per round) would spend more on start-up than the
    work it parallelises. Use as a context manager, or call :meth:`close`.
    """

    def __init__(self, n_workers: int, shutdown_timeout: float = 30.0):
        self.n_workers = max(1, int(n_workers))
        self.shutdown_timeout = max(0.0, float(shutdown_timeout))
        self._ex: ProcessPoolExecutor | None = None
        self._dir: str | None = None
        self._paths: dict[str, str] = {}

    def __enter__(self) -> "CandidatePool":
        return self

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False

    # -- lifecycle --
    def _executor(self) -> ProcessPoolExecutor:
        if self._ex is None:
            self._ex = ProcessPoolExecutor(max_workers=self.n_workers,
                                           initializer=_worker_init)
        return self._ex

    def close(self) -> None:
        if self._ex is not None:
            executor = self._ex
            self._ex = None
            # ProcessPoolExecutor has no bounded shutdown API. Ask it to stop,
            # then bound joins ourselves and terminate/kill workers that ignore
            # the request. This prevents interpreter shutdown from waiting
            # forever on a wedged backtest worker.
            process_map = getattr(executor, "_processes", None) or {}
            processes = list(process_map.values())
            executor.shutdown(wait=False, cancel_futures=True)
            control = current_control()
            deadline = time.monotonic() + (
                0 if control is not None and control.cancelled.is_set()
                else self.shutdown_timeout)
            for proc in processes:
                proc.join(max(0.0, deadline - time.monotonic()))
            alive = [proc for proc in processes if proc.is_alive()]
            for proc in alive:
                proc.terminate()
            for proc in alive:
                proc.join(1.0)
            for proc in alive:
                if proc.is_alive() and hasattr(proc, "kill"):
                    proc.kill()
                    proc.join(1.0)
        if self._dir is not None:
            shutil.rmtree(self._dir, ignore_errors=True)
            self._dir = None
        self._paths.clear()

    # -- work --
    def _panel_path(self, panel) -> str:
        """Where this panel is pickled for the workers, writing it at most once.

        Keyed by ``panel.version``, which is a hash of the data --- so every date
        of a walk (each of which reloads an equal panel as a NEW object) shares
        one file, while genuinely different data gets its own.
        """
        key = str(getattr(panel, "version", "") or id(panel))
        path = self._paths.get(key)
        if path is None:
            if self._dir is None:
                self._dir = tempfile.mkdtemp(prefix="zora-pool-")
            path = os.path.join(self._dir, f"panel-{len(self._paths)}.pkl")
            with open(path, "wb") as fh:
                pickle.dump(panel, fh, protocol=pickle.HIGHEST_PROTOCOL)
            self._paths[key] = path
        return path

    def metrics(self, formulas, panel, config) -> list[tuple[bool, object]]:
        """In-sample metrics for each formula, **in the order given**.

        One ``(ok, payload)`` per input: ``(True, metrics)``, or ``(False, msg)``
        for a formula the evaluator could not compute. Order is guaranteed by
        ``executor.map`` (submission order, not completion order) and the caller
        depends on it --- see the module docstring.

        A dead worker raises ``BrokenProcessPool`` straight through: the pool is
        all-or-nothing, and quietly re-running the batch inline would make a run
        that hit an OOM kill look exactly like one that did not.
        """
        items = list(formulas)
        if not items:
            return []
        path = self._panel_path(panel)
        cfg = config.to_dict()
        check_cancelled()
        executor = self._executor()
        with _spawn_environment():
            futures = [executor.submit(_metrics_task, (path, f, cfg)) for f in items]
        results = []
        for future in futures:
            while True:
                check_cancelled()
                try:
                    results.append(future.result(timeout=0.1))
                    break
                except TimeoutError:
                    if future.done():  # the TASK itself raised TimeoutError
                        raise
        check_cancelled()
        return results


def pool_for(config, log=None) -> CandidatePool | None:
    """The pool this config asks for, or ``None`` for the inline path.

    ``None`` whenever parallelism cannot pay: ``n_jobs == 1`` (the default, which
    runs the historic inline code byte for byte), or a breadth of 1, where a
    round holds a single backtest and there is nothing to overlap. More workers
    than candidates is capped rather than honoured --- the extra processes would
    each pay the multi-second engine import to then sit idle.
    """
    n_jobs = max(1, int(getattr(config, "n_jobs", 1) or 1))
    k = max(1, int(getattr(config, "candidates_per_round", 1) or 1))
    if n_jobs == 1:
        return None
    if k == 1:
        if log:
            log(f"n_jobs={n_jobs} ignored: candidates_per_round=1 leaves one "
                "backtest per round to run. Raise breadth (K) to use workers.")
        return None
    workers = min(n_jobs, k)
    if log:
        log(f"parallel: {workers} worker process(es) for {k} candidates/round")
    return CandidatePool(workers)
