"""Spawn entry point deliberately importing NumPy before the initializer."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
THREAD_NAMES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
BEFORE_NUMPY = {name: os.environ.get(name) for name in THREAD_NAMES}
import numpy
import threadpoolctl


def no_initialization():
    pass


def environment_task(_payload):
    return True, {"environment": BEFORE_NUMPY,
                  "threads": [info["num_threads"] for info in threadpoolctl.threadpool_info()]}


if __name__ == "__main__":
    import harness.parallel as parallel
    from harness.config import RunConfig

    original = {name: os.environ.get(name) for name in THREAD_NAMES}
    parallel._worker_init = no_initialization
    parallel._metrics_task = environment_task
    with parallel.CandidatePool(2) as pool:
        results = pool.metrics(["one", "two"], SimpleNamespace(version="probe"), RunConfig())
    assert {name: os.environ.get(name) for name in THREAD_NAMES} == original
    print(json.dumps(results))
