"""Early worker import pinning without mutating the caller's lasting settings."""
from concurrent.futures import Future
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from harness.config import RunConfig
from harness.parallel import CandidatePool, _THREAD_ENV


@pytest.mark.parametrize("fail", [False, True])
def test_submit_has_pinned_environment_and_restores_on_error(monkeypatch, fail):
    for i, name in enumerate(_THREAD_ENV):
        if i % 2:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, "3")
    original = {name: os.environ.get(name) for name in _THREAD_ENV}

    class Executor:
        def submit(self, *_args):
            assert {name: os.environ.get(name) for name in _THREAD_ENV} == dict.fromkeys(_THREAD_ENV, "1")
            if fail:
                raise RuntimeError("submission failed")
            future = Future()
            future.set_result((True, {}))
            return future

    pool = CandidatePool(2)
    monkeypatch.setattr(pool, "_panel_path", lambda _panel: "unused")
    monkeypatch.setattr(pool, "_executor", lambda: Executor())
    if fail:
        with pytest.raises(RuntimeError, match="submission failed"):
            pool.metrics(["one", "two"], SimpleNamespace(), RunConfig())
    else:
        assert pool.metrics(["one", "two"], SimpleNamespace(), RunConfig()) == [(True, {}), (True, {})]
    assert {name: os.environ.get(name) for name in _THREAD_ENV} == original


def test_real_spawn_main_imports_numpy_with_one_thread():
    probe = Path(__file__).with_name("parallel_bootstrap_probe.py")
    environment = dict(os.environ)
    environment.update(dict.fromkeys(_THREAD_ENV, "2"))
    result = subprocess.run([sys.executable, str(probe)], env=environment,
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    records = json.loads(result.stdout)
    assert len(records) == 2
    for ok, record in records:
        assert ok and record["environment"] == dict.fromkeys(_THREAD_ENV, "1")
        assert record["threads"] and all(count == 1 for count in record["threads"])
