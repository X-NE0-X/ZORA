"""Regression coverage for model controls, cancellation, and yf cache isolation."""
import asyncio
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from textual.widgets import Button, Select, TabbedContent
from textual.worker import WorkerCancelled

from harness import tui, runner, memory, proposer
from harness.cancellation import RunCancelled, RunControl, cancellation_scope
from harness.config import RunConfig
from harness.data import make_synthetic
from harness.parallel import CandidatePool
from harness.providers.base import LLMProvider, ProviderResponse
from harness.providers.cli_base import run_cli
from harness.factor.protocol import STRICT
from harness.factor.contract import POLICY_HASH
from harness.tui import HarnessTUI
from tests.runtime_probe import blocking_metrics

PROBE = str(pathlib.Path(__file__).with_name("runtime_probe.py"))


def _alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            code = wintypes.DWORD()
            return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_reasoning_reaches_generation_and_memory_paths():
    cfg = RunConfig(provider="codex", model="gpt-6-luna", reasoning_effort="max", math_contract=STRICT)
    factor = {"formula": "rank(close)", "parameters": {}, "rationale": "r", "mechanism": "m",
              "expected_sign": 1}

    class Capture(LLMProvider):
        name = "codex"

        def __init__(self):
            self.calls = []
            answers = [json.dumps(factor), json.dumps({"candidates": [factor, {**factor, "formula": "rank(volume)"}]})]
            self.answers = iter([*answers, "Synthesis test"])

        def complete(self, system, messages, **kwargs):
            self.calls.append(kwargs)
            return ProviderResponse(next(self.answers), kwargs["model"], self.name)

    capture = Capture()
    proposal, _ = proposer.propose(capture, cfg)
    proposer.propose_many(capture, RunConfig.from_dict(
        {**cfg.to_dict(), "candidates_per_round": 2}))
    with tempfile.TemporaryDirectory() as root:
        memory.archive({"run_name": "effort", "config": cfg.to_dict(), "math_policy_hash": POLICY_HASH,
                        "model": cfg.model, "provider": "codex", "data_version": "fixture"},
                       [{**factor, "research_date": "2024-01-01", "verdict": {
                           "passed": False, "label": "FAIL", "objective": "sortino",
                           "is_value": 0.5, "oos_value": -1.0, "reasons": []}}], root)
        assert memory.synthesize(capture, cfg, root)["text"] == "Synthesis test"
    assert len(capture.calls) == 3
    assert all(call["reasoning_effort"] == "max" for call in capture.calls)


def test_cancel_cli_kills_owned_children_and_prevents_next_call():
    control = RunControl()
    stopped = []
    with tempfile.TemporaryDirectory() as root:
        ready = pathlib.Path(root) / "ready.json"

        def run():
            try:
                with cancellation_scope(control):
                    run_cli([sys.executable, PROBE, "tree", str(ready)])
                    stopped.append("continued")
            except RunCancelled:
                stopped.append("cancelled")

        thread = threading.Thread(target=run)
        thread.start()
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pids = json.loads(ready.read_text(encoding="utf-8"))
        assert all(_alive(pid) for pid in pids), "live-process probe is not reliable"
        control.cancel()
        thread.join(12)
        assert not thread.is_alive()
        assert stopped == ["cancelled"]
        assert not any(_alive(pid) for pid in pids), pids

    # Cancel during retry backoff rather than burning the remaining model calls.
    calls = []

    class Failing(LLMProvider):
        name = "stub"

        def complete(self, system, messages, **kwargs):
            calls.append(True)
            raise RuntimeError("temporary endpoint failure")

    control = RunControl()
    cfg = RunConfig(max_iters=4, retry_backoff=600, retry_max_delay=600)
    panel = make_synthetic(["A", "B"], "2024-01-01", "2024-02-01")
    start = time.monotonic()
    try:
        with cancellation_scope(control):
            runner.optimize(Failing(), cfg, panel,
                            log=lambda msg: control.cancel() if "provider error" in msg else None)
    except RunCancelled:
        pass
    else:
        raise AssertionError("cancelled backoff continued")
    assert calls == [True] and time.monotonic() - start < 2


async def test_tui_reasoning_is_on_model_page_and_restores():
    app = HarnessTUI(config=RunConfig(provider="codex", model="gpt-6-luna", reasoning_effort="max"))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        field = app.query_one("#reasoning_effort", Select)
        assert any(a.id == "tab-model" for a in field.ancestors)
        assert not field.disabled and field.value == "max"
        app.query_one(TabbedContent).active = "tab-model"
        await pilot.pause()
        assert field.region.height >= 3
        assert app._build_config().reasoning_effort == "max"
        field.value = "high"
        assert app._build_config().reasoning_effort == "high"
        app._apply_config(RunConfig(provider="codex", reasoning_effort="low"))
        await pilot.pause()
        assert field.value == "low"


async def test_tui_quit_waits_for_real_provider_process_cleanup():
    calls = []
    with tempfile.TemporaryDirectory() as root:
        ready = pathlib.Path(root) / "ready.json"

        class BlockingProvider(LLMProvider):
            name = "codex"

            def complete(self, system, messages, **kwargs):
                calls.append(kwargs)
                return ProviderResponse(run_cli([sys.executable, PROBE, "tree", str(ready)]), "m", "codex")

        cfg = RunConfig(provider="codex", model="gpt-6-luna", max_iters=4,
                        data_source="synthetic", memory=False, run_name="quit_test")
        app = HarnessTUI(config=cfg, artifacts_root=root)
        with patch.object(tui, "get_provider", return_value=BlockingProvider()), \
             patch.object(tui.preflight, "check_config", return_value=SimpleNamespace(ok=True)), \
             patch.object(tui, "check_data_coverage"):
            async with app.run_test() as pilot:
                app.action_run()
                deadline = time.monotonic() + 12
                while not ready.exists() and time.monotonic() < deadline:
                    await asyncio.sleep(0.02)
                assert ready.exists(), "provider process never started"
                pids = json.loads(ready.read_text(encoding="utf-8"))
                assert all(_alive(pid) for pid in pids)
                await pilot.press("ctrl+q")
                try:
                    await app.workers.wait_for_complete()
                except WorkerCancelled:
                    # App exit cancels Textual's async wrapper AFTER our owned
                    # resources have been cleaned and _finish_run has run.
                    pass
        assert len(calls) == 1, "Quit started another model call"
        assert app._run_control is None
        pids = json.loads(ready.read_text(encoding="utf-8"))
        assert not any(_alive(pid) for pid in pids)
        ledger = pathlib.Path(root) / "runs/quit_test/factors.jsonl"
        assert not ledger.exists() or not ledger.read_text(encoding="utf-8").strip()


def test_multiticker_real_sqlite_cache_is_serialized():
    with tempfile.TemporaryDirectory() as root:
        processes = [subprocess.Popen([sys.executable, PROBE, "yf-cache", root],
                     stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                     text=True, encoding="utf-8") for _ in range(2)]
        try:
            for proc in processes:
                stdout, stderr = proc.communicate(timeout=30)
                assert proc.returncode == 0, stderr
                assert json.loads(stdout)["cache_writes"] == 6
        finally:
            for proc in processes:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()


def test_cancel_parallel_pool_stops_waiting_and_terminates_workers():
    control = RunControl()
    cfg = RunConfig()
    panel = make_synthetic(["A", "B"], "2024-01-01", "2024-02-01")
    pool = CandidatePool(2)
    stopped = []

    root = tempfile.TemporaryDirectory()
    ready = [pathlib.Path(root.name) / f"worker-{i}.pid" for i in range(2)]

    def run():
        try:
            with cancellation_scope(control):
                try:
                    pool.metrics([str(path) for path in ready], panel, cfg)
                finally:
                    pool.close()
        except RunCancelled:
            stopped.append(True)

    try:
        with patch("harness.parallel._metrics_task", blocking_metrics):
            thread = threading.Thread(target=run)
            thread.start()
            deadline = time.monotonic() + 20
            while not all(path.exists() for path in ready) and time.monotonic() < deadline:
                time.sleep(0.02)
            started = all(path.exists() for path in ready)
            processes = list(pool._ex._processes.values())
            control.cancel()
            thread.join(12)
            assert started, "both workers must be inside blocked tasks before Quit"
            assert not thread.is_alive() and stopped == [True]
            assert all(not p.is_alive() for p in processes)
            assert pool._ex is None and pool._dir is None
    finally:
        control.cancel()
        root.cleanup()
