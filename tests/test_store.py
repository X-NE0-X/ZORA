"""Store hardening: run_name can't escape the artifacts tree; a corrupt ledger
line doesn't sink the whole read."""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import warnings

from harness import store as store_module
from harness.store import Store

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_run_name_rejects_traversal_and_absolute():
    with tempfile.TemporaryDirectory() as root:
        bad_names = [
            "..", "../evil", "a/b", "a\\b", os.path.abspath(os.sep),
            # Windows drive-relative names: isabs() says False but os.path.join
            # still resets to the drive root (escapes artifacts/runs)
            "C:foo", "a:b", "C:", ".",
        ]
        for bad in bad_names:
            try:
                Store(root, bad)
            except ValueError:
                continue
            raise AssertionError(f"run_name {bad!r} should have been rejected")

        # a clean single segment is fine and stays under <root>/runs/
        s = Store(root, "run_01")
        s.ensure()
        assert os.path.normpath(s.run_dir) == os.path.normpath(
            os.path.join(root, "runs", "run_01")
        )
        assert os.path.isdir(s.run_dir)


def test_read_factors_skips_corrupt_line():
    with tempfile.TemporaryDirectory() as root:
        s = Store(root, "run_corrupt")
        s.append_factor({"formula": "-returns", "ok": 1})
        # inject a broken line between two good ones
        with open(s.ledger_path, "a", encoding="utf-8") as fh:
            fh.write("{ this is not valid json \n")
        s.append_factor({"formula": "rank(close)", "ok": 2})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            recs = s.read_factors()
        assert [r["ok"] for r in recs] == [1, 2], "good lines must survive"
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)


def test_same_run_name_is_locked_across_processes():
    with tempfile.TemporaryDirectory() as root:
        store = Store(root, "owned")
        code = (
            "from harness.store import Store, RunLockedError; "
            f"s=Store({root!r}, 'owned'); "
            "\ntry:\n s.acquire()\nexcept RunLockedError:\n raise SystemExit(7)\n"
            "else:\n s.release(); raise SystemExit(0)\n"
        )
        with store.ownership():
            child = subprocess.run(
                [sys.executable, "-c", code], cwd=ROOT,
                capture_output=True, text=True, encoding="utf-8",
                check=False,
            )
        assert child.returncode == 7, child.stderr


def test_manifest_and_trace_replace_atomically():
    with tempfile.TemporaryDirectory() as root:
        store = Store(root, "atomic")
        store.write_manifest({"version": "old"})
        store.append_trace({"text": "old", "prompt_hash": "abc"})
        manifest_before = pathlib.Path(store.manifest_path).read_bytes()
        trace_before = pathlib.Path(store.trace_path).read_bytes()
        original = store_module.os.replace

        def fail_replace(_src, _dst):
            raise OSError("simulated replace failure")

        store_module.os.replace = fail_replace
        try:
            for write in (
                lambda: store.write_manifest({"version": "new"}),
                lambda: store.append_trace({"text": "new", "prompt_hash": "def"}),
            ):
                try:
                    write()
                except OSError:
                    pass
                else:
                    raise AssertionError("simulated atomic replace failure must surface")
        finally:
            store_module.os.replace = original

        assert pathlib.Path(store.manifest_path).read_bytes() == manifest_before
        assert pathlib.Path(store.trace_path).read_bytes() == trace_before


def test_ledger_append_flushes_to_stable_storage():
    with tempfile.TemporaryDirectory() as root:
        store = Store(root, "durable")
        calls = []
        original = store_module.os.fsync
        store_module.os.fsync = lambda fd: calls.append(fd)
        try:
            store.append_factor({"formula": "rank(close)"})
        finally:
            store_module.os.fsync = original
        assert calls, "a successful ledger append must fsync before returning"
