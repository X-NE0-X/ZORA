"""Second-order hardening surfaced by the adversarial review:

  * append heals a missing trailing newline, so a write interrupted mid-flush
    cannot merge onto and silently swallow the next record;
  * truncate_trace ALWAYS rewrites, dropping a half-written tail line even when
    the parseable count already equals n;
  * reset() clears the manifest, so a crashed fresh run cannot replay against a
    stale config;
  * resume refuses a config change that would desync replay, but allows merely
    extending the horizon (t_p);
  * replay warns when the reloaded data version drifts from the recorded run.
"""
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import tomllib
import warnings
import pytest

from harness.config import RunConfig
from harness.runner import replay, run_walk_forward
from harness.store import Store

SYMS = ["A", "B", "C", "D", "E", "F"]


def _cfg(**kw):
    base = dict(
        symbols=SYMS, data_source="synthetic",
        data_start="2015-01-01", data_end="2023-12-31",
        is_years=5, oos_days=126, min_oos_days=20,
        t_0="2021-01-01", t_p="2021-01-01", frequency="YS",
        provider="scripted", seed=17, max_iters=2,
        run_name="test_hardening", logging=False,
    )
    base.update(kw)
    return RunConfig(**base)


# --- durable append / truncate (store) -------------------------------------
def test_append_factor_heals_missing_newline():
    with tempfile.TemporaryDirectory() as root:
        store = Store(root, "heal")
        store.ensure()
        # a prior append cut off mid-write: bytes on disk with NO trailing newline
        with open(store.ledger_path, "w", encoding="utf-8") as fh:
            fh.write('{"research_date": "partial"')       # invalid JSON, no "\n"
        store.append_factor({"research_date": "2021-01-01", "ok": True})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            recs = store.read_factors()
        # the fresh record survived: the fragment was isolated + skipped, not
        # glued onto the new row (which would have lost BOTH)
        assert len(recs) == 1
        assert recs[0]["research_date"] == "2021-01-01"


def test_truncate_trace_drops_half_written_tail():
    with tempfile.TemporaryDirectory() as root:
        store = Store(root, "trunc")
        store.ensure()
        store.append_trace({"text": "a"})
        store.append_trace({"text": "b"})
        # an orphan tail line left half-written by a hard kill (no newline)
        with open(store.trace_path, "a", encoding="utf-8") as fh:
            fh.write('{"text": "orph')
        # only 2 entries parse, so the old early-return would NOT rewrite; the
        # fixed version always rewrites and drops the fragment
        with pytest.warns(RuntimeWarning, match="skipping unreadable trace line 3"):
            store.truncate_trace(2)
        with open(store.trace_path, encoding="utf-8") as fh:
            raw = fh.read()
        assert "orph" not in raw, "truncate must rewrite out the half-written tail"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            assert [e["text"] for e in store.read_trace()] == ["a", "b"]


def test_reset_removes_manifest():
    with tempfile.TemporaryDirectory() as root:
        store = Store(root, "rst")
        store.write_manifest({"config": {"run_name": "rst"}})
        store.append_factor({"x": 1})
        assert os.path.exists(store.manifest_path)
        store.reset()
        assert not os.path.exists(store.manifest_path)
        assert not os.path.exists(store.ledger_path)


# --- resume config guard ---------------------------------------------------
def test_resume_refuses_conflicting_config():
    with tempfile.TemporaryDirectory() as root:
        run_walk_forward(_cfg(max_iters=2), artifacts_root=root)
        try:
            run_walk_forward(_cfg(max_iters=3), artifacts_root=root, resume=True)
        except ValueError as exc:
            assert "max_iters" in str(exc), str(exc)
        else:
            raise AssertionError("resume must refuse a conflicting max_iters")


def test_resume_refuses_ledger_without_manifest():
    with tempfile.TemporaryDirectory() as root:
        cfg = _cfg(run_name="missing_manifest")
        Store(root, cfg.run_name).append_factor({
            "research_date": "2021-01-01", "verdict": {"passed": False},
        })
        try:
            run_walk_forward(cfg, artifacts_root=root, resume=True)
        except ValueError as exc:
            assert "manifest.json is missing" in str(exc)
        else:
            raise AssertionError("resume must not reuse an unverifiable ledger")


def test_resume_allows_extending_horizon():
    # t_p may grow on resume (extend the walk-forward) without a conflict
    with tempfile.TemporaryDirectory() as root:
        run_walk_forward(_cfg(t_0="2021-01-01", t_p="2021-01-01"),
                         artifacts_root=root)
        out = run_walk_forward(_cfg(t_0="2021-01-01", t_p="2022-01-01"),
                               artifacts_root=root, resume=True)
        assert [r["research_date"] for r in out["records"]] == \
            ["2021-01-01", "2022-01-01"]


def test_resume_tolerates_an_absolute_path_respelt_as_relative():
    """Re-spelling ``parquet_daily`` must not read as "different data".

    The shipped configs carry repo-relative parquet paths (no machine-specific
    absolute path). A run recorded before that rewrite pinned the SAME file as an
    absolute path, so both sides are resolved before comparison --- while a
    genuinely different file must still block the resume.
    """
    from harness.data import _harness_root
    from harness.runner import _config_conflicts

    same = "artifacts/cache/Data_EQT_US_D_abc.parquet"
    absolute = str(_harness_root() / "artifacts" / "cache" / "Data_EQT_US_D_abc.parquet")
    recorded = _cfg(parquet_daily=[absolute]).to_dict()

    assert _config_conflicts(recorded, _cfg(parquet_daily=[same]).to_dict()) == []
    assert _config_conflicts(recorded, _cfg(parquet_daily=same).to_dict()) == [], \
        "a bare string and a one-element list name the same dataset"

    other = _cfg(parquet_daily=["artifacts/cache/Data_EQT_US_D_zzz.parquet"]).to_dict()
    assert _config_conflicts(recorded, other) == ["parquet_daily"], \
        "a different FILE is still a real conflict"


def test_resume_refuses_shrinking_horizon():
    # t_p may grow but NOT shrink: dropping an already-recorded date would leave
    # its ledger/trace rows orphaned and desync the manifest (breaking replay).
    with tempfile.TemporaryDirectory() as root:
        run_walk_forward(_cfg(t_0="2021-01-01", t_p="2022-01-01"),
                         artifacts_root=root)
        try:
            run_walk_forward(_cfg(t_0="2021-01-01", t_p="2021-01-01"),
                             artifacts_root=root, resume=True)
        except ValueError as exc:
            assert "2022-01-01" in str(exc), str(exc)
        else:
            raise AssertionError("resume must refuse a shrunk t_p horizon")


# --- shipped-tree hygiene ---------------------------------------------------
def test_gitignore_has_no_inline_comments():
    """git has NO inline comments: a trailing ``# ...`` swallows the pattern.

    ``ArchiveDeck/   # CTX pipeline logs`` read as ignored and was not --- the
    whole line became one literal pattern that matches nothing, so ten CTX logs
    (each embedding an absolute local path) were committed. Comments must sit on
    their own line, and this guard is here so the next one cannot slip through.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / ".gitignore"
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "#" not in stripped, (
            f".gitignore:{n} has an inline comment, so the entire line is one "
            f"literal pattern and matches nothing: {line!r}"
        )


def test_canonical_runner_treats_import_failure_as_failure(tmp_path):
    root = pathlib.Path(__file__).resolve().parents[1]
    broken = tmp_path / "test_broken.py"
    broken.write_text('raise ImportError("forced import failure")\n', encoding="utf-8")
    reports = tmp_path / "receipts"
    result = subprocess.run(
        [sys.executable, str(root / "tests" / "run_tests.py"), str(broken),
         "--report-dir", str(reports)], cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode != 0
    assert "forced import failure" in result.stdout
    receipt = json.loads((reports / "results.json").read_text(encoding="utf-8"))
    assert receipt["collection_errors"] == 1
    assert receipt["exit_code"] != 0
    assert receipt["counts"].get("passed", 0) == 0


def test_reference_requirements_are_fully_hashed_and_safe_versions():
    root = pathlib.Path(__file__).resolve().parents[1]
    lines = (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
    pin = re.compile(r"^([A-Za-z0-9_.-]+)==([^\\\s]+) \\$")
    versions = {}
    groups = 0
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw or raw.startswith("#"):
            i += 1
            continue
        match = pin.fullmatch(raw)
        assert match is not None, f"unhashed or inexact requirement: {raw!r}"
        versions[match.group(1).lower()] = match.group(2)
        groups += 1
        i += 1
        n_hashes = 0
        while i < len(lines) and lines[i].lstrip().startswith("--hash=sha256:"):
            digest = lines[i].strip().removesuffix("\\").strip()
            assert re.fullmatch(r"--hash=sha256:[0-9a-f]{64}", digest), digest
            n_hashes += 1
            continued = lines[i].rstrip().endswith("\\")
            i += 1
            if not continued:
                break
        assert n_hashes, f"{match.group(1)} has no artifact hash"
    assert groups >= 70
    assert versions["build"] == "1.6.0"
    assert versions["pillow"] == "12.3.0"
    assert versions["setuptools"] == "84.0.0"


def test_release_hardening_configuration_is_present():
    root = pathlib.Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["build-system"]["requires"] == ["setuptools==84.0.0"]
    assert project["tool"]["pytest"]["ini_options"]["asyncio_mode"] == "auto"
    assert project["tool"]["pytest"]["ini_options"]["cache_dir"] == \
        "artifacts/.pytest_cache"
    ignored = (root / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "!.env.template" in ignored
    assert "memory/raw/*" in ignored
    assert "memory/ResearchNotes.md" in ignored
    assert (root / ".github" / "workflows" / "ci.yml").is_file()
    assert (root / "THIRD_PARTY_NOTICES.md").is_file()


# --- replay data-version drift guard ---------------------------------------
def test_replay_warns_on_data_version_drift():
    with tempfile.TemporaryDirectory() as root:
        run_walk_forward(_cfg(), artifacts_root=root)
        store = Store(root, "test_hardening")
        trace = store.read_trace()
        manifest = {**store.read_manifest(), "data_version": "STALE-no-match"}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            replay(manifest, trace)
        assert any("data drift" in str(w.message) for w in caught), \
            "replay must warn when the reloaded data version != the recorded one"
