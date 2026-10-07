"""Canonical current-only pytest suite; optional disjoint file shards.

python tests/run_tests.py                  # all cases once, one process
python tests/run_tests.py --workers 4      # the same cases, non-overlapping shards
python tests/run_tests.py -m current       # explicitly scoped current-only check
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _invoke(arguments, destination, name, env):
    from tests.regression_support import source_fingerprint
    before = source_fingerprint()
    command = [sys.executable, "-m", "pytest", "-p", "tests.regression_support", "-q", *arguments,
               "--regression-report", str(destination / f"{name}.json"),
               "--basetemp", str(destination / f"{name}-temp"),
               "-o", f"cache_dir={destination / (name + '-cache')}",
               "--junitxml", str(destination / f"{name}.xml")]
    execution_error = None
    try:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", timeout=5400)
        code, output = result.returncode, result.stdout
    except subprocess.TimeoutExpired as error:
        code, output = 124, error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        execution_error = "Regression subprocess exceeded its time limit"
    except OSError as error:
        code, output = 1, ""
        execution_error = f"Regression subprocess could not start: {error}"
    (destination / f"{name}.txt").write_text(output, encoding="utf-8")
    try:
        receipt = json.loads((destination / f"{name}.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        execution_error = execution_error or "Regression subprocess produced no valid receipt"
        receipt = {"schema": 1, "source_before": before, "source_after": source_fingerprint(),
                   "cases": [], "complete": False}
    if execution_error:
        code = code or 1
        receipt.update(complete=False, exit_code=code, execution_error=execution_error)
        (destination / f"{name}.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: exit={code}", flush=True)
    if code:
        print(execution_error or output[-5000:], flush=True)
    return code, receipt


def _parallel(workers, destination):
    from tests.regression_support import counts, merge_results, plan_shards, source_fingerprint
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    code, collection = _invoke(["--collect-only"], destination, "collection", environment)
    if code or collection.get("scope") != "full":
        code = code or 1
        failure = {"schema": 1, "scope": "full", "complete": False, "exit_code": code,
                   "error": collection.get("execution_error", "Full-suite collection required; remove pytest selectors and filters")}
        (destination / "summary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        return code
    groups = plan_shards(collection["cases"], workers)
    if not groups:
        return 5
    with ThreadPoolExecutor(max_workers=len(groups)) as executor:
        futures = [executor.submit(_invoke, group, destination, f"shard-{i}", environment)
                   for i, group in enumerate(groups)]
        finished = [future.result() for future in futures]
    receipts = [receipt for _, receipt in finished]
    try:
        if any(receipt["source_before"] != collection["source_before"] for receipt in receipts):
            raise ValueError("Shard sources differ from the collected version")
        if source_fingerprint() != collection["source_before"]:
            raise ValueError("Source changed since collection")
        cases = merge_results(collection["cases"], receipts)
        summary = counts(cases)
        code = 0 if all(value == 0 for value, _ in finished) and summary == {"passed": len(cases)} else 1
        payload = {"schema": 1, "scope": "full", "complete": code == 0,
                   "python": collection["python"], "pytest": collection["pytest"],
                   "source": collection["source_before"],
                   "unique_cases": len(cases), "counts": summary, "exit_code": code,
                   "warning_count": sum(receipt.get("warning_count", 0) for receipt in receipts),
                   "contracts": dict(Counter(c["contract"] for c in cases)),
                   "shards": groups, "cases": cases}
    except ValueError as error:
        code = 1
        payload = {"schema": 1, "scope": "full", "complete": False, "exit_code": code, "error": str(error)}
    (destination / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"cases", "shards"}}), flush=True)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--report-dir", type=Path)
    options, extra = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if options.workers < 1:
        parser.error("--workers must be positive")
    if options.workers > 1 and any(arg not in {"-q", "-v"} for arg in extra):
        parser.error("Parallel execution is full-suite only; use --workers 1 for pytest selectors")
    reports = ROOT / "artifacts" / "regression"
    reports.mkdir(parents=True, exist_ok=True)
    destination = options.report_dir.resolve() if options.report_dir else Path(tempfile.mkdtemp(prefix="run-", dir=reports))
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        parser.error("--report-dir must be empty; existing regression evidence is never overwritten")
    print(f"Regression evidence: {destination}", flush=True)
    if options.workers > 1:
        return _parallel(options.workers, destination)
    import pytest
    pytest.register_assert_rewrite("tests.regression_support")
    from tests import regression_support
    previous = Path.cwd()
    try:
        os.chdir(ROOT)
        return int(pytest.main([*(extra or ["-q"]), "--regression-report", str(destination / "results.json"),
                               "-o", f"cache_dir={destination / 'cache'}",
                               "--basetemp", str(destination / "temp"),
                               "--junitxml", str(destination / "results.xml")], plugins=[regression_support]))
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    raise SystemExit(main())
