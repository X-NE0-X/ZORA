"""Unique-case accounting shared by pytest and the canonical regression runner."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform

import pytest

ROOT = Path(__file__).resolve().parents[1]


def partition_contracts(items):
    """Never silently deduplicate: repeated execution or ambiguous modes fails."""
    repeated = [nodeid for nodeid, n in Counter(item.nodeid for item in items).items() if n > 1]
    if repeated:
        raise pytest.UsageError("Duplicate regression node IDs: " + ", ".join(repeated))
    for item in items:
        current = item.get_closest_marker("current") is not None
        if item.get_closest_marker("legacy") is not None:
            raise pytest.UsageError(f"Unsupported regression contract: {item.nodeid}")
        if not current:
            item.add_marker(pytest.mark.current)


def source_fingerprint():
    paths = [ROOT / "pyproject.toml"]
    for directory in ("harness", "tests"):
        paths.extend(p for p in (ROOT / directory).rglob("*")
                     if p.is_file() and p.suffix in {".py", ".json"} and "__pycache__" not in p.parts)
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def case_outcome(reports):
    if any(n > 1 for n in Counter(r["phase"] for r in reports).values()):
        return "error"
    if any(r["outcome"] == "failed" and r["phase"] != "call" for r in reports):
        return "error"
    if any(r["outcome"] == "failed" for r in reports):
        return "failed"
    if any(r["outcome"] == "skipped" for r in reports):
        return "skipped"
    if {r["phase"] for r in reports if r["outcome"] == "passed"} == {"setup", "call", "teardown"}:
        return "passed"
    return "not_run"


def merge_results(expected, receipts):
    """A full run needs exact once-only coverage, not a sum of historical totals."""
    wanted = {case["nodeid"]: case for case in expected}
    if len(wanted) != len(expected):
        raise ValueError("Duplicate expected regression cases")
    results = {}
    for receipt in receipts:
        if receipt["source_before"] != receipt["source_after"]:
            raise ValueError("Source changed during a regression shard")
        for case in receipt["cases"]:
            nodeid = case["nodeid"]
            if nodeid in results:
                raise ValueError(f"Overlapping regression shards: {nodeid}")
            if nodeid not in wanted or case["contract"] != wanted[nodeid]["contract"]:
                raise ValueError(f"Unexpected regression case or contract: {nodeid}")
            if case.get("outcome") not in {"passed", "failed", "error", "skipped"}:
                raise ValueError(f"Regression case did not finish: {nodeid}")
            results[nodeid] = case
    missing = wanted.keys() - results.keys()
    if missing:
        raise ValueError("Missing regression cases: " + ", ".join(sorted(missing)))
    return [results[nodeid] for nodeid in wanted]


def counts(cases):
    return dict(Counter(case.get("outcome", "not_run") for case in cases))


def plan_shards(cases, workers):
    if workers < 1:
        raise ValueError("workers must be positive")
    if len({case["nodeid"] for case in cases}) != len(cases):
        raise ValueError("Duplicate expected regression cases")
    by_file = {}
    for case in cases:
        by_file.setdefault(case["nodeid"].split("::", 1)[0], []).append(case)
    groups = [[] for _ in range(min(workers, len(by_file)))]
    loads = [0] * len(groups)
    for filename, selected in sorted(by_file.items(), key=lambda item: (-len(item[1]), item[0])):
        index = min(range(len(groups)), key=lambda i: loads[i])
        groups[index].append(filename)
        loads[index] += len(selected)
    return groups


def regression_scope(config, deselected=0):
    targets = [Path(arg).resolve() for arg in config.args if "::" not in arg]
    filtered = any(getattr(config.option, name, None) for name in (
        "keyword", "markexpr", "deselect", "ignore", "ignore_glob", "lf", "stepwise",
    ))
    return "full" if not deselected and not filtered and targets == [ROOT / "tests"] \
        and len(targets) == len(config.args) else "selected"


class RegressionAudit:
    def __init__(self, destination):
        self.destination = Path(destination)
        self.cases = []
        self.reports = {}
        self.collection_errors = 0
        self.deselected = 0
        self.warnings = []
        self.source_before = source_fingerprint()

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items):
        partition_contracts(items)
        self.cases = [{"nodeid": item.nodeid,
                       "contract": "strict-math-v2"}
                      for item in items]

    def pytest_collectreport(self, report):
        if report.failed:
            self.collection_errors += 1

    def pytest_deselected(self, items):
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report):
        self.reports.setdefault(report.nodeid, []).append({"phase": report.when, "outcome": report.outcome})

    def pytest_warning_recorded(self, warning_message, when, nodeid, location):
        self.warnings.append({"nodeid": nodeid, "when": when,
                              "category": warning_message.category.__name__,
                              "message": str(warning_message.message)})

    def pytest_sessionfinish(self, session, exitstatus):
        after = source_fingerprint()
        for case in self.cases:
            case["outcome"] = case_outcome(self.reports.get(case["nodeid"], []))
        scope = regression_scope(session.config, self.deselected)
        complete = bool(self.cases) and all(c["outcome"] == "passed" for c in self.cases) \
            and not self.collection_errors and not session.config.option.collectonly
        if after != self.source_before or (scope == "full" and not session.config.option.collectonly
                                          and not complete and session.exitstatus == 0):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
        payload = {"schema": 1, "python": platform.python_version(), "pytest": pytest.__version__,
                   "scope": scope, "complete": complete and after == self.source_before,
                   "unique_cases": len(self.cases), "deselected": self.deselected,
                   "warning_count": len(self.warnings), "warnings": self.warnings,
                   "source_before": self.source_before, "source_after": after,
                   "collection_only": session.config.option.collectonly,
                   "collection_errors": self.collection_errors, "exit_code": int(session.exitstatus),
                   "counts": counts(self.cases), "cases": self.cases}
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        self.destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def pytest_addoption(parser):
    parser.addoption("--regression-report", help="UTF-8 unique-case regression receipt (JSON)")


def pytest_configure(config):
    destination = config.getoption("--regression-report")
    if destination:
        config.pluginmanager.register(RegressionAudit(destination), "zora-regression-audit")
