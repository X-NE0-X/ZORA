"""Regression accounting is fail-closed and never adds overlapping executions."""
from collections import Counter
from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.regression_support import (
    ROOT, case_outcome, merge_results, partition_contracts, plan_shards, regression_scope,
)


class Item:
    def __init__(self, nodeid, *markers):
        self.nodeid = nodeid
        self.markers = set(markers)

    def get_closest_marker(self, name):
        return name if name in self.markers else None

    def add_marker(self, marker):
        self.markers.add(marker.name)


def test_contract_classification_is_exclusive_and_defaults_current():
    items = [Item("a"), Item("b", "current"), Item("c")]
    partition_contracts(items)
    assert [item.markers for item in items] == [{"current"}, {"current"}, {"current"}]


@pytest.mark.parametrize("items, message", [
    ([Item("a"), Item("a")], "Duplicate regression node IDs"),
    ([Item("a", "legacy", "current")], "Unsupported regression contract"),
])
def test_ambiguous_or_duplicate_collection_is_rejected(items, message):
    with pytest.raises(pytest.UsageError, match=message):
        partition_contracts(items)


def inventory():
    return [{"nodeid": "tests/a.py::test_a", "contract": "strict-math-v2"},
            {"nodeid": "tests/b.py::test_b", "contract": "strict-math-v2"}]


def receipts():
    return [{"source_before": "frozen", "source_after": "frozen",
             "cases": [{**case, "outcome": "passed"}]} for case in inventory()]


def test_complete_disjoint_receipts_match_inventory_order_once():
    merged = merge_results(inventory(), list(reversed(receipts())))
    assert merged == [case for receipt in receipts() for case in receipt["cases"]]
    assert len(merged) == len({case["nodeid"] for case in merged}) == 2


@pytest.mark.parametrize("defect, message", [
    ("duplicate_inventory", "Duplicate expected"),
    ("overlap", "Overlapping regression shards"),
    ("missing", "Missing regression cases"),
    ("unexpected", "Unexpected regression case"),
    ("wrong_contract", "Unexpected regression case"),
    ("unfinished", "did not finish"),
    ("changed_source", "Source changed"),
])
def test_incomplete_or_overlapping_receipts_cannot_pass(defect, message):
    expected, actual = deepcopy(inventory()), deepcopy(receipts())
    if defect == "duplicate_inventory":
        expected.append(expected[0])
    elif defect == "overlap":
        actual.append(actual[0])
    elif defect == "missing":
        actual.pop()
    elif defect == "unexpected":
        actual[0]["cases"][0]["nodeid"] = "unknown"
    elif defect == "wrong_contract":
        actual[0]["cases"][0]["contract"] = "unsupported-contract"
    elif defect == "unfinished":
        actual[0]["cases"][0]["outcome"] = "not_run"
    elif defect == "changed_source":
        actual[0]["source_after"] = "changed"
    with pytest.raises(ValueError, match=message):
        merge_results(expected, actual)


@pytest.mark.parametrize("reports, expected", [
    ([], "not_run"),
    ([{"phase": "setup", "outcome": "passed"}], "not_run"),
    ([{"phase": "setup", "outcome": "failed"}], "error"),
    ([{"phase": "call", "outcome": "failed"}], "failed"),
    ([{"phase": "setup", "outcome": "skipped"}], "skipped"),
    ([{"phase": phase, "outcome": "passed"} for phase in ("setup", "call", "teardown")], "passed"),
    ([{"phase": "call", "outcome": "passed"}, {"phase": "teardown", "outcome": "failed"}], "error"),
    ([{"phase": "call", "outcome": "passed"}] * 2, "error"),
])
def test_case_outcome_requires_a_finished_unambiguous_lifecycle(reports, expected):
    assert case_outcome(reports) == expected


@pytest.mark.parametrize("workers", [1, 2, 4, 100])
def test_shard_plan_keeps_each_file_and_parametrized_case_in_one_owner(workers):
    cases = [{"nodeid": f"tests/{filename}.py::test[{number}]", "contract": "strict-math-v2"}
             for filename, total in (("a", 7), ("b", 3), ("c", 1)) for number in range(total)]
    groups = plan_shards(cases, workers)
    assert groups == plan_shards(list(reversed(cases)), workers)
    files = [filename for group in groups for filename in group]
    assert Counter(files) == {"tests/a.py": 1, "tests/b.py": 1, "tests/c.py": 1}
    assert sum(sum(case["nodeid"].split("::")[0] in group for case in cases) for group in groups) == len(cases)
    assert 1 <= len(groups) <= min(workers, 3)


@pytest.mark.parametrize("cases, workers, message", [
    (inventory(), 0, "positive"),
    (inventory() * 2, 4, "Duplicate expected"),
])
def test_shard_plan_rejects_invalid_requests(cases, workers, message):
    with pytest.raises(ValueError, match=message):
        plan_shards(cases, workers)


@pytest.mark.parametrize("args, option, deselected, expected", [
    ([str(ROOT / "tests")], {}, 0, "full"),
    ([str(ROOT / "tests")], {"markexpr": "current"}, 0, "selected"),
    ([str(ROOT / "tests")], {"keyword": "strict"}, 0, "selected"),
    ([str(ROOT / "tests")], {}, 1, "selected"),
    ([str(ROOT / "tests" / "test_config.py")], {}, 0, "selected"),
    ([str(ROOT / "tests") + "::test_a"], {}, 0, "selected"),
])
def test_scope_never_labels_a_selection_as_full(args, option, deselected, expected):
    assert regression_scope(SimpleNamespace(args=args, option=SimpleNamespace(**option)), deselected) == expected


def test_runner_propagates_teardown_failure_and_writes_truthful_receipt(tmp_path):
    target = tmp_path / "test_teardown.py"
    target.write_text(
        'import pytest\n@pytest.fixture\ndef broken():\n    yield\n    raise RuntimeError("teardown probe")\n'
        'def test_body(broken):\n    assert True\n', encoding="utf-8",
    )
    destination = tmp_path / "receipts"
    result = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "run_tests.py"), str(target),
         "--report-dir", str(destination)], cwd=tmp_path,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", timeout=60,
    )
    assert result.returncode != 0, result.stdout
    receipt = json.loads((destination / "results.json").read_text(encoding="utf-8"))
    assert receipt["counts"] == {"error": 1}
    assert receipt["scope"] == "selected" and not receipt["complete"]
    assert receipt["unique_cases"] == 1
    assert receipt["source_before"] == receipt["source_after"]


@pytest.mark.parametrize("defect", ["missing_receipt", "timeout", "spawn_error"])
def test_subprocess_failure_always_leaves_a_failing_receipt(tmp_path, monkeypatch, defect):
    from tests.run_tests import _invoke

    def fail(*args, **kwargs):
        if defect == "timeout":
            raise subprocess.TimeoutExpired(args[0], 1, output=b"timeout probe")
        if defect == "spawn_error":
            raise OSError("spawn probe")
        return SimpleNamespace(returncode=0, stdout="missing receipt probe")

    monkeypatch.setattr(subprocess, "run", fail)
    code, receipt = _invoke([], tmp_path, "probe", {})
    assert code != 0 and not receipt["complete"]
    assert receipt["cases"] == [] and receipt["execution_error"]
    assert json.loads((tmp_path / "probe.json").read_text(encoding="utf-8")) == receipt


def test_parallel_full_suite_rejects_environmental_selection(tmp_path, monkeypatch):
    from tests import run_tests

    calls = []
    monkeypatch.setattr(run_tests, "_invoke", lambda args, *rest: calls.append(args) or (
        0, {"scope": "selected", "collection_only": True, "cases": inventory()},
    ))
    assert run_tests._parallel(4, tmp_path) != 0
    assert calls == [["--collect-only"]]
    receipt = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert not receipt["complete"] and "Full-suite collection required" in receipt["error"]


def test_runner_refuses_to_reuse_existing_evidence(tmp_path):
    from tests import run_tests

    old = tmp_path / "results.json"
    old.write_text('{"old":true}\n', encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        run_tests.main(["--report-dir", str(tmp_path)])
    assert error.value.code != 0
    assert old.read_text(encoding="utf-8") == '{"old":true}\n'
