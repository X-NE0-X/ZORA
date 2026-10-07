"""Release hygiene fails closed; no model calls, publishing or history edits."""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import zipfile

import pytest

from tools import check_release as release
from tools import download_gitleaks as scanner
from tools.scan_git_history import main as scan_history, scoped_git_env, verified_coverage
from tools.smoke_wheel import verify_location, verify_metrics


@pytest.mark.parametrize("name", ["../key", "/root/key", "C:secret", "a\\b",
                                  "artifacts/result.json", "memory/raw/run.json",
                                  "harness/.env", ".env.template", "data/prices.parquet",
                                  ".git/config", ".aws/credentials", "a/../b"])
def test_private_and_unsafe_members_rejected(name):
    with pytest.raises(release.ReleaseError):
        release.safe_name(name)


def test_only_documented_env_template_allowed():
    assert release.safe_name(release.TEMPLATE) == release.TEMPLATE


def test_current_public_selection_includes_untracked_modules_but_no_private_state():
    source = release.public_files(release.ROOT)
    release.validate_source(release.ROOT, source)
    assert "harness/factor/contract.py" in source
    assert "harness/providers/structured.py" in source
    assert "tests/test_rejected_bindings.py" in source
    assert "tests/test_release_preparation.py" in source
    assert "tools/smoke_wheel.py" in source
    assert not any(part in release.PRIVATE_PARTS for name in source for part in Path(name).parts)


@pytest.mark.parametrize("line_ending", ["\n", "\r\n"], ids=["LF", "CRLF"])
def test_public_identity_and_license_are_anonymous_and_combined(line_ending):
    source = release.public_files(release.ROOT)
    source["LICENSE"] = source["LICENSE"].decode("utf-8").replace("\r\n", "\n") \
        .replace("\n", line_ending).encode("utf-8")
    release.validate_source(release.ROOT, source)
    text = source["LICENSE"].decode("utf-8").replace("\r\n", "\n")
    assert "Copyright 2026 X-NE0-X" in text
    assert '"Commons Clause" License Condition v1.0' in text
    assert "right to\nSell the Software" in text
    assert "Apache License" in text and "END OF TERMS AND CONDITIONS" in text
    provenance = json.loads(source["provenance.json"])
    assert provenance["public_rights_holder"] == "X-NE0-X"
    assert len(provenance["components"]) == 4


def _source():
    return {"harness/__init__.py": b"# package\n", "LICENSE": b"combined license",
            "NOTICE": b"notice", "THIRD_PARTY_NOTICES.md": b"dependency notices",
            "provenance.json": b"{}", ".gitignore": b"artifacts/\n"}


def _metadata():
    return ("Metadata-Version: 2.4\nName: zora-harness\nVersion: 0.1.0\n"
            f"License-Expression: {release.LICENSE_EXPRESSION}\n" +
            "".join(f"License-File: {name}\n" for name in
                    ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "provenance.json"))).encode()


def _wheel(path, source, extra=None):
    files = {name: payload for name, payload in source.items() if name.startswith("harness/")}
    files.update({"zora_harness-0.1.0.dist-info/licenses/" + name: source[name]
                  for name in ("LICENSE", "NOTICE", "THIRD_PARTY_NOTICES.md", "provenance.json")})
    files["zora_harness-0.1.0.dist-info/METADATA"] = _metadata()
    files.update(extra or {})
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)


def test_wheel_legal_assets_and_current_source_checked(tmp_path):
    path = tmp_path / "good.whl"
    source = _source()
    _wheel(path, source)
    assert release.verify_archive(path, source)["complete"]
    source["harness/__init__.py"] = b"changed"
    with pytest.raises(release.ReleaseError, match="stale"):
        release.verify_archive(path, source)


@pytest.mark.parametrize("name", ["harness/.env.local", "unexpected.txt", "data/raw.csv"])
def test_unapproved_wheel_files_fail_closed(tmp_path, name):
    path = tmp_path / "bad.whl"
    source = _source()
    _wheel(path, source, {name: b"private"})
    with pytest.raises(release.ReleaseError):
        release.verify_archive(path, source)


def test_duplicate_wheel_member_rejected(tmp_path):
    path = tmp_path / "duplicate.whl"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("harness/__init__.py", b"one")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("harness/__init__.py", b"two")
    with pytest.raises(release.ReleaseError, match="Duplicate"):
        release.archive_files(path)


def test_sdist_source_and_no_git_history_checked(tmp_path):
    path = tmp_path / "good.tar.gz"
    source = _source()
    with tarfile.open(path, "w:gz") as archive:
        for name, payload in {**{n: p for n, p in source.items() if n != ".gitignore"},
                              "PKG-INFO": _metadata()}.items():
            info = tarfile.TarInfo("zora_harness-0.1.0/" + name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    assert release.verify_archive(path, source)["complete"]


@pytest.mark.parametrize("name", ["zora/../escape", "/absolute", "zora/.git/config"])
def test_sdist_unsafe_names_rejected(tmp_path, name):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        archive.addfile(tarfile.TarInfo(name))
    with pytest.raises(release.ReleaseError):
        release.archive_files(path)


def test_sdist_link_rejected(tmp_path):
    path = tmp_path / "link.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo("zora/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../private"
        archive.addfile(info)
    with pytest.raises(release.ReleaseError, match="link"):
        release.archive_files(path)


def test_export_never_overwrites_prior_source(tmp_path):
    target = tmp_path / "candidate"
    release.export_snapshot(target, {"README.md": b"public"})
    with pytest.raises(release.ReleaseError, match="already exist"):
        release.export_snapshot(target, {"README.md": b"replacement"})
    assert (target / "README.md").read_bytes() == b"public"
    assert not (target / ".git").exists()


def test_export_rejects_credential_path(tmp_path):
    with pytest.raises(release.ReleaseError):
        release.export_snapshot(tmp_path / "candidate", {".env": b"private"})
    assert not (tmp_path / "candidate" / ".env").exists()


def test_scanner_checksum_precedes_executable_read(monkeypatch):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("gitleaks.exe", b"test scanner")
    content = payload.getvalue()
    with pytest.raises(ValueError, match="checksum"):
        scanner.verified_executable(content)
    monkeypatch.setattr(scanner, "SHA256", hashlib.sha256(content).hexdigest())
    assert scanner.verified_executable(content) == b"test scanner"


def test_smoke_location_rejects_checkout_even_when_environment_is_nested(tmp_path):
    prefix = tmp_path / ".venv"
    verify_location(str(prefix / "Lib/site-packages/harness/__init__.py"), str(prefix), tmp_path)
    with pytest.raises(ValueError, match="checkout"):
        verify_location(str(tmp_path / "harness/__init__.py"), str(tmp_path), tmp_path)


@pytest.mark.parametrize("bad_label", [None, "is_metrics", "oos_metrics"])
def test_wheel_metric_floors_use_full_is_and_oos_names(bad_label):
    from types import SimpleNamespace
    config = SimpleNamespace(min_is_days=35, min_oos_days=24)
    sample = {"n_active": 40, "cagr": -0.05, "sortino": -0.2,
              "avg_gross": 1.0, "avg_turnover": 0.3}
    record = {label: dict(sample) for label in ("is_metrics", "oos_metrics")}
    if bad_label:
        record[bad_label]["n_active"] = 23
        with pytest.raises(ValueError, match="active bars"):
            verify_metrics(record, config)
    else:
        verify_metrics(record, config)  # a strategy FAIL is not a software error


def test_ci_actions_pinned_and_new_gates_present():
    workflow = (release.ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    import re
    actions = re.findall(r"uses: ([^\s]+)", workflow)
    assert actions and all(re.fullmatch(r"actions/[^@]+@[0-9a-f]{40}", action) for action in actions)
    for gate in ("pip_audit", "--require-hashes", "tools/check_release.py", "tools/smoke_wheel.py",
                 "tools/scan_git_history.py", "python tests/run_tests.py --workers 4"):
        assert gate in workflow


def test_canonical_runner_uses_independent_shard_caches():
    runner = (release.ROOT / "tests/run_tests.py").read_text(encoding="utf-8")
    assert "name + '-cache'" in runner
    assert "cache_dir={destination / 'cache'}" in runner


@pytest.mark.parametrize("output,expected", [("0 commits scanned.\nno leaks found", 1),
                                            ("1 commits scanned.\nERR fatal: git failed", 1),
                                            ("no leaks found", 1), ("1 commits scanned.", 2),
                                            ("0 commits scanned.", 0),
                                            ("1 commits scanned.\n1 commits scanned.", 1),
                                            ("\x1b[1m0 commits scanned.\x1b[0m", 1),
                                            ("1 commits scanned.\n\x1b[31mERR\x1b[0m git failed", 1),
                                            ("\x1b[1m1 commits scanned.\x1b[0m\n"
                                             "\x1b[1m1 commits scanned.\x1b[0m", 1),
                                            ("\x1b[1m1 commits scanned.\x1b[0m", 2)])
def test_secret_history_false_green_rejected(output, expected):
    with pytest.raises(ValueError, match="complete"):
        verified_coverage(output, expected)


def test_secret_history_coverage_and_scoped_trust(tmp_path, monkeypatch):
    assert verified_coverage("1 commits scanned.\nno leaks found", 1) == 1
    assert verified_coverage("\x1b[1m1 commits scanned.\x1b[0m\nno leaks found", 1) == 1
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "other.setting")
    environment = scoped_git_env(tmp_path)
    assert environment["GIT_CONFIG_COUNT"] == "2"
    assert environment["GIT_CONFIG_KEY_1"] == "safe.directory"
    assert environment["GIT_CONFIG_VALUE_1"] == tmp_path.resolve().as_posix()
    assert environment["GIT_CONFIG_KEY_0"] == "other.setting"


@pytest.mark.parametrize("output,exit_code,passed", [
    ("1 commits scanned.\nno leaks found", 0, True),
    ("1 commits scanned.\nno leaks found", 1, False),
    ("0 commits scanned.\nno leaks found", 0, False),
    ("no leaks found", 0, False),
    ("1 commits scanned.\nERR [git] fatal: detected dubious ownership", 0, False),
    ("\x1b[1m1 commits scanned.\x1b[0m\nno leaks found", 0, True),
    ("\x1b[1m1 commits scanned.\x1b[0m\n\x1b[31mERR\x1b[0m [git] failed", 0, False),
])
def test_history_receipt_preserves_fail_closed_coverage_and_redacts_output(
        tmp_path, monkeypatch, capsys, output, exit_code, passed):
    from types import SimpleNamespace
    from tools import scan_git_history as history

    private_text = "hidden-token-should-never-be-persisted"
    results = iter([
        SimpleNamespace(returncode=0, stdout="1\n", stderr=""),
        SimpleNamespace(returncode=exit_code, stdout=output, stderr=private_text),
    ])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return next(results)

    monkeypatch.setattr(history.subprocess, "run", run)
    report = tmp_path / "history.json"
    result = scan_history(["--source", str(tmp_path), "--scanner", str(tmp_path / "scanner"),
                           "--report", str(report)])
    assert result == (0 if passed else 1)
    text = report.read_text(encoding="utf-8")
    receipt = json.loads(text)
    assert receipt["complete"] is passed
    assert receipt["redacted"] is True
    assert receipt["diagnostics"]["expected_commits"] == 1
    assert receipt["diagnostics"]["scanner_exit_code"] == exit_code
    if "1 commits scanned" in output:
        assert receipt["diagnostics"]["reported_commits"] == [1]
    assert receipt["diagnostics"]["scanner_error"] is ("ERR" in output)
    assert "--no-color" in calls[1] and "--redact=100" in calls[1]
    assert private_text not in text + capsys.readouterr().out
    assert "stdout" not in receipt and "stderr" not in receipt


@pytest.mark.parametrize("failure", [ValueError, OSError])
def test_history_subprocess_exceptions_never_persist_private_text(tmp_path, monkeypatch, capsys,
                                                                 failure):
    from tools import scan_git_history as history

    private_text = "hidden-token-in-subprocess-exception"

    def fail(*args, **kwargs):
        raise failure(private_text)

    monkeypatch.setattr(history.subprocess, "run", fail)
    report = tmp_path / "history.json"
    assert scan_history(["--source", str(tmp_path), "--scanner", str(tmp_path / "scanner"),
                         "--report", str(report)]) == 1
    text = report.read_text(encoding="utf-8")
    assert json.loads(text)["complete"] is False
    assert private_text not in text + capsys.readouterr().out
