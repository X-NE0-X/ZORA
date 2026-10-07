"""Local redacted history scan; exit zero without actual coverage is rejected."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess


class HistoryScanError(ValueError):
    """A fixed, operator-safe failure message without subprocess output."""


def plain_scan_output(output: str) -> str:
    """Remove terminal styling before checking coverage and error markers."""
    return re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)


def verified_coverage(output: str, expected: int) -> int:
    output = plain_scan_output(output)
    counts = re.findall(r"\b(\d+) commits scanned\b", output)
    if expected < 1 or len(counts) != 1 or int(counts[0]) != expected \
            or re.search(r"\bERR\b|fatal:", output):
        raise HistoryScanError("History scanner did not verify the complete selected Git history")
    return int(counts[0])


def scan_diagnostics(output: str, expected: int, returncode: int) -> dict:
    """Persist only allowlisted counters/flags, never raw Git/scanner text."""
    output = plain_scan_output(output)
    lower = output.lower()
    return {
        "expected_commits": expected,
        "reported_commits": [int(value) for value in
                             re.findall(r"\b(\d+) commits scanned\b", output)],
        "scanner_exit_code": returncode,
        "scanner_error": bool(re.search(r"\bERR\b|fatal:", output)),
        "git_ownership_error": "dubious ownership" in lower,
        "git_stderr_error": "[git]" in lower,
        "git_patch_parse_error": "parse" in lower and "error" in lower,
        "remote_url_error": "remote url" in lower and "unable" in lower,
    }


def scoped_git_env(root: Path) -> dict[str, str]:
    # An exact, process-local trust exception for the user-selected checkout.
    # No global safe.directory, Git config, ownership or history is changed.
    env = dict(os.environ)
    index = int(env.get("GIT_CONFIG_COUNT", "0"))
    env["GIT_CONFIG_COUNT"] = str(index + 1)
    env[f"GIT_CONFIG_KEY_{index}"] = "safe.directory"
    env[f"GIT_CONFIG_VALUE_{index}"] = root.resolve().as_posix()
    return env


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    parser.add_argument("--scanner", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        parser.error("Existing scan evidence is never overwritten")
    root = args.source.resolve()
    payload = {"complete": False, "redacted": True}
    try:
        env = scoped_git_env(root)
        git = subprocess.run(["git", "-C", str(root), "rev-list", "--all", "--count"],
                             env=env, capture_output=True, text=True, encoding="utf-8", timeout=30)
        if git.returncode != 0:
            raise HistoryScanError("Cannot read selected Git history")
        if not re.fullmatch(r"\d+", git.stdout.strip()):
            raise HistoryScanError("Cannot parse selected Git history count")
        expected = int(git.stdout.strip())
        result = subprocess.run([str(args.scanner.resolve()), "git", str(root), "--log-opts=--all",
                                 "--redact=100", "--no-banner", "--no-color"],
                                env=env, capture_output=True, text=True, encoding="utf-8", timeout=120)
        output = result.stdout + result.stderr
        payload["diagnostics"] = scan_diagnostics(output, expected, result.returncode)
        covered = verified_coverage(output, expected)
        if result.returncode != 0:
            raise HistoryScanError("History secret scan failed or found candidate credentials")
        payload.update(complete=True, expected_commits=expected, scanned_commits=covered,
                       scanner="gitleaks", personal_metadata_reviewed=False)
    except HistoryScanError as error:
        payload["error"] = str(error)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        payload.update(error="History scan could not be completed", error_type=type(error).__name__)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))
    return 0 if payload["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
