# Release procedure

## License and identity

ZORA is source-available under the combined LICENSE (Apache License 2.0 terms
subject to Commons Clause v1.0). It is not OSI-approved open source. The approved
public copyright holder and maintainer is X-NE0-X; no personal email is used.
First-party engine ownership is based on the direct rights-holder confirmation
recorded in provenance.json, not an independent historical authorship audit.
Dependencies and services keep their own terms; see THIRD_PARTY_NOTICES.md.

## Validated scope and CI

Reference engine regression targets Windows / CPython 3.14 and requirements.txt.
Python 3.11-3.13 CI jobs compile source only. Neither those jobs nor the declared
resolver range establish engine support for Linux/macOS or those Python versions.
CI definitions become evidence only after they actually run on the selected Git
tree. Local test results must not be described as successful GitHub Actions runs.

The workflow has SHA-pinned Actions, read-only permissions, the canonical full
suite, dependency advisory gates, archive/metadata checks and an installed-wheel
offline execution/replay check. Release tools are isolated and locked separately
for Windows / CPython 3.14 in requirements-release.txt. Syntax-only CI does not
install that Windows-specific tool closure.

## Local candidate checks (no publication)

Run from a source checkout with the reference .venv. Use new, empty paths for
receipts so earlier evidence is preserved. The canonical runner assigns a fresh
cache per shard; do not override it to a shared path in PYTEST_ADDOPTS.

```powershell
.venv/Scripts/python tests/run_tests.py --workers 4 --report-dir artifacts/release/regression
.venv/Scripts/python -m venv artifacts/release/tools-env
artifacts/release/tools-env/Scripts/python -m pip install --require-hashes -r requirements-release.txt
artifacts/release/tools-env/Scripts/python -m pip check
artifacts/release/tools-env/Scripts/python -m pip_audit -r requirements.txt --disable-pip --strict -f json -o artifacts/release/runtime-audit.json
artifacts/release/tools-env/Scripts/python -m pip_audit -r requirements-release.txt --disable-pip --strict -f json -o artifacts/release/tools-audit.json
artifacts/release/tools-env/Scripts/python -m build --outdir artifacts/release/dist
artifacts/release/tools-env/Scripts/python -m twine check --strict artifacts/release/dist/*
.venv/Scripts/python tools/check_release.py --dist artifacts/release/dist --report artifacts/release/archive-check.json
.venv/Scripts/python tools/download_gitleaks.py --output artifacts/release/scanner
.venv/Scripts/python tools/scan_git_history.py --scanner artifacts/release/scanner/gitleaks.exe --report artifacts/release/history-scan.json
```

The build frontend uses an isolated environment and the exact setuptools backend
pin in pyproject.toml. The runtime lock and release-tool lock use PyPI SHA-256
hashes. Updating them is an explicit version/hash review, not a floating install.
Advisory checks query published package vulnerabilities; they do not prove
project-code security or authenticate every upstream author.

The archive checker compares current authored files (including untracked files)
to the wheel and sdist, includes the tests/tools/legal assets, and refuses private
state, links, unsafe paths and unexpected members. Its source-file hash manifest
is local evidence, not a Git commit or publication authorization. Run Gitleaks on
the chosen public source candidate too; reports must remain redacted. The history
wrapper verifies actual coverage against git rev-list --all --count; a scanner's
zero exit status with no readable commits is not acceptance. Its exact-path
safe.directory exception applies only to the child process, not global Git config.

Installed-wheel acceptance requires a new venv with the hashed runtime lock,
the wheel installed with --no-deps, and a working directory outside the source
package. Invoke tools/smoke_wheel.py with that venv's Python and a new --output
directory. It refuses checkout imports and checks doctor, all prompts, active
finite IS/OOS metrics and exact replay-core equality. A strategy FAIL is valid
software-smoke output and must not be promoted to profitable alpha.

## Existing-repository baseline

Release preparation reuses X-NE0-X/ZORA with its existing repository URL and
identity. For the current one-root-commit history, an owner-authorized amend can
replace the root with the reviewed current source and explicitly reset both
author and committer to X-NE0-X and the account's GitHub noreply address. Rebase
and squash are unnecessary for a single commit. Changing Git identity settings
alone does not remove the email already stored in a commit.

Before rewriting, preserve the original Git metadata and all current tracked
and untracked non-ignored source in a hash-verified, local private recovery
archive. Never publish the recovery archive or push backup branches/tags that
retain the old identity. Include every accepted source/test/tool change through
the reviewed release selection; preserve private data and research state on
disk. The empty memory/raw/.gitkeep marker is safe to retain; research contents
are not part of the release tree.

Replace only refs/heads/main using --force-with-lease with an explicitly reviewed
expected remote SHA. Abort if the remote has changed; do not use --mirror or a
blanket force push. Keep the repository private throughout this preparation.

Review existing Actions runs separately: their stored head_commit metadata can
retain old author/committer emails after a history rewrite. Preserve private
evidence before any owner-authorized deletion of affected runs. Force pushing
does not certify removal of GitHub cached commits or other references; check
residual access and request GitHub Support cleanup only with authorization.

## Human publication gates

1. Independently verify the rewritten main branch and its entire reachable
   history, including author and committer identities. Keep the existing
   repository private until publication receives separate explicit approval.
2. The selected Git tree must contain every accepted source/test/tool change.
   Compare its file hashes to the reviewed current source, including formerly
   untracked implementation. Local recovery archives and research state must
   remain outside the Git tree and published packages.
3. Review the selected tree and complete Git history for credentials and personal
   metadata. Gitleaks does not certify absence of personal information or catch
   all possible secrets. Review commit authors separately.
4. Enable GitHub private vulnerability reporting when public access is authorized
   and verify the Report a vulnerability entry. The configured URL alone does not
   establish an active channel. No personal-email fallback is published.
5. Run the complete CI and installed-package acceptance on the final release
   baseline. Review warnings, archive manifests and any skipped scan records.
6. Obtain explicit authorization for visibility changes, push, tags and package
   upload. Nothing in these tools publishes, commits or rewrites history.

These are source-release gates. Live provider compatibility, statistical alpha
validity, market-data redistribution rights and suitability for live trading
remain separate claims requiring their own evidence.
