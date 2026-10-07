# Contributing

ZORA is source-available under the combined license in LICENSE, including
Commons Clause. It is not unmodified Apache-2.0 or OSI-approved open source.
Contributions intentionally submitted for inclusion are under those same terms;
submit only code you are entitled to license. Third-party code must keep its
original license/attribution and include verifiable source/version provenance.

Use Windows / CPython 3.14 and an isolated environment:

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install --require-hashes --only-binary=TA-Lib -r requirements.txt
.venv/Scripts/python -m pip install --no-deps --no-build-isolation -e .
.venv/Scripts/python -m pip check
.venv/Scripts/python tests/run_tests.py --workers 4
```

Keep changes focused and add a regression for changed behavior. The canonical
runner requires exact once-only coverage; collection failure or skipped cases do
not count as full-suite acceptance. Review raw logs as well as JSON receipts.
Do not submit credentials, personal paths, market-data caches, model traces,
research journals, provider login state or generated artifacts. Keep tests and
release tools in the public source tree so recipients can verify it.

For package checks follow RELEASE.md. Report vulnerabilities privately via the
channel in SECURITY.md; ordinary bugs may use repository issues with a minimal,
sanitized reproduction. Live paid-provider tests require a separate decision.
