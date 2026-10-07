# Security

## Scope and supported runtime

ZORA is a single-user local research tool running with the user's permissions.
The harness itself is not sandboxed or a multi-tenant service. The OpenCode
adapter can start an owned, temporary loopback server; external provider APIs
and CLI binaries have their own trust boundaries.

The reference runtime is Windows / CPython 3.14 with the hashed runtime lock.
See RELEASE.md for the validation scope. Successful tests or dependency scans
do not establish that every input, external service or vendored kernel is safe.

## Trusted inputs

Treat run configs, prompt assets, research journals, replay traces and market
data as operator-controlled inputs. Configs choose provider endpoints and file
paths. A provider may receive the prompts, formulas, metrics and journal text.
Do not run another party's config or research state without reviewing it.

Model-written rationale, mechanism and journal text is included in subsequent
prompts. That creates a prompt-injection boundary. The model proposes candidates;
local code controls admission and scoring. Containment does not establish the
truth of a model's economic explanation.

A trace or manifest can be edited to falsify research provenance. Prompt hashes
are integrity checks, not signatures or proof of authorship. Hashless traces
are rejected and prompt mismatches raise ReplayDesyncError in the current replay
provider. Data-version mismatches are also surfaced, and historical records
must not be relabeled as validation of current data or code.

## Formula admission and local persistence

- The untrusted formula passes the strict-math-v2 compiler in
  harness/factor/contract.py. It checks a closed AST grammar, operator slots,
  typed bindings, units and structure. It rejects attributes, indexing,
  comprehensions and arbitrary callable names. There is no Python eval of the
  formula. harness/factor/checked_evaluate.py applies the admitted mathematical
  construction with masks and numerical-domain checks.
- run_name must be a single path segment. Store rejects path separators,
  traversal, absolute paths and Windows drive-relative forms.
- A run ownership lock covers reset and writes. Manifests and traces use
  replace-based persistence. These controls do not authenticate files that an
  operator can edit outside the harness.

## Credentials and private state

Keys are read from the process environment first. A real provider loads .env
lazily; package import does not load it. Source installs use
harness/_vendor/ENV_MGMT/.env. Wheel installs use a per-user configuration
directory (APPDATA on Windows; XDG_CONFIG_HOME or ~/.config elsewhere).

env.save_key creates files with POSIX mode 0600 and replaces duplicate key
entries. Windows chmod does not enforce an owner-only ACL. Windows protection
depends on the actual inherited directory ACL; do not assume it from a path
name. The harness does not establish a Windows ACL guarantee.

Store recursively redacts recognized secrets before serializing JSON artifacts.
RecordingProvider redacts completion/error strings and CLI errors are redacted.
The redactor uses managed live key values and known credential patterns. It is
not an exhaustive detector for arbitrary credentials or sensitive research prose.
Review exported files; runtime redaction alone does not authorize publication.

.env and .env variants are ignored; only the placeholder .env.template travels
with the package. The package data list is explicit. The release checker rejects
credentials, parquet, live memory, artifacts, local-agent state and unapproved
archive members. It also verifies package bytes against current source.

memory/ and artifacts/ are private runtime state. Never copy them or a provider's
cached login into a public tree. Selected research records require a separate
review and export decision. If a key leaks, revoke it with its issuer; removing a
file or commit does not invalidate the credential.

## CLI and HTTP providers

Child environments use an operational-variable allowlist, with only the selected
backend credential re-admitted. Providers may still read their own cached login
and configuration. Codex uses its read-only sandbox. OpenCode normally pins the
custom harness-llm agent with known file/shell/research tools disabled in its
inline config. This is application-level control, not a kernel sandbox or a
guarantee about arbitrary plugins or altered third-party binaries.

Provider calls have explicit timeouts. The run lifecycle closes owned OpenCode
processes; Windows CLI containment uses an owned Job Object. HTTP cancellation
can return before a blocking transport thread finishes. It does not prove that
remote generation, billing or all underlying network activity stopped.

## Network egress

scripted + synthetic is the offline path and needs no API key or market-data
download. yfinance fetches Yahoo data; real providers send prompts to the chosen
service. CTX reads configured data and uses its installed dependency stack.
The release tools additionally query official package metadata/advisories and
download a checksum-pinned scanner. Secret scanning itself is local.

## Private vulnerability reporting

Maintainer: X-NE0-X. No personal name or email is published.

Use GitHub private vulnerability reporting:
https://github.com/X-NE0-X/ZORA/security/advisories/new

This is the selected reporting channel, not a claim that it is already active
while the repository remains private. A public release must enable and verify
the Report a vulnerability entry before announcing availability. There is no
public-email fallback. Do not put secrets or exploit details in a public issue.

A useful report includes the version/commit, a small reproduction, impact and
required attacker/victim actions. There is no bounty or guaranteed response SLA.
