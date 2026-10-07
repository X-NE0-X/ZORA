# Licensing and dependency notices

## Project-authored code

Copyright 2026 X-NE0-X. ZORA is source-available, with Apache License 2.0
terms subject to Commons Clause v1.0. The complete grant is in LICENSE;
LicenseRef-ZORA-Source-Available identifies this combination in package
metadata. It is not OSI-approved open source or Apache-2.0 alone.

Internal company research is allowed. The restriction applies to "Sell" as
defined in the license, including products/services whose value derives
entirely or substantially from the software. It is not a blanket prohibition
of all commercial use or every related service. The license text controls.
References: https://commonsclause.com/ and
https://www.apache.org/licenses/LICENSE-2.0.txt

## Engine and support directories

The rights holder confirmed on 2026-10-07 that all authored code is their own,
apart from separately imported packages. BacktestEngine, CTX, FactorEngine and
ENV_MGMT under harness/_vendor are recorded as first-party code covered by
LICENSE, despite the directory name. See provenance.json. No external origin
URL, original source commit or independent authorship verification is claimed.
Preserve any actual third-party attribution if later discovered.

## Separately installed dependencies

Dependencies retain their own licenses. The ZORA wheel does not bundle the
installed dependency tree. requirements.txt records the exact runtime versions
and PyPI artifact hashes. requirements-release.txt separately locks the Windows
reference release-tool environment. Package metadata/version inventories support
review; they do not establish rights to redistribute a bundled environment.
This procedure does not publish such a bundle.

vectorbt 1.1.0 is licensed by Oleg Polakow under Apache License 2.0 with Commons
Clause. Its independent restrictions remain applicable:
https://github.com/polakowo/vectorbt/blob/v1.1.0/LICENSE.md
Do not describe the complete runnable stack as unconditionally OSI open source.

## External providers and data

Provider services/binaries are separate products under their own terms. No
provider binary, cached login, credential, market-data cache or research journal
is included in the release. A code license does not grant permission to publish
Yahoo or other vendor market data. Use appropriately licensed data.
