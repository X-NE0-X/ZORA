#================================= ZORA HARNESS (vendored) ==============================
# INFRASTRUCTURE (CTX)
# Data Processing | Notebook Runtime | Payload
#========================================================================================

# No import-time bootstrap lives here on purpose. The upstream package opened with a
# monorepo-root search plus a dotenv credential loader that reached OUTSIDE the
# repository; neither is wanted in a self-contained, publishable tree.
# `harness._infra.ensure_infra()` is the one supported bootstrap and it already puts
# `_vendor` on sys.path, which is what makes the shared ENV_MGMT bundle resolvable.

# ---- public API re-exports (facade keeps `from CTX import CTX` stable) ----
from .Config import (
    DEFAULT_DATA_CONFIG, DEFAULT_FACTOR_CONFIG, DEFAULT_EXECUTION_CONFIG,
    DEFAULT_SIGNAL_CONFIG, DEFAULT_TRAVERSAL_CONFIG,
    DEFAULT_PARQUET_CATALOG, DEFAULT_TIME_PROFILES, DATA_ARCHIVES_ROOT,
    DEFAULT_PARQUET_PATTERNS, SESSION_SIGNATURE_FIELDS, _deep_config_merge,
)
from .DataOps import _DataOps
from .MarketData import MarketData
from .CTX import CTX
