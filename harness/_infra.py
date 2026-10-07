"""Bootstrap so the vendored engine packages import in this environment.

The whole engine is vendored *inside* the harness package, under
``harness/harness/_vendor/``:

    _vendor/
      CTX/  FactorEngine/  BacktestEngine/    # the engine + data packages
      ENV_MGMT/                                # the shared imports bundle

This makes the harness fully self-contained: it reaches only *down* into its
own ``_vendor`` dir, never *up* into the surrounding Zora tree.

The vendored packages are plain top-level packages that sit side by side in
``_vendor`` and import each other by bare name (``import FactorEngine``,
``from ENV_MGMT.imports import *``). All this bootstrap has to do, therefore,
is put ``_vendor`` itself on ``sys.path`` --- one entry covers the three engine
packages *and* the shared imports bundle.

The packages were carved down to the subset the harness actually runs and
renamed to functional names (CTX / FactorEngine / BacktestEngine); their logic
is otherwise unchanged. Call :func:`ensure_infra` once before importing any
vendored package.
"""
from __future__ import annotations

import pathlib
import sys

_READY = False


def _vendor_dir() -> pathlib.Path:
    # this file: <root>/harness/harness/_infra.py -> siblings/_vendor
    return pathlib.Path(__file__).resolve().parent / "_vendor"


def infra_available() -> bool:
    """True if the vendored engine packages appear to be present in ``_vendor``."""
    vendor = _vendor_dir()
    return all((vendor / pkg).is_dir()
               for pkg in ("CTX", "FactorEngine", "BacktestEngine"))


def ensure_infra() -> pathlib.Path:
    """Make CTX/FactorEngine/BacktestEngine importable; return the ``_vendor`` dir.

    Idempotent. Raises RuntimeError with an actionable message if the shared
    ENV_MGMT bundle cannot be located inside ``_vendor``.
    """
    global _READY
    vendor = _vendor_dir()
    if not (vendor / "ENV_MGMT" / "imports.py").exists():
        raise RuntimeError(
            "cannot locate ENV_MGMT/imports.py (the shared imports bundle every "
            f"vendored module starts with) under {vendor}; the vendored engine "
            "is unavailable"
        )

    # one sys.path entry: _vendor holds all three engine packages AND ENV_MGMT,
    # so both the top-level `import FactorEngine` and `import ENV_MGMT` resolve
    # here. sys.path is inherited by spawned worker processes, so this single
    # entry is also what keeps the vendored imports resolvable inside a
    # multiprocessing pool -- an injected sys.modules entry would not be.
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    _READY = True
    return vendor
