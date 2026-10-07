#================================= ZORA HARNESS (vendored) ==============================
# INFRASTRUCTURE (shared imports bundle)
#========================================================================================

"""Marker that makes ``ENV_MGMT`` a real package --- purely for PACKAGING.

``ENV_MGMT`` worked as an implicit namespace package in a source checkout, so
``from ENV_MGMT.imports import *`` (the first line of every vendored module)
resolved fine. Namespace packages are invisible to setuptools' ``PackageFinder``
though, so ``imports.py`` was omitted from the wheel --- and it is the one file
the whole vendored tree depends on.

Deliberately empty of re-exports: ``imports.py`` is a star-import bundle that
pulls in TA-Lib, numba, duckdb, matplotlib and vectorbt. Re-exporting it from
here would move that cost to ``import ENV_MGMT`` and run it even for callers
that only need the package to exist.
"""
