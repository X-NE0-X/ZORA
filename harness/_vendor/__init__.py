#================================= ZORA HARNESS (vendored) ==============================
# INFRASTRUCTURE (container)
#========================================================================================

"""Marker that makes ``_vendor`` a real package --- purely for PACKAGING.

Nothing imports ``harness._vendor``. The vendored engines are consumed as
top-level packages (``import CTX``) after :func:`harness._infra.ensure_infra`
puts this directory on ``sys.path``, and that keeps working whether or not this
file exists.

It has to exist anyway because setuptools' ``PackageFinder`` walks the tree and
treats a directory WITHOUT ``__init__.py`` as a non-package it must not descend
into. Without this file ``find(include=["harness*"])`` returned exactly
``['harness', 'harness.factor', 'harness.providers']`` --- the built wheel
contained no ``_vendor/`` at all, so an installed harness had no backtest
engine, no factor engine and no market-data layer. The failure was invisible in
the source checkout (where ``_vendor`` is simply a directory on disk) and only
appeared after ``pip install``.

Deliberately free of imports: an ``__init__`` that pulled in the engines would
make merely *finding* the package drag in TA-Lib, numba and vectorbt.
"""
