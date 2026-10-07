"""ZORA factor-discovery harness.

An LLM invents a factor as a math formula (over a whitelisted operator set),
the harness backtests it on real data, iterates to improve in-sample Sortino,
then validates on a held-out out-of-sample window and returns Pass / Fail.

Provider-agnostic, real execution, reproducible. Standalone: depends on nothing
outside this package.
"""

__version__ = "0.1.0"
