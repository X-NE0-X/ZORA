"""Operator library --- the math building blocks a factor formula is composed of.

Design contract (this is the whole safety story):
  * A "panel" is a set of named fields. Each field is a pandas DataFrame indexed
    by date (rows, ascending time) x symbol (columns). So ``close`` is a
    (T x N) DataFrame.
  * Cross-sectional operators act ACROSS symbols (axis=1) at a single date.
  * Time-series operators act DOWN time (axis=0) using only PAST rows
    (rolling / shift with positive lag). No operator can look forward, so a
    signal value at date t never depends on data after t --- the structural
    no-look-ahead guarantee (verified by tests/test_lookahead.py).

Every operator returns a DataFrame of the same (T x N) shape (or a scalar op
scalar). The model composes these into novel factors; the operators themselves
are just arithmetic --- humans do NOT predefine factors here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- field names the formula may reference ---------------------------------
# ``typical_price`` is always (high+low+close)/3 --- no volume term. ``vwap`` is a
# genuine volume-weighted price only when the source supplies one (CTX parquet
# carrying a Vwap column); on yfinance/synthetic panels it falls back to the same
# HLC3 proxy, and Panel.vwap_source records which one a run actually got.
FIELDS = frozenset(
    {"open", "high", "low", "close", "volume", "vwap", "typical_price", "returns"}
)


def _is_df(x) -> bool:
    return isinstance(x, pd.DataFrame)


# --- arithmetic ------------------------------------------------------------
def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    """Safe divide: division by zero -> NaN (never raises / never +inf)."""
    if _is_df(b):
        b = b.replace(0.0, np.nan)
    elif b == 0:
        b = np.nan
    return a / b


def neg(a):
    return -a


def _abs(a):
    return a.abs() if _is_df(a) else abs(a)


def signed_log(a):
    """Sign-preserving log1p(|a|) --- defined for negatives, stable at 0."""
    if _is_df(a):
        return np.sign(a) * np.log1p(a.abs())
    return float(np.sign(a) * np.log1p(abs(a)))


def sign(a):
    return np.sign(a)


def ordinary_power(a, p):
    """Ordinary real power; the compiler and evaluator enforce its domain."""
    return a ** p


def signed_power(a, p):
    """Explicit sign(a) * abs(a)**p, with no hybrid integer-power behavior."""
    return np.sign(a) * a.abs() ** p


def emin(a, b):
    return np.minimum(a, b)


def emax(a, b):
    return np.maximum(a, b)


# --- cross-sectional (axis=1, across symbols at one date) ------------------
def rank(a):
    """Cross-sectional percentile rank in (0, 1]."""
    return a.rank(axis=1, pct=True)


def zscore(a):
    mu = a.mean(axis=1)
    sd = a.std(axis=1, ddof=1).replace(0.0, np.nan)
    return a.sub(mu, axis=0).div(sd, axis=0)


def demean(a):
    return a.sub(a.mean(axis=1), axis=0)


def scale(a, k=1.0):
    """Scale each date's cross-section so sum(|weights|) == k."""
    s = a.abs().sum(axis=1).replace(0.0, np.nan)
    return a.div(s, axis=0) * float(k)


# --- time-series (axis=0, rolling over PAST rows) --------------------------
def _win(d) -> int:
    """Coerce a window/lag to a positive int.

    This is load-bearing for the no-look-ahead guarantee: a negative shift
    (e.g. ``delay(close, -1)``) would pull a FUTURE value into the present.
    Rejecting d < 1 closes that hole for every time-series operator.
    """
    d = int(d)
    if d < 1:
        raise ValueError(f"time-series window/lag must be >= 1, got {d}")
    return d


def delay(a, d):
    return a.shift(_win(d))


def delta(a, d):
    d = _win(d)
    return a - a.shift(d)


def ts_mean(a, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).mean()


def ts_std(a, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).std(ddof=1)


def ts_sum(a, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).sum()


def ts_min(a, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).min()


def ts_max(a, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).max()


def _rank_last(x: np.ndarray) -> float:
    # percentile of the last value within the window, in (0, 1]
    return float(np.mean(x <= x[-1]))


def ts_rank(a, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).apply(_rank_last, raw=True)


def ts_corr(a, b, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).corr(b)


def ts_cov(a, b, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).cov(b)


def decay_linear(a, d):
    d = _win(d)
    if _is_df(a) and d > len(a):
        # A window longer than the panel can never fill (min_periods=d), so the
        # rolling result is all-NaN anyway; short-circuit BEFORE allocating a
        # length-d weight vector so a pathological d can't exhaust memory.
        return a * np.nan
    w = np.arange(1, d + 1, dtype=float)
    w /= w.sum()

    def _wavg(x):
        return float(np.dot(x, w))

    return a.rolling(d, min_periods=d).apply(_wavg, raw=True)


def product(a, d):
    d = _win(d)
    return a.rolling(d, min_periods=d).apply(np.prod, raw=True)


# --- registry: DSL name -> implementation ----------------------------------
# The compiler's operator specs determine which names and argument types exist.
FUNCS = {
    "abs": _abs,
    "signed_log1p": signed_log,
    "sign": sign,
    "pow": ordinary_power,
    "signed_pow": signed_power,
    "min": emin,
    "max": emax,
    "rank": rank,
    "zscore": zscore,
    "demean": demean,
    "scale": scale,
    "delay": delay,
    "delta": delta,
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_sum": ts_sum,
    "ts_min": ts_min,
    "ts_max": ts_max,
    "ts_rank": ts_rank,
    "ts_corr": ts_corr,
    "ts_cov": ts_cov,
    "decay_linear": decay_linear,
    "product": product,
}
