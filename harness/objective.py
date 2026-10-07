"""Objective metrics registry --- the optimisation target is a free choice.

The harness no longer hardcodes Sortino. Any metric below can be the thing the
model optimises in-sample and the thing the OOS Pass Line is checked against.
Each metric declares its direction and objective-space neutral baseline.
Economic profitability is checked independently on compounded net wealth
growth (CAGR), including for win rates and arithmetic-return ratios.
"""
from __future__ import annotations

import math

METRICS: dict[str, dict] = {
    "sortino":      {"higher_is_better": True,  "neutral": 0.0, "label": "Sortino"},
    "sharpe":       {"higher_is_better": True,  "neutral": 0.0, "label": "Sharpe"},
    "calmar":       {"higher_is_better": True,  "neutral": 0.0, "label": "Calmar"},
    "cagr":         {"higher_is_better": True,  "neutral": 0.0, "label": "CAGR"},
    "ann_return":   {"higher_is_better": True,  "neutral": 0.0, "label": "AnnReturn"},
    "hit_rate":     {"higher_is_better": True,  "neutral": 0.5, "label": "HitRate"},
    "maxdd":        {"higher_is_better": True,  "neutral": None,
                     "profitability_metric": "cagr", "label": "MaxDD"},
    "ann_vol":      {"higher_is_better": False, "neutral": None,
                     "profitability_metric": "cagr", "label": "AnnVol"},
    "avg_turnover": {"higher_is_better": False, "neutral": None,
                     "profitability_metric": "cagr", "label": "Turnover"},
}

DEFAULT_OBJECTIVE = "sortino"

# A strategy whose average per-bar gross exposure is at or below this is treated
# as "inactive" (an empty / degenerate book that holds essentially nothing). Its
# metrics are vacuous, so it must never win the optimisation or clear the gate.
ACTIVITY_EPS = 1e-8


def is_valid(name: str) -> bool:
    return name in METRICS


def label(name: str) -> str:
    return METRICS[name]["label"]


def higher_is_better(name: str) -> bool:
    return METRICS[name]["higher_is_better"]


def value(metrics: dict, name: str) -> float:
    """Pull a metric out of a metrics dict; missing/None -> NaN."""
    v = metrics.get(name, float("nan"))
    return float("nan") if v is None else float(v)


def goodness(v: float, name: str) -> float:
    """Direction-normalised score where HIGHER is always better.

    NaN maps to -inf so an uncomputable metric never wins a comparison.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return -math.inf
    return v if higher_is_better(name) else -v


def better(a: float, b: float, name: str) -> bool:
    """True if metric value ``a`` is strictly better than ``b``."""
    return goodness(a, name) > goodness(b, name)


def passes(v: float, pass_line: float, name: str) -> bool:
    """Does value ``v`` clear the Pass Line for this metric?"""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    return v >= pass_line if higher_is_better(name) else v <= pass_line


def is_good(v: float, name: str) -> bool:
    """Is ``v`` on the favorable side of an objective-space no-edge baseline?

    Pure risk/cost metrics have no such baseline: finite drawdown, volatility or
    turnover says nothing about whether the strategy made money. They return
    ``False`` here; callers deciding profitability must use
    :func:`profitability_metric` and inspect the full metrics dictionary.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return False
    neutral = METRICS[name]["neutral"]
    if neutral is None:
        return False
    if higher_is_better(name):
        return v > neutral
    return v < neutral


def profitability_metric(name: str) -> str:
    """Metric that proves economic profitability for objective ``name``."""
    if not is_valid(name):
        raise KeyError(name)
    return "cagr"


def is_profitable(metrics: dict, name: str) -> bool:
    """Whether ``metrics`` is profitable under ``name``'s evidence rule."""
    proof = profitability_metric(name)
    growth = value(metrics, proof)
    return math.isfinite(growth) and growth > 0.0


def is_active(metrics: dict) -> bool:
    """True if the book actually held exposure (average gross > ``ACTIVITY_EPS``).

    Guards against the degenerate empty book (e.g. ``demean`` of a constant
    signal -> all-zero weights): every ratio is then either NaN or a vacuous
    0/-inf that some objectives (``avg_turnover``, ``ann_vol``) would happily
    reward. A metrics dict without an ``avg_gross`` entry (a non-backtest slice)
    is treated as inactive, so callers must supply it to assert activity.
    """
    g = metrics.get("avg_gross")
    if g is None:
        return False
    try:
        g = float(g)
    except (TypeError, ValueError):
        return False
    return math.isfinite(g) and g > ACTIVITY_EPS


def names() -> list[str]:
    return list(METRICS)
