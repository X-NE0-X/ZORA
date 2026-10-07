"""In-sample / out-of-sample split and the Pass/Fail verdict.

The one discipline that makes a run mean anything: the signal is backtested once
over the full panel (operators never look forward, so this is safe), then the
strategy returns are sliced into the IS window --- where the model is allowed to
optimise the chosen objective --- and the OOS window, which is scored exactly
once and compared against the Pass Line to decide Pass/Fail.

The OOS window is always constrained to start strictly AFTER the IS window ends,
so no bar can belong to both regardless of how the windows were configured, and
the ``min_oos_days`` floor is counted in bars the book actually held exposure on
--- not calendar bars, which a mostly-flat strategy would clear on zeros.
"""
from __future__ import annotations

from . import objective as _obj
from .backtest import metrics, run_backtest
from .factor.protocol import configured_contract
from .factor.checked_evaluate import ResearchScope
from .factor.evaluate import evaluate


def _slice(bt: dict, mask) -> dict:
    sub = {"strategy_returns": bt["strategy_returns"][mask],
           "turnover": bt["turnover"][mask]}
    if "gross" in bt:
        sub["gross"] = bt["gross"][mask]
    return sub


@configured_contract
def evaluate_windows(formula, panel, config) -> dict:
    """Backtest ``formula`` and return IS/OOS metrics computed on one pass."""
    signal = evaluate(formula, panel, scope=ResearchScope(*config.oos_window(), phase="frozen-OOS"))
    bt = run_backtest(
        signal, panel, config.cost_bps, config.gross, config.initial_cash,
        weight_config=config.backtest_weight_config(),
        close_delisted_at_last=config.close_delisted_at_last,
    )

    is_start, is_end = config.is_window()
    oos_start, oos_end = config.oos_window()
    idx = bt["strategy_returns"].index

    is_mask = (idx >= is_start) & (idx <= is_end)
    # OOS starts strictly after IS ends -> guaranteed no overlap / no leakage.
    oos_mask = (idx > is_end) & (idx >= oos_start) & (idx <= oos_end)

    result = {
        "is_metrics": metrics(_slice(bt, is_mask), config.annualization),
        "oos_metrics": metrics(_slice(bt, oos_mask), config.annualization),
        # the ACTUAL bars scored in each window (exposed so callers/tests can
        # assert disjointness against real output, not a reimplementation).
        "is_index": idx[is_mask],
        "oos_index": idx[oos_mask],
    }
    result["math_admission"] = signal.attrs["zora_admission"]
    return result


def verdict(is_metrics: dict, oos_metrics: dict, config) -> dict:
    """Decide Pass/Fail from IS + OOS metrics. OOS is the arbiter."""
    obj = config.objective
    reasons: list[str] = []

    is_v = _obj.value(is_metrics, obj)
    oos_v = _obj.value(oos_metrics, obj)

    # The floor is on bars the book was actually AT RISK on, not on calendar
    # bars: the engine scores a flat bar as a 0.0 return rather than NaN, so a
    # strategy holding exposure on three days out of sixty would otherwise clear
    # a 20-day floor on 57 vacuous zeros and be judged on three days of evidence.
    # A metrics dict with no `n_active` cannot prove exposure, so it counts as
    # none --- the same stance objective.is_active takes on a missing avg_gross.
    n_is_live = int(is_metrics.get("n_active", 0))
    enough_is = n_is_live >= config.min_is_days
    if not enough_is:
        reasons.append(
            f"insufficient IS observations with live exposure "
            f"({n_is_live} < {config.min_is_days}; {is_metrics['n']} bar(s) scored)"
        )

    n_live = int(oos_metrics.get("n_active", 0))
    enough_oos = n_live >= config.min_oos_days
    if not enough_oos:
        reasons.append(
            f"insufficient OOS observations with live exposure "
            f"({n_live} < {config.min_oos_days}; {oos_metrics['n']} bar(s) scored)"
        )

    # An empty / degenerate book (~zero GROSS exposure) has vacuous metrics that
    # some objectives (turnover, vol) would happily reward. Require the book to
    # actually hold exposure in BOTH windows before any Pass is possible.
    active = _obj.is_active(is_metrics) and _obj.is_active(oos_metrics)
    if not active:
        reasons.append(
            "inactive strategy: average gross exposure is ~0 (empty/degenerate book)"
        )

    meets = _obj.passes(oos_v, config.pass_line, obj)
    if not meets:
        cmp = ">=" if _obj.higher_is_better(obj) else "<="
        reasons.append(
            f"OOS {_obj.label(obj)} {oos_v:.4f} fails Pass Line "
            f"({cmp} {config.pass_line:.4f})"
        )

    sign_ok = True
    if config.require_sign_consistency:
        proof = _obj.profitability_metric(obj)
        is_profit_v = _obj.value(is_metrics, proof)
        oos_profit_v = _obj.value(oos_metrics, proof)
        sign_ok = (_obj.is_profitable(is_metrics, obj)
                   and _obj.is_profitable(oos_metrics, obj))
        if not sign_ok:
            reasons.append(
                f"IS/OOS not consistently profitable on {_obj.label(proof)} "
                f"(IS {is_profit_v:.4f}, OOS {oos_profit_v:.4f}); "
                f"{_obj.label(obj)} remains the optimisation/Pass-Line objective"
            )

    passed = bool(enough_is and enough_oos and active and meets and sign_ok)
    return {
        "passed": passed,
        "label": "PASS" if passed else "FAIL",
        "objective": obj,
        "pass_line": config.pass_line,
        "is_value": None if is_v != is_v else is_v,     # None if NaN
        "oos_value": None if oos_v != oos_v else oos_v,
        "profitability_metric": _obj.profitability_metric(obj),
        "reasons": reasons,
    }
