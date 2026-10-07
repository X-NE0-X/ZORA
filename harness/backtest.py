"""Signal -> positions -> returns -> risk metrics, executed by BacktestEngine (vbt).

The harness turns a factor signal into a cross-sectional, dollar-neutral
signal-strength panel (``signal_to_positions``), hands that to BacktestEngine's own
``Position.compute_weight_panel`` to build the executable
target-weight book (the infra's signal->position stage), then replays it through
the BacktestEngine vectorbt portfolio for NAV; performance metrics come from BacktestEngine
and are normalised to the objective registry. Factor -> position -> replay ->
metrics is the vendored infra's pipeline, reused stage-by-stage.

No-look-ahead lives in two places: operators can't peek forward inside the
signal (see factor/operators.py), and BacktestEngine executes weight row ``t`` at that
bar's OPEN -- so we lag the weight book by one bar here (``shift(1)``), i.e. the
target for day ``t`` is decided from information available at close ``t-1``.
"""
from __future__ import annotations

import pandas as pd

from . import infra_engine
from .factor import operators


def signal_to_positions(signal: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Dollar-neutral cross-sectional signal strength: demean then scale to sum|w|=gross.

    This is the *strength* the harness feeds to the infra's weight constructor;
    it is also a valid (unregularised) weight book on its own, which the tests
    use as the un-lagged "cheat" comparison. :func:`run_backtest` calls it at
    ``gross=1.0`` on purpose --- the run's gross budget is applied downstream, by
    the engine, exactly once (see there).
    """
    return operators.scale(operators.demean(signal), gross)


def run_backtest(
    signal: pd.DataFrame,
    panel,
    cost_bps: float = 3.0,
    gross: float = 1.0,
    initial_cash: float = infra_engine.DEFAULT_INITIAL_CASH,
    weight_config: dict | None = None,
    close_delisted_at_last: bool = False,
) -> dict:
    """Backtest a signal on a panel via BacktestEngine's full signal->position->replay.

    The cross-sectional strength is turned into an executable weight book by
    BacktestEngine's ``Position`` (``infra_engine.compute_weight_panel``),
    lagged one bar for no-look-ahead, then replayed through the vbt portfolio.
    Returns aligned ``strategy_returns`` (daily net portfolio returns), the
    ``turnover`` series, the ``nav`` path, the lagged ``positions`` book and the
    raw BacktestEngine ``metric_map``.

    **The gross budget is applied exactly once, and the engine owns it.**
    ``Position`` already multiplies the strength panel by ``gross_target`` and
    caps each row there, so what we hand it is normalised to *unit* gross, not to
    ``gross``: scaling in both layers ran a gross-squared book (executed
    sum|w| = gross**2 -- invisible at the default gross=1.0 because 1**2 == 1,
    but a quarter-sized book at gross=0.5, since the engine's row cap only ever
    scales an over-budget row DOWN). The same ``gross`` is then handed to the vbt
    portfolio as its ``max_gross_exposure`` ceiling, so the book the engine
    executes is the book the ``turnover`` / ``gross`` series below report; with
    the ceiling pinned at 1.0 the engine clamped every over-budget book back to
    1.0 while the report still described the un-clamped panel, i.e. raising
    ``gross`` moved ``avg_turnover`` and ``avg_gross`` without moving a trade.
    """
    dates = panel.dates
    from .factor.checked_evaluate import require_validated_signal
    require_validated_signal(signal)
    # Unit-gross strength: the ``gross`` budget belongs to Position, below.
    strength = signal_to_positions(signal, 1.0)
    strength = strength.reindex(index=dates, columns=panel.symbols)

    # infra signal->position stage: Position builds the target-weight
    # book (gross cap, warm-up, hold/rebalance, event masks, calendar alignment).
    weights = infra_engine.compute_weight_panel(
        panel.test_data, strength, gross=gross, weight_config=weight_config,
    )
    weights = weights.reindex(index=dates, columns=panel.symbols)

    # BacktestEngine executes row t at OPEN[t]; lag one bar so the target for t uses
    # only information available at close t-1 (no look-ahead).
    weight_panel = weights.shift(1).fillna(0.0)

    res = infra_engine.run_vbt(
        panel.test_data, weight_panel, cost_bps=cost_bps, initial_cash=initial_cash,
        max_gross_exposure=gross,
        close_delisted_at_last=close_delisted_at_last,
    )

    strat = pd.Series(res["returns"]).reindex(dates)
    executed = pd.DataFrame(res["executed_weight_panel"]).reindex(
        index=dates, columns=panel.symbols,
    ).fillna(0.0)
    # Turnover is filled order notional / portfolio value, calculated inside the
    # engine. Target delta misses rebalance orders and invents trades that were
    # rejected, so it is not an execution statistic.
    turnover = pd.Series(res["turnover"]).reindex(dates).fillna(0.0)
    # Per-bar gross exposure of the realized open-position book. This is the
    # authority for avg_gross / n_active and therefore for the OOS evidence gate.
    gross_exposure = executed.abs().sum(axis=1, min_count=1)

    return {
        "strategy_returns": strat,
        "turnover": turnover,
        "gross": gross_exposure,
        "nav": res["nav"],
        "positions": executed,
        "target_positions": weight_panel,
        "metric_map": res.get("metric_map", {}),
    }


def metrics(bt: dict, ann: int = 252) -> dict:
    """Objective metrics from a run_backtest result (or a windowed slice of one).

    ``ann`` pins the annualisation (periods per year) so IS/OOS slices are scored
    on the trading-day clock the returns live on, not BacktestEngine's
    365-calendar-day default. A ``gross`` series (if present) feeds the
    ``avg_gross`` activity measure and the ``n_active`` live-exposure bar count.
    """
    return infra_engine.window_metrics(
        bt["strategy_returns"], bt.get("turnover"), bt.get("gross"),
        periods_per_year=ann,
    )
