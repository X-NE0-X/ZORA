"""Bridge to the vendored engine (FactorEngine factors, BacktestEngine vbt backtest).

The harness invents a *free-form* factor with its own safe-AST evaluator; this
module is the seam where that work meets the user's own infrastructure:

  * **FactorEngine** is the factor-computation engine. The harness's free-form
    signal is registered into a ``FactorManager`` as a *custom* factor cube, so
    FactorEngine owns the factor cube / cache / manifest (lineage). Only the
    compute + custom-factor path is used --- the predefined indicator library is
    intentionally not exposed to the model.
  * **BacktestEngine** is the backtest engine. A signal-strength panel is turned into
    an executable target-weight book by ``Position.compute_weight_panel`` (the
    infra's own signal->position stage), then executed through its vectorbt
    portfolio (``PortfolioEngine.portfolio_inputs`` -> ``portfolio_bt``); NAV and the
    performance metric map come straight from BacktestEngine.
  * **CTX** cleans/loads data (wired in ``data.py``).

Every infra import is lazy and goes through :mod:`harness._infra` bootstrap, so
the pure DSL/objective layers stay import-clean when the infra isn't needed.
Metric names/units are normalised to the harness ``objective`` registry here, in
one place, so the rest of the code never sees BacktestEngine's raw field names.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

from . import _infra
from . import objective as _obj

DEFAULT_INITIAL_CASH = 1_000_000.0


class _PortfolioValuationWarningFilter(logging.Filter):
    """Drop repeated sparse-panel valuation noise before it corrupts TUI redraws."""

    _zora_portfolio_warning_filter = True
    _prefix = "[WARNING] Portfolio valuation price panel contains "

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._prefix)


def _install_logging_filters() -> None:
    root = logging.getLogger()
    if not any(getattr(f, "_zora_portfolio_warning_filter", False)
               for f in root.filters):
        root.addFilter(_PortfolioValuationWarningFilter())


_install_logging_filters()


class InfraUnavailable(RuntimeError):
    """Raised when the vendored engine packages cannot be located/imported."""


def available() -> bool:
    """True if the vendored infra packages appear to be present."""
    return _infra.infra_available()


_INFRA_IMPORTED = False


def _ensure() -> None:
    if not _infra.infra_available():
        raise InfraUnavailable(
            "vendored infra (FactorEngine/BacktestEngine/CTX) not found under "
            "the harness project root; cannot run the vbt backtest / factor engine"
        )
    _infra.ensure_infra()
    # The packages are present on disk, but a runtime dependency of the vendored
    # stack (talib / numba / duckdb / vectorbt / ...) may be missing. Probe-import
    # the three engines once and turn any ImportError into a graceful
    # InfraUnavailable, so the caller gets a clean diagnostic instead of a raw
    # ModuleNotFoundError bubbling out of the middle of a backtest.
    global _INFRA_IMPORTED
    if not _INFRA_IMPORTED:
        try:
            import CTX          # noqa: F401
            import FactorEngine  # noqa: F401
            import BacktestEngine  # noqa: F401
        except ImportError as exc:
            raise InfraUnavailable(
                "vendored infra is present but a runtime dependency is missing "
                f"({exc}); install the infra extras (pip install '.[infra]')"
            ) from exc
        _INFRA_IMPORTED = True


# --- FactorEngine: factor-computation engine ----------------------------------
def build_manager(test_data: pd.DataFrame, *, attach: bool = True):
    """Build (and optionally attach) a FactorEngine ``FactorManager`` for a panel."""
    _ensure()
    from FactorEngine import FactorManager

    return FactorManager.from_test_data(test_data, attach=attach, memory_cache=True)


def get_manager(test_data: pd.DataFrame):
    """Return the FactorManager attached to ``test_data``, building one if absent."""
    _ensure()
    from FactorEngine import FactorManager

    mgr = FactorManager.get_attached_store(test_data)
    return mgr if mgr is not None else build_manager(test_data)


def register_signal(manager, feature_id: str, signal: pd.DataFrame) -> pd.DataFrame:
    """Register a materialised T x N factor panel; return the aligned cube panel.

    The signal is force-aligned to the manager's calendar/asset axes and stored
    as a ``CubeArtifact`` (so it shows up in ``manager.manifest()``); the value
    read back from ``get_cube`` is the canonical signal used downstream.
    """
    from .factor.checked_evaluate import require_validated_signal
    require_validated_signal(signal)
    sig = signal.reindex(index=manager.calendar, columns=manager.asset_keys)
    manager.register(
        feature_id, sig, semantic_name=feature_id, timing_semantics="t_close"
    )
    cube = np.asarray(manager.get_cube(feature_id))  # (P=1, T, N)
    return pd.DataFrame(cube[0], index=manager.calendar, columns=manager.asset_keys)


# --- BacktestEngine: signal-strength -> target weights (Position) ----------------
def compute_weight_panel(
    test_data: pd.DataFrame,
    signal_strength: pd.DataFrame,
    *,
    gross: float = 1.0,
    weight_config: dict | None = None,
) -> pd.DataFrame:
    """Turn a signal-strength panel into an executable target-weight book.

    Reuses BacktestEngine's ``Position.compute_weight_panel`` --- the infra's own
    signal->position stage --- so weight construction (per-row gross cap, warm-up
    zeroing, hold/rebalance smoothing, event-risk masks, calendar alignment) is
    the vendored engine's, not hand-rolled here.

    ``signal_strength`` is expected to be an already cross-sectional,
    dollar-neutral strength panel (T x N), so the default ``raw_strength`` sizing
    just gross-caps it at ``gross`` per bar. The returned panel is *not* lagged;
    the caller applies the one-bar execution lag (weight[t] uses info <= close
    t-1) before handing it to :func:`run_vbt`.

    ``weight_config`` keys are read straight out of the vendored engine with
    ``.get(key, default)``, so a *misspelt* key is silently ignored rather than
    rejected. Note the hold/rebalance knob is ``hold_every`` (it was ``hold_days``
    in older engine builds); passing the old name changes nothing and falls back
    to every-bar rebalancing without any warning.
    """
    _ensure()
    from BacktestEngine.Position import Position

    cfg = {"weight_mode": "raw_strength", "selection_mode": "none",
           "gross_target": float(gross)}
    if weight_config:
        cfg.update(weight_config)
    if str(cfg.get("selection_mode", "none")).strip().lower() == "top_k":
        # BacktestEngine uses separate long/short K controls. Harness exposes
        # one symmetric Top K so UI/config behavior stays unambiguous.
        k = int(cfg.get("top_k", 0))
        cfg.setdefault("long_k", k)
        cfg.setdefault("short_k", k)
    sr = Position()
    panel = sr.compute_weight_panel(
        test_data=test_data, signal_strength_panel=signal_strength, weight_config=cfg
    )
    return pd.DataFrame(panel)


# --- BacktestEngine: vbt backtest engine -----------------------------------------
def run_vbt(
    test_data: pd.DataFrame,
    weight_panel: pd.DataFrame,
    *,
    cost_bps: float = 3.0,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    max_gross_exposure: float = 1.0,
    close_delisted_at_last: bool = False,
) -> dict:
    """Execute a target-weight panel through BacktestEngine's vectorbt portfolio.

    ``weight_panel`` rows are executed at each bar's OPEN, so the caller must
    already have lagged the weights (weight[t] uses info <= t-1) to avoid
    look-ahead. ``cost_bps`` maps to vbt slippage (BacktestEngine models slippage, not
    commission). Returns BacktestEngine's raw result dict (``nav`` / ``returns`` /
    ``metric_map`` / ...).

    ``max_gross_exposure`` is the engine's per-row sum|w| ceiling for the
    cash-shared ``targetpercent`` book: over-budget rows are scaled down before
    execution. It is a *parameter*, not a pin, because the caller's gross budget
    has to reach the portfolio -- with it hardcoded to 1.0 the engine silently
    clamped every leveraged book back to 1.0 while the harness went on reporting
    turnover / gross exposure from the un-clamped panel, so the ``gross`` knob
    changed the report and not one trade.
    """
    _ensure()
    from BacktestEngine.PortfolioEngine import PortfolioEngine
    from BacktestEngine.VBTEngine import BacktestEngine_VBT

    engine = BacktestEngine_VBT(
        test_data,
        initial_cash=float(initial_cash),
        slippage=float(cost_bps) / 1e4,
        spread=0.0,
        signal_start_t=2,
    )
    inp = PortfolioEngine.portfolio_inputs(test_data, weight_panel, cal_column="Close")
    return PortfolioEngine.portfolio_bt(
        btengine=engine,
        weight_panel=inp["weight_panel"],
        open_panel=inp["open_panel"],
        close_panel=inp["close_panel"],
        force_flat_mask=inp["force_flat_mask"],
        prevent_open_mask=inp["prevent_open_mask"],
        max_gross_exposure=float(max_gross_exposure),
        # pinned, not defaulted: both are engine defaults today, but an upstream
        # default flip would silently move NAV (forced end-of-life flattening) and
        # every Sharpe/Sortino (the excess-return base), breaking replay in silence.
        risk_free_rate=0.02,
        close_delisted_at_last=bool(close_delisted_at_last),
    )


def performance(nav_series: pd.Series, *, periods_per_year: float | None = None) -> dict:
    """BacktestEngine performance metrics for a NAV series (raw field names).

    ``periods_per_year`` pins the annualisation factor (e.g. 252 trading days);
    when ``None`` the engine infers it from the calendar spacing of the NAV
    index (its historical behaviour). The harness threads
    ``config.annualization`` through here so IS/OOS slices are annualised on the
    trading-day clock the returns actually live on, not the 365-calendar-day
    default.
    """
    _ensure()
    from BacktestEngine.Manager import BacktestEngineManager

    return BacktestEngineManager.performance_metrics(
        nav_series=nav_series, periods_per_year=periods_per_year
    )


def nav_from_returns(
    returns: pd.Series, *, initial_cash: float = DEFAULT_INITIAL_CASH
) -> pd.Series:
    """Compound a return series into a NAV path, preserving the FIRST return.

    A leading baseline point (starting capital, one step before the first
    return) is prepended so that the engine's internal ``nav.pct_change()``
    recovers all ``T`` returns instead of silently dropping ``returns[0]``. The
    step size is inferred from the index spacing (falling back to one unit) so
    the prepended timestamp stays on the series' own clock.
    """
    r = pd.Series(returns).dropna()
    nav_after = (1.0 + r).cumprod() * float(initial_cash)
    if r.shape[0] == 0:
        return nav_after
    i0 = r.index[0]
    if r.shape[0] > 1:
        step = r.index[1] - i0
    else:
        step = pd.Timedelta(days=1) if isinstance(i0, pd.Timestamp) else 1
    base = pd.Series([float(initial_cash)], index=[i0 - step])
    return pd.concat([base, nav_after])


# --- metric adapter: BacktestEngine field names/units -> objective registry ------
def _num(mapping: dict, key: str) -> float:
    v = mapping.get(key)
    return float("nan") if v is None else float(v)


def _resolve_no_downside(out: dict, metric_map: dict) -> None:
    """In place: rescue the ratios that are NaN only because there was NO downside.

    BacktestEngine leaves ``sortino_ratio`` NaN when the downside deviation of the
    excess returns is 0, ``calmar_ratio`` NaN when max drawdown is 0, and
    ``sharpe_ratio`` NaN when the excess-return volatility is 0 --- in every case
    it is the *denominator* that vanished, not the book that broke.
    Left as NaN they are indistinguishable from an uncomputable metric, and
    :func:`harness.objective.goodness` maps NaN to ``-inf`` "so an uncomputable
    metric never wins": the one window without a single losing bar would then
    rank below every losing window and auto-FAIL the Pass Line.

    The two NaNs are separated here, using only fields BacktestEngine itself
    reports (no re-derivation of its formulas):

      * the downside/drawdown/volatility is identically zero
        (``yield_downside_deviation`` / ``maxdd_rate`` / ``yield_volatility``
        == 0), the window actually made money
        (``annualized_return_rate`` > 0) and the book held exposure
        (:func:`harness.objective.is_active`) -> the ratio's limit is ``+inf``,
        which is exactly what "no downside at all" is worth. It wins comparisons
        and clears any finite Pass Line, as it should.
      * anything else --- empty/degenerate book, fewer than two bars, invalid NAV
        --- stays NaN, i.e. still the worst possible score.

    The activity gate is what keeps that honest. Against the pinned 2% risk-free
    a flat all-zero book has excess returns of ``-rf`` every bar, so its downside
    is non-zero and this branch is unreachable for it today; the moment the
    risk-free rate is 0 it *would* be reachable, and promoting the empty book to
    ``+inf`` would hand the optimiser a perfect score for holding nothing.
    """
    if not _obj.is_active(out):
        return
    # NaN-safe: a NaN annualised return fails this test, as it must.
    if not (_num(metric_map, "annualized_return_rate") > 0.0):
        return
    if (math.isnan(out["sortino"])
            and _num(metric_map, "yield_downside_deviation") == 0.0):
        out["sortino"] = math.inf
    if math.isnan(out["calmar"]) and _num(metric_map, "maxdd_rate") == 0.0:
        out["calmar"] = math.inf
    # Sharpe for the same reason. A literally zero-variance return stream is a
    # rarer degeneracy than "no losing bar" --- it needs EVERY bar identical ---
    # but sharpe is a first-class objective in the registry, so leaving it out
    # would keep exactly the bug this function exists to fix, just on a metric
    # nobody happened to test.
    if math.isnan(out["sharpe"]) and _num(metric_map, "yield_volatility") == 0.0:
        out["sharpe"] = math.inf


def adapt_metrics(
    metric_map: dict,
    *,
    returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    gross: pd.Series | None = None,
) -> dict:
    """Translate a BacktestEngine ``metric_map`` into the harness objective keys.

    BacktestEngine reports ratios directly, but returns/drawdown as *percent* and
    max-drawdown as a positive magnitude. We normalise to the registry's
    conventions (fractions; max-drawdown a negative fraction) and fill the four
    metrics BacktestEngine doesn't produce (daily hit-rate, average turnover,
    average gross exposure, live-exposure bar count).

    ``gross`` (per-bar sum|w| of the *executed* book) is filled here rather than
    patched in by the caller, so ``avg_gross`` --- the sole input to
    :func:`harness.objective.is_active` --- has exactly one owner; a metrics dict
    built without it keeps ``avg_gross`` NaN and therefore reads as inactive.

    ``n_active`` counts the bars on which the book actually held exposure,
    alongside ``n`` (bars in the window). The engine forces an undefined bar's
    return to 0.0, so ``n`` alone cannot tell a fully-invested window from one
    where the strategy sat flat through most of it.
    """
    ann_ret_pct = _num(metric_map, "annualized_return_rate")
    maxdd_pct = _num(metric_map, "maxdd_rate")
    out = {
        "sortino": _num(metric_map, "sortino_ratio"),
        "sharpe": _num(metric_map, "sharpe_ratio"),
        "calmar": _num(metric_map, "calmar_ratio"),
        "cagr": ann_ret_pct / 100.0,
        "ann_return": ann_ret_pct / 100.0,
        "maxdd": (float("nan") if math.isnan(maxdd_pct) else -abs(maxdd_pct) / 100.0),
        "ann_vol": _num(metric_map, "yield_volatility"),
        "hit_rate": float("nan"),
        "avg_turnover": float("nan"),
        "avg_gross": float("nan"),
        "n": 0,
        "n_active": 0,
    }
    if returns is not None:
        r = pd.Series(returns).dropna()
        out["n"] = int(r.shape[0])
        if r.shape[0]:
            out["hit_rate"] = float((r > 0).mean())
    if turnover is not None:
        tu = pd.Series(turnover).dropna()
        out["avg_turnover"] = float(tu.mean()) if tu.shape[0] else float("nan")
    if gross is not None:
        g = pd.Series(gross).dropna().abs()
        if g.shape[0]:
            out["avg_gross"] = float(g.mean())
            # same epsilon as the activity gate, so "the book was alive on this
            # bar" means the same thing per-bar as it does on the window average.
            out["n_active"] = int((g > _obj.ACTIVITY_EPS).sum())
    # must run last: it consults avg_gross, so the dict has to be complete first.
    _resolve_no_downside(out, metric_map)
    return out


def window_metrics(
    returns: pd.Series,
    turnover: pd.Series | None = None,
    gross: pd.Series | None = None,
    *,
    initial_cash: float = DEFAULT_INITIAL_CASH,
    periods_per_year: float | None = None,
) -> dict:
    """Objective metrics for a (windowed) daily-return series via BacktestEngine.

    Rebuilds a NAV path from the returns (preserving the first return, see
    :func:`nav_from_returns`) and runs it back through BacktestEngine's metric
    engine, so in-sample and out-of-sample slices are scored with exactly the
    same formulas as the full backtest. ``periods_per_year`` pins annualisation
    (trading-day clock); ``gross`` (per-bar sum|w| of the executed book) feeds
    both the ``avg_gross`` activity measure used to reject degenerate empty books
    and the ``n_active`` live-exposure bar count (see :func:`adapt_metrics`).
    """
    r = pd.Series(returns).dropna()
    if r.shape[0] < 2:
        metric_map: dict = {}
    else:
        nav = nav_from_returns(r, initial_cash=initial_cash)
        metric_map = performance(nav, periods_per_year=periods_per_year)
    return adapt_metrics(metric_map, returns=r, turnover=turnover, gross=gross)
