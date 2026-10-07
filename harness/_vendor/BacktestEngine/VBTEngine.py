from ENV_MGMT.imports import *
import FactorEngine







# VBT Back-test Engine Class
#----------------------------------------------------------------------------------------
class BacktestEngine_VBT:
    """
    ### What It Does
    Executes a vectorbt-backed backtest from aligned price data and target weight series.

    #### Responsibility
    Keeps vectorbt portfolio construction, order sizing, NAV extraction, trade-account reporting, and execution assumptions behind one reusable engine object.

    #### How To Use
    Instantiate it with a working price frame and cash baseline, then call `vectorbt(...)` directly or let `BacktestEngineManager.vbt_testcycle(...)` resolve weights and dispatch it.

    #### Usage Example
    `obj = BacktestEngine_VBT(...)`

    ---

    ### Parameters
    - `test_data`: **pd.DataFrame**.
    - `initial_cash`: **float**.

    #### Optional Parameters
    - `slippage`: **float** = *0.0*.
    - `spread`: **float** = *0.0*.
    - `fees`: **float** = *0.0*.
    - `signal_start_t`: **int** = *2*.
    """

    def __init__(self,
                 test_data: pd.DataFrame,
                 initial_cash: float,

                 slippage: float = 0.0,
                 spread: float = 0.0,
                 fees: float = 0.0,
                 signal_start_t: int = 2) -> None:

        self.test_data = test_data
        self.initial_cash = initial_cash
        self.slippage = float(slippage)
        self.spread = float(spread)
        self.fees = float(fees)
        self.benchmark_series_override: Optional[pd.Series] = None
        self.event_price_panel: Optional[pd.DataFrame] = None
        self.event_weight_panel: Optional[pd.DataFrame] = None

        self.signal_start_t = signal_start_t

        self.reset()


    def reset(self) -> None:
        """
        ### What It Does
        Resets vectorbt-engine state for another run.

        #### Responsibility
        Clears portfolio, NAV, trade, and diagnostic fields before rebuilding a vectorbt portfolio.

        #### How To Use
        Call it before reusing a `BacktestEngine_VBT` instance.

        #### Usage Example
        `result = reset(...)`
        """

        nav_idx_source = self.test_data["Datetime"] if "Datetime" in self.test_data.columns else self.test_data.index
        nav_index = pd.to_datetime(nav_idx_source, errors = "coerce")

        self.nav = []
        self.nav_line = pd.DataFrame(
                                     {
                                        'NAV': np.nan,
                                        'Yield': np.nan,
                                        'BM_Buy_n_Hold': np.nan,
                                     }, index = nav_index
                                    )

        self.holding_period_h_mean = None
        self.holding_period_h_median = None
        self.holding_period_h_max = None
        self.holding_period_h_min = None
        self.holding_period_d_mean = None
        self.holding_period_d_median = None
        self.holding_period_d_max = None
        self.holding_period_d_min = None

        self.nav_total = None
        self.nav_return = None
        self.nav_rate = None
        self.nav_total_bm = None
        self.nav_total_bm_return = None
        self.nav_total_bm_rate = None
        self.rel_return = None
        self.rel_return_rate = None
        self.annualized_return_rate = None
        self.win_rate = None
        self.calmar_ratio = None
        self.sharpe_ratio = None
        self.sortino_ratio = None
        self.yield_volatility = None
        self.yield_downside_deviation = None
        self.var95 = None
        self.cvar95 = None
        self.var99 = None
        self.cvar99 = None
        self.pl_ratio = None
        self.maxdd_total = None
        self.maxdd_rate = None
        self.maxdd_period = None

        self.cover_trades = 0
        self.open_trades = 0
        self.win_trades = 0
        self.max_profit_trade = 0
        self.min_profit_trade = 0
        self.max_dd_trade = 0
        self.min_dd_trade = 0

        self.vbt_portfolio: Optional[Any] = None


    def TradeAccount(self,
                     portfolio: Optional[Any] = None,
                     position_panel: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        ### What It Does
        Summarizes vectorbt trade records into the engine's trade metrics.

        #### Responsibility
        Normalizes trade counts, P/L, holding periods, and drawdown per trade for reports.

        #### How To Use
        Call it after `vectorbt(...)` has produced a portfolio object.

        #### Key Parameters In Practice
        - `portfolio`
          - Portfolio object or result payload. Use it as the source of trade records, NAV, orders, and final performance extraction.
          - Expected shape/type: `Optional[Any]`.
        - `position_panel`
          - Aligned panel input. Keep row timestamps and asset columns synchronized with the price/calendar contract before calling this function.
          - Expected shape/type: `Optional[pd.DataFrame]`.

        #### Usage Example
        `result = TradeAccount(...)`

        ---

        #### Optional Parameters
        - `portfolio`: **Optional[Any]** = *None*.
        - `position_panel`: **Optional[pd.DataFrame]** = *None*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        from .PortfolioEngine import PortfolioEngine  # deferred import: breaks circular dependency
        trade_stats = PortfolioEngine.TradeAccount(portfolio = self.vbt_portfolio if portfolio is None else portfolio, position_panel = position_panel)

        self.open_trades = trade_stats["open_trades"]
        self.cover_trades = trade_stats["cover_trades"]
        self.win_trades = trade_stats["win_trades"]
        self.win_rate = trade_stats["win_rate"]
        self.pl_ratio = trade_stats["pl_ratio"]
        self.holding_period_h_mean = trade_stats["holding_period_h_mean"]
        self.holding_period_h_median = trade_stats["holding_period_h_median"]
        self.holding_period_h_max = trade_stats["holding_period_h_max"]
        self.holding_period_h_min = trade_stats["holding_period_h_min"]
        self.holding_period_d_mean = trade_stats["holding_period_d_mean"]
        self.holding_period_d_median = trade_stats["holding_period_d_median"]
        self.holding_period_d_max = trade_stats["holding_period_d_max"]
        self.holding_period_d_min = trade_stats["holding_period_d_min"]
        self.max_profit_trade = trade_stats["max_profit_trade"]
        self.min_profit_trade = trade_stats["min_profit_trade"]
        self.max_dd_trade = trade_stats["max_dd_trade"]
        self.min_dd_trade = trade_stats["min_dd_trade"]

        return trade_stats


    def vectorbt(self,
                test_data: pd.DataFrame,
                *,
                weight_series: pd.Series,
                cal_column: str = "Close") -> Tuple[pd.Series, Any]:
        """
        ### What It Does
        Executes a vectorbt backtest from prices and target weights.

        #### Responsibility
        Translates engine inputs into vectorbt portfolio construction and records execution output.

        #### How To Use
        Call it with aligned test data and weight inputs for VBT-mode execution.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `weight_series`
          - Single-asset target-weight series. Its index should align to the price series used by the selected backtest backend.
          - Expected shape/type: `pd.Series`.
        - `cal_column`
          - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = vectorbt(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `weight_series`: **pd.Series**.

        #### Optional Parameters
        - `cal_column`: **str** = *"Close"*.

        ---

        ### Returns
        - `result`: **Tuple[pd.Series, Any]**.
        """

        from .Manager import BacktestEngineManager  # deferred import: breaks circular dependency
        if vbt is None:

            raise ImportError("[WARNING] vectorbt is unavailable; VBT backtest path cannot run.")

        price_column = "Open" if "Open" in test_data.columns else cal_column

        time_index = BacktestEngineManager._time_index(test_data)
        close_series = pd.Series(pd.to_numeric(test_data[cal_column], errors = "coerce").to_numpy(dtype = float, copy = False), index = time_index, dtype = float)
        price_series = pd.Series(pd.to_numeric(test_data[price_column], errors = "coerce").to_numpy(dtype = float, copy = False), index = time_index, dtype = float)
        weight_series = BacktestEngineManager._align_series_to_test_data(weight_series, test_data).fillna(0.0).astype(float)
        resolved_freq = BacktestEngineManager._infer_vbt_freq(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index)
        weight_series.index = time_index

        # signal_start_t is a warmup cutoff: the NAV over [0, start_t) is blanked below. Zero the warmup
        # weights too so VBT cannot open positions during warmup on caller-injected non-zero weights. The
        # normal Signal path already zeros them; this makes the engine self-defending and
        # keeps executed trades consistent with the blanked NAV window.
        warmup_cutoff = max(0, int(self.signal_start_t))

        if warmup_cutoff > 0:
            weight_series.iloc[: min(warmup_cutoff, len(weight_series))] = 0.0

        portfolio = vbt.Portfolio.from_orders(
                                                close = close_series,
                                                size = weight_series,
                                                size_type = "targetpercent",
                                                direction = "both",
                                                price = price_series,
                                                val_price = price_series,
                                                init_cash = float(self.initial_cash),
                                                cash_sharing = False,
                                                slippage = float(self.slippage) + (float(self.spread) / 2.0),
                                                fees = float(self.fees),
                                                freq = resolved_freq,
                                             )

        nav_series = portfolio.value()

        nav_series = pd.to_numeric(pd.Series(nav_series, index = time_index), errors = "coerce")
        nav_series.iloc[: max(0, int(self.signal_start_t))] = np.nan

        self.test_data = test_data.copy()

        self.test_data["weights"] = weight_series.to_numpy()
        self.nav_line = pd.DataFrame({"NAV": nav_series.values}, index = test_data.index)
        self.nav = float(nav_series.dropna().iloc[-1]) if not nav_series.dropna().empty else np.nan
        self.vbt_portfolio = portfolio

        self.TradeAccount(portfolio = portfolio)


        return nav_series, portfolio
