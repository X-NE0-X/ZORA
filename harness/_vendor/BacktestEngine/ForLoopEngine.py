from ENV_MGMT.imports import *







# For-Loop Back-test Engine Class
#----------------------------------------------------------------------------------------
class BacktestEngine_ForLoop:
    """
    ### What It Does
    Executes a bar-by-bar backtest with explicit cash, position, trade, NAV, and holiday/timeout handling.

    #### Responsibility
    Maintains the mutable account state used by for-loop execution, including entry/exit accounting, slippage/spread application, forced covers, holding-period statistics, and NAV/yield reporting.

    #### How To Use
    Instantiate it with one test frame and starting cash, call `reset()` before each run, then drive positions through `entry(...)`, `exit(...)`, or `BacktestEngineManager.testcycle(...)`.

    #### Usage Example
    `obj = BacktestEngine_ForLoop(...)`

    ---

    ### Parameters
    - `test_data`: **pd.DataFrame**.
    - `initial_cash`: **float**.

    #### Optional Parameters
    - `risk_free_rate`: **float** = *0.02*.
    - `holidays`: **Optional[List[datetime.date]]** = *None*.
    - `slippage`: **float** = *0.0*.
    - `spread`: **float** = *0.0*.
    - `fees`: **float** = *0.0*.
    - `benchmark_series`: **Optional[pd.Series]** = *None*.
    - `signal_start_t`: **int** = *2*.
    """

    # BacktestEngine Define
    @staticmethod
    def nav_and_yield(t: int,
                      cash: float,
                      amt: float,
                      test_data: pd.DataFrame,
                      nav: List[float],
                      nav_line: pd.DataFrame,
                      cal_column: str = "Close") -> Tuple[List[float], pd.DataFrame]:
        """
        ### What It Does
        Computes NAV_t = cash + amt * Close_t and updates per-bar yield from previous NAV.

        #### Responsibility
        Maintains valuation timeline consumed by all downstream performance metrics.

        #### How To Use
        Call once per bar after all trade actions are applied.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `cash`
          - Available cash state. It must remain consistent with position amount and NAV updates.
          - Expected shape/type: `float`.
        - `amt`
          - Position amount. Positive/negative sign encodes exposure direction and is updated on entry/exit.
          - Expected shape/type: `float`.
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `nav`
          - Net asset value series or scalar state. It is the account equity basis for returns, drawdown, and metrics.
          - Expected shape/type: `List[float]`.
        - `nav_line`
          - NAV series for one candidate or replay. It should be indexed to the evaluation calendar before metrics are computed.
          - Expected shape/type: `pd.DataFrame`.
        - `cal_column`
          - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = nav_and_yield(...)`

        ---

        ### Parameters
        - `t`: **int**.
        - `cash`: **float**.
        - `amt`: **float**.
        - `test_data`: **pd.DataFrame**.
        - `nav`: **List[float]**.
        - `nav_line`: **pd.DataFrame**.

        #### Optional Parameters
        - `cal_column`: **str** = *"Close"*.

        ---

        ### Returns
        - `result`: **Tuple[List[float], pd.DataFrame]**.
        """

        resolved_cal_column = str(cal_column).strip() if isinstance(cal_column, str) else "Close"

        if not resolved_cal_column:
            resolved_cal_column = "Close"

        price_column = resolved_cal_column

        if price_column not in test_data.columns:

            if "Close" in test_data.columns:
                price_column = "Close"

            else:

                raise KeyError(
                    f"[WARNING] price column '{resolved_cal_column}' not found and fallback 'Close' is unavailable."
                )

        current_nav = cash + amt * test_data[price_column].iloc[t]

        if 'Datetime' in test_data.columns:
            current_datetime = test_data['Datetime'].iloc[t]

        else:
            current_datetime = test_data.index[t]

        nav.append(current_nav)

        nav_line.loc[current_datetime, 'NAV'] = current_nav

        if len(nav) >= 2:
            current_yield = (nav[-1] - nav[-2]) / nav[-2]
            nav_line.loc[current_datetime, 'Yield'] = current_yield


        return nav, nav_line


    @staticmethod
    def max_drawdown(nav_series: pd.Series) -> Tuple[float, float, float]:
        """
        ### What It Does
        Calculates maximum drawdown statistics for a NAV series.

        #### Responsibility
        Finds peak-to-trough loss and duration using the for-loop engine's reporting conventions.

        #### How To Use
        Call it with a NAV or equity curve after a test run.

        #### Key Parameters In Practice
        - `nav_series`
          - NAV time series used for metric calculation. Ensure it is numeric and ordered by time.
          - Expected shape/type: `pd.Series`.

        #### Usage Example
        `result = max_drawdown(...)`

        ---

        ### Parameters
        - `nav_series`: **pd.Series**.

        ---

        ### Returns
        - `result`: **Tuple[float, float, float]**.
        """

        nav = pd.to_numeric(pd.Series(nav_series).copy(), errors = "coerce").dropna()

        if not nav.empty:

            try:
                nav.index = pd.to_datetime(nav.index, errors = "coerce")

            except Exception:
                pass

        if nav.empty:

            return np.nan, np.nan, np.nan

        arr = np.asarray(nav.values, dtype = float)
        roll_max = np.maximum.accumulate(arr)

        with np.errstate(divide = "ignore", invalid = "ignore"):
            drawdown_pct = (roll_max - arr) / roll_max

        valid_idx = np.where(np.isfinite(drawdown_pct))[0]

        if len(valid_idx) == 0:

            return np.nan, np.nan, np.nan

        local_arg = int(np.argmax(drawdown_pct[valid_idx]))
        maxdd_end = int(valid_idx[local_arg])
        maxdd_start = int(np.argmax(arr[:maxdd_end + 1])) if maxdd_end > 0 else 0

        maxdd_total = float(roll_max[maxdd_end] - arr[maxdd_end])
        maxdd_rate = float(drawdown_pct[maxdd_end] * 100.0)
        maxdd_period = float(maxdd_end - maxdd_start)

        return maxdd_total, maxdd_rate, maxdd_period


    def __init__(self,
                test_data: pd.DataFrame,
                initial_cash: float,

                risk_free_rate: float = 0.02,
                holidays: Optional[List[datetime.date]] = None,
                slippage: float = 0.0,
                spread: float = 0.0,
                fees: float = 0.0,
                benchmark_series: Optional[pd.Series] = None,

                signal_start_t: int = 2) -> None:

        self.test_data = test_data
        self.initial_cash = initial_cash
        self.risk_free_rate = risk_free_rate
        self.slippage = float(slippage)
        self.spread = float(spread)
        self.fees = float(fees)
        self.benchmark_series_override = benchmark_series

        self.holidays = [] if holidays is None else holidays

        self.signal_start_t = signal_start_t

        # Initialize mutable runtime state for the first cycle.
        self.reset()


    def reset(self) -> None:
        """
        ### What It Does
        Resets mutable runtime state (account/nav/trade/signal cache) while preserving engine configuration and data references.

        #### Responsibility
        Prevents cross-cycle state contamination when one BacktestEngine instance is reused by repeated test cycles.

        #### How To Use
        Call at the start of each new backtest cycle when reusing the same engine instance.

        #### Usage Example
        `result = reset(...)`
        """

        nav_idx_source = self.test_data["Datetime"] if "Datetime" in self.test_data.columns else self.test_data.index
        nav_index = pd.to_datetime(nav_idx_source, errors = "coerce")

        self.nav = []
        self.nav_line = pd.DataFrame({
            'NAV': np.nan,
            'Yield': np.nan,
            'BM_Buy_n_Hold': np.nan,
        }, index = nav_index)

        self.t_hold = []
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

        # TradeAccount runtime registers
        self.cash = self.initial_cash
        self.amt = 0
        self.position = 0
        self.p_open = None
        self.p_cover = None
        self.open_time = None

        self.open_long_rec = []
        self.cover_long_rec = []
        self.open_short_rec = []
        self.cover_short_rec = []
        self.trade_rec = []
        self.trade_return = []

        self.cover_trades = 0
        self.open_trades = 0
        self.win_trades = 0
        self.max_profit_trade = 0
        self.min_profit_trade = 0
        self.max_dd_trade = 0
        self.min_dd_trade = 0

        self.is_weekend_count = 0
        self.is_holiday_count = 0


    def calculate_holding_period(self) -> None:
        """
        ### What It Does
        Computes holding-time statistics from open/cover records for both long and short trades.

        #### Responsibility
        Quantifies exposure duration and turnover characteristics.

        #### How To Use
        Call after trade records are complete for the cycle.

        #### Usage Example
        `result = calculate_holding_period(...)`
        """

        # Long positions
        for i in range(min(len(self.open_long_rec), len(self.cover_long_rec))):

            open_long_t = self.open_long_rec[i][1]
            cover_long_t = self.cover_long_rec[i][1]
            holding_seconds = abs((cover_long_t - open_long_t).total_seconds())
            self.t_hold.append(holding_seconds)

        # Short positions
        for i in range(min(len(self.open_short_rec), len(self.cover_short_rec))):

            open_short_t = self.open_short_rec[i][1]
            cover_short_t = self.cover_short_rec[i][1]
            holding_seconds = abs((cover_short_t - open_short_t).total_seconds())
            self.t_hold.append(holding_seconds)

        self.holding_period_h_mean = pd.Series(self.t_hold).mean() / 3600
        self.holding_period_h_median = pd.Series(self.t_hold).median() / 3600
        self.holding_period_h_max = np.max(pd.Series(self.t_hold)) / 3600
        self.holding_period_h_min = np.min(pd.Series(self.t_hold)) / 3600

        self.holding_period_d_mean = self.holding_period_h_mean / 24
        self.holding_period_d_median = self.holding_period_h_median / 24
        self.holding_period_d_max = self.holding_period_h_max / 24
        self.holding_period_d_min = self.holding_period_h_min / 24


    # TradeAccount Define
    def _update_trade_stats(self, pnl: float) -> None:

        if pnl > 0:
            self.win_trades += 1

        if (self.max_profit_trade == 0 and 0 < pnl) or pnl > self.max_profit_trade > 0:
            self.max_profit_trade = pnl

        if (self.min_profit_trade == 0 and 0 < pnl) or self.min_profit_trade > pnl > 0:
            self.min_profit_trade = pnl

        if (self.max_dd_trade == 0 and 0 > pnl) or pnl < self.max_dd_trade < 0:
            self.max_dd_trade = pnl

        if (self.min_dd_trade == 0 and 0 > pnl) or self.min_dd_trade < pnl < 0:
            self.min_dd_trade = pnl


    def open_long(self,
                  t: int,
                  p_open: float,
                  current_time: datetime.datetime,

                  slippage: float = 0.0,
                  spread: float = 0.0,
                  fees: float = 0.0,
                  weight: float = 1.0) -> None:
        """
        ### What It Does
        Opens long when flat: p_trade = p_open*(1+slippage+spread/2), amt = cash/p_trade, cash -> 0.

        #### Responsibility
        Implements canonical long-entry accounting and log writes.

        #### How To Use
        Call only when position is flat; method raises on invalid state.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `p_open`
          - Entry/open execution price. It is adjusted by slippage and spread before position size is calculated.
          - Expected shape/type: `float`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `datetime.datetime`.
        - `slippage`
          - Per-trade price-impact assumption. Increase it when execution should be penalized for market impact; leave zero only for frictionless research runs.
          - Expected shape/type: `float`.
        - `spread`
          - Bid/ask spread assumption. It affects open/cover prices, so set it in price-return units consistent with the tested asset and frequency.
          - Expected shape/type: `float`.

        #### Usage Example
        `result = open_long(...)`

        ---

        ### Parameters
        - `t`: **int**.
        - `p_open`: **float**.
        - `current_time`: **datetime.datetime**.

        #### Optional Parameters
        - `slippage`: **float** = *0.0*.
        - `spread`: **float** = *0.0*.
        """

        if p_open is None or not np.isfinite(p_open):

            raise ValueError("[WARNING] Cannot open long position: p_open is None or non-finite.")

        if slippage is None:
            slippage = self.slippage

        if spread is None:
            spread = self.spread

        if slippage is None:
            slippage = self.slippage

        if spread is None:
            spread = self.spread

        if slippage < 0 or spread < 0 or fees < 0:

            raise ValueError("[WARNING] Slippage/Spread/Fees must be NON-NEGATIVE.")

        if self.position == 0:
            p_trade = p_open * (1 + slippage + spread / 2)
            # scale notional by the target weight (fraction of equity); weight=1.0 reproduces
            # the legacy all-in behaviour exactly. open only fires when flat, so cash == equity.
            w = min(max(float(weight), 0.0), 1.0)
            self.amt = w * self.cash / p_trade
            self.cash = self.cash - self.amt * p_trade * (1.0 + fees)

            self.p_open = p_trade
            self.position = 1
            self.open_trades += 1
            self.open_time = current_time

            self.open_long_rec.append((t, current_time, p_trade))
            self.trade_rec.append((t, current_time, p_trade, self.amt, 'open_long'))

        else:

            raise Exception("[WARNING] Position already open.")


    def cover_long(self,
                   t: int,
                   p_cover: float,
                   current_time: datetime.datetime,

                   slippage: float = 0.0,
                   spread: float = 0.0,
                   fees: float = 0.0) -> None:
        """
        ### What It Does
        Closes long: p_trade = p_cover*(1-slippage-spread/2), pnl = (p_trade- p_open)/p_open, then reset to flat.

        #### Responsibility
        Finalizes long lifecycle and updates return/trade records.

        #### How To Use
        Call only when position is long.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `p_cover`
          - Exit/cover execution price. It is adjusted by slippage and spread before realized PnL is recorded.
          - Expected shape/type: `float`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `datetime.datetime`.
        - `slippage`
          - Per-trade price-impact assumption. Increase it when execution should be penalized for market impact; leave zero only for frictionless research runs.
          - Expected shape/type: `float`.
        - `spread`
          - Bid/ask spread assumption. It affects open/cover prices, so set it in price-return units consistent with the tested asset and frequency.
          - Expected shape/type: `float`.

        #### Usage Example
        `result = cover_long(...)`

        ---

        ### Parameters
        - `t`: **int**.
        - `p_cover`: **float**.
        - `current_time`: **datetime.datetime**.

        #### Optional Parameters
        - `slippage`: **float** = *0.0*.
        - `spread`: **float** = *0.0*.
        """

        if self.p_open is None:

            raise ValueError("[WARNING] Cannot cover long position: no open_long recorded.")

        if p_cover is None or not np.isfinite(p_cover):

            raise ValueError("[WARNING] Cannot cover long position: p_cover is None or non-finite.")

        if slippage < 0 or spread < 0 or fees < 0:

            raise ValueError("[WARNING] Slippage/Spread/Fees must be NON-NEGATIVE.")

        if self.position == 1:
            p_trade = p_cover * (1 - slippage - spread / 2)
            # add liquidation proceeds to the un-invested cash retained at open (which is 0
            # under legacy weight=1.0, so this stays byte-identical to the all-in path).
            self.cash = self.amt * p_trade * (1.0 - fees) + self.cash
            pnl = (p_trade - self.p_open) / self.p_open
            self._update_trade_stats(pnl)
            self.trade_return.append((t, current_time, p_trade, self.p_open, pnl, -self.amt, self.max_profit_trade, self.min_profit_trade, self.max_dd_trade, self.min_dd_trade, 'cover_long'))

            self.cover_long_rec.append((t, current_time, p_trade))
            self.trade_rec.append((t, current_time, p_trade, -self.amt, 'cover_long'))

            self.amt = 0
            self.p_open = None
            self.p_cover = p_trade
            self.position = 0
            self.cover_trades += 1

        else:

            raise Exception("[WARNING] No long position to close.")


    def open_short(self,
                   t: int,
                   p_open: float,
                   current_time: datetime.datetime,

                   slippage: float = 0.0,
                   spread: float = 0.0,
                   fees: float = 0.0,
                   weight: float = 1.0) -> None:
        """
        ### What It Does
        Opens short when flat: p_trade = p_open*(1-slippage-spread/2), amt becomes negative, sale proceeds credited to cash.

        #### Responsibility
        Implements canonical short-entry accounting and log writes.

        #### How To Use
        Call only when position is flat; method raises on invalid state.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `p_open`
          - Entry/open execution price. It is adjusted by slippage and spread before position size is calculated.
          - Expected shape/type: `float`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `datetime.datetime`.
        - `slippage`
          - Per-trade price-impact assumption. Increase it when execution should be penalized for market impact; leave zero only for frictionless research runs.
          - Expected shape/type: `float`.
        - `spread`
          - Bid/ask spread assumption. It affects open/cover prices, so set it in price-return units consistent with the tested asset and frequency.
          - Expected shape/type: `float`.

        #### Usage Example
        `result = open_short(...)`

        ---

        ### Parameters
        - `t`: **int**.
        - `p_open`: **float**.
        - `current_time`: **datetime.datetime**.

        #### Optional Parameters
        - `slippage`: **float** = *0.0*.
        - `spread`: **float** = *0.0*.
        """

        if p_open is None or not np.isfinite(p_open):

            raise ValueError("[WARNING] Cannot open short position: p_open is None or non-finite.")

        if slippage < 0 or spread < 0 or fees < 0:

            raise ValueError("[WARNING] Slippage/Spread/Fees must be NON-NEGATIVE.")

        if self.position == 0:
            p_trade = p_open * (1 - slippage - spread / 2)
            # scale short notional by the target weight; weight=1.0 reproduces the legacy
            # full-notional short exactly. open only fires when flat, so cash == equity.
            w = min(max(float(weight), 0.0), 1.0)
            amount = w * self.cash / p_trade
            self.amt = -amount
            self.cash += amount * p_trade * (1.0 - fees)

            self.p_open = p_trade
            self.position = -1
            self.open_trades += 1
            self.open_time = current_time
            self.open_short_rec.append((t, current_time, p_trade))
            self.trade_rec.append((t, current_time, p_trade, self.amt, 'open_short'))

        else:

            raise Exception("[WARNING] Position already open.")


    def cover_short(self,
                    t: int,
                    p_cover: float,
                    current_time: datetime.datetime,

                    slippage: float = 0.0,
                    spread: float = 0.0,
                    fees: float = 0.0) -> None:
        """
        ### What It Does
        Closes short: p_trade = p_cover*(1+slippage+spread/2), pnl = (p_open- p_trade)/p_open, then reset to flat.

        #### Responsibility
        Finalizes short lifecycle and updates return/trade records.

        #### How To Use
        Call only when position is short.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `p_cover`
          - Exit/cover execution price. It is adjusted by slippage and spread before realized PnL is recorded.
          - Expected shape/type: `float`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `datetime.datetime`.
        - `slippage`
          - Per-trade price-impact assumption. Increase it when execution should be penalized for market impact; leave zero only for frictionless research runs.
          - Expected shape/type: `float`.
        - `spread`
          - Bid/ask spread assumption. It affects open/cover prices, so set it in price-return units consistent with the tested asset and frequency.
          - Expected shape/type: `float`.

        #### Usage Example
        `result = cover_short(...)`

        ---

        ### Parameters
        - `t`: **int**.
        - `p_cover`: **float**.
        - `current_time`: **datetime.datetime**.

        #### Optional Parameters
        - `slippage`: **float** = *0.0*.
        - `spread`: **float** = *0.0*.
        """

        if self.p_open is None:

            raise ValueError("[WARNING] Cannot close short position: no open_short recorded.")

        if p_cover is None or not np.isfinite(p_cover):

            raise ValueError("[WARNING] Cannot close short position: p_cover is None or non-finite.")

        if slippage < 0 or spread < 0 or fees < 0:

            raise ValueError("[WARNING] Slippage/Spread/Fees must be NON-NEGATIVE.")

        if self.position == -1:
            p_trade = p_cover * (1 + slippage + spread / 2)
            self.cash = self.amt * p_trade * (1.0 + fees) + self.cash

            pnl = (self.p_open - p_trade) / self.p_open
            self._update_trade_stats(pnl)
            self.trade_return.append((t, current_time, p_trade, self.p_open, pnl, -self.amt, self.max_profit_trade, self.min_profit_trade, self.max_dd_trade, self.min_dd_trade, 'cover_short'))

            self.cover_short_rec.append((t, current_time, p_trade))
            self.trade_rec.append((t, current_time, p_trade, -self.amt, 'cover_short'))

            self.amt = 0
            self.p_open = None
            self.p_cover = p_trade
            self.position = 0
            self.cover_trades += 1

        else:

            raise Exception("[WARNING] No short position to close.")


    def entry(self,
              t: int,
              direction: str | int,

              p_open: Optional[float] = None,
              current_time: Optional[datetime.datetime] = None,
              cover_reverse: Optional[bool] = False) -> None:
        """
        ### What It Does
        Normalizes direction and dispatches entry: BUY/1 -> open_long, SELL/-1 -> open_short; optional reverse cover when cover_reverse=True.

        #### Responsibility
        Centralizes entry branching and default price/time fallback extraction.

        #### How To Use
        Call once per bar with strategy direction; supply p_open/current_time explicitly if needed.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `direction`
          - Trade-side selector. Use explicit long/short/buy/sell conventions so dispatch opens or closes the intended side.
          - Expected shape/type: `str | int`.
        - `p_open`
          - Entry/open execution price. It is adjusted by slippage and spread before position size is calculated.
          - Expected shape/type: `Optional[float]`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `Optional[datetime.datetime]`.
        - `cover_reverse`
          - Reverse-position handling switch. Enable it only when a new opposite signal should close the current exposure immediately.
          - Expected shape/type: `Optional[bool]`.

        #### Usage Example
        `result = entry(...)`

        ---

        ### Parameters
        - `t`: **int**.
        - `direction`: **str | int**.

        #### Optional Parameters
        - `p_open`: **Optional[float]** = *None*.
        - `current_time`: **Optional[datetime.datetime]** = *None*.
        - `cover_reverse`: **Optional[bool]** = *False*.
        """

        if p_open is None:

            if "Open" in self.test_data.columns:
                p_open = self.test_data["Open"].iloc[t]

            else:
                p_open = self.test_data["Close"].iloc[t]

        if current_time is None:

            if "Datetime" in self.test_data.columns:
                current_time = self.test_data["Datetime"].iloc[t]

            else:
                current_time = self.test_data.index[t]

        if p_open is None:

            raise ValueError("[WARNING] p_open not found in test_data.")

        if current_time is None:

            raise ValueError("[WARNING] current_time not found in test_data.")

        if self.slippage < 0 or self.spread < 0:

            raise ValueError("[WARNING] Slippage/Spread must be NON-NEGATIVE.")

        # per-bar target exposure: honour a "weights" column (same series VBT uses as
        # targetpercent) so the ForLoop caliber matches the VBT path. Absent/NaN -> full notional.
        target_weight = 1.0

        if "weights" in self.test_data.columns:
            raw_weight = self.test_data["weights"].iloc[t]

            if raw_weight is not None and np.isfinite(raw_weight):
                target_weight = abs(float(raw_weight))

        direction_norm = direction

        if isinstance(direction, str):
            direction_norm = direction.strip().upper()

        if direction_norm in ("BUY", 1):

            if self.position == 1:

                return

            if cover_reverse and self.position == -1:
                self.cover_short(t, p_open, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)

            if self.position == 0:
                self.open_long(t, p_open, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees, weight = target_weight)

            return

        if direction_norm in ("SELL", -1):

            if self.position == -1:

                return

            if cover_reverse and self.position == 1:
                self.cover_long(t, p_open, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)

            if self.position == 0:
                self.open_short(t, p_open, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees, weight = target_weight)

            return


        raise ValueError("[WARNING] Entry direction must be 'BUY'/'SELL' or 1/-1.")


    def exit(self,
             t: int,
             direction: str | int,

             p_cover: Optional[float] = None,
             current_time: Optional[datetime.datetime] = None) -> None:
        """
        ### What It Does
        Normalizes direction and dispatches exit: LONG/L/1 -> cover_long, SHORT/S/-1 -> cover_short.

        #### Responsibility
        Centralizes close-side branching and fallback extraction.

        #### How To Use
        Call when strategy emits exit intent for current position side.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `direction`
          - Trade-side selector. Use explicit long/short/buy/sell conventions so dispatch opens or closes the intended side.
          - Expected shape/type: `str | int`.
        - `p_cover`
          - Exit/cover execution price. It is adjusted by slippage and spread before realized PnL is recorded.
          - Expected shape/type: `Optional[float]`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `Optional[datetime.datetime]`.

        #### Usage Example
        `result = exit(...)`

        ---

        ### Parameters
        - `t`: **int**.
        - `direction`: **str | int**.

        #### Optional Parameters
        - `p_cover`: **Optional[float]** = *None*.
        - `current_time`: **Optional[datetime.datetime]** = *None*.
        """

        if p_cover is None:

            if "Open" in self.test_data.columns:
                p_cover = self.test_data["Open"].iloc[t]

            else:
                p_cover = self.test_data["Close"].iloc[t]

        if current_time is None:

            if "Datetime" in self.test_data.columns:
                current_time = self.test_data["Datetime"].iloc[t]

            else:
                current_time = self.test_data.index[t]

        if p_cover is None:

            raise ValueError("[WARNING] p_cover not found in test_data.")

        if current_time is None:

            raise ValueError("[WARNING] current_time not found in test_data.")

        if self.slippage < 0 or self.spread < 0:

            raise ValueError("[WARNING] Slippage/Spread must be NON-NEGATIVE.")

        direction_norm = direction

        if isinstance(direction, str):
            direction_norm = direction.strip().upper()

        if direction_norm in ("LONG", "L", 1):

            if self.position != 1:

                return

            self.cover_long(t, p_cover, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)

            return

        if direction_norm in ("SHORT", "S", -1):

            if self.position != -1:

                return

            self.cover_short(t, p_cover, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)

            return


        raise ValueError("[WARNING] Exit direction must be 'LONG'/'SHORT' or 1/-1.")


    def force_cover_timeout(self,
                            t: int,

                            current_time: Optional[datetime.datetime] = None,
                            forced_close_time: Optional[datetime.datetime] = None,
                            p_cover: Optional[float] = None) -> None:
        """
        ### What It Does
        Forces close when current_time >= forced_close_time for any open position.

        #### Responsibility
        Implements time-cutoff liquidation control.

        #### How To Use
        Call per bar when cutoff policy is active; no-op before cutoff.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `Optional[datetime.datetime]`.
        - `forced_close_time`
          - Timestamp after which an open position must be closed. Use it to enforce intraday/session risk limits.
          - Expected shape/type: `Optional[datetime.datetime]`.
        - `p_cover`
          - Exit/cover execution price. It is adjusted by slippage and spread before realized PnL is recorded.
          - Expected shape/type: `Optional[float]`.

        #### Usage Example
        `result = force_cover_timeout(...)`

        ---

        ### Parameters
        - `t`: **int**.

        #### Optional Parameters
        - `current_time`: **Optional[datetime.datetime]** = *None*.
        - `forced_close_time`: **Optional[datetime.datetime]** = *None*.
        - `p_cover`: **Optional[float]** = *None*.
        """

        if p_cover is None:

            if "Open" in self.test_data.columns:
                p_cover = self.test_data["Open"].iloc[t]

            else:
                p_cover = self.test_data["Close"].iloc[t]

        if current_time is None:

            if "Datetime" in self.test_data.columns:
                current_time = self.test_data["Datetime"].iloc[t]

            else:
                current_time = self.test_data.index[t]

        if p_cover is None:

            raise ValueError("[WARNING] p_cover not found in test_data.")

        if current_time is None:

            raise ValueError("[WARNING] current_time not found in test_data.")

        if self.slippage < 0 or self.spread < 0:

            raise ValueError("[WARNING] Slippage/Spread must be NON-NEGATIVE.")

        if forced_close_time is not None and current_time >= forced_close_time:

            if self.position == 1:

                if self.p_open is None:

                    raise ValueError("[WARNING] Cannot cover long position: No Open_long Recorded.")

                self.cover_long(t, p_cover, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)

            elif self.position == -1:

                if self.p_open is None:

                    raise ValueError("[WARNING] Cannot cover short position: No Open_short Recorded.")

                self.cover_short(t, p_cover, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)


    def force_cover_holiday(self,
                            t: int,

                            current_time: Optional[datetime.datetime] = None,
                            p_cover: Optional[float] = None) -> None:
        """
        ### What It Does
        Forces close on pre-holiday day (holiday minus one day) for open positions.

        #### Responsibility
        Reduces overnight/event exposure around holiday closures.

        #### How To Use
        Call per bar when holiday calendar is configured.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `Optional[datetime.datetime]`.
        - `p_cover`
          - Exit/cover execution price. It is adjusted by slippage and spread before realized PnL is recorded.
          - Expected shape/type: `Optional[float]`.

        #### Usage Example
        `result = force_cover_holiday(...)`

        ---

        ### Parameters
        - `t`: **int**.

        #### Optional Parameters
        - `current_time`: **Optional[datetime.datetime]** = *None*.
        - `p_cover`: **Optional[float]** = *None*.
        """

        if not self.holidays:

            return

        if p_cover is None:

            if "Open" in self.test_data.columns:
                p_cover = self.test_data["Open"].iloc[t]

            else:
                p_cover = self.test_data["Close"].iloc[t]

        if current_time is None:

            if "Datetime" in self.test_data.columns:
                current_time = self.test_data["Datetime"].iloc[t]

            else:
                current_time = self.test_data.index[t]

        if p_cover is None:

            raise ValueError("[WARNING] p_cover not found in test_data.")

        if current_time is None:

            raise ValueError("[WARNING] current_time not found in test_data.")

        if self.slippage < 0 or self.spread < 0:

            raise ValueError("[WARNING] Slippage/Spread must be NON-NEGATIVE.")

        current_date = current_time.date()

        # self.holidays holds ACTUAL calendar dates (multi-year NYSE closures, incl. floating
        # holidays like Good Friday / Thanksgiving). Never rebuild them into current_time.year:
        # that misaligns every floating holiday, off-by-ones the Jan-1 / Dec-31 pre-holiday across
        # the year boundary, and crashes on a Feb-29 entry in a non-leap year. Compare against the
        # real date; datetime / pd.Timestamp entries are normalized to date() first.
        for holiday in self.holidays:
            holiday_date = holiday.date() if isinstance(holiday, datetime.datetime) else holiday
            pre_holiday = holiday_date - datetime.timedelta(days = 1)

            if current_date == pre_holiday:

                if self.position == 1:

                    if self.p_open is None:

                        raise ValueError("[WARNING] Cannot cover long position: No open_long Recorded.")

                    self.cover_long(t, p_cover, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)

                elif self.position == -1:

                    if self.p_open is None:

                        raise ValueError("[WARNING] Cannot cover short position: No open_short Recorded.")

                    self.cover_short(t, p_cover, current_time, slippage = self.slippage, spread = self.spread, fees = self.fees)


    def prevent_new_open_positions(self,
                                   t: int,

                                   current_time: Optional[datetime.datetime] = None) -> bool:
        """
        ### What It Does
        Returns True when opening should be blocked (Friday, holiday date, or pre-holiday date).

        #### Responsibility
        Acts as pre-entry guard while preserving exit/forced-close operations.

        #### How To Use
        Check before entry call; skip opens when this returns True.

        #### Key Parameters In Practice
        - `t`
          - Current bar index or timestamp location. Use it to update stateful engine fields in chronological order.
          - Expected shape/type: `int`.
        - `current_time`
          - Timestamp of the current bar. Force-cover, holiday, timeout, and prevent-open checks depend on it.
          - Expected shape/type: `Optional[datetime.datetime]`.

        #### Usage Example
        `result = prevent_new_open_positions(...)`

        ---

        ### Parameters
        - `t`: **int**.

        #### Optional Parameters
        - `current_time`: **Optional[datetime.datetime]** = *None*.

        ---

        ### Returns
        - `result`: **bool**.
        """

        if current_time is None:

            if "Datetime" in self.test_data.columns:
                current_time = self.test_data["Datetime"].iloc[t]

            else:
                current_time = self.test_data.index[t]

        if current_time is None:

            raise ValueError("[WARNING] current_time not found in test_data.")

        if current_time.weekday() == 4:
            self.is_weekend_count += 1

            return True

        if self.holidays:

            current_date = current_time.date()

            # ACTUAL calendar dates (see force_cover_holiday): compare directly, never rebuild into
            # current_time.year (floating-holiday misalignment + Feb-29 crash + year-edge off-by-one).
            for holiday in self.holidays:
                holiday_date = holiday.date() if isinstance(holiday, datetime.datetime) else holiday
                pre_holiday = holiday_date - datetime.timedelta(days = 1)

                if current_date == holiday_date:
                    self.is_holiday_count += 1

                    return True

                if current_date == pre_holiday:

                    return True

        else:

            return False


        return False
