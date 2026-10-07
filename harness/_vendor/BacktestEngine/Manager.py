from ENV_MGMT.imports import *
from .Signal import Signal
from .Position import Position
from .Calendars import default_holidays, default_macro_release_dates
from .Utils import _strategy_pack_for_transport, _strategy_resolve_callable, _is_signature_mismatch_type_error, multi_log_set_top
import FactorEngine

if TYPE_CHECKING:  # type-only import; runtime import is deferred inside methods to break circular dependency
    from .ForLoopEngine import BacktestEngine_ForLoop







# Back-test Engine Manager Class
#----------------------------------------------------------------------------------------
class BacktestEngineManager:
    """
    ### What It Does
    BacktestEngineManager orchestrates one-run execution plus protocol-first fixed-IS / rolling-OOS workflows through `window_processing(...)`.

    #### Responsibility
    It coordinates parameter-context generation, traversal search, stitched final replay, and consolidated outputs.

    #### How To Use
    Use `window_processing(...)` for search workflows, or instantiate manager for reusable per-cycle test execution.

    #### Usage Example
    `obj = BacktestEngineManager(...)`

    ---

    ### Parameters
    - `Data_Config`: **Dict[str, Any]**.
    - `Factor_Config`: **Dict[str, Any]**.
    - `Execution_Config`: **Dict[str, Any]**.
    - `factor_param_ranges`: **Dict[str, List[Any]]**.

    #### Optional Parameters
    - `Signal_Config`: **Optional[Dict[str, Any]]** = *None*.
    - `payload`: **Optional[Dict[str, Any]]** = *None*.
    - `backtest_strategy`: **Optional[Callable[[int, "BacktestEngine_ForLoop", pd.DataFrame], Any]]** = *None*.
    """

    _btmgr: Optional["BacktestEngineManager"] = None
    # Persistent-traversal-pool worker state, set in each worker process by
    # trav_init_worker_persistent. _trav_worker_token gates a per-window _btmgr
    # rebuild so ONE pool serves every walk-forward window without respawning.
    _trav_window_state: Any = None
    _trav_worker_token: Any = None

    @staticmethod
    def _resolve_independent_books(*, independent_books: Any = False) -> bool:

        if independent_books is None:

            return False

        if isinstance(independent_books, bool):

            return independent_books

        raise ValueError(f"[WARNING] independent_books must be a boolean (True = per-asset independent books with cash_sharing=False; False = shared-cash cross-sectional portfolio); got {independent_books!r}.")


    @staticmethod
    def _resolve_cal_column(*, cal_column: Any = "Close") -> str:

        text = str(cal_column).strip()

        if not text:

            raise ValueError("[WARNING] cal_column cannot be empty.")


        return text


    @staticmethod
    def _time_index(test_data: pd.DataFrame) -> pd.Index:

        if "Datetime" in test_data.columns:
            dt_index = pd.DatetimeIndex(pd.to_datetime(test_data["Datetime"], errors = "coerce"))

            if not bool(pd.isna(dt_index).all()):

                return dt_index

        return pd.Index(test_data.index)


    @staticmethod
    def _align_series_to_test_data(raw_series: pd.Series, test_data: pd.DataFrame) -> pd.Series:

        time_index = BacktestEngineManager._time_index(test_data)
        weight_series = pd.Series(pd.to_numeric(raw_series, errors = "coerce"), index = raw_series.index, dtype = float)

        aligned = weight_series.reindex(time_index)

        if aligned.isna().all():
            aligned = weight_series.reindex(test_data.index)

        if len(weight_series) > 0 and bool(weight_series.notna().any()) and bool(aligned.isna().all()):

            logging.warning(
                            "[WARNING] weight/signal series index does not overlap the test calendar; "
                            "result is forced all-flat (source rows=%d).",
                            len(weight_series),
                            )

        aligned_values = pd.to_numeric(aligned, errors = "coerce").to_numpy(dtype = float, copy = False)

        return pd.Series(aligned_values, index = test_data.index, dtype = float)


    @staticmethod
    def _infer_vbt_freq(index_like: Any) -> str:

        idx = pd.DatetimeIndex(pd.to_datetime(index_like, errors = "coerce"))
        idx = idx[idx.notna()]

        if len(idx) <= 1:

            return "1D"

        if len(idx) < 3:
            delta = idx[1] - idx[0]
            total_seconds = int(delta.total_seconds())

            if total_seconds % 86400 == 0:

                return f"{max(1, total_seconds // 86400)}D"

            if total_seconds % 3600 == 0:

                return f"{max(1, total_seconds // 3600)}H"

            if total_seconds % 60 == 0:

                return f"{max(1, total_seconds // 60)}min"

            return f"{max(1, total_seconds)}S"

        inferred = pd.infer_freq(idx)

        if isinstance(inferred, str) and inferred.strip():

            try:
                pd.Timedelta(inferred)

                return inferred

            except Exception:
                pass

        deltas = pd.Series(idx[1:] - idx[:-1])

        if deltas.empty:

            return "1D"

        try:
            delta = deltas.mode(dropna = True).iloc[0]

        except Exception:
            delta = deltas.iloc[0]

        if not isinstance(delta, pd.Timedelta) or pd.isna(delta) or delta <= pd.Timedelta(0):

            return "1D"

        total_seconds = int(delta.total_seconds())

        if total_seconds % 86400 == 0:

            return f"{max(1, total_seconds // 86400)}D"

        if total_seconds % 3600 == 0:

            return f"{max(1, total_seconds // 3600)}H"

        if total_seconds % 60 == 0:

            return f"{max(1, total_seconds // 60)}min"

        return f"{max(1, total_seconds)}S"


    @staticmethod
    def _target_metric_value(metric_map: Dict[str, Any], target_metric: str) -> float:

        metric_name = str(target_metric or "sharpe_ratio").strip() or "sharpe_ratio"
        candidates = [metric_name, "sharpe_ratio", "nav_rate"]
        seen: set[str] = set()

        for key in candidates:
            key_text = str(key)

            if key_text in seen:

                continue

            seen.add(key_text)
            raw_value = metric_map.get(key_text, np.nan)
            value = cast(Any, raw_value).item() if isinstance(raw_value, np.generic) else raw_value

            try:
                numeric = float(value)

            except Exception:

                continue

            if np.isfinite(numeric):

                return numeric

        return np.nan


    @staticmethod
    def _target_metric_rank_value(target_metric: str, metric_value: Any) -> float:

        value = cast(Any, metric_value).item() if isinstance(metric_value, np.generic) else metric_value

        try:
            numeric = float(value)

        except Exception:

            return np.nan

        if not np.isfinite(numeric):

            return np.nan

        metric_name = str(target_metric or "sharpe_ratio").strip().lower()

        if metric_name in {
                            "maxdd",
                            "maxdd_rate",
                            "max_drawdown",
                            "max_drawdown_rate",
                            "drawdown",
                            "drawdown_rate",
                          }:

            return -numeric


        return numeric


    @staticmethod
    def _weighted_panel_sum(panel_specs: List[Tuple[pd.DataFrame, float]],
                            *,
                            index: pd.DatetimeIndex,
                            columns: List[str]) -> pd.DataFrame:

        out = pd.DataFrame(0.0, index = index, columns = columns, dtype = float)

        for panel, weight in panel_specs:
            aligned = pd.DataFrame(panel, copy = True).reindex(index = index, columns = columns).fillna(0.0).astype(float)
            out = out.add(aligned * float(weight), fill_value = 0.0)


        return out.astype(float)


    @staticmethod
    def _build_protocol_windows(*,
                                test_data: pd.DataFrame,
                                start_date: str,
                                end_date: str,
                                train_months: Optional[int],
                                test_months: Optional[int],
                                step_months: Optional[int]) -> Tuple[pd.DataFrame, List[Dict[str, Any]], pd.Timestamp, pd.Timestamp]:

        if test_data is None or len(test_data) == 0:

            raise ValueError("[WARNING] window_processing requires non-empty test_data.")

        prepared = pd.DataFrame(test_data).copy()
        prepared_dt = pd.Series(pd.to_datetime(prepared["Datetime"] if "Datetime" in prepared.columns else prepared.index, errors = "coerce"), index = prepared.index)
        prepared["__zora_dt__"] = prepared_dt.dt.normalize()

        if prepared["__zora_dt__"].isna().any():

            raise ValueError("[WARNING] window_processing test_data contains invalid Datetime values.")

        start_ts = pd.Timestamp(start_date).normalize()
        end_ts = pd.Timestamp(end_date).normalize()

        if end_ts < start_ts:

            raise ValueError("[WARNING] window_processing requires end_date >= start_date.")

        in_range = prepared.loc[(prepared["__zora_dt__"] >= start_ts) & (prepared["__zora_dt__"] <= end_ts)].copy()

        if in_range.empty:

            raise ValueError("[WARNING] window_processing produced empty protocol slice.")

        windows: List[Dict[str, Any]] = []

        if test_months is None:

            if train_months is None:
                train_slice = in_range.copy()

            else:
                train_start = max(start_ts, end_ts - pd.DateOffset(months = int(train_months)) + pd.Timedelta(days = 1))
                train_slice = in_range.loc[in_range["__zora_dt__"] >= train_start].copy()

            if train_slice.empty:

                raise RuntimeError("[WARNING] window_processing produced empty fixed-IS train slice.")

            train_start_ts = pd.Timestamp(train_slice["__zora_dt__"].iloc[0]).normalize()
            train_end_ts = pd.Timestamp(train_slice["__zora_dt__"].iloc[-1]).normalize()
            windows.append(
                            {
                                "window_id": "W001",
                                "train_start": train_start_ts,
                                "train_end": train_end_ts,
                                "test_start": None,
                                "test_end": None,
                                "train_df": train_slice.drop(columns = ["__zora_dt__"]).copy(),
                                "test_df": None,
                            }
                          )

            return in_range.drop(columns = ["__zora_dt__"]).copy(), windows, start_ts, end_ts

        if step_months is None:
            step_months = int(test_months)

        if int(test_months) <= 0 or int(step_months) <= 0:

            raise ValueError("[WARNING] test_months and step_months must be positive when rolling protocol is enabled.")

        previous_test_end: Optional[pd.Timestamp] = None
        current_test_start = start_ts
        window_id = 1

        while current_test_start <= end_ts:

            train_start = start_ts if train_months is None else current_test_start - pd.DateOffset(months = int(train_months))
            test_end = min((current_test_start + pd.DateOffset(months = int(test_months))) - pd.Timedelta(days = 1), end_ts)
            train_df = prepared.loc[(prepared["__zora_dt__"] >= train_start) & (prepared["__zora_dt__"] < current_test_start)].copy()
            test_df = prepared.loc[(prepared["__zora_dt__"] >= current_test_start) & (prepared["__zora_dt__"] <= test_end)].copy()

            if (not train_df.empty) and (not test_df.empty):
                actual_test_start = pd.Timestamp(test_df["__zora_dt__"].iloc[0]).normalize()

                if previous_test_end is not None and actual_test_start <= previous_test_end:

                    raise ValueError("[WARNING] window_processing requires non-overlapping test windows.")

                previous_test_end = pd.Timestamp(test_df["__zora_dt__"].iloc[-1]).normalize()
                windows.append(
                                {
                                    "window_id": f"W{window_id:03d}",
                                    "train_start": pd.Timestamp(train_df["__zora_dt__"].iloc[0]).normalize(),
                                    "train_end": pd.Timestamp(train_df["__zora_dt__"].iloc[-1]).normalize(),
                                    "test_start": pd.Timestamp(test_df["__zora_dt__"].iloc[0]).normalize(),
                                    "test_end": pd.Timestamp(test_df["__zora_dt__"].iloc[-1]).normalize(),
                                    "train_df": train_df.drop(columns = ["__zora_dt__"]).copy(),
                                    "test_df": test_df.drop(columns = ["__zora_dt__"]).copy(),
                                }
                              )

                window_id += 1

            current_test_start = current_test_start + pd.DateOffset(months = int(step_months))

        if len(windows) == 0:

            raise RuntimeError("[WARNING] window_processing produced no valid rolling windows.")


        return in_range.drop(columns = ["__zora_dt__"]).copy(), windows, start_ts, end_ts


    @staticmethod
    def _normalize_protocol_param_dict(raw: Mapping[str, Any]) -> Dict[str, Any]:

        return {

                str(key): (
                    value.unwrap() if isinstance(value, FactorEngine.AutoParam.ParamValue)

                    else (cast(Any, value).item() if isinstance(value, np.generic) else value)
                )
                for key, value in dict(raw).items()
                }


    @staticmethod
    def _extract_best_params(perf_rec: pd.DataFrame,
                            *,
                            target_metric: str) -> Tuple[Dict[str, Any], Dict[str, Any], pd.DataFrame]:

        perf = pd.DataFrame(perf_rec).copy()

        if perf.empty:

            return {}, {}, perf

        # Resolve the target to its canonical performance column BEFORE the strict
        # selection below. The only whitelisted target whose column name differs from
        # the target name is the max-drawdown family -> "maxdd_rate". Without this
        # canonicalization a maxdd-family target on a flat/zero-trade window would fail
        # the literal column match, fall back to sharpe_ratio (NaN on that window), and
        # be wrongly dropped by the strict filter -- whereas maxdd_rate = 0 is finite
        # and keeps the window. Keeps sharpe_ratio and every metric that is already its
        # own column byte-identical.
        maxdd_aliases = {"maxdd", "maxdd_rate", "max_drawdown", "max_drawdown_rate", "drawdown", "drawdown_rate"}

        if target_metric in perf.columns:
            metric_name = target_metric

        elif str(target_metric or "").strip().lower() in maxdd_aliases and "maxdd_rate" in perf.columns:
            metric_name = "maxdd_rate"

        elif "sharpe_ratio" in perf.columns:
            metric_name = "sharpe_ratio"

        else:
            metric_name = "nav_rate"

        for column in ("sharpe_ratio", "nav_rate", "sortino_ratio", "maxdd_rate", metric_name):

            if column in perf.columns:
                perf[column] = pd.to_numeric(perf[column], errors = "coerce")

        # NaN-intolerant selection: rank STRICTLY by the
        # resolved target-metric column, WITHOUT the per-row nav_rate fallback that
        # _target_metric_value applies. A candidate whose target metric is non-finite
        # (e.g. an early window whose training slice is too short to clear the signal
        # warmup -> no trades -> zero-variance NAV -> NaN sharpe) yields a NaN score,
        # is dropped by the .notna() filter below, and -- when EVERY candidate in the
        # window is non-finite -- leaves an empty frame so the caller's
        # `if len(best_params) == 0: continue` skips the window instead of stitching an
        # unvalidatable pick. This matches combo_processing's strict
        # `perf_rec[target_metric]` ranking.
        perf["__selection_score__"] = (
                                        pd.to_numeric(perf[metric_name], errors = "coerce")
                                        if metric_name in perf.columns
                                        else pd.Series(np.nan, index = perf.index)
                                      )
        perf["__selection_rank_score__"] = perf["__selection_score__"].map(lambda value: BacktestEngineManager._target_metric_rank_value(target_metric, value))
        perf = perf.loc[perf["__selection_rank_score__"].notna()].copy()

        if perf.empty:

            return {}, {}, perf

        sort_cols: List[str] = ["__selection_rank_score__"]
        ascending: List[bool] = [False]

        for column in ("sharpe_ratio", "nav_rate"):

            if column in perf.columns:
                sort_cols.append(column)
                ascending.append(False)

        perf = perf.sort_values(sort_cols, ascending = ascending).reset_index(drop = True)
        best_row = {str(key): value for key, value in perf.iloc[0].to_dict().items()}
        raw_params = best_row.get("factor_param_ranges")
        best_params: Dict[str, Any] = {}

        if isinstance(raw_params, dict):
            best_params = BacktestEngineManager._normalize_protocol_param_dict(raw_params)

        elif isinstance(best_row.get("factor_param_json"), str):

            try:
                parsed = json.loads(str(best_row["factor_param_json"]))

                if isinstance(parsed, dict):
                    best_params = BacktestEngineManager._normalize_protocol_param_dict(parsed)

            except Exception:
                best_params = {}


        return best_params, best_row, perf


    @staticmethod
    def performance_metrics(nav_series: pd.Series,
                            benchmark_series: Optional[pd.Series] = None,
                            risk_free_rate: float = 0.02,
                            periods_per_year: Optional[float] = None) -> Dict[str, float]:
        """
        ### What It Does
        Calculates performance and risk metrics from a NAV series.

        #### Responsibility
        Centralizes return, volatility, drawdown, benchmark-relative, and optional vectorbt metrics.

        #### How To Use
        Call it after a backtest produces a NAV series.

        #### Key Parameters In Practice
        - `nav_series`
          - NAV time series used for metric calculation. Ensure it is numeric and ordered by time.
          - Expected shape/type: `pd.Series`.
        - `benchmark_series`
          - Benchmark NAV/price series for relative reporting. Align it to the tested period instead of filling missing pre-history.
          - Expected shape/type: `Optional[pd.Series]`.
        - `risk_free_rate`
          - Annual risk-free assumption for metrics. Use a strategy-level rate aligned with the reporting period, not a per-bar return.
          - Expected shape/type: `float`.

        #### Usage Example
        `result = performance_metrics(...)`

        ---

        ### Parameters
        - `nav_series`: **pd.Series**.

        #### Optional Parameters
        - `benchmark_series`: **Optional[pd.Series]** = *None*.
        - `risk_free_rate`: **float** = *0.02*.

        ---

        ### Returns
        - `result`: **Dict[str, float]**.
        """

        from .ForLoopEngine import BacktestEngine_ForLoop  # deferred import: breaks circular dependency
        calendar_year_days = 365.0

        nav_raw = pd.to_numeric(pd.Series(nav_series).copy(), errors = "coerce")
        nav_valid_mask = nav_raw.notna()

        if nav_valid_mask.any():
            first_valid_pos = int(np.flatnonzero(nav_valid_mask.to_numpy())[0])
            nav = nav_raw.iloc[first_valid_pos:]

        else:
            nav = nav_raw.iloc[0:0]

        invalid_nav_path = bool(not nav.empty and nav.isna().any())

        if not nav.empty:

            try:
                nav.index = pd.to_datetime(nav.index, errors = "coerce")

            except Exception:
                pass

        result: Dict[str, float] = {
                                    "nav_total": np.nan,
                                    "nav_return": np.nan,
                                    "nav_rate": np.nan,
                                    "nav_total_bm": np.nan,
                                    "nav_total_bm_return": np.nan,
                                    "nav_total_bm_rate": np.nan,
                                    "rel_return": np.nan,
                                    "rel_return_rate": np.nan,
                                    "annualized_return_rate": np.nan,
                                    "maxdd_total": np.nan,
                                    "maxdd_rate": np.nan,
                                    "maxdd_period": np.nan,
                                    "calmar_ratio": np.nan,
                                    "sharpe_ratio": np.nan,
                                    "sortino_ratio": np.nan,
                                    "yield_volatility": np.nan,
                                    "yield_downside_deviation": np.nan,
                                    "var95": np.nan,
                                    "cvar95": np.nan,
                                    "var99": np.nan,
                                    "cvar99": np.nan,
                                    }

        if nav.empty or invalid_nav_path:

            return result

        def _periods_per_year(index_like: pd.Index, observed_periods: int) -> float:

            # Explicit override (e.g. 252 trading days) pins annualisation
            # instead of inferring it from the calendar spacing of the index.
            if (periods_per_year is not None
                    and np.isfinite(periods_per_year) and periods_per_year > 0):

                return float(periods_per_year)

            if observed_periods <= 0:

                return calendar_year_days

            if not isinstance(index_like, pd.DatetimeIndex) or len(index_like) <= 1:

                return calendar_year_days

            idx = pd.DatetimeIndex(index_like).dropna().sort_values()

            if len(idx) <= 1:

                return calendar_year_days

            date_counts = pd.Series(1, index = idx).groupby(idx.normalize()).sum()

            if not date_counts.empty and float(date_counts.median()) > 1.0:

                return calendar_year_days * float(date_counts.median())

            deltas = pd.Series(idx[1:] - idx[:-1])

            if deltas.empty:

                return calendar_year_days

            median_gap = float(deltas.dt.total_seconds().median() / 86400.0)

            if not np.isfinite(median_gap) or median_gap <= 0:
                median_gap = 1.0


            return calendar_year_days / median_gap


        nav_first = float(nav.iloc[0])
        nav_last = float(nav.iloc[-1])
        result["nav_total"] = nav_last

        if nav_first != 0 and pd.notna(nav_first) and pd.notna(nav_last):
            nav_return = nav_last / nav_first
            result["nav_return"] = float(nav_return)
            result["nav_rate"] = float((nav_return - 1.0) * 100.0)
            elapsed_periods = max(1, len(nav) - 1)
            periods_per_year_for_return = _periods_per_year(nav.index, elapsed_periods)

            if nav_return > 0 and periods_per_year_for_return > 0:
                result["annualized_return_rate"] = float(((nav_return ** (periods_per_year_for_return / elapsed_periods)) - 1.0) * 100.0)

        maxdd_total, maxdd_rate, maxdd_period = BacktestEngine_ForLoop.max_drawdown(nav)
        result["maxdd_total"] = maxdd_total
        result["maxdd_rate"] = maxdd_rate
        result["maxdd_period"] = maxdd_period

        if pd.notna(result["annualized_return_rate"]) and pd.notna(maxdd_rate) and maxdd_rate != 0:
            result["calmar_ratio"] = float(result["annualized_return_rate"] / maxdd_rate)

        period_returns = nav.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

        if not period_returns.empty:

            periods_per_year = _periods_per_year(period_returns.index, len(period_returns))
            bar_year_fraction = 1 / periods_per_year

            period_rf = (1 + risk_free_rate) ** bar_year_fraction - 1
            excess_returns = period_returns - period_rf
            excess_std = np.std(excess_returns, ddof = 1)
            annualizer = np.sqrt(periods_per_year)

            if excess_std > 0:
                result["sharpe_ratio"] = float(np.mean(excess_returns) / excess_std * annualizer)

            downside_returns = np.minimum(excess_returns, 0)
            downside_std = float(np.sqrt(np.mean(np.square(downside_returns))))

            if downside_std > 0:
                result["sortino_ratio"] = float(np.mean(excess_returns) / downside_std * annualizer)

            result["yield_volatility"] = float(np.std(period_returns, ddof = 1) * annualizer)
            result["yield_downside_deviation"] = float(np.sqrt(np.mean(np.square(np.minimum(period_returns, 0)))) * annualizer)
            var95_raw = float(np.percentile(period_returns, 5))
            var99_raw = float(np.percentile(period_returns, 1))
            result["var95"] = float(var95_raw * 100.0)
            result["var99"] = float(var99_raw * 100.0)
            tail95 = period_returns[period_returns < var95_raw]
            tail99 = period_returns[period_returns < var99_raw]
            result["cvar95"] = float(tail95.mean() * 100.0) if len(tail95) > 0 else np.nan
            result["cvar99"] = float(tail99.mean() * 100.0) if len(tail99) > 0 else np.nan

        if benchmark_series is not None:

            bm = pd.to_numeric(pd.Series(benchmark_series), errors = "coerce")
            bm = bm.reindex(nav.index).dropna()

            if not bm.empty:

                bm_first = float(bm.iloc[0])
                bm_last = float(bm.iloc[-1])
                result["nav_total_bm"] = bm_last

                if bm_first != 0 and pd.notna(bm_first) and pd.notna(bm_last):

                    bm_return = bm_last / bm_first
                    result["nav_total_bm_return"] = float(bm_return)
                    result["nav_total_bm_rate"] = float((bm_return - 1.0) * 100.0)

                    if pd.notna(result["nav_return"]):
                        result["rel_return"] = float(result["nav_return"] - bm_return)

                    if pd.notna(result["nav_rate"]):
                        result["rel_return_rate"] = float(result["nav_rate"] - result["nav_total_bm_rate"])


        return result


    @staticmethod
    def report(performance_rec: Any) -> None:
        """
        ### What It Does
        Prints grouped summary report (return, risk, trade details) from one finished BacktestEngine run.

        #### Responsibility
        Standardizes human-readable diagnostics across optimization paths.

        #### How To Use
        Call after performance_metrics has populated report fields on engine.

        #### Key Parameters In Practice
        - `performance_rec`
          - Performance record table. Downstream protocol selection and reporting read objective columns from it.
          - Expected shape/type: `Any`.

        #### Usage Example
        `result = report(...)`

        ---

        ### Parameters
        - `performance_rec`: **Any**.
        """

        if isinstance(performance_rec, pd.DataFrame):

            if performance_rec.empty:
                print("[WARNING] Empty performance record frame.")

                return

            performance_rec = performance_rec.iloc[-1]

        def _value(field: str, default: Any = np.nan) -> Any:

            if isinstance(performance_rec, pd.Series):

                return performance_rec.get(field, default)

            if isinstance(performance_rec, dict):

                return performance_rec.get(field, default)

            return default


        def _safe_round(v: Any, ndigits: int = 4) -> Any:

            try:

                if v is None or bool(pd.isna(v)):

                    return "N/A"

                return round(float(v), ndigits)

            except Exception:

                return "N/A"


        def _safe_int(v: Any, default: int = 0) -> int:

            try:

                if v is None or bool(pd.isna(v)):

                    return int(default)

                return int(v)

            except Exception:

                return int(default)


        def _safe_pct(v: Any, ndigits: int = 4) -> Any:

            try:

                if v is None or bool(pd.isna(v)):

                    return "N/A"

                return round(float(v) * 100.0, ndigits)

            except Exception:

                return "N/A"


        print(
                "BACK-TEST PERFORMANCE", '\n',
                "---", '\n',
                "Net Asset Value:", _safe_round(_value("nav_total")), '\n',
                "Rate of Return:", _safe_round(_value("nav_rate")), "%", '\n',
                "Annualized Rate of Return:", _safe_round(_value("annualized_return_rate")), "%", '\n',
                "Strategy VS Benchmark:", "Benchmark:", _safe_round(_value("nav_total_bm_return")), "|", "Relative Return:", _safe_round(_value("rel_return")), _safe_round(_value("rel_return_rate")), "%", '\n',
                "Winning Rates:", _safe_round(_value("win_rate")), "%", '\n',
                "Sharpe Ratio:", _safe_round(_value("sharpe_ratio")), '\n',
                "Sortino Ratio:", _safe_round(_value("sortino_ratio")), '\n',
                '\n',
                "RISK", '\n',
                "---", '\n',
                "Maximum Drawdown:", _safe_round(_value("maxdd_rate")), "%", '\n',
                "Maximum Drawdown Period:", _safe_round(_value("maxdd_period")), "bars", '\n',
                "Yield Volatility:", _safe_round(_value("yield_volatility")), '\n',
                "Yield Downside Deviation:", _safe_round(_value("yield_downside_deviation")), '\n',
                "VaR 95%:", _safe_round(_value("var95")), "%", "|", "VaR 99%:", _safe_round(_value("var99")), "%", '\n',
                "Expected Shortfall (cVaR 95%):", _safe_round(_value("cvar95")), "%", "|", "Expected Shortfall (cVaR 99%)", _safe_round(_value("cvar99")), "%", '\n',
                '\n',
                "TRADES DETAILS", '\n',
                "---", '\n',
                "Profit / Loss Ratio:", _safe_round(_value("pl_ratio")), "%", '\n',
                "Open Trade Counts:", _safe_int(_value("open_trades", 0), 0), '\n',
                "Cover Trade Counts:", _safe_int(_value("cover_trades", 0), 0), '\n',
                "Average Holding Period:", _safe_round(_value("holding_period_h_mean"), 2), "Hours", "|", _safe_round(_value("holding_period_d_mean"), 2), "Days", '\n',
                "Median Holding Period:", _safe_round(_value("holding_period_h_median"), 2), "Hours", "|", _safe_round(_value("holding_period_d_median"), 2), "Days", '\n',
                "Maximum Holding Period:", _safe_round(_value("holding_period_h_max"), 2), "Hours", "|", _safe_round(_value("holding_period_d_max"), 2), "Days", '\n',
                "Minimum Holding Period:", _safe_round(_value("holding_period_h_min"), 2), "Hours", "|", _safe_round(_value("holding_period_d_min"), 2), "Days", '\n',
                "Maximum Drawdown Per Trade:", _safe_pct(_value("max_dd_trade")), "%", '\n',
                "Minimum Drawdown Per Trade:", _safe_pct(_value("min_dd_trade")), "%", '\n',
                "Maximum Profit Per Trade:", _safe_pct(_value("max_profit_trade")), "%", '\n',
                "Minimum Profit Per Trade:", _safe_pct(_value("min_profit_trade")), "%", '\n'
             )


    @classmethod
    def window_processing(cls,
                          *,
                          Data_Config: Dict[str, Any],
                          Factor_Config: Dict[str, Any],
                          factor_param_ranges: Dict[str, List[Any]],
                          Execution_Config: Dict[str, Any],

                          Signal_Config: Optional[Dict[str, Any]] = None,
                          Traversal_Config: Optional[Dict[str, Any]] = None,
                          payload: Optional[Dict[str, Any]] = None,
                          backtest_strategy: Optional[Callable[[int, "BacktestEngine_ForLoop", pd.DataFrame], Any]] = None,) -> Dict[str, Any]:
        """
        ### What It Does
        Runs the unified protocol-first workflow for both fixed-IS and rolling-OOS. It builds windows, runs the traversal search per window, stitches selected `weight_panel` slices, and executes one final replay.

        #### Responsibility
        Provides one canonical search controller for: - `test_months = None` fixed-IS - `test_months != None` rolling-OOS

        #### How To Use
        Use this for any hypertuning or OOS protocol. Runner facades forward here directly.

        #### Key Parameters In Practice
        - `backtest_strategy`
          - For-loop strategy callable. Supply it only when `vbt` is false or when a custom Python entry/exit path is required; VBT signal/weight paths normally do not need it.
          - Expected shape/type: `Optional[Callable[[int, "BacktestEngine_ForLoop", pd.DataFrame], Any]]`.
        - `Data_Config`
          - Top-level data-domain contract. Put dataset identity, prepared `test_data`, benchmark inputs, date bounds, and universe labels here; do not put signal rules or optimizer domains into it.
          - Expected shape/type: `Dict[str, Any]`.
        - `Factor_Config`
          - Factor-computation contract. Use it to pass `cal_column`, the active `FactorManager`/`FactorLibrary`, and factor-local constants that factor code actually consumes.
          - Expected shape/type: `Dict[str, Any]`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Dict[str, List[Any]]`.
        - `Execution_Config`
          - Execution-domain contract. Use it for capital, backend selection, runtime worker count, logging paths, and rolling protocol shape.
          - Expected shape/type: `Dict[str, Any]`.
        - `Signal_Config`
          - Signal-construction contract. Use it for SR thresholds, signal gates, signal artifact paths, temporal cover/prevent-open rules, and portfolio weighting rules.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `payload`
          - Runtime artifact container. Use it for precomputed signal/weight arrays, metadata paths, calendar/indexer/asset keys, and other objects that should stay outside config domains.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `Traversal_Config`
          - Traversal optimizer contract. Use it for the fixed-grid objective and traversal-specific selection settings.
          - Expected shape/type: `Optional[Dict[str, Any]]`.

        #### Usage Example
        `result = window_processing(...)`

        ---

        ### Parameters
        - `Data_Config`: **Dict[str, Any]**.
        - `Factor_Config`: **Dict[str, Any]**.
        - `factor_param_ranges`: **Dict[str, List[Any]]**.
        - `Execution_Config`: **Dict[str, Any]**.

        #### Optional Parameters
        - `Signal_Config`: **Optional[Dict[str, Any]]** = *None*.
        - `Traversal_Config`: **Optional[Dict[str, Any]]** = *None*.
        - `payload`: **Optional[Dict[str, Any]]** = *None*.
        - `backtest_strategy`: **Optional[Callable[[int, "BacktestEngine_ForLoop", pd.DataFrame], Any]]** = *None*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        from .VBTEngine import BacktestEngine_VBT  # deferred import: breaks circular dependency
        from .PortfolioEngine import PortfolioEngine  # deferred import: breaks circular dependency
        data_config = dict(Data_Config or {})
        factor_config = dict(Factor_Config or {})
        signal_config = dict(Signal_Config or {})
        execution_config = dict(Execution_Config or {})
        backtest_config = dict(execution_config.get("BacktestEngine", {}) or {})
        runtime_config = dict(execution_config.get("Runtime", {}) or {})
        protocol_config = dict(execution_config.get("Protocol", {}) or {})
        traversal_config = dict(Traversal_Config or {})

        runtime_payload = dict(payload or {})

        test_data = data_config.get("test_data")
        benchmark_series = data_config.get("benchmark_series")
        benchmark_name = data_config.get("benchmark_name")
        start_date = data_config.get("start_date")
        end_date = data_config.get("end_date")

        factor_manager = factor_config.get("factor_manager")
        cal_column = factor_config.get("cal_column", "Close")

        signal_z_window = signal_config.get("signal_z_window", 120)
        signal_start_t = signal_config.get("signal_start_t", 2)
        signal_cooldown = signal_config.get("signal_cooldown", 1)
        signal_epsilon = signal_config.get("signal_epsilon", 1e-9)
        signal_confirm_mode = signal_config.get("signal_confirm_mode", "mean")
        signal_gate_mode = signal_config.get("signal_gate_mode", "mean")
        signal_gate_style = signal_config.get("signal_gate_style", "soft")
        signal_gate_threshold = signal_config.get("signal_gate_threshold", 0.0)
        portfolio_config = signal_config.get("portfolio_config")

        initial_cash = backtest_config.get("initial_cash")
        risk_free_rate = backtest_config.get("risk_free_rate", 0.02)
        slippage = backtest_config.get("slippage", 0.0)
        spread = backtest_config.get("spread", 0.0)
        fees = backtest_config.get("fees", 0.0)
        vbt = backtest_config.get("vbt", True)
        independent_books = backtest_config.get("independent_books", False)

        testcycle_sec = runtime_config.get("testcycle_sec", 10.0)
        logical_processors = runtime_config.get("logical_processors", 1)
        log_queue = runtime_config.get("log_queue")
        log_path = str(runtime_config.get("log_path", "") or "")
        output_dir = str(runtime_config.get("output_dir", "") or "")

        optimizer_name = str(protocol_config.get("optimizer", "traversal") or "traversal").strip().lower() or "traversal"

        # Traversal is the only backend vendored here (the genetic backend was not
        # carried over). Reject anything else UP FRONT -- before the manager and the
        # worker pool are built -- rather than silently falling through to the
        # exhaustive grid: a protocol that asked for a sampled search and quietly got
        # the full Cartesian product burns hours of compute before anyone notices.
        if optimizer_name != "traversal":

            raise ValueError(f"[WARNING] unsupported optimizer {optimizer_name!r}; only 'traversal' is available.")

        train_months = protocol_config.get("train_months")
        test_months = protocol_config.get("test_months")
        step_months = protocol_config.get("step_months")
        combo_pool_size = protocol_config.get("combo_pool_size", 1)
        min_combo_size = protocol_config.get("min_combo_size", 1)
        max_combo_size = protocol_config.get("max_combo_size", 1)
        combo_weight_mode = protocol_config.get("combo_weight_mode", "equal")

        target = traversal_config.get("target", "sharpe_ratio")

        signal_strength = runtime_payload.get("signal_strength")
        weight = runtime_payload.get("weight")
        calendar = runtime_payload.get("calendar")
        indexer = runtime_payload.get("indexer")
        asset_keys = runtime_payload.get("asset_keys")

        if test_data is None or len(test_data) == 0:

            raise ValueError("[WARNING] window_processing requires non-empty test_data.")

        if factor_param_ranges is None:

            raise ValueError("[WARNING] window_processing requires factor_param_ranges.")

        try:
            initial_cash = float(cast(Any, initial_cash))

        except Exception as exc:

            raise ValueError("[WARNING] window_processing requires positive numeric initial_cash.") from exc

        if (not np.isfinite(initial_cash)) or initial_cash <= 0.0:

            raise ValueError("[WARNING] window_processing requires positive numeric initial_cash.")

        effective_test_data = pd.DataFrame(test_data).copy()
        dt_index = pd.DatetimeIndex(pd.to_datetime(effective_test_data["Datetime"] if "Datetime" in effective_test_data.columns else effective_test_data.index, errors = "coerce")).normalize()

        if start_date is None:
            start_date = str(pd.Timestamp(dt_index.min()).date())

        if end_date is None:
            end_date = str(pd.Timestamp(dt_index.max()).date())

        manager_data_config = dict(data_config)

        manager_data_config.update({
                                    "test_data": effective_test_data,
                                    "benchmark_series": benchmark_series,
                                    "benchmark_name": benchmark_name,
                                  })

        manager_factor_config = dict(factor_config)
        manager_factor_config.update({
                                      "cal_column": cal_column,
                                      "factor_manager": factor_manager,
                                     })

        manager_backtest_config = dict(backtest_config)

        manager_backtest_config.update({
                                        "initial_cash": initial_cash,
                                        "risk_free_rate": risk_free_rate,
                                        "slippage": slippage,
                                        "spread": spread,
                                        "fees": fees,
                                        "independent_books": independent_books,
                                        "vbt": vbt,
                                      })

        manager_execution_config = dict(execution_config)
        manager_execution_config["BacktestEngine"] = manager_backtest_config
        manager_signal_config = dict(signal_config)

        manager_signal_config.update({
                                      "signal_z_window": signal_z_window,
                                      "signal_start_t": signal_start_t,
                                      "signal_cooldown": signal_cooldown,
                                      "signal_epsilon": signal_epsilon,
                                      "signal_confirm_mode": signal_confirm_mode,
                                      "signal_gate_mode": signal_gate_mode,
                                      "signal_gate_style": signal_gate_style,
                                      "signal_gate_threshold": signal_gate_threshold,
                                      "portfolio_config": portfolio_config,
                                    })

        manager_payload = dict(runtime_payload)

        manager_payload.update({
                                "signal_strength": signal_strength,
                                "weight": weight,
                                "calendar": calendar,
                                "indexer": indexer,
                                "asset_keys": asset_keys,
                              })

        manager = cls(
                        Data_Config = manager_data_config,
                        Factor_Config = manager_factor_config,
                        Execution_Config = manager_execution_config,
                        Signal_Config = manager_signal_config,

                        payload = manager_payload,
                        backtest_strategy = backtest_strategy,
                        factor_param_ranges = factor_param_ranges,
                     )

        signal_strength = manager.signal_strength
        weight = manager.weight
        calendar = manager.calendar
        indexer = manager.indexer
        asset_keys = manager.asset_keys

        independent_books = cls._resolve_independent_books(independent_books = manager.independent_books)
        resolved_vbt = bool(manager.vbt if vbt is None else vbt)
        resolved_portfolio_config = dict(manager.portfolio_config if portfolio_config is None else (portfolio_config or {}))
        target_metric = str(target or "sharpe_ratio").strip() or "sharpe_ratio"
        optimizer_log_path = str(log_path).strip() if isinstance(log_path, str) else ""

        protocol_slice, windows, start_ts, end_ts = cls._build_protocol_windows(
                                                                                test_data = effective_test_data,
                                                                                start_date = str(start_date),
                                                                                end_date = str(end_date),
                                                                                train_months = train_months,
                                                                                test_months = test_months,
                                                                                step_months = step_months,
                                                                                )

        selection_rows: List[Dict[str, Any]] = []
        combo_metric_parts: List[pd.DataFrame] = []
        param_cycle_count = len(list(itertools.product(*factor_param_ranges.values())))
        train_eval_count = int(param_cycle_count * len(windows))
        final_replay_count = 1 if test_months is not None else 0
        protocol_step_count = int(len(windows) + final_replay_count)
        total_combinations_num = int(np.prod([len(param_range) for param_range in factor_param_ranges.values()], dtype = object))

        protocol_summary_lines = [
            "\n========================================================================================",
            f"Protocol Windows: {len(windows)}",
            f"Parameter Cycles Per Window: {param_cycle_count}",
            f"Train Evaluation Cycles: {train_eval_count}",
            f"Final Stitched Replay Cycles: {final_replay_count}",
            f"Protocol Progress Steps: {protocol_step_count}",
            "",
            "HyperTuning Parameter Space:",
            *[f"{param_key}: {param_range} | Count: {len(param_range)}" for param_key, param_range in factor_param_ranges.items()],
            "",
            "BacktestEngine Parameters:",
            f"initial_cash: {initial_cash}",
            f"risk_free_rate: {risk_free_rate}",
            f"slippage: {slippage}",
            f"spread: {spread}",
            f"fees: {fees}",
            f"logical_processors: {logical_processors}",
            f"signal_z_window: {signal_z_window}",
            f"signal_start_t: {signal_start_t}",
            f"signal_cooldown: {signal_cooldown}",
            f"signal_epsilon: {signal_epsilon}",
            f"signal_confirm_mode: {signal_confirm_mode}",
            f"signal_gate_mode: {signal_gate_mode}",
            f"signal_gate_style: {signal_gate_style}",
            f"signal_gate_threshold: {signal_gate_threshold}",
            "",
            "Optimizer Parameters:",
            f"optimizer: {optimizer_name}",
            f"target: {target_metric}",
            f"total_combinations: {total_combinations_num}",
            f"total_train_cycles: {train_eval_count}",
            f"estimated_eta_hours: {round((train_eval_count * testcycle_sec / max(1, logical_processors)) / 3600, 2)}",
            "========================================================================================\n",
        ]

        for protocol_summary_line in protocol_summary_lines:

            print(protocol_summary_line)
            logging.info(protocol_summary_line)

        progress_bar = Progress(
                                    TextColumn("[bold]{task.description}"),
                                    BarColumn(bar_width = None, style = "#36454F", complete_style = "#EF5350", finished_style = "#26A69A"),
                                    MofNCompleteColumn(),
                                    TimeElapsedColumn(),
                                    TimeRemainingColumn(),
                                    )

        protocol_task = progress_bar.add_task("OOS PROTOCOL PROGRESS", total = protocol_step_count)
        progress_bar.start()

        def progress_reporter(message: str) -> None:
            """
            ### What It Does
            Handles protocol reporter behavior for `the module`.

            #### Responsibility
            Centralizes the validation, alignment, and dispatch rules used by `progress_reporter`.

            #### How To Use
            Call `progress_reporter(...)` from the workflow path that needs protocol reporter output.

            #### Key Parameters In Practice
            - `message`
              - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
              - Expected shape/type: `str`.

            #### Usage Example
            `result = progress_reporter(...)`

            ---

            ### Parameters
            - `message`: **str**.
            """

            progress_bar.print(message)


        # Persistent worker pool: build ONE pool for the whole walk-forward instead of
        # respawning per window (each respawn re-pays the Windows-spawn vectorbt/numba
        # cold import ~5s, which is what makes small grids slower than serial). Each
        # window's test_data reaches the living workers via a shared proxy plus a
        # per-window token refresh.
        traversal_executor = None
        traversal_window_state = None
        pool_manager = None

        if int(logical_processors) > 1:
            pool_manager = multiprocessing.Manager()

            if log_queue is None:
                log_queue = pool_manager.Queue()

            shared_window_state = pool_manager.dict()
            traversal_window_state = shared_window_state
            traversal_executor = ProcessPoolExecutor(
                                                        max_workers = int(logical_processors),
                                                        initializer = trav_init_worker_persistent,
                                                        initargs = ((log_queue, shared_window_state),),
                                                    )

        # Crash recovery for the persistent pool. A worker dying hard (segfault /
        # OOM-kill / native crash in numba/BLAS) breaks the whole ProcessPoolExecutor;
        # with one pool shared across every window, that would otherwise take down all
        # remaining windows (traversal aborts the run at the next submit). The old
        # per-window ephemeral pools contained a crash to
        # a single window, so we restore that isolation: on BrokenExecutor rebuild the
        # pool once and retry the same window; if it breaks again, skip that window
        # (loud-logged) and continue. pool_manager / shared_window_state / log_queue
        # survive a worker crash (the Manager is a separate process), so only the
        # executor is rebuilt.
        _POOL_DEAD = object()

        def _rebuild_pool() -> None:
            nonlocal traversal_executor

            try:
                if traversal_executor is not None:
                    traversal_executor.shutdown(wait = False, cancel_futures = True)

            except Exception:
                pass

            traversal_executor = ProcessPoolExecutor(
                                                        max_workers = int(logical_processors),
                                                        initializer = trav_init_worker_persistent,
                                                        initargs = ((log_queue, shared_window_state),),
                                                    )

        def _call_with_recovery(window_token: int, window_id: Any, thunk: Callable[[], Any]) -> Any:
            # Run one window's traversal backend, recovering from a hard worker crash:
            # attempt 0 = original run; on BrokenExecutor rebuild the pool and retry once;
            # a second BrokenExecutor returns _POOL_DEAD so the caller skips this window.
            # (When logical_processors <= 1 there is no pool, so BrokenExecutor cannot be
            # raised and the thunk simply runs once.)
            for _attempt in range(2):

                try:
                    return thunk()

                except BrokenExecutor:

                    if _attempt == 0 and int(logical_processors) > 1:
                        crash_msg = (f"[WARNING] window_processing: worker pool crashed on window {window_token} "
                                     f"(window_id={window_id}); rebuilding pool and retrying this window once.")
                        sys.stderr.write(crash_msg + "\n")
                        sys.stderr.flush()
                        logging.warning(crash_msg)
                        _rebuild_pool()

                        continue

                    skip_msg = (f"[WARNING] window_processing: worker pool crashed AGAIN on window {window_token} "
                                f"(window_id={window_id}) after rebuild+retry; SKIPPING this window and continuing.")
                    sys.stderr.write(skip_msg + "\n")
                    sys.stderr.flush()
                    logging.warning(skip_msg)

                    return _POOL_DEAD

            return _POOL_DEAD

        for window_token, window in enumerate(windows):
            train_slice = pd.DataFrame(window["train_df"]).copy()

            trav_result = _call_with_recovery(window_token, window.get("window_id"), lambda: manager._traversal_backend(
                                                            test_data = train_slice,
                                                            initial_cash = initial_cash,
                                                            backtest_strategy = backtest_strategy,
                                                            factor_param_ranges = factor_param_ranges,
                                                            risk_free_rate = risk_free_rate,
                                                            slippage = slippage,
                                                            spread = spread,
                                                            fees = fees,
                                                            benchmark_series = benchmark_series,
                                                            benchmark_name = benchmark_name,
                                                            signal_z_window = signal_z_window,
                                                            signal_start_t = signal_start_t,
                                                            signal_cooldown = signal_cooldown,
                                                            signal_epsilon = signal_epsilon,
                                                            signal_confirm_mode = signal_confirm_mode,
                                                            signal_gate_mode = signal_gate_mode,
                                                            signal_gate_style = signal_gate_style,
                                                            signal_gate_threshold = signal_gate_threshold,
                                                            signal_strength = signal_strength,
                                                            weight = weight,
                                                            factor_manager = factor_manager,
                                                            calendar = calendar,
                                                            indexer = indexer,
                                                            asset_keys = asset_keys,
                                                            independent_books = independent_books,
                                                            cal_column = cal_column,
                                                            portfolio_config = resolved_portfolio_config,
                                                            vbt = resolved_vbt,
                                                            logical_processors = logical_processors,
                                                            log_queue = log_queue,
                                                            log_path = log_path,
                                                            terminal_reporter = progress_reporter,
                                                            traversal_executor = traversal_executor,
                                                            traversal_window_state = traversal_window_state,
                                                            traversal_token = window_token,
                                                          ))

            if trav_result is _POOL_DEAD:

                continue

            nav_rec, perf_rec = trav_result

            if int(combo_pool_size) == 1 and int(min_combo_size) == 1 and int(max_combo_size) == 1:
                best_params, best_metric_map, optimizer_trace_df = cls._extract_best_params(pd.DataFrame(perf_rec), target_metric = target_metric)

                if len(best_params) == 0:

                    continue

                member_params_list = [best_params]
                weights = [1.0]

            else:
                combo_payload = manager.combo_processing(
                                                        hypertuning_nav_rec = nav_rec,
                                                        performance_rec = perf_rec,
                                                        test_data = train_slice,
                                                        factor_param_ranges = factor_param_ranges,
                                                        combo_pool_size = combo_pool_size,
                                                        min_combo_size = min_combo_size,
                                                        max_combo_size = max_combo_size,
                                                        target = target_metric,
                                                        combo_weight_mode = combo_weight_mode,
                                                        logical_processors = logical_processors,
                                                        independent_books = independent_books,
                                                        portfolio_config = resolved_portfolio_config,
                                                        vbt = resolved_vbt,
                                                        log_path = log_path,
                                                        )

                combo_perf_rec = pd.DataFrame(combo_payload.get("combo_performance_rec", pd.DataFrame())).copy()

                if not combo_perf_rec.empty:

                    best_combo = combo_perf_rec.iloc[0].to_dict()
                    raw_member_param_jsons = json.loads(str(best_combo.get("member_param_jsons", "[]")))
                    member_params_list = []

                    for raw_json in raw_member_param_jsons:
                        parsed: Any = raw_json

                        if isinstance(parsed, str):
                            parsed = json.loads(parsed)

                        if isinstance(parsed, dict):
                            member_params_list.append(cls._normalize_protocol_param_dict(parsed))

                    raw_weights = json.loads(str(best_combo.get("weights_json", "[]")))
                    weights = [float(x) for x in raw_weights] if isinstance(raw_weights, list) else []

                    if len(member_params_list) == 0:

                        continue

                    if len(weights) != len(member_params_list):
                        weights = [1.0 / float(len(member_params_list))] * len(member_params_list)

                    else:
                        weight_sum = float(np.sum(weights))
                        weights = [float(x) / weight_sum for x in weights] if weight_sum > 0.0 else [1.0 / float(len(member_params_list))] * len(member_params_list)

                    best_metric_map = best_combo
                    optimizer_trace_df = combo_perf_rec

                else:
                    best_params, best_metric_map, optimizer_trace_df = cls._extract_best_params(
                        pd.DataFrame(perf_rec),
                        target_metric = target_metric,
                    )

                    if len(best_params) == 0:

                        continue

                    member_params_list = [best_params]
                    weights = [1.0]

            if len(member_params_list) == 0:

                continue

            combo_score = BacktestEngineManager._target_metric_value(cast(Dict[str, Any], best_metric_map), target_metric)
            combo_rank_score = BacktestEngineManager._target_metric_rank_value(target_metric, combo_score)
            combo_label_parts = [
                                f"M{member_idx}:{','.join(f'{key}={param_dict[key]}' for key in param_dict.keys())}"
                                for member_idx, param_dict in enumerate(member_params_list, start = 1)
                                ]
            selected_combo = str(best_metric_map.get("combo_name", " | ".join(combo_label_parts) if combo_label_parts else "combo_empty"))
            test_start_ts = start_ts if window.get("test_start") is None else pd.Timestamp(window["test_start"]).normalize()
            test_end_ts = end_ts if window.get("test_end") is None else pd.Timestamp(window["test_end"]).normalize()

            if not optimizer_trace_df.empty:

                trace_df = optimizer_trace_df.copy()
                trace_df["window_id"] = str(window["window_id"])
                combo_metric_parts.append(trace_df)

            selection_rows.append(
                                    {
                                    "window_id": str(window["window_id"]),
                                    "train_start": str(pd.Timestamp(window["train_start"]).date()),
                                    "train_end": str(pd.Timestamp(window["train_end"]).date()),
                                    "test_start": str(test_start_ts.date()),
                                    "test_end": str(test_end_ts.date()),
                                    "selected_combo": selected_combo,
                                    "member_count": int(len(member_params_list)),
                                    "member_params_list": member_params_list,
                                    "member_params_jsons": json.dumps([json.dumps(param_dict, ensure_ascii = True, default = str) for param_dict in member_params_list], ensure_ascii = True),
                                    "weights": [float(x) for x in weights],
                                    "weights_json": json.dumps([float(x) for x in weights], ensure_ascii = True),
                                    "target": target_metric,
                                    "combo_score": combo_score,
                                    "combo_rank_score": combo_rank_score,
                                    }
                                 )

            # Progress Bar Checkpoint
            progress_bar.advance(protocol_task)

        # Tear down the persistent worker pool now the window loop is complete.
        # (window_processing is normally driven from a runner subprocess, so a mid-loop
        # exception is reaped by that subprocess' teardown; this covers the normal path
        # and the empty-selection error path just below.)
        if traversal_executor is not None:
            traversal_executor.shutdown()

        if pool_manager is not None:
            pool_manager.shutdown()

        if len(selection_rows) == 0:

            progress_bar.stop()

            raise RuntimeError("[WARNING] window_processing produced no valid selector windows.")

        oos_slice = protocol_slice.copy()
        oos_index = pd.DatetimeIndex(pd.to_datetime(oos_slice["Datetime"] if "Datetime" in oos_slice.columns else oos_slice.index, errors = "coerce"))
        selected_weight_panel, weight_combo_daily, weight_window_daily = manager._stitch_selected_panel(values = manager.weight, selection_rows = selection_rows, oos_slice = oos_slice, oos_index = oos_index)
        selected_signal_strength_panel, signal_strength_combo_daily, signal_strength_window_daily = manager._stitch_selected_panel(values = manager.signal_strength, selection_rows = selection_rows, oos_slice = oos_slice, oos_index = oos_index)
        selected_combo_daily = weight_combo_daily if selected_weight_panel is not None else signal_strength_combo_daily
        selected_window_daily = weight_window_daily if selected_weight_panel is not None else signal_strength_window_daily

        if selected_weight_panel is None and selected_signal_strength_panel is None:

            raise RuntimeError("[WARNING] window_processing requires signal_strength or weight to build stitched final replay.")

        replay_weight_panel = selected_weight_panel
        replay_signal_strength_panel = None if replay_weight_panel is not None else selected_signal_strength_panel

        if not resolved_vbt and replay_weight_panel is not None and replay_signal_strength_panel is None:

            raise ValueError("[WARNING] non-vbt final replay requires stitched signal-strength artifacts, not weight-only artifacts.")

        reference_panel = replay_weight_panel if replay_weight_panel is not None else replay_signal_strength_panel
        assert reference_panel is not None
        replay_asset_keys = [str(col) for col in reference_panel.columns]
        replay_factor_param_ranges = {"__selected__": [0]}
        portfolio_payload: Dict[str, Any] = {}
        final_position_panel = pd.DataFrame()

        if (not independent_books) and resolved_vbt:
            direct_manager = cls(
                                Data_Config = {
                                                'test_data': oos_slice,
                                                'benchmark_series': benchmark_series,
                                                'benchmark_name': benchmark_name,
                                              },
                                Factor_Config = {'cal_column': cal_column},
                                factor_param_ranges = replay_factor_param_ranges,
                                backtest_strategy = backtest_strategy,
                                Execution_Config = {
                                                    'BacktestEngine': {
                                                                        'initial_cash': initial_cash,
                                                                        'risk_free_rate': risk_free_rate,
                                                                        'slippage': slippage,
                                                                        'spread': spread,
                                                                        'fees': fees,
                                                                        'independent_books': independent_books,
                                                                        'vbt': resolved_vbt,
                                                                      },
                                                    },
                                Signal_Config = {
                                                'signal_z_window': signal_z_window,
                                                'signal_start_t': signal_start_t,
                                                'signal_cooldown': signal_cooldown,
                                                'signal_epsilon': signal_epsilon,
                                                'signal_confirm_mode': signal_confirm_mode,
                                                'signal_gate_mode': signal_gate_mode,
                                                'signal_gate_style': signal_gate_style,
                                                'signal_gate_threshold': signal_gate_threshold,
                                                'asset_keys': replay_asset_keys,
                                                'portfolio_config': resolved_portfolio_config,
                                                },
                                )

            effective_replay_weight_panel = replay_weight_panel

            if effective_replay_weight_panel is None:

                assert replay_signal_strength_panel is not None

                effective_replay_weight_panel = direct_manager._build_position().compute_weight_panel(test_data = oos_slice, signal_strength_panel = replay_signal_strength_panel, weight_config = resolved_portfolio_config)

            portfolio_inputs = PortfolioEngine.portfolio_inputs(test_data = oos_slice, weight_panel = cast(pd.DataFrame, effective_replay_weight_panel), cal_column = cal_column, portfolio_frames = None, portfolio_config = (resolved_portfolio_config if replay_weight_panel is not None else None))

            portfolio_payload = PortfolioEngine.portfolio_bt(
                                                                btengine = cast(BacktestEngine_VBT, direct_manager._resolve_btengine(True)),
                                                                weight_panel = portfolio_inputs["weight_panel"],
                                                                open_panel = portfolio_inputs["open_panel"],
                                                                close_panel = portfolio_inputs["close_panel"],
                                                                benchmark_series = benchmark_series,
                                                                risk_free_rate = risk_free_rate,
                                                                force_flat_mask = portfolio_inputs.get("force_flat_mask"),
                                                                prevent_open_mask = portfolio_inputs.get("prevent_open_mask"),
                                                                max_gross_exposure = resolved_portfolio_config.get("max_gross_exposure"),
                                                                close_delisted_at_last = bool(resolved_portfolio_config.get("close_delisted_at_last", False)),
                                                              )

            final_nav_series = pd.Series(pd.to_numeric(portfolio_payload["nav"], errors = "coerce"), index = portfolio_payload["nav"].index, dtype = float).sort_index()
            nav_metric_map = portfolio_payload["metric_map"]
            underlying_metrics = portfolio_payload["underlying_metrics"]
            final_position_panel = pd.DataFrame(portfolio_payload["position_panel"]).copy()

            final_perf_rec = direct_manager.performance_rec_df(
                                                                cycle_name = "cycle_1|selected",
                                                                factor_param_ranges = {"__selected__": 0},
                                                                test_cycle = 1,
                                                                metric_map = nav_metric_map,
                                                                extra_columns = {
                                                                                "active_days": int(final_nav_series.pct_change().fillna(0.0).abs().gt(1e-15).sum()),
                                                                                "portfolio_underlying_count": len(underlying_metrics),
                                                                                },
                                                              )

        else:
            replay_manager = cls(
                                Data_Config = {
                                                'test_data': oos_slice,
                                                'benchmark_series': benchmark_series,
                                                'benchmark_name': benchmark_name,
                                              },
                                Factor_Config = {'cal_column': cal_column},
                                factor_param_ranges = replay_factor_param_ranges,
                                backtest_strategy = backtest_strategy,
                                Execution_Config = {
                                                    'BacktestEngine': {
                                                                        'initial_cash': initial_cash,
                                                                        'risk_free_rate': risk_free_rate,
                                                                        'slippage': slippage,
                                                                        'spread': spread,
                                                                        'fees': fees,
                                                                        'independent_books': independent_books,
                                                                        'vbt': resolved_vbt,
                                                                      },
                                                    },
                                Signal_Config = {
                                                'signal_z_window': signal_z_window,
                                                'signal_start_t': signal_start_t,
                                                'signal_cooldown': signal_cooldown,
                                                'signal_epsilon': signal_epsilon,
                                                'signal_confirm_mode': signal_confirm_mode,
                                                'signal_gate_mode': signal_gate_mode,
                                                'signal_gate_style': signal_gate_style,
                                                'signal_gate_threshold': signal_gate_threshold,
                                                'portfolio_config': resolved_portfolio_config,
                                                },
                                payload = {
                                                'signal_strength': None if replay_signal_strength_panel is None else np.asarray(replay_signal_strength_panel.to_numpy(dtype = np.float32, copy = True)[np.newaxis, :, :], dtype = np.float32),
                                                'weight': None if replay_weight_panel is None else np.asarray(replay_weight_panel.to_numpy(dtype = np.float32, copy = True)[np.newaxis, :, :], dtype = np.float32),
                                                'calendar': oos_index,
                                                'asset_keys': replay_asset_keys,
                                          },
                                )

            final_nav, _, _, final_perf_rec = replay_manager.test_run(factor_param_ranges = {"__selected__": 0}, test_cycle = 1)
            final_nav_series = pd.Series(pd.to_numeric(final_nav, errors = "coerce"), index = final_nav.index, dtype = float).sort_index()

        daily_frame = pd.DataFrame({"nav": final_nav_series, "ret": final_nav_series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)})
        daily_frame["window_id"] = selected_window_daily.reindex(daily_frame.index)
        daily_frame["selected_combo"] = selected_combo_daily.reindex(daily_frame.index)

        if daily_frame.index.duplicated().any():

            raise ValueError("[WARNING] window_processing generated overlapping OOS daily rows.")

        position_panel = final_position_panel if "final_position_panel" in locals() else pd.DataFrame()

        if not position_panel.empty:
            trade_col = "trade_date"

            if trade_col not in position_panel.columns:
                for candidate_col in ("Datetime", "datetime", "date"):

                    if candidate_col in position_panel.columns:
                        position_panel = position_panel.rename(columns = {candidate_col: trade_col})

                        break

            if trade_col in position_panel.columns:

                trade_dates = pd.to_datetime(position_panel[trade_col], errors = "coerce")
                combo_map = selected_combo_daily.to_dict()
                window_map = selected_window_daily.to_dict()
                normalized_combo_map = pd.Series(selected_combo_daily.to_numpy(dtype = object), index = pd.DatetimeIndex(selected_combo_daily.index).normalize(), dtype = "object").to_dict()
                normalized_window_map = pd.Series(selected_window_daily.to_numpy(dtype = object), index = pd.DatetimeIndex(selected_window_daily.index).normalize(), dtype = "object").to_dict()

                position_panel["window_id"] = trade_dates.map(window_map)
                position_panel["member_label"] = trade_dates.map(combo_map)
                position_panel["selected_combo"] = trade_dates.map(combo_map)
                missing_window = position_panel["window_id"].isna()

                if bool(missing_window.any()):
                    normalized_trade_dates = trade_dates.dt.normalize()
                    position_panel.loc[missing_window, "window_id"] = normalized_trade_dates.loc[missing_window].map(normalized_window_map)
                    position_panel.loc[missing_window, "member_label"] = normalized_trade_dates.loc[missing_window].map(normalized_combo_map)
                    position_panel.loc[missing_window, "selected_combo"] = normalized_trade_dates.loc[missing_window].map(normalized_combo_map)

        selection_df = pd.DataFrame(
                                    [
                                        {
                                            "window_id": row["window_id"],
                                            "train_start": row["train_start"],
                                            "train_end": row["train_end"],
                                            "test_start": row["test_start"],
                                            "test_end": row["test_end"],
                                            "selected_combo": row["selected_combo"],
                                            "member_count": row["member_count"],
                                            "member_params_jsons": row["member_params_jsons"],
                                            "weights_json": row["weights_json"],
                                            "target": row["target"],
                                            "combo_score": row["combo_score"],
                                            "combo_rank_score": row["combo_rank_score"],
                                        }
                                        for row in selection_rows
                                    ]
                                   )

        metric_rows: List[Dict[str, Any]] = []

        if not daily_frame.empty and len(selection_rows) > 0:

            for row in selection_rows:

                test_start = pd.Timestamp(row["test_start"]).normalize()
                test_end = pd.Timestamp(row["test_end"])

                if test_end == test_end.normalize():
                    test_end = test_end + pd.Timedelta(days = 1) - pd.Timedelta(nanoseconds = 1)

                window_slice = daily_frame.loc[(daily_frame.index >= test_start) & (daily_frame.index <= test_end)].copy()

                if window_slice.empty:

                    continue

                metric_map = BacktestEngineManager.performance_metrics(nav_series = pd.Series(pd.to_numeric(window_slice["nav"], errors = "coerce"), index = window_slice.index, dtype = float), risk_free_rate = risk_free_rate,)
                metric_rows.append(
                                    {
                                        "window_id": row.get("window_id"),
                                        "train_start": row.get("train_start"),
                                        "train_end": row.get("train_end"),
                                        "test_start": row.get("test_start"),
                                        "test_end": row.get("test_end"),
                                        "selected_combo": row.get("selected_combo"),
                                        "member_count": row.get("member_count", np.nan),
                                        "train_combo_score": row.get("combo_score", np.nan),
                                        "train_combo_rank_score": row.get("combo_rank_score", np.nan),
                                        "oos_nav_rate": metric_map.get("nav_rate", np.nan),
                                        "oos_sharpe_ratio": metric_map.get("sharpe_ratio", np.nan),
                                        "oos_maxdd_rate": metric_map.get("maxdd_rate", np.nan),
                                    }
                                  )

        window_metrics_df = pd.DataFrame(metric_rows)
        combo_metrics_df = pd.concat(combo_metric_parts, ignore_index = True) if combo_metric_parts else pd.DataFrame()
        optimizer_audit_df = combo_metrics_df.copy()

        if not optimizer_audit_df.empty:

            selected_context_cols = [
                                    col
                                    for col in [
                                                "window_id",
                                                "train_start",
                                                "train_end",
                                                "test_start",
                                                "test_end",
                                                "selected_combo",
                                                "member_count",
                                                "member_params_jsons",
                                                "weights_json",
                                                "target",
                                                "combo_score",
                                                "combo_rank_score",
                                                ]

                                    if col in selection_df.columns
                                    ]

            optimizer_audit_df = optimizer_audit_df.merge(selection_df[selected_context_cols], on = "window_id", how = "left", suffixes = ("", "_selected"))
            candidate_label = (
                                optimizer_audit_df["combo_name"]

                                if "combo_name" in optimizer_audit_df.columns

                                else optimizer_audit_df["cycle_name"] if "cycle_name" in optimizer_audit_df.columns

                                else pd.Series("", index = optimizer_audit_df.index, dtype = "object")
                              )

            label_selected = candidate_label.astype(str).eq(optimizer_audit_df["selected_combo"].astype(str))

            param_selected = (
                                optimizer_audit_df.apply(lambda row: str(row.get("factor_param_json", "")) in str(row.get("member_params_jsons", "")), axis = 1)

                                if "factor_param_json" in optimizer_audit_df.columns and "member_params_jsons" in optimizer_audit_df.columns

                                else pd.Series(False, index = optimizer_audit_df.index)
                             )

            optimizer_audit_df["selected"] = label_selected | param_selected

            if not window_metrics_df.empty and "window_id" in window_metrics_df.columns:
                oos_cols = [
                            col
                            for col in [
                                        "window_id",
                                        "oos_nav_rate",
                                        "oos_sharpe_ratio",
                                        "oos_maxdd_rate",
                                       ]

                            if col in window_metrics_df.columns
                            ]

                selected_oos = window_metrics_df[oos_cols].drop_duplicates(subset = ["window_id"], keep = "last")
                optimizer_audit_df = optimizer_audit_df.merge(selected_oos, on = "window_id", how = "left")

                for col in ["oos_nav_rate", "oos_sharpe_ratio", "oos_maxdd_rate"]:

                    if col in optimizer_audit_df.columns:
                        optimizer_audit_df[col] = optimizer_audit_df[col].where(optimizer_audit_df["selected"], np.nan)

        elif not selection_df.empty:

            optimizer_audit_df = selection_df.copy()
            optimizer_audit_df["selected"] = True

            if not window_metrics_df.empty and "window_id" in window_metrics_df.columns:

                oos_cols = [
                            col
                            for col in [
                                        "window_id",
                                        "oos_nav_rate",
                                        "oos_sharpe_ratio",
                                        "oos_maxdd_rate",
                                       ]

                            if col in window_metrics_df.columns
                            ]

                optimizer_audit_df = optimizer_audit_df.merge(window_metrics_df[oos_cols].drop_duplicates(subset = ["window_id"], keep = "last"), on = "window_id", how = "left")

        member_return_frame = (pd.DataFrame(portfolio_payload.get("member_return_frame", pd.DataFrame())).copy() if (not independent_books) and resolved_vbt else pd.DataFrame())

        # Progress Bar Checkpoint
        if final_replay_count:
            progress_bar.advance(protocol_task)

        progress_bar.stop()

        if output_dir:

            out_dir = pathlib.Path(output_dir)
            out_dir.mkdir(parents = True, exist_ok = True)
            daily_frame.to_csv(out_dir / "final_nav_stitched.csv", index = True)
            position_panel.to_csv(out_dir / "final_position_stitched.csv", index = False)
            member_return_frame.to_csv(out_dir / "assets_detailed.csv", index = True)

        if optimizer_log_path and not optimizer_audit_df.empty:

            pathlib.Path(optimizer_log_path).parent.mkdir(parents = True, exist_ok = True)
            optimizer_audit_df.to_csv(optimizer_log_path, index = False)

            logging.info(f"OPTIMIZER AUDIT RECORDS: {optimizer_log_path}")

        result_payload: Dict[str, Any] = {
                                        "daily_frame": daily_frame,
                                        "selection_df": selection_df,
                                        "window_metrics_df": window_metrics_df,
                                        "combo_metrics_df": combo_metrics_df,
                                        "optimizer_audit_df": optimizer_audit_df,
                                        "member_return_frame": member_return_frame,
                                        "position_panel": position_panel,
                                        "performance_rec": pd.DataFrame(final_perf_rec),
                                        }


        return result_payload


    def __init__(self,
                *,
                Data_Config: Dict[str, Any],
                Factor_Config: Dict[str, Any],
                Execution_Config: Dict[str, Any],
                factor_param_ranges: Dict[str, List[Any]],

                Signal_Config: Optional[Dict[str, Any]] = None,
                payload: Optional[Dict[str, Any]] = None,
                backtest_strategy: Optional[Callable[[int, "BacktestEngine_ForLoop", pd.DataFrame], Any]] = None) -> None:

        from .ForLoopEngine import BacktestEngine_ForLoop  # deferred import: breaks circular dependency
        from .VBTEngine import BacktestEngine_VBT  # deferred import: breaks circular dependency
        self.Data_Config = dict(Data_Config or {})
        self.Factor_Config = dict(Factor_Config or {})
        self.Execution_Config = dict(Execution_Config or {})
        self.Signal_Config = dict(Signal_Config or {})
        self.payload = dict(payload or {})
        self._owned_memmaps: List[np.memmap] = []  # only handles THIS manager opens (see _load_cached_artifact); closed in close()
        loaded_indexers: List[Signal.ParamIndexer] = []
        loaded_assets: List[List[str]] = []
        loaded_calendars: List[pd.DatetimeIndex] = []

        def _load_cached_artifact(meta_path_raw: str | pathlib.Path) -> Tuple[np.memmap, Signal.ParamIndexer, List[str], pd.DatetimeIndex]:

            meta_path = pathlib.Path(meta_path_raw)

            if not meta_path.exists():

                raise FileNotFoundError(f"[WARNING] Configured artifact metadata path missing......{meta_path}")

            with meta_path.open("r", encoding = "utf-8") as f:
                meta = json.load(f)

            artifact_path_raw = meta.get("artifact_path")

            if not artifact_path_raw:

                raise ValueError(f"[WARNING] Artifact metadata missing artifact_path: {meta_path}")

            artifact_path = pathlib.Path(artifact_path_raw)

            if not artifact_path.is_absolute():
                artifact_path = (meta_path.parent / artifact_path).resolve()

            if not artifact_path.exists():

                raise FileNotFoundError(f"[WARNING] Artifact mmap file missing......{artifact_path}")

            shape_raw = meta.get("shape")

            if not shape_raw or len(shape_raw) != 3:

                raise ValueError(f"[WARNING] Invalid shape in artifact metadata: {meta_path}")

            shape = (int(shape_raw[0]), int(shape_raw[1]), int(shape_raw[2]))
            dtype = str(meta.get("dtype", "float32"))
            param_keys = list(meta.get("param_keys", []))
            param_values = [list(v) for v in meta.get("param_values", [])]
            artifact_assets = [str(v) for v in meta.get("asset_keys", [])]

            if len(param_keys) != len(param_values):

                raise ValueError(f"[WARNING] Artifact metadata param axis mismatch: {meta_path}")

            combo_total = 1

            for values in param_values:

                if len(values) == 0:

                    raise ValueError(f"[WARNING] Artifact metadata contains empty parameter axis: {meta_path}")

                combo_total *= len(values)

            if int(shape[0]) != int(combo_total):

                raise ValueError(f"[WARNING] Artifact combo axis mismatch: shape[0]={shape[0]} param_total={combo_total} meta={meta_path}")

            if len(artifact_assets) != int(shape[2]):

                raise ValueError(f"[WARNING] Artifact asset axis mismatch: shape[2]={shape[2]} asset_keys={len(artifact_assets)} meta={meta_path}")

            calendar_raw = meta.get("calendar")

            if not calendar_raw:

                raise ValueError(f"[WARNING] Artifact metadata missing row-axis calendar: {meta_path}")

            artifact_calendar = pd.DatetimeIndex(pd.to_datetime(calendar_raw, errors = "coerce"))

            if artifact_calendar.tz is not None:
                # Reloaded ISO calendars can carry a tz offset while test_data is tz-naive; a tz-aware vs
                # tz-naive date join matches ZERO rows -> weights silently all-zero (flat NAV, no error).
                # Drop the tz so the join stays on wall-clock dates.
                artifact_calendar = artifact_calendar.tz_localize(None)

            if len(artifact_calendar) != shape[1] or bool(pd.isna(artifact_calendar).any()):

                raise ValueError(f"[WARNING] Artifact metadata calendar does not match row axis: {meta_path}")

            artifact_indexer = Signal.ParamIndexer(param_keys = param_keys, param_values = param_values, asset_keys = None)

            if int(artifact_indexer.total) != int(shape[0]):

                raise ValueError(f"[WARNING] Artifact indexer total mismatch: indexer.total={artifact_indexer.total} shape[0]={shape[0]} meta={meta_path}")

            return np.memmap(artifact_path, dtype = dtype, mode = "r", shape = shape), artifact_indexer, artifact_assets, artifact_calendar


        for artifact_key, meta_key in (("signal_strength", "signal_strength_meta_path"), ("weight", "weight_meta_path")):

            if self.payload.get(artifact_key) is not None or not self.payload.get(meta_key):

                continue

            artifact, artifact_indexer, artifact_assets, artifact_calendar = _load_cached_artifact(self.payload[meta_key])
            self.payload[artifact_key] = artifact
            self._owned_memmaps.append(artifact)
            loaded_indexers.append(artifact_indexer)
            loaded_assets.append(artifact_assets)
            loaded_calendars.append(artifact_calendar)

        if self.payload.get("signal_strengths") is None and isinstance(self.payload.get("signal_strength_meta_paths"), dict):
            split_signal_strengths: Dict[str, np.memmap] = {}

            for signal_key, meta_path in dict(self.payload["signal_strength_meta_paths"]).items():
                artifact, artifact_indexer, artifact_assets, artifact_calendar = _load_cached_artifact(meta_path)
                split_signal_strengths[str(signal_key)] = artifact
                self._owned_memmaps.append(artifact)
                loaded_indexers.append(artifact_indexer)
                loaded_assets.append(artifact_assets)
                loaded_calendars.append(artifact_calendar)

            if split_signal_strengths:
                self.payload["signal_strengths"] = split_signal_strengths

        if len(loaded_indexers) > 1:
            left = loaded_indexers[0]

            for right in loaded_indexers[1:]:
                same_indexer = (
                                list(left.param_keys) == list(right.param_keys)
                                and [list(v) for v in left.param_values] == [list(v) for v in right.param_values]
                                and list(left.asset_keys or []) == list(right.asset_keys or [])
                               )

                if not same_indexer:

                    raise ValueError("[WARNING] signal_strength and weight metadata indexers do not match.")

        if len(loaded_assets) > 1:
            first_assets = loaded_assets[0]

            for current_assets in loaded_assets[1:]:

                if first_assets != current_assets:

                    raise ValueError("[WARNING] signal_strength and weight metadata asset_keys do not match.")

        if len(loaded_calendars) > 1:
            first_calendar = loaded_calendars[0]

            for current_calendar in loaded_calendars[1:]:

                if not first_calendar.equals(current_calendar):

                    raise ValueError("[WARNING] signal_strength and weight metadata calendars do not match.")

        if self.payload.get("indexer") is None and loaded_indexers:
            self.payload["indexer"] = loaded_indexers[0]

        if self.payload.get("asset_keys") is None and loaded_assets:
            self.payload["asset_keys"] = loaded_assets[0]

        if self.payload.get("calendar") is None and loaded_calendars:
            self.payload["calendar"] = loaded_calendars[0]

        self.BacktestEngine_Config = dict(self.Execution_Config.get("BacktestEngine", {}) or {})
        self.test_data = self.Data_Config.get("test_data")

        if self.test_data is None or len(self.test_data) == 0:

            raise ValueError("[WARNING] BacktestEngineManager requires non-empty Data_Config['test_data'].")

        try:
            self.initial_cash = float(cast(Any, self.BacktestEngine_Config.get("initial_cash")))

        except Exception as exc:

            raise ValueError("[WARNING] BacktestEngineManager requires positive numeric initial_cash.") from exc

        if (not np.isfinite(self.initial_cash)) or self.initial_cash <= 0.0:

            raise ValueError("[WARNING] BacktestEngineManager requires positive numeric initial_cash.")

        self.backtest_strategy = _strategy_resolve_callable(backtest_strategy) if backtest_strategy is not None else None
        self.factor_param_ranges = factor_param_ranges or {}

        self.benchmark_series = self.Data_Config.get("benchmark_series")
        self.benchmark_name = self.Data_Config.get("benchmark_name")

        self.risk_free_rate = float(self.BacktestEngine_Config.get("risk_free_rate", 0.02))
        self.slippage = float(self.BacktestEngine_Config.get("slippage", 0.0))
        self.spread = float(self.BacktestEngine_Config.get("spread", 0.0))
        self.fees = float(self.BacktestEngine_Config.get("fees", 0.0))
        self.independent_books = BacktestEngineManager._resolve_independent_books(independent_books = self.BacktestEngine_Config.get("independent_books", False))
        self.vbt = bool(self.BacktestEngine_Config.get("vbt", True))

        self.signal_z_window = int(self.Signal_Config.get("signal_z_window", 120))
        self.signal_start_t = int(self.Signal_Config.get("signal_start_t", 2))
        self.signal_cooldown = int(self.Signal_Config.get("signal_cooldown", 1))
        self.signal_epsilon = float(self.Signal_Config.get("signal_epsilon", 1e-9))
        self.signal_confirm_mode = self.Signal_Config.get("signal_confirm_mode", "mean")
        self.signal_gate_mode = self.Signal_Config.get("signal_gate_mode", "mean")
        self.signal_gate_style = self.Signal_Config.get("signal_gate_style", "soft")
        self.signal_gate_threshold = float(self.Signal_Config.get("signal_gate_threshold", 0.0))

        self.portfolio_config = dict(self.Signal_Config.get("portfolio_config") or {})
        self.force_cover_holidays = self.portfolio_config.get("force_cover_holidays", False)

        if self.force_cover_holidays is True:
            self.force_cover_holidays = default_holidays()

        elif self.force_cover_holidays is False or self.force_cover_holidays is None:
            self.force_cover_holidays = []

        self.cal_column = BacktestEngineManager._resolve_cal_column(cal_column = self.Factor_Config.get("cal_column", "Close"))
        self.factor_manager = self.Factor_Config.get("factor_manager") if isinstance(self.Factor_Config.get("factor_manager"), FactorEngine.FactorManager) else FactorEngine.FactorManager.get_attached_store(self.test_data)

        self.signal_strength = self.payload.get("signal_strength")
        self.weight = self.payload.get("weight")
        raw_calendar = self.payload.get("calendar")
        self.calendar = None if raw_calendar is None else pd.DatetimeIndex(pd.to_datetime(raw_calendar, errors = "coerce"))
        raw_asset_keys = self.payload.get("asset_keys")
        self.asset_keys = [str(asset).strip() for asset in raw_asset_keys] if raw_asset_keys is not None else None

        self.param_keys = list(self.factor_param_ranges.keys())

        indexer = self.payload.get("indexer")

        if indexer is not None:
            self.indexer = indexer

            cube_asset_keys = getattr(indexer, "asset_keys", None)

            if self.asset_keys is None and cube_asset_keys is not None:
                self.asset_keys = [str(asset).strip() for asset in cube_asset_keys if str(asset).strip()]

        elif (self.signal_strength is not None or self.weight is not None) and self.factor_param_ranges:

            param_keys = list(self.factor_param_ranges.keys())
            param_values: List[List[Any]] = []

            for values in self.factor_param_ranges.values():

                if isinstance(values, range):
                    param_values.append(list(values))

                elif isinstance(values, np.ndarray):
                    param_values.append(np.asarray(values).tolist())

                elif isinstance(values, (list, tuple, set)):
                    param_values.append(list(values))

                else:
                    param_values.append([values])

            asset_keys: Optional[List[str]] = None
            raw_underlyings = None
            attrs = getattr(self.test_data, "attrs", None)

            if isinstance(attrs, dict):
                raw_underlyings = attrs.get("__zora_ctx_underlyings__")

            if isinstance(raw_underlyings, (list, tuple, set)):
                cleaned_underlyings = [str(u).strip() for u in raw_underlyings if str(u).strip()]

                if len(cleaned_underlyings) > 1:
                    asset_keys = list(dict.fromkeys(cleaned_underlyings))

            if asset_keys is None:

                if self.asset_keys is not None:
                    cleaned_asset_keys = [str(asset).strip() for asset in self.asset_keys if str(asset).strip()]

                    if len(cleaned_asset_keys) > 0:
                        asset_keys = list(dict.fromkeys(cleaned_asset_keys))

            self.indexer = Signal.ParamIndexer(param_keys = param_keys, param_values = param_values, asset_keys = asset_keys)

        else:
            self.indexer = None

        self.engine_kwargs_forloop: Dict[str, Any] = dict(
                                            initial_cash = self.initial_cash,
                                            test_data = self.test_data,
                                            risk_free_rate = self.risk_free_rate,
                                            holidays = self.force_cover_holidays,
                                            slippage = self.slippage,
                                            spread = self.spread,
                                            fees = self.fees,
                                            benchmark_series = self.benchmark_series,
                                            signal_start_t = self.signal_start_t,
                                        )

        self.engine_kwargs_vbt: Dict[str, Any] = dict(
                                        initial_cash = self.initial_cash,
                                        test_data = self.test_data,
                                        slippage = self.slippage,
                                        spread = self.spread,
                                        fees = self.fees,
                                        signal_start_t = self.signal_start_t,
                                    )

        self.btengine_forloop: Optional[BacktestEngine_ForLoop] = None
        self.btengine_vbt: Optional[BacktestEngine_VBT] = None
        self.btengine: Optional[Any] = None


    def close(self) -> None:
        """
        Release the disk-backed artifact handles this manager OPENED itself. _load_cached_artifact
        (mode='r', from payload meta-paths) is the only site that opens memmaps the manager owns; they
        stay open for the whole run so the combo loop can re-slice them, and are freed here once the
        performance record is materialized -- otherwise on Windows the files stay locked (WinError 32)
        against recompute and temp cleanup. Artifacts passed into the payload as live objects are the
        CALLER's to release: a manager must never close a handle it merely borrows and may share with
        other managers (e.g. best_mgr borrows the sweep's signal_strength/weight). Idempotent; called by
        __exit__ / __del__. Do NOT call between combos of one run -- the payload is reused across it.
        """
        for handle in getattr(self, "_owned_memmaps", ()):
            Signal._close_memmap(handle)

        self._owned_memmaps = []


    def __enter__(self) -> "BacktestEngineManager":

        return self


    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> None:

        self.close()


    def __del__(self) -> None:

        try:
            self.close()

        except Exception:
            pass


    def _resolve_btengine(self, use_vbt: bool) -> Any:

        from .ForLoopEngine import BacktestEngine_ForLoop  # deferred import: breaks circular dependency
        from .VBTEngine import BacktestEngine_VBT  # deferred import: breaks circular dependency
        if use_vbt:

            if self.btengine_vbt is None:
                self.btengine_vbt = BacktestEngine_VBT(**self.engine_kwargs_vbt)

            return self.btengine_vbt

        if self.btengine_forloop is None:
            self.btengine_forloop = BacktestEngine_ForLoop(**self.engine_kwargs_forloop)


        return self.btengine_forloop


    def _build_position(self) -> Position:

        return Position(

                                    signal_z_window = self.signal_z_window,
                                    signal_start_t = self.signal_start_t,
                                    signal_epsilon = self.signal_epsilon,
                                    )


    def _resolve_vbt_weight_series(self,
                                   *,
                                   test_data: pd.DataFrame,
                                   weight_series: Optional[pd.Series] = None,
                                   weight_col: str = "weights",
                                   signal_strength_col: Optional[np.ndarray] = None,
                                   position: Optional[Position] = None,
                                   weight_config: Optional[Dict[str, Any]] = None) -> pd.Series:


        if weight_series is not None:
            resolved = BacktestEngineManager._align_series_to_test_data(weight_series, test_data)

            if not resolved.isna().all():

                return resolved.fillna(0.0).astype(float)

        if isinstance(weight_col, str) and weight_col in test_data.columns:
            existing_weights = pd.to_numeric(test_data[weight_col], errors = "coerce")

            if not existing_weights.isna().all():

                return existing_weights.fillna(0.0).astype(float)

        if signal_strength_col is not None:

            position = position if isinstance(position, Position) else self._build_position()
            signal_strength_panel = pd.DataFrame(
                {
                    "SINGLE": pd.Series(
                        pd.to_numeric(signal_strength_col, errors = "coerce"),
                        index = pd.DatetimeIndex(
                            pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce")
                        ),
                        dtype = float,
                    )
                }
            )
            weight_panel = position.compute_weight_panel(
                test_data = test_data,
                signal_strength_panel = signal_strength_panel,
                weight_config = weight_config,
            )

            return pd.Series(weight_panel.iloc[:, 0].to_numpy(dtype = float), index = test_data.index, name = "weights", dtype = float)

        raise ValueError("[WARNING] vbt=True requires one of: weight_series, test_data['weights'], or explicit signal_strength_col input.")


    def _slice_panel(self,
                    *,
                    test_data: pd.DataFrame,
                    values: np.ndarray | np.memmap,
                    factor_param_ranges: Dict[str, Any]) -> pd.DataFrame:

        from .PortfolioEngine import PortfolioEngine  # deferred import: breaks circular dependency
        manager_test_data = self.test_data

        if manager_test_data is None:

            raise ValueError("[WARNING] Artifact slicing requires manager test_data.")

        if self.indexer is None:

            raise ValueError("[WARNING] artifact slicing requires indexer.")

        calendar = self.calendar
        asset_keys = self.asset_keys

        if calendar is None:
            dt_source = manager_test_data["Datetime"] if "Datetime" in manager_test_data.columns else manager_test_data.index
            calendar = pd.DatetimeIndex(pd.to_datetime(dt_source, errors = "coerce"))

        if asset_keys is None:

            if isinstance(self.factor_manager, FactorEngine.FactorManager):
                asset_keys = list(self.factor_manager.asset_keys)

            else:
                asset_keys = list(PortfolioEngine.split_portfolio_test_data(manager_test_data, cal_column = self.cal_column).keys())
                asset_keys = [str(asset) for asset in asset_keys]

        asset_key_list = [str(asset) for asset in asset_keys]

        if len(asset_key_list) == 0:

            raise ValueError("[WARNING] artifact contract requires non-empty asset_keys.")


        return Signal.slice_panel(

                                                test_data = test_data,
                                                values = values,
                                                indexer = self.indexer,
                                                factor_param_ranges = factor_param_ranges,
                                                calendar = calendar,
                                                asset_keys = asset_key_list,
                                                )


    def _stitch_selected_panel(self,
                                *,
                                values: Optional[np.ndarray],
                                selection_rows: List[Dict[str, Any]],
                                oos_slice: pd.DataFrame,
                                oos_index: pd.DatetimeIndex) -> Tuple[Optional[pd.DataFrame], pd.Series, pd.Series]:

        selected_combo_daily = pd.Series(pd.NA, index = oos_index, dtype = "object")
        selected_window_daily = pd.Series(pd.NA, index = oos_index, dtype = "object")

        if values is None:

            return None, selected_combo_daily, selected_window_daily

        panel_parts: List[pd.DataFrame] = []

        for row in selection_rows:
            test_start = pd.Timestamp(row["test_start"]).normalize()
            test_end = pd.Timestamp(row["test_end"])

            if test_end == test_end.normalize():
                test_end = test_end + pd.Timedelta(days = 1) - pd.Timedelta(nanoseconds = 1)

            window_mask = (oos_index >= test_start) & (oos_index <= test_end)
            window_slice = oos_slice.loc[window_mask].copy()

            if window_slice.empty:

                continue

            panel_specs: List[Tuple[pd.DataFrame, float]] = []

            for member_params, weight in zip(list(row["member_params_list"]), list(row["weights"])):
                sliced_panel = self._slice_panel(
                    test_data = window_slice,
                    values = values,
                    factor_param_ranges = member_params,
                ).astype(float)
                panel_specs.append((sliced_panel, float(weight)))

            if len(panel_specs) == 0:

                continue

            columns = self.asset_keys if self.asset_keys is not None else list(panel_specs[0][0].columns)
            window_panel = BacktestEngineManager._weighted_panel_sum(panel_specs, index = pd.DatetimeIndex(pd.to_datetime(window_slice["Datetime"] if "Datetime" in window_slice.columns else window_slice.index, errors = "coerce")), columns = list(columns))

            if window_panel.empty:

                continue

            panel_parts.append(window_panel)
            selected_combo_daily.loc[window_panel.index] = str(row["selected_combo"])
            selected_window_daily.loc[window_panel.index] = str(row["window_id"])

        if len(panel_parts) == 0:

            return None, selected_combo_daily, selected_window_daily

        stitched_panel = pd.concat(panel_parts, axis = 0).sort_index()

        if stitched_panel.index.duplicated().any():

            raise ValueError("[WARNING] window_processing generated overlapping selected artifact rows.")

        stitched_panel = stitched_panel.reindex(oos_index).fillna(0.0).astype(float)


        return stitched_panel, selected_combo_daily, selected_window_daily


    def _traversal_backend(self,
                            # Data_Config
                            test_data: Optional[pd.DataFrame],
                            benchmark_series: Optional[pd.Series] = None,
                            benchmark_name: Optional[str] = None,

                            # Factor_Config
                            factor_param_ranges: Optional[Dict[str, List[Any]]] = None,
                            cal_column: Any = "Close",
                            factor_manager: Optional[FactorEngine.FactorManager] = None,

                            # Execution_Config.BacktestEngine
                            initial_cash: float = 0.0,
                            backtest_strategy: Optional[Callable[[int, "BacktestEngine_ForLoop", pd.DataFrame], Any]] = None,
                            risk_free_rate: float = 0.02,
                            slippage: float = 0.0,
                            spread: float = 0.0,
                            fees: float = 0.0,
                            independent_books: bool = False,
                            portfolio_config: Optional[Dict[str, Any]] = None,
                            vbt: bool = True,

                            # Execution_Config.Runtime
                            logical_processors: int = 1,
                            log_queue: Optional[Any] = None,
                            log_path: str = "",
                            terminal_reporter: Optional[Callable[[str], Any]] = None,

                            # Signal_Config
                            signal_z_window: int = 120,
                            signal_start_t: int = 2,
                            signal_cooldown: int = 1,
                            signal_epsilon: float = 1e-9,
                            signal_confirm_mode: str = "mean",
                            signal_gate_mode: str = "mean",
                            signal_gate_style: str = "soft",
                            signal_gate_threshold: float = 0.0,
                            signal_strength: Optional[np.ndarray] = None,
                            weight: Optional[np.ndarray] = None,
                            calendar: Optional[pd.Index] = None,
                            indexer: Optional[Signal.ParamIndexer] = None,
                            asset_keys: Optional[List[str]] = None,

                            # Persistent traversal pool (owned by window_processing across the
                            # walk-forward window loop). When traversal_executor is provided the
                            # pool is reused and this window's test_data is delivered to living
                            # workers via traversal_window_state["args"] + traversal_token; when
                            # None the method falls back to an ephemeral per-call pool.
                            traversal_executor: Optional[Any] = None,
                            traversal_window_state: Optional[Any] = None,
                            traversal_token: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:

        resolved_cal_column = BacktestEngineManager._resolve_cal_column(cal_column = cal_column)
        independent_books = BacktestEngineManager._resolve_independent_books(independent_books = independent_books)

        backend_portfolio_config = dict(portfolio_config or {})
        holidays = backend_portfolio_config.get("force_cover_holidays", False)

        if holidays is True:
            holidays = default_holidays()

        elif holidays is False or holidays is None:
            holidays = []

        if test_data is None:

            raise ValueError("[WARNING] `test_data` is required.")

        if factor_param_ranges is None:

            raise ValueError("[WARNING] `factor_param_ranges` is required.")

        if (not vbt) and backtest_strategy is None:

            raise ValueError("[WARNING] `backtest_strategy` is required.")

        effective_test_data = test_data
        _tmp_log_manager = None

        owns_log_queue = log_queue is None

        if log_queue is None:

            if logical_processors <= 1:
                log_queue = queue.Queue()

            else:
                _tmp_log_manager = multiprocessing.Manager()
                log_queue = cast(Any, _tmp_log_manager.Queue())

        param_keys = list(factor_param_ranges.keys())
        factor_param_combinations = itertools.product(*factor_param_ranges.values())

        future_to_combo = {}
        idx_source = effective_test_data["Datetime"] if "Datetime" in effective_test_data.columns else effective_test_data.index
        hypertuning_nav_rec = pd.DataFrame(index = pd.to_datetime(idx_source, errors = "coerce"), columns = [])
        performance_rec = pd.DataFrame()
        nav_log_path = ""
        log_path_text = str(log_path).strip() if isinstance(log_path, str) else ""
        write_log_enabled = bool(log_path_text)

        if write_log_enabled:
            raw_log_path = log_path_text

            if raw_log_path.lower().endswith(".csv"):
                nav_log_path = f"{raw_log_path[:-4]}_nav.csv"

            else:
                nav_log_path = f"{raw_log_path}_nav.csv"

        else:
            warn_msg = "[WARNING] log_path is empty; traversal performance CSV will be skipped."
            print(warn_msg)
            logging.warning(warn_msg)

        strategy_transport = _strategy_pack_for_transport(backtest_strategy)

        worker_signal_config = {
                                'signal_z_window': signal_z_window,
                                'signal_start_t': signal_start_t,
                                'signal_cooldown': signal_cooldown,
                                'signal_epsilon': signal_epsilon,
                                'signal_confirm_mode': signal_confirm_mode,
                                'signal_gate_mode': signal_gate_mode,
                                'signal_gate_style': signal_gate_style,
                                'signal_gate_threshold': signal_gate_threshold,
                                'signal_strength': signal_strength,
                                'weight': weight,
                                'calendar': calendar,
                                'indexer': indexer,
                                'asset_keys': asset_keys,
                                'portfolio_config': backend_portfolio_config,
                                'signal_strength_meta_path': self.payload.get("signal_strength_meta_path"),
                                'signal_strength_meta_paths': self.payload.get("signal_strength_meta_paths"),
                                'weight_meta_path': self.payload.get("weight_meta_path"),
                               }

        if logical_processors > 1 and (worker_signal_config.get("signal_strength_meta_path") or worker_signal_config.get("signal_strength_meta_paths") or worker_signal_config.get("weight_meta_path")):

            worker_signal_config["signal_strength"] = None
            worker_signal_config["weight"] = None
            worker_signal_config["calendar"] = None
            worker_signal_config["indexer"] = None
            worker_signal_config["asset_keys"] = None

        worker_payload = {
                            key: worker_signal_config.get(key)
                            for key in (
                                        "signal_strength",
                                        "weight",
                                        "calendar",
                                        "indexer",
                                        "asset_keys",
                                        "signal_strength_meta_path",
                                        "signal_strength_meta_paths",
                                        "weight_meta_path",
                                       )

                            if worker_signal_config.get(key) is not None
                         }

        init_btmgr_args = {
                            'backtest_strategy': strategy_transport,
                            'Data_Config': {
                                            'test_data': effective_test_data,
                                            'holidays': holidays,
                                            'benchmark_series': benchmark_series,
                                            'benchmark_name': benchmark_name,
                                            },
                            'Factor_Config': {
                                                'cal_column': resolved_cal_column,
                                                'factor_manager': factor_manager,
                                             },
                            'factor_param_ranges': factor_param_ranges,
                            'Execution_Config': {
                                                'BacktestEngine': {
                                                    'initial_cash': initial_cash,
                                                    'risk_free_rate': risk_free_rate,
                                                    'slippage': slippage,
                                                    'spread': spread,
                                                    'fees': fees,
                                                    'independent_books': independent_books,
                                                    'vbt': vbt,
                                                },
                                                },
                            'Signal_Config': worker_signal_config,
                            'payload': worker_payload,
                          }

        logging.info(f"HyperTuning Initializing Succeeded.\n")

        reporter: Optional[Callable[[str], None]] = cast(Callable[[str], None], terminal_reporter) if callable(terminal_reporter) else None

        if True:
            BacktestEngineManager._btmgr = None

            def _apply_result(result: Any, combo_ctx: Tuple[Any, ...]) -> None:

                nonlocal performance_rec

                if result is None:

                    sys.stderr.write(f"[WARNING] HyperTuning WARNING: No result returned for {combo_ctx}\n")
                    sys.stderr.flush()

                    logging.warning(f"[WARNING] HyperTuning WARNING: No result returned for {combo_ctx}\n")

                    return

                current_nav_series, result_factor_param_ranges, result_column_name, performance_rec_df = result

                if not isinstance(performance_rec_df, pd.DataFrame) or performance_rec_df.empty:

                    sys.stderr.write(f"[WARNING] HyperTuning WARNING: Empty performance record for {combo_ctx}\n")
                    sys.stderr.flush()

                    logging.warning(f"[WARNING] HyperTuning WARNING: Empty performance record for {combo_ctx}\n")

                    return

                try:
                    nav_rate = performance_rec_df['nav_rate'].iloc[-1]
                    sharpe_ratio = performance_rec_df['sharpe_ratio'].iloc[-1]

                except Exception as e:
                    sys.stderr.write(f"[WARNING] HyperTuning Results Parsing ERROR: {combo_ctx}\n")
                    sys.stderr.write(f"Exception: {e.__class__.__name__}: {e}\n")
                    traceback.print_exc(file = sys.stderr)
                    sys.stderr.flush()

                    logging.error(f"[WARNING] HyperTuning Results Parsing ERROR: {combo_ctx}: {e}\n", exc_info = True)

                    return

                logging.info(f"For: {result_factor_param_ranges} in {result_column_name} cycle, Performance: {nav_rate}, Sharpe: {sharpe_ratio}\n")

                if reporter is not None:
                    reporter(f"For: {result_factor_param_ranges} in {result_column_name} cycle, Performance: {nav_rate}, Sharpe: {sharpe_ratio}\n")

                hypertuning_nav_rec[str(result_column_name)] = current_nav_series
                performance_rec = pd.concat([performance_rec, performance_rec_df], axis = 0)


            # Fallback to serial execution when strategy functions are notebook-defined
            # and cannot be pickled under Windows spawn multiprocessing.

            if logical_processors <= 1:

                logging.info("[TRAV] Evaluate mode: Serial")

                if BacktestEngineManager._btmgr is None:
                    trav_init_worker((log_queue, init_btmgr_args))

                for idx, combo in enumerate(factor_param_combinations):
                    test_cycle = idx
                    combo_ctx = tuple(combo)

                    logging.info(f"Starting Computation Cycle: {test_cycle}......")

                    if reporter is not None:
                        reporter(f"Starting Computation Cycle: {test_cycle}......")

                    try:
                        result = trav_worker(combo_ctx, test_cycle)
                        _apply_result(result, combo_ctx)

                    except Exception as e:

                        if _is_signature_mismatch_type_error(e):

                            raise RuntimeError(
                                                "[WARNING] backtest_strategy signature mismatch. "
                                                "Expected compatible form: (t, btengine, test_data)."
                                              ) from e

                        sys.stderr.write(f"[WARNING] Serial HyperTuning FAILED for {combo_ctx}\n")
                        sys.stderr.write(f"Exception: {e.__class__.__name__}: {e}\n")
                        traceback.print_exc(file = sys.stderr)
                        sys.stderr.flush()

                        logging.error(f"[WARNING] Serial HyperTuning FAILED for {combo_ctx}: {e}\n", exc_info = True)

                        continue

            else:

                if traversal_executor is not None:
                    # Persistent pool (owned by window_processing): reused across every
                    # walk-forward window. Publish this window's init_btmgr_args to the
                    # shared proxy, then submit with a per-window token so each living
                    # worker rebuilds its _btmgr exactly once for this window. This ships
                    # the SAME per-window test_data the baked initializer would, so OOS
                    # stays bit-identical; only the delivery path changes. NOT torn down
                    # here (window_processing owns its lifecycle).
                    traversal_window_state["args"] = init_btmgr_args
                    executor = traversal_executor
                    submit_token = traversal_token
                    owns_pool = False

                else:
                    # Ephemeral per-call pool (standalone / combo_processing fallback): the
                    # initializer bakes _btmgr and submit_token stays None so workers use
                    # that baked _btmgr with no refresh.
                    executor = ProcessPoolExecutor(max_workers = logical_processors, initializer = trav_init_worker, initargs = ((log_queue, init_btmgr_args),))
                    submit_token = None
                    owns_pool = True

                try:

                    for idx, combo in enumerate(factor_param_combinations):

                        test_cycle = idx
                        combo_ctx = tuple(combo)

                        logging.info(f"Starting Computation Cycle: {test_cycle}......")

                        if reporter is not None:
                            reporter(f"Starting Computation Cycle: {test_cycle}......")

                        try:
                            # Submit Task for Each Combination
                            future = executor.submit(trav_worker, combo_ctx, test_cycle, submit_token)

                        except Exception as e:
                            sys.stderr.write(f"[WARNING] Parallel HyperTuning WARNING for {combo_ctx}\n")
                            sys.stderr.write(f"Exception: {e.__class__.__name__}: {e}\n")
                            traceback.print_exc(file = sys.stderr)
                            sys.stderr.flush()

                            logging.warning(f"[WARNING] Parallel HyperTuning WARNING for {combo_ctx}: {e}\n", exc_info = True)

                            raise

                        # Log Result for Each Calculation
                        future_to_combo[future] = combo_ctx

                    for future in as_completed(future_to_combo):

                        combo_ctx = future_to_combo[future]

                        try:
                            # Unpack Tuning Results
                            result = future.result()
                            _apply_result(result, combo_ctx)

                        except BrokenExecutor:
                            # A worker died hard (segfault / OOM-kill / native crash) -> the
                            # persistent pool is broken, NOT a per-combo error. Propagate so
                            # window_processing rebuilds the pool and retries this window,
                            # instead of swallowing every remaining combo and then aborting
                            # the whole walk-forward at the next window's submit.
                            raise

                        except Exception as e:

                            if _is_signature_mismatch_type_error(e):

                                raise RuntimeError(
                                                    "[WARNING] backtest_strategy signature mismatch. "
                                                    "Expected compatible form: (t, btengine, test_data)."
                                                  ) from e

                            sys.stderr.write(f"[WARNING] HyperTuning Results Unpacking ERROR: {combo_ctx}\n")
                            sys.stderr.write(f"Exception: {e.__class__.__name__}: {e}\n")
                            traceback.print_exc(file = sys.stderr)
                            sys.stderr.flush()

                            logging.error(f"[WARNING] HyperTuning Results Unpacking ERROR: {combo_ctx}: {e}\n", exc_info = True)

                            continue

                finally:
                    if owns_pool:
                        executor.shutdown()

        print(f"\n========================================================================================\n")

        if performance_rec.empty:

            warn_msg = "[WARNING] No valid traversal results were produced."
            print(warn_msg)
            logging.warning(warn_msg)

            performance_rec = pd.DataFrame(
                                            columns = [
                                                        "nav_rate",
                                                        "rel_return_rate",
                                                        "maxdd_rate",
                                                        "sharpe_ratio",
                                                        "sortino_ratio",
                                                      ]
                                          )

            if write_log_enabled:
                performance_rec.to_csv(log_path_text, index = False)

                logging.info(f"PERFORMANCE RECORDS: {log_path_text}")

            else:
                logging.warning("[WARNING] PERFORMANCE RECORDS WRITE SKIPPED (empty log_path).")

            if nav_log_path:

                nav_out_df = hypertuning_nav_rec.copy()
                nav_out_df = nav_out_df.reindex(
                                                columns = sorted(
                                                                nav_out_df.columns,
                                                                key = lambda col: (
                                                                                    int(match.group(1))

                                                                                    if (match := re.search(r"cycle_(\d+)", str(col))) is not None

                                                                                    else 10**12,
                                                                                    str(col),
                                                                                   ),
                                                               )
                                               )
                nav_out_df.index.name = "Datetime"
                nav_out_df.to_csv(nav_log_path, index = True)

                logging.info(f"NAV RECORDS: {nav_log_path}")

            if _tmp_log_manager is not None:

                try:
                    shutdown = getattr(_tmp_log_manager, "shutdown", None)

                    if callable(shutdown):
                        shutdown()

                except Exception:
                    pass

            if owns_log_queue and log_queue is not None and hasattr(log_queue, "close"):
                root_logger = logging.getLogger()

                for handler in list(getattr(root_logger, "handlers", [])):

                    if getattr(handler, "queue", None) is log_queue:

                        try:
                            root_logger.removeHandler(handler)

                        except Exception:
                            pass

                        try:
                            handler.close()

                        except Exception:
                            pass

                try:
                    log_queue.close()

                except Exception:
                    pass

                try:
                    log_queue.join_thread()

                except Exception:
                    pass

            raise RuntimeError(
                                "[WARNING] Traversal failed: no valid cycles were produced. "
                                "Please check worker error logs and backtest_strategy signature compatibility."
                              )

        print(f"BEST NAV: {performance_rec['nav_rate'].max()}")
        print(f"BEST SHARPE: {performance_rec['sharpe_ratio'].max()}")
        print(f"BEST SORTINO: {performance_rec['sortino_ratio'].max()}")

        logging.info("\n========================================================================================\n")

        logging.info(f"BEST NAV: {performance_rec['nav_rate'].max()}")
        logging.info(f"BEST SHARPE: {performance_rec['sharpe_ratio'].max()}")
        logging.info(f"BEST SORTINO: {performance_rec['sortino_ratio'].max()}")

        logging.info("\n========================================================================================\n")

        factor_params_raw = performance_rec["factor_param_ranges"].tolist()
        factor_rows: List[List[Any]] = []
        max_param_len = 0

        for params in factor_params_raw:

            if isinstance(params, dict):
                row = [params.get(k, np.nan) for k in param_keys]

            elif isinstance(params, (list, tuple, np.ndarray)):
                row = list(params)

            else:
                row = [params]

            max_param_len = max(max_param_len, len(row))
            factor_rows.append(row)

        if max_param_len > 0:

            padded_rows = [row + [np.nan] * (max_param_len - len(row)) for row in factor_rows]
            factor_cols = list(param_keys[:max_param_len]) if param_keys and len(param_keys) >= max_param_len else [f"p{i}" for i in range(max_param_len)]
            factor_params_df = pd.DataFrame(padded_rows, index = performance_rec.index, columns = factor_cols)
            performance_rec = pd.concat([performance_rec.drop(columns = ["factor_param_ranges"]), factor_params_df], axis = 1)

        else:
            performance_rec = performance_rec.drop(columns = ["factor_param_ranges"])

        if write_log_enabled:
            performance_rec.to_csv(log_path_text, index = False)

            logging.info(f"PERFORMANCE RECORDS: {log_path_text}")

        else:
            logging.warning("[WARNING] PERFORMANCE RECORDS WRITE SKIPPED (empty log_path).")

        if nav_log_path:
            nav_out_df = hypertuning_nav_rec.copy()
            nav_out_df = nav_out_df.reindex(
                                            columns = sorted(
                                                            nav_out_df.columns,
                                                            key = lambda col: (
                                                                                int(match.group(1))

                                                                                if (match := re.search(r"cycle_(\d+)", str(col))) is not None

                                                                                else 10**12,
                                                                                str(col),
                                                                               ),
                                                           )
                                           )
            nav_out_df.index.name = "Datetime"
            nav_out_df.to_csv(nav_log_path, index = True)

            logging.info(f"NAV RECORDS: {nav_log_path}")

        if _tmp_log_manager is not None:

            try:
                shutdown = getattr(_tmp_log_manager, "shutdown", None)

                if callable(shutdown):
                    shutdown()

            except Exception:
                pass

        if owns_log_queue and log_queue is not None and hasattr(log_queue, "close"):
            root_logger = logging.getLogger()

            for handler in list(getattr(root_logger, "handlers", [])):

                if getattr(handler, "queue", None) is log_queue:

                    try:
                        root_logger.removeHandler(handler)

                    except Exception:
                        pass

                    try:
                        handler.close()

                    except Exception:
                        pass

            try:
                log_queue.close()

            except Exception:
                pass

            try:
                log_queue.join_thread()

            except Exception:
                pass


        return hypertuning_nav_rec, performance_rec


    def performance_rec_df(self,
                           cycle_name: str,
                           factor_param_ranges: Optional[Dict[str, Any]],
                           test_cycle: int,
                           metric_map: Dict[str, float],
                           extra_columns: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        ### What It Does
        Builds a one-row performance record DataFrame.

        #### Responsibility
        Normalizes metric dictionaries and selected parameter metadata into a tabular report row.

        #### How To Use
        Call it when appending a test result to CSV or an in-memory report table.

        #### Key Parameters In Practice
        - `cycle_name`
          - Evaluation label. Use it to identify single, train, OOS, combo, or replay cycles in logs and result rows.
          - Expected shape/type: `str`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `test_cycle`
          - Cycle input payload. It represents the concrete data slice and parameter selection being evaluated.
          - Expected shape/type: `int`.
        - `metric_map`
          - Mapping from metric names to report values. Use it to control what appears in summary output.
          - Expected shape/type: `Dict[str, float]`.
        - `extra_columns`
          - Additional columns to preserve. Use it when downstream reporting needs fields beyond the core price/signal columns.
          - Expected shape/type: `Optional[Dict[str, Any]]`.

        #### Usage Example
        `result = performance_rec_df(...)`

        ---

        ### Parameters
        - `cycle_name`: **str**.
        - `factor_param_ranges`: **Optional[Dict[str, Any]]**.
        - `test_cycle`: **int**.
        - `metric_map`: **Dict[str, float]**.

        #### Optional Parameters
        - `extra_columns`: **Optional[Dict[str, Any]]** = *None*.

        ---

        ### Returns
        - `result`: **pd.DataFrame**.
        """

        record: Dict[str, Any] = {
                                    "cycle_name": str(cycle_name),
                                    "factor_param_ranges": factor_param_ranges,
                                    "factor_param_json": json.dumps(factor_param_ranges or {}, ensure_ascii = True, default = str),
                                    "nav_total": metric_map.get("nav_total", np.nan),
                                    "nav_return": metric_map.get("nav_return", np.nan),
                                    "nav_rate": metric_map.get("nav_rate", np.nan),
                                    "nav_total_bm": metric_map.get("nav_total_bm", np.nan),
                                    "nav_total_bm_return": metric_map.get("nav_total_bm_return", np.nan),
                                    "nav_total_bm_rate": metric_map.get("nav_total_bm_rate", np.nan),
                                    "rel_return": metric_map.get("rel_return", np.nan),
                                    "rel_return_rate": metric_map.get("rel_return_rate", np.nan),
                                    "annualized_return_rate": metric_map.get("annualized_return_rate", np.nan),
                                    "maxdd_total": metric_map.get("maxdd_total", np.nan),
                                    "maxdd_rate": metric_map.get("maxdd_rate", np.nan),
                                    "maxdd_period": metric_map.get("maxdd_period", np.nan),
                                    "calmar_ratio": metric_map.get("calmar_ratio", np.nan),
                                    "sharpe_ratio": metric_map.get("sharpe_ratio", np.nan),
                                    "sortino_ratio": metric_map.get("sortino_ratio", np.nan),
                                    "yield_volatility": metric_map.get("yield_volatility", np.nan),
                                    "yield_downside_deviation": metric_map.get("yield_downside_deviation", np.nan),
                                    "var95": metric_map.get("var95", np.nan),
                                    "cvar95": metric_map.get("cvar95", np.nan),
                                    "var99": metric_map.get("var99", np.nan),
                                    "cvar99": metric_map.get("cvar99", np.nan),
                                    "win_rate": metric_map.get("win_rate", np.nan),
                                    "pl_ratio": metric_map.get("pl_ratio", np.nan),
                                    "open_trades": metric_map.get("open_trades", np.nan),
                                    "cover_trades": metric_map.get("cover_trades", np.nan),
                                    "win_trades": metric_map.get("win_trades", np.nan),
                                    "holding_period_h_mean": metric_map.get("holding_period_h_mean", np.nan),
                                    "holding_period_h_median": metric_map.get("holding_period_h_median", np.nan),
                                    "holding_period_h_max": metric_map.get("holding_period_h_max", np.nan),
                                    "holding_period_h_min": metric_map.get("holding_period_h_min", np.nan),
                                    "holding_period_d_mean": metric_map.get("holding_period_d_mean", np.nan),
                                    "holding_period_d_median": metric_map.get("holding_period_d_median", np.nan),
                                    "holding_period_d_max": metric_map.get("holding_period_d_max", np.nan),
                                    "holding_period_d_min": metric_map.get("holding_period_d_min", np.nan),
                                    "max_profit_trade": metric_map.get("max_profit_trade", np.nan),
                                    "min_profit_trade": metric_map.get("min_profit_trade", np.nan),
                                    "max_dd_trade": metric_map.get("max_dd_trade", np.nan),
                                    "min_dd_trade": metric_map.get("min_dd_trade", np.nan),
                                 }

        if extra_columns:
            record.update(extra_columns)


        return pd.DataFrame(record, index = pd.Index([test_cycle]))


    def test_run(self,
                 factor_param_ranges: Optional[Dict[str, Any]] = None,
                 test_cycle: int = 1) -> Tuple[pd.Series, Optional[Dict[str, Any]], str, pd.DataFrame]:
        """
        ### What It Does
        Runs one parameter combination through the selected backtest path.

        #### Responsibility
        Resolves signal artifacts, factor params, engine mode, and portfolio config before returning NAV and metrics.

        #### How To Use
        Call it from traversal or manual probes for a single combo.

        #### Key Parameters In Practice
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `test_cycle`
          - Cycle input payload. It represents the concrete data slice and parameter selection being evaluated.
          - Expected shape/type: `int`.

        #### Usage Example
        `result = test_run(...)`

        ---

        #### Optional Parameters
        - `factor_param_ranges`: **Optional[Dict[str, Any]]** = *None*.
        - `test_cycle`: **int** = *1*.

        ---

        ### Returns
        - `result`: **Tuple[pd.Series, Optional[Dict[str, Any]], str, pd.DataFrame]**.
        """

        from .ForLoopEngine import BacktestEngine_ForLoop  # deferred import: breaks circular dependency
        from .VBTEngine import BacktestEngine_VBT  # deferred import: breaks circular dependency
        from .PortfolioEngine import PortfolioEngine  # deferred import: breaks circular dependency
        manager_test_data = self.test_data

        if manager_test_data is None:

            raise ValueError("[WARNING] test_run requires manager test_data.")

        effective_test_data = pd.DataFrame(manager_test_data)
        resolved_vbt = self.vbt
        resolved_cal_column = self.cal_column
        independent_books = BacktestEngineManager._resolve_independent_books(independent_books = self.independent_books)
        resolved_portfolio_config = dict(self.portfolio_config or {})
        strategy_callable = None if resolved_vbt else self.backtest_strategy

        btengine = self._resolve_btengine(resolved_vbt)

        if resolved_vbt and not isinstance(btengine, BacktestEngine_VBT):

            raise TypeError("[WARNING] vbt=True requires a BacktestEngine_VBT instance.")

        if (not resolved_vbt) and not isinstance(btengine, BacktestEngine_ForLoop):

            raise TypeError("[WARNING] vbt=False requires a BacktestEngine_ForLoop instance.")

        if (not resolved_vbt) and strategy_callable is None:

            raise TypeError("[WARNING] vbt=False requires backtest_strategy.")

        vbt_engine: Optional[BacktestEngine_VBT] = None
        forloop_engine: Optional[BacktestEngine_ForLoop] = None

        if resolved_vbt:
            assert isinstance(btengine, BacktestEngine_VBT)
            vbt_engine = btengine

        else:
            assert isinstance(btengine, BacktestEngine_ForLoop)
            forloop_engine = btengine

        resolved_factor_param_ranges: Dict[str, Any] = {}

        if factor_param_ranges is not None:

            if not isinstance(factor_param_ranges, dict):

                raise TypeError("[WARNING] test_run factor_param_ranges must be a dict.")

            for key, raw_value in factor_param_ranges.items():
                value = raw_value.unwrap() if isinstance(raw_value, FactorEngine.AutoParam.ParamValue) else raw_value

                if isinstance(value, FactorEngine.AutoParam.ParamList):
                    values = list(value)

                elif isinstance(value, range):
                    values = list(value)

                elif isinstance(value, np.ndarray):
                    values = value.tolist()

                elif isinstance(value, (list, tuple, set)):
                    values = list(value)

                else:
                    values = [value]

                cleaned_values: List[Any] = []

                for current in values:
                    item = current.unwrap() if isinstance(current, FactorEngine.AutoParam.ParamValue) else current
                    item = cast(Any, item).item() if isinstance(item, np.generic) else item
                    cleaned_values.append(FactorEngine.AutoParam.clean_value(item))

                if len(cleaned_values) == 0:

                    raise ValueError("[WARNING] test_run factor_param_ranges cannot contain empty parameter ranges.")

                if len(cleaned_values) != 1:

                    raise ValueError("[WARNING] test_run factor_param_ranges requires exactly one value per key.")

                resolved_factor_param_ranges[str(key)] = cleaned_values[0]

        ordered_param_keys = [k for k in self.param_keys if k in resolved_factor_param_ranges]
        ordered_key_set = set(ordered_param_keys)

        ordered_param_keys.extend(sorted(k for k in resolved_factor_param_ranges.keys() if k not in ordered_key_set))
        param_signature = ",".join(f"{key}={resolved_factor_param_ranges[key]}" for key in ordered_param_keys) if ordered_param_keys else "default"
        column_name = f"cycle_{test_cycle}|{param_signature}"

        resolved_factor_param_ranges_payload: Optional[Dict[str, Any]] = resolved_factor_param_ranges if resolved_factor_param_ranges else None
        resolved_portfolio_config = PortfolioEngine.resolve_portfolio_config(resolved_portfolio_config, resolved_factor_param_ranges_payload)
        position = self._build_position()

        if not independent_books:

            if not resolved_vbt:

                raise ValueError("[WARNING] true portfolio mode currently requires vbt=True.")

            weight_panel: Optional[pd.DataFrame] = None
            portfolio_frames: Optional[Dict[str, pd.DataFrame]] = None

            if resolved_factor_param_ranges_payload is not None and (self.weight is not None or self.signal_strength is not None):

                if self.weight is not None:

                    weight_panel = self._slice_panel(
                                                    test_data = effective_test_data,
                                                    values = self.weight,
                                                    factor_param_ranges = resolved_factor_param_ranges_payload,
                                                    ).fillna(0.0).astype(float)

                else:

                    assert self.signal_strength is not None

                    signal_strength_panel = self._slice_panel(
                                                                test_data = effective_test_data,
                                                                values = self.signal_strength,
                                                                factor_param_ranges = resolved_factor_param_ranges_payload,
                                                             ).astype(float)

                    weight_panel = position.compute_weight_panel(
                                                                            test_data = effective_test_data,
                                                                            signal_strength_panel = signal_strength_panel,
                                                                            weight_config = resolved_portfolio_config,
                                                                          )

            else:

                raise ValueError("[WARNING] portfolio mode now requires precomputed artifacts; legacy 2D runtime has been removed.")

            portfolio_inputs = PortfolioEngine.portfolio_inputs(
                                                                    test_data = effective_test_data,
                                                                    weight_panel = cast(pd.DataFrame, weight_panel),
                                                                    cal_column = resolved_cal_column,
                                                                    portfolio_frames = portfolio_frames,
                                                                    portfolio_config = (resolved_portfolio_config if self.weight is not None else None),
                                                                 )

            portfolio_payload = PortfolioEngine.portfolio_bt(
                                                                btengine = cast(BacktestEngine_VBT, vbt_engine),
                                                                weight_panel = portfolio_inputs["weight_panel"],
                                                                open_panel = portfolio_inputs["open_panel"],
                                                                close_panel = portfolio_inputs["close_panel"],
                                                                benchmark_series = self.benchmark_series,
                                                                risk_free_rate = self.risk_free_rate,
                                                                force_flat_mask = portfolio_inputs.get("force_flat_mask"),
                                                                prevent_open_mask = portfolio_inputs.get("prevent_open_mask"),
                                                                max_gross_exposure = resolved_portfolio_config.get("max_gross_exposure"),
                                                                close_delisted_at_last = bool(resolved_portfolio_config.get("close_delisted_at_last", False)),
                                                              )

            current_nav_series = portfolio_payload["nav"]
            nav_metric_map = portfolio_payload["metric_map"]
            underlying_metrics = portfolio_payload["underlying_metrics"]

            performance_rec_df = self.performance_rec_df(
                                                        cycle_name = column_name,
                                                        factor_param_ranges = resolved_factor_param_ranges_payload,
                                                        test_cycle = test_cycle,
                                                        metric_map = nav_metric_map,
                                                        extra_columns = {
                                                                        "active_days": int(pd.Series(current_nav_series).pct_change().fillna(0.0).abs().gt(1e-15).sum()),
                                                                        "portfolio_underlying_count": len(underlying_metrics),
                                                                        },
                                                        )

            return current_nav_series, resolved_factor_param_ranges_payload, column_name, performance_rec_df

        else:

            independent_selection_mode = str((resolved_portfolio_config or {}).get("selection_mode", "none")).strip().lower() or "none"

            if independent_selection_mode in {"top_k", "top_q", "percentile"}:

                raise ValueError(f"[WARNING] independent_books = True runs each asset in its own wallet, so cross-sectional selection_mode = {independent_selection_mode!r} is meaningless (a single-asset book cannot be rank-selected). Use shared-cash mode (independent_books = False) for cross-sectional selection, or set selection_mode = 'none'.")

            portfolio_frames = PortfolioEngine.split_portfolio_test_data(test_data = effective_test_data, cal_column = resolved_cal_column)

            if len(portfolio_frames) < 1:

                raise ValueError("[WARNING] independent_books mode requires >=1 underlying in test_data.")

            signal_panel: Optional[pd.DataFrame] = None
            weight_panel: Optional[pd.DataFrame] = None

            if resolved_factor_param_ranges_payload is not None and (self.weight is not None or self.signal_strength is not None):

                if self.signal_strength is not None:
                    signal_panel = self._slice_panel(test_data = effective_test_data, values = self.signal_strength, factor_param_ranges = resolved_factor_param_ranges_payload).astype(float)

                if self.weight is not None:
                    weight_panel = self._slice_panel(test_data = effective_test_data, values = self.weight, factor_param_ranges = resolved_factor_param_ranges_payload).fillna(0.0).astype(float)

            nav_parts: List[pd.Series] = []
            benchmark_parts: List[pd.Series] = []
            underlying_metrics: Dict[str, Dict[str, float]] = {}

            batch_engine: Any = vbt_engine if resolved_vbt else forloop_engine
            assert batch_engine is not None
            original_test_data = batch_engine.test_data

            try:
                for underlying in sorted(portfolio_frames.keys()):

                    frame = portfolio_frames[underlying]
                    batch_engine.test_data = frame
                    batch_engine.reset()
                    frame_index = pd.DatetimeIndex(pd.to_datetime(frame["Datetime"] if "Datetime" in frame.columns else frame.index, errors = "coerce"))
                    signal_strength_col: Optional[np.ndarray] = None
                    explicit_weight_series: Optional[pd.Series] = None

                    if signal_panel is not None:

                        if underlying not in signal_panel.columns:

                            raise KeyError(f"[WARNING] batch signal panel missing underlying: {underlying}")

                        signal_strength_col = pd.Series(
                                                        pd.to_numeric(signal_panel[underlying], errors = "coerce"),
                                                        index = signal_panel.index,
                                                        dtype = float,
                                                       ).reindex(frame_index).fillna(0.0).to_numpy(dtype = float)

                    if weight_panel is not None:

                        if underlying not in weight_panel.columns:

                            raise KeyError(f"[WARNING] batch weight panel missing underlying: {underlying}")

                        explicit_weight_series = pd.Series(
                                                            pd.to_numeric(weight_panel[underlying], errors = "coerce"),
                                                            index = weight_panel.index,
                                                            dtype = float,
                                                          ).reindex(frame_index).fillna(0.0).astype(float)

                    if signal_strength_col is None and explicit_weight_series is None:

                        raise ValueError("[WARNING] batch mode now requires precomputed artifacts; legacy 2D runtime has been removed.")

                    if resolved_vbt:

                        if explicit_weight_series is not None:
                            weight_series = explicit_weight_series

                        else:
                            assert signal_strength_col is not None

                            derived_weight_panel = position.compute_weight_panel(
                                                                                            test_data = frame,
                                                                                            signal_strength_panel = pd.DataFrame({"SINGLE": pd.Series(signal_strength_col, index = frame_index, dtype = float)}),
                                                                                            weight_config = resolved_portfolio_config,
                                                                                          )

                            weight_series = pd.Series(derived_weight_panel.iloc[:, 0].to_numpy(dtype = float), index = frame.index, dtype = float)

                        nav_underlying, basic_map, _ = self.vbt_testcycle(
                                                                            test_data = frame,
                                                                            btengine = batch_engine,
                                                                            weight_series = weight_series,
                                                                            signal_strength_col = signal_strength_col,
                                                                            cal_column = resolved_cal_column,
                                                                            position = position,
                                                                         )

                    else:

                        if signal_strength_col is None:

                            raise ValueError("[WARNING] batch mode with vbt=False requires signal_strength input, not weight-only input.")

                        nav_underlying = self.testcycle(
                                                        test_data = frame,
                                                        btengine = batch_engine,
                                                        backtest_strategy = cast(Callable[..., Any], strategy_callable),
                                                        net_col = signal_strength_col,
                                                        cal_column = resolved_cal_column,
                                                        position = position,
                                                        )

                        bm_underlying: Optional[pd.Series] = None
                        bm_underlying_source: Optional[Any] = batch_engine.benchmark_series_override

                        if bm_underlying_source is None and "Close" in frame.columns:
                            bm_underlying_source = cast(pd.Series, frame["Close"])

                        if bm_underlying_source is not None:

                            bm_underlying_raw = pd.Series(bm_underlying_source).copy()

                            bm_tmp = pd.Series(
                                               pd.to_numeric(bm_underlying_raw, errors = "coerce"),
                                               index = bm_underlying_raw.index,
                                               dtype = float,
                                               )

                            idx_raw = bm_tmp.index
                            idx_try = pd.to_datetime(idx_raw, errors = "coerce")
                            idx_dtype = getattr(idx_raw, "dtype", None)
                            idx_is_numeric = bool(idx_dtype is not None and pd.api.types.is_numeric_dtype(idx_dtype))
                            use_parsed_idx = isinstance(idx_raw, pd.DatetimeIndex) or ((not idx_is_numeric) and bool(pd.notna(idx_try).any()))

                            if use_parsed_idx:
                                bm_tmp.index = idx_try

                            else:
                                idx_src = frame["Datetime"] if "Datetime" in frame.columns else frame.index

                                if len(bm_tmp) == len(idx_src):
                                    bm_tmp.index = pd.to_datetime(idx_src, errors = "coerce")

                                elif (not idx_is_numeric) and bool(pd.notna(idx_try).any()):
                                    bm_tmp.index = idx_try

                            bm_tmp = bm_tmp.sort_index()
                            bm_tmp = bm_tmp.reindex(nav_underlying.index)
                            bm_valid = bm_tmp.dropna()

                            if not bm_valid.empty:
                                bm_first = float(bm_valid.iloc[0])

                                if bm_first != 0.0:

                                    scaled_benchmark = pd.Series(
                                                                batch_engine.initial_cash * (bm_tmp / bm_first),
                                                                index = bm_tmp.index,
                                                                dtype = float,
                                                                )

                                    scaled_benchmark.name = "BM_Buy_n_Hold"
                                    bm_underlying = scaled_benchmark
                                    benchmark_parts.append(scaled_benchmark.rename(str(underlying)))

                        basic_map = BacktestEngineManager.performance_metrics(
                                                                                nav_series = nav_underlying,
                                                                                benchmark_series = bm_underlying,
                                                                                risk_free_rate = self.risk_free_rate,
                                                                              )

                    nav_parts.append(nav_underlying.rename(str(underlying)))

                    underlying_metrics[str(underlying)] = {
                                                            "nav_rate": float(basic_map["nav_rate"]) if pd.notna(basic_map["nav_rate"]) else np.nan,
                                                            "maxdd_rate": float(basic_map["maxdd_rate"]) if pd.notna(basic_map["maxdd_rate"]) else np.nan,
                                                          }

            finally:
                batch_engine.test_data = original_test_data
                batch_engine.reset()

            def _aggregate_committed_books(parts: List[pd.Series]) -> pd.Series:
                # Capital-conserving aggregation for staggered-start universes. Each per-underlying book is
                # seeded with batch_engine.initial_cash; before its coin lists the book simply holds that seed
                # as idle cash (leading gap -> initial_cash), and after it delists it converts to cash and holds
                # its last NAV (trailing / interior gap -> ffill). Summing the filled levels keeps committed
                # capital constant at (N * initial_cash), so a newly-listed coin no longer injects a fresh
                # initial_cash step that performance_metrics would misread as portfolio return. For single-start
                # (non-staggered) universes every column spans the full window with no gaps, so ffill/fillna are
                # no-ops and the result is identical to the prior skipna sum.
                if not parts:

                    return pd.Series(dtype = float)

                level_frame = pd.concat(parts, axis = 1).sort_index()
                level_frame = level_frame.ffill().fillna(float(batch_engine.initial_cash))

                return level_frame.sum(axis = 1)

            current_nav_series = _aggregate_committed_books(nav_parts)

            if resolved_vbt:
                nav_metric_map = BacktestEngineManager.performance_metrics(nav_series = current_nav_series, risk_free_rate = self.risk_free_rate)

                for metric_key, metric_value in nav_metric_map.items():
                    setattr(batch_engine, metric_key, metric_value)

            else:
                bm_portfolio_series: Optional[pd.Series] = None

                if benchmark_parts:
                    bm_portfolio_series = _aggregate_committed_books(benchmark_parts)

                elif batch_engine.benchmark_series_override is not None and not current_nav_series.empty:

                    bm_underlying_raw = pd.Series(batch_engine.benchmark_series_override).copy()

                    bm_tmp = pd.Series(
                                       pd.to_numeric(bm_underlying_raw, errors = "coerce"),
                                       index = bm_underlying_raw.index,
                                       dtype = float,
                                       )

                    idx_raw = bm_tmp.index
                    idx_try = pd.to_datetime(idx_raw, errors = "coerce")
                    idx_dtype = getattr(idx_raw, "dtype", None)
                    idx_is_numeric = bool(idx_dtype is not None and pd.api.types.is_numeric_dtype(idx_dtype))
                    use_parsed_idx = isinstance(idx_raw, pd.DatetimeIndex) or ((not idx_is_numeric) and bool(pd.notna(idx_try).any()))

                    if use_parsed_idx:
                        bm_tmp.index = idx_try

                    else:
                        idx_src = effective_test_data["Datetime"] if "Datetime" in effective_test_data.columns else effective_test_data.index

                        if len(bm_tmp) == len(effective_test_data):
                            bm_tmp.index = pd.to_datetime(idx_src, errors = "coerce")

                        elif (not idx_is_numeric) and bool(pd.notna(idx_try).any()):
                            bm_tmp.index = idx_try

                    bm_tmp = bm_tmp.sort_index()
                    bm_tmp = bm_tmp.reindex(current_nav_series.index)
                    bm_valid = bm_tmp.dropna()

                    if not bm_valid.empty:
                        bm_first = float(bm_valid.iloc[0])

                        if bm_first != 0.0:
                            bm_portfolio_tmp = (batch_engine.initial_cash * len(portfolio_frames)) * (bm_tmp / bm_first)
                            bm_portfolio_tmp.name = "BM_Buy_n_Hold"
                            bm_portfolio_series = bm_portfolio_tmp

                nav_metric_map = BacktestEngineManager.performance_metrics(
                                                                            nav_series = current_nav_series,
                                                                            benchmark_series = bm_portfolio_series,
                                                                            risk_free_rate = self.risk_free_rate,
                                                                          )

            performance_rec_df = self.performance_rec_df(
                                                        cycle_name = column_name,
                                                        factor_param_ranges = resolved_factor_param_ranges_payload,
                                                        test_cycle = test_cycle,
                                                        metric_map = nav_metric_map,
                                                        extra_columns = {
                                                                        "active_days": int(pd.Series(current_nav_series).pct_change().fillna(0.0).abs().gt(1e-15).sum()),
                                                                        "portfolio_underlying_count": len(underlying_metrics),
                                                                        },
                                                        )

            return current_nav_series, resolved_factor_param_ranges_payload, column_name, performance_rec_df


    def combo_processing(self,
                         hypertuning_nav_rec: Optional[pd.DataFrame] = None,
                         performance_rec: Optional[pd.DataFrame] = None,
                         *,
                         test_data: Optional[pd.DataFrame] = None,
                         factor_param_ranges: Optional[Dict[str, List[Any]]] = None,
                         combo_pool_size: int = 5,
                         min_combo_size: int = 1,
                         max_combo_size: int = 3,
                         target: str = "sharpe_ratio",
                         combo_weight_mode: str = "equal",
                         logical_processors: int = 1,
                         independent_books: Optional[bool] = None,
                         portfolio_config: Optional[Dict[str, Any]] = None,
                         vbt: Optional[bool] = None,
                         log_path: str = "") -> Dict[str, Any]:
        """
        ### What It Does
        Builds train-sample combo candidates on top of traversal results and returns combo NAV/performance tables.

        #### Responsibility
        Adds portfolio-agnostic research protocol for parameter-cycle combination without changing execution semantics.

        #### How To Use
        Either pass existing traversal outputs or let the manager run traversal first, then combine the top cycles.

        #### Key Parameters In Practice
        - `hypertuning_nav_rec`
          - Optimizer NAV record collection. Use it to compare parameter candidates and build train/OOS selection results.
          - Expected shape/type: `Optional[pd.DataFrame]`.
        - `performance_rec`
          - Performance record table. Downstream protocol selection and reporting read objective columns from it.
          - Expected shape/type: `Optional[pd.DataFrame]`.
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `Optional[pd.DataFrame]`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Optional[Dict[str, List[Any]]]`.
        - `combo_pool_size`
          - Number of top candidates retained for combo construction. Increase it only when ensemble-style selection is intended.
          - Expected shape/type: `int`.
        - `min_combo_size`
          - Minimum candidate count in a combo. Keep it at one for ordinary best-candidate selection.
          - Expected shape/type: `int`.
        - `max_combo_size`
          - Maximum candidate count in a combo. Raise it when protocol should evaluate candidate ensembles.
          - Expected shape/type: `int`.
        - `target`
          - Objective or dependent variable. In optimizers it names the performance metric to maximize; in analysis it names the response column.
          - Expected shape/type: `str`.
        - `combo_weight_mode`
          - Combo weighting rule. Use `equal` for simple ensemble averaging or a configured mode when performance-weighted replay is intended.
          - Expected shape/type: `str`.
        - `logical_processors`
          - Worker budget. Use `1` for deterministic local debugging; increase it only when payloads and strategy callables are process-safe.
          - Expected shape/type: `int`.
        - `independent_books`
          - Execution topology switch. `False` (default) runs one shared-cash cross-sectional portfolio; `True` runs each asset in its own wallet (`cash_sharing=False`) and sums the books. `None` inherits the manager's configured value.
          - Expected shape/type: `Optional[bool]`.
        - `portfolio_config`
          - Portfolio execution configuration. Put shared-cash, force-cover, prevent-open, sleeve, and report-epsilon behavior here.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `vbt`
          - Backend switch. Use `True` for vectorbt execution and portfolio shared-cash workflows; use `False` only when a custom for-loop strategy path is needed.
          - Expected shape/type: `Optional[bool]`.
        - `log_path`
          - CSV/progress output path. Runners fill it with the run CSV path before the search starts.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = combo_processing(...)`

        ---

        #### Optional Parameters
        - `hypertuning_nav_rec`: **Optional[pd.DataFrame]** = *None*.
        - `performance_rec`: **Optional[pd.DataFrame]** = *None*.
        - `test_data`: **Optional[pd.DataFrame]** = *None*.
        - `factor_param_ranges`: **Optional[Dict[str, List[Any]]]** = *None*.
        - `combo_pool_size`: **int** = *5*.
        - `min_combo_size`: **int** = *1*.
        - `max_combo_size`: **int** = *3*.
        - `target`: **str** = *"sharpe_ratio"*.
        - `combo_weight_mode`: **str** = *"equal"*.
        - `logical_processors`: **int** = *1*.
        - `independent_books`: **Optional[bool]** = *None*.
        - `portfolio_config`: **Optional[Dict[str, Any]]** = *None*.
        - `vbt`: **Optional[bool]** = *None*.
        - `log_path`: **str** = *""*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        from .PortfolioEngine import PortfolioEngine  # deferred import: breaks circular dependency
        effective_test_data = self.test_data if test_data is None else test_data
        resolved_vbt = self.vbt if vbt is None else bool(vbt)
        independent_books = BacktestEngineManager._resolve_independent_books(independent_books = self.independent_books if independent_books is None else independent_books)
        resolved_portfolio_config = dict(self.portfolio_config if portfolio_config is None else (portfolio_config or {}))
        resolved_factor_param_ranges = self.factor_param_ranges if factor_param_ranges is None else factor_param_ranges
        target_metric = str(target or "sharpe_ratio").strip() or "sharpe_ratio"
        combo_weight_mode = str(combo_weight_mode or "equal").strip().lower() or "equal"

        if combo_weight_mode not in {"equal", "score"}:

            raise ValueError(f"[WARNING] unsupported combo_weight_mode: {combo_weight_mode!r}")

        nav_rec = None if hypertuning_nav_rec is None else pd.DataFrame(hypertuning_nav_rec).copy()
        perf_rec = None if performance_rec is None else pd.DataFrame(performance_rec).copy()

        if nav_rec is None or perf_rec is None:

            nav_rec, perf_rec = self._traversal_backend(
                                                        test_data = effective_test_data,
                                                        initial_cash = self.initial_cash,
                                                        backtest_strategy = self.backtest_strategy,
                                                        factor_param_ranges = resolved_factor_param_ranges,
                                                        risk_free_rate = self.risk_free_rate,
                                                        slippage = self.slippage,
                                                        spread = self.spread,
                                                        fees = self.fees,
                                                        benchmark_series = self.benchmark_series,
                                                        benchmark_name = self.benchmark_name,
                                                        signal_z_window = self.signal_z_window,
                                                        signal_start_t = self.signal_start_t,
                                                        signal_cooldown = self.signal_cooldown,
                                                        signal_epsilon = self.signal_epsilon,
                                                        signal_confirm_mode = self.signal_confirm_mode,
                                                        signal_gate_mode = self.signal_gate_mode,
                                                        signal_gate_style = self.signal_gate_style,
                                                        signal_gate_threshold = self.signal_gate_threshold,
                                                        signal_strength = self.signal_strength,
                                                        weight = self.weight,
                                                        factor_manager = self.factor_manager,
                                                        calendar = self.calendar,
                                                        indexer = self.indexer,
                                                        asset_keys = self.asset_keys,
                                                        logical_processors = logical_processors,
                                                        independent_books = independent_books,
                                                        cal_column = self.cal_column,
                                                        portfolio_config = resolved_portfolio_config,
                                                        vbt = resolved_vbt,
                                                        log_path = log_path,
                                                        )

        nav_rec = pd.DataFrame(nav_rec).copy()
        perf_rec = pd.DataFrame(perf_rec).copy()

        if nav_rec.empty or perf_rec.empty:

            return {

                    "hypertuning_nav_rec": nav_rec,
                    "performance_rec": perf_rec,
                    "pool_performance_rec": pd.DataFrame(),
                    "combo_nav_rec": pd.DataFrame(index = nav_rec.index),
                    "combo_performance_rec": pd.DataFrame(),
                    }

        if "cycle_name" not in perf_rec.columns:

            raise KeyError("[WARNING] performance_rec missing required column: 'cycle_name'.")

        if target_metric not in perf_rec.columns:

            raise KeyError(f"[WARNING] performance_rec missing required target metric: {target_metric!r}")

        perf_rec = perf_rec.copy()
        perf_rec["cycle_name"] = perf_rec["cycle_name"].astype(str)
        perf_rec[target_metric] = pd.to_numeric(perf_rec[target_metric], errors = "coerce")
        perf_rec["__selection_rank_score__"] = perf_rec[target_metric].map(lambda value: BacktestEngineManager._target_metric_rank_value(target_metric, value))
        perf_rec = perf_rec.loc[perf_rec["cycle_name"].isin(nav_rec.columns)].copy()

        pool_rec = (
                    perf_rec.loc[perf_rec["__selection_rank_score__"].notna()]
                    .sort_values(["__selection_rank_score__", "sharpe_ratio", "nav_rate"], ascending = [False, False, False])
                    .head(max(1, int(combo_pool_size)))
                    .reset_index(drop = True)
                    )

        if pool_rec.empty:

            return {

                    "hypertuning_nav_rec": nav_rec,
                    "performance_rec": perf_rec,
                    "pool_performance_rec": pool_rec,
                    "combo_nav_rec": pd.DataFrame(index = nav_rec.index),
                    "combo_performance_rec": pd.DataFrame(),
                    }

        combo_nav_rec = pd.DataFrame(index = nav_rec.index)
        combo_rows: List[Dict[str, Any]] = []
        combo_counter = 0
        min_size = max(1, int(min_combo_size))
        max_size = min(max(min_size, int(max_combo_size)), len(pool_rec))

        for combo_size in range(min_size, max_size + 1):

            for member_idx in itertools.combinations(range(len(pool_rec)), combo_size):

                member_rec = pool_rec.iloc[list(member_idx)].copy()
                member_names = member_rec["cycle_name"].astype(str).tolist()
                member_nav = nav_rec.reindex(columns = member_names).apply(pd.to_numeric, errors = "coerce")

                if combo_weight_mode == "score":
                    raw_weights = pd.to_numeric(member_rec[target_metric], errors = "coerce").fillna(0.0).to_numpy(dtype = float)
                    raw_weights = np.clip(raw_weights, 0.0, None)

                    if raw_weights.sum() <= 0.0:
                        weights = np.full(len(member_names), 1.0 / float(len(member_names)), dtype = float)

                    else:
                        weights = raw_weights / raw_weights.sum()

                else:
                    weights = np.full(len(member_names), 1.0 / float(len(member_names)), dtype = float)

                member_ret, combo_ret, combo_nav = PortfolioEngine.weighted_return_nav(member_nav = member_nav, weights = weights, initial_cash = float(self.initial_cash))

                combo_counter += 1

                combo_name = f"combo_{combo_counter:03d}"
                combo_nav_rec[combo_name] = combo_nav
                metric_map = BacktestEngineManager.performance_metrics(nav_series = combo_nav, risk_free_rate = self.risk_free_rate)
                combo_score = BacktestEngineManager._target_metric_value(metric_map, target_metric)
                combo_rank_score = BacktestEngineManager._target_metric_rank_value(target_metric, combo_score)

                combo_rows.append(
                                    {
                                    "combo_name": combo_name,
                                    "target": target_metric,
                                    "combo_score": combo_score,
                                    "combo_rank_score": combo_rank_score,
                                    "member_count": int(len(member_names)),
                                    "members_json": json.dumps(member_names, ensure_ascii = True),
                                    "member_param_jsons": json.dumps(member_rec.get("factor_param_json", pd.Series(["{}"] * len(member_rec))).astype(str).tolist(), ensure_ascii = True),
                                    "weights_json": json.dumps([float(x) for x in weights], ensure_ascii = True),
                                    "combo_weight_mode": combo_weight_mode,
                                    "nav_rate": metric_map.get("nav_rate", np.nan),
                                    "annualized_return_rate": metric_map.get("annualized_return_rate", np.nan),
                                    "maxdd_rate": metric_map.get("maxdd_rate", np.nan),
                                    "sharpe_ratio": metric_map.get("sharpe_ratio", np.nan),
                                    "sortino_ratio": metric_map.get("sortino_ratio", np.nan),
                                    }
                                 )

        combo_perf_rec = pd.DataFrame(combo_rows)

        if not combo_perf_rec.empty:
            combo_perf_rec = combo_perf_rec.sort_values(["combo_rank_score", "sharpe_ratio", "nav_rate"], ascending = False).reset_index(drop = True)


        return {

                "hypertuning_nav_rec": nav_rec,
                "performance_rec": perf_rec,
                "pool_performance_rec": pool_rec,
                "combo_nav_rec": combo_nav_rec,
                "combo_performance_rec": combo_perf_rec,
               }


    def testcycle(self,
                  test_data: pd.DataFrame,
                  btengine: "BacktestEngine_ForLoop",
                  backtest_strategy: Callable[..., Any],
                  net_col: Optional[np.ndarray] = None,
                  cal_column: str = "Close",
                  position: Optional[Position] = None) -> pd.Series:
        """
        ### What It Does
        Runs the for-loop backtest cycle for one parameter combination.

        #### Responsibility
        Executes strategy callbacks or precomputed net signals through the event-style engine and returns reports.

        #### How To Use
        Call it when non-VBT execution is required.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `btengine`
          - Backtest engine instance or class. It defines which execution backend actually consumes prices, signals, and weights.
          - Expected shape/type: `"BacktestEngine_ForLoop"`.
        - `backtest_strategy`
          - For-loop strategy callable. Supply it only when `vbt` is false or when a custom Python entry/exit path is required; VBT signal/weight paths normally do not need it.
          - Expected shape/type: `Callable[..., Any]`.
        - `net_col`
          - Net signal column name. Point it to the column that should be converted into signal strength or weights.
          - Expected shape/type: `Optional[np.ndarray]`.
        - `cal_column`
          - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
          - Expected shape/type: `str`.
        - `position`
          - Position instance used when a path must derive signal strength or weights from runtime signal columns/artifacts.
          - Expected shape/type: `Optional[Position]`.

        #### Usage Example
        `result = testcycle(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `btengine`: **"BacktestEngine_ForLoop"**.
        - `backtest_strategy`: **Callable[..., Any]**.

        #### Optional Parameters
        - `net_col`: **Optional[np.ndarray]** = *None*.
        - `cal_column`: **str** = *"Close"*.
        - `position`: **Optional[Position]** = *None*.

        ---

        ### Returns
        - `result`: **pd.Series**.
        """

        from .ForLoopEngine import BacktestEngine_ForLoop  # deferred import: breaks circular dependency
        start_t = max(0, int(getattr(btengine, "signal_start_t", 2)))
        position = position if isinstance(position, Position) else self._build_position()

        if net_col is not None:

            weight_panel = position.compute_weight_panel(
                                                                    test_data = test_data,
                                                                    signal_strength_panel = pd.DataFrame(
                                                                        {
                                                                            "SINGLE": pd.Series(
                                                                                pd.to_numeric(net_col, errors = "coerce"),
                                                                                index = pd.DatetimeIndex(
                                                                                    pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce")
                                                                                ),
                                                                                dtype = float,
                                                                            )
                                                                        }
                                                                    ),
                                                                  )

            weight_series = pd.Series(weight_panel.iloc[:, 0].to_numpy(dtype = float), index = test_data.index, dtype = float)
            test_data["weights"] = BacktestEngineManager._align_series_to_test_data(weight_series, test_data).to_numpy(dtype = float)

        for t in range(start_t, len(test_data)):

            backtest_strategy(t = t, btengine = btengine, test_data = test_data)

            btengine.nav, btengine.nav_line = BacktestEngine_ForLoop.nav_and_yield(
                                                                                    t = t,
                                                                                    cash = btengine.cash,
                                                                                    amt = btengine.amt,
                                                                                    test_data = test_data,
                                                                                    nav = btengine.nav,
                                                                                    nav_line = btengine.nav_line,
                                                                                    cal_column = cal_column,
                                                                                  )


        return btengine.nav_line["NAV"].copy()


    def vbt_testcycle(self,
                      test_data: pd.DataFrame,
                      btengine: Any,
                      weight_series: Optional[pd.Series] = None,
                      signal_strength_col: Optional[np.ndarray] = None,
                      cal_column: str = "Close",
                      position: Optional[Position] = None,
                      weight_config: Optional[Dict[str, Any]] = None) -> Tuple[pd.Series, Dict[str, float], Any]:
        """
        ### What It Does
        Runs the vectorbt backtest cycle for one parameter combination.

        #### Responsibility
        Builds signal-derived or explicit weights, dispatches vectorbt execution, and returns NAV plus diagnostics.

        #### How To Use
        Call it when VBT execution is enabled for a single combo.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `btengine`
          - Backtest engine instance or class. It defines which execution backend actually consumes prices, signals, and weights.
          - Expected shape/type: `Any`.
        - `weight_series`
          - Single-asset target-weight series. Its index should align to the price series used by the selected backtest backend.
          - Expected shape/type: `Optional[pd.Series]`.
        - `signal_strength_col`
          - Column name or series used as the signal source for VBT/for-loop conversion. Use it when signal strength lives inside `test_data` rather than an artifact.
          - Expected shape/type: `Optional[np.ndarray]`.
        - `cal_column`
          - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
          - Expected shape/type: `str`.
        - `position`
          - Position instance used when a path must derive signal strength or weights from runtime signal columns/artifacts.
          - Expected shape/type: `Optional[Position]`.
        - `weight_config`
          - Signal-to-weight configuration. Use it to choose weighting mode, clipping, normalization, and other portfolio weight construction behavior.
          - Expected shape/type: `Optional[Dict[str, Any]]`.

        #### Usage Example
        `result = vbt_testcycle(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `btengine`: **Any**.

        #### Optional Parameters
        - `weight_series`: **Optional[pd.Series]** = *None*.
        - `signal_strength_col`: **Optional[np.ndarray]** = *None*.
        - `cal_column`: **str** = *"Close"*.
        - `position`: **Optional[Position]** = *None*.
        - `weight_config`: **Optional[Dict[str, Any]]** = *None*.

        ---

        ### Returns
        - `result`: **Tuple[pd.Series, Dict[str, float], Any]**.
        """

        from .VBTEngine import BacktestEngine_VBT  # deferred import: breaks circular dependency
        if isinstance(btengine, BacktestEngine_VBT):
            vbt_engine = btengine

        else:

            if hasattr(self, "_resolve_btengine"):
                vbt_engine = self._resolve_btengine(True)

            else:

                raise TypeError(
                                "[WARNING] vbt_testcycle requires a BacktestEngine_VBT instance "
                                "or a BacktestEngineManager capable of resolving one."
                                )

        assert isinstance(vbt_engine, BacktestEngine_VBT)

        vbt_engine.test_data = test_data.copy()
        vbt_engine.reset()

        if weight_series is not None:
            resolved_weight_series = BacktestEngineManager._align_series_to_test_data(weight_series, test_data).fillna(0.0).astype(float)

        else:
            resolved_weight_series = self._resolve_vbt_weight_series(
                                                                    test_data = test_data,
                                                                    weight_series = None,
                                                                    signal_strength_col = signal_strength_col,
                                                                    position = position,
                                                                    weight_config = weight_config,
                                                                    )

        nav_series, portfolio = vbt_engine.vectorbt(
                                                    test_data = test_data,
                                                    weight_series = resolved_weight_series,
                                                    cal_column = cal_column,
                                                    )

        # Build the buy-and-hold benchmark so the VBT cycle reports the same benchmark-relative metrics
        # (rel_return, rel_return_rate, nav_total_bm, ...) as the ForLoop path; without it those fields come
        # back NaN and the two engines' performance records are not comparable. Mirrors the single-ForLoop
        # benchmark alignment (benchmark_series_override, fallback Close, scaled by initial_cash).
        bm_series: Optional[pd.Series] = None
        bm_source: Optional[pd.Series] = vbt_engine.benchmark_series_override

        if bm_source is None and "Close" in test_data.columns:
            bm_source = cast(pd.Series, test_data["Close"])

        if bm_source is not None:

            bm_raw = pd.Series(bm_source).copy()
            bm_tmp = pd.Series(pd.to_numeric(bm_raw, errors = "coerce"), index = bm_raw.index, dtype = float)

            idx_raw = bm_tmp.index
            idx_try = pd.to_datetime(idx_raw, errors = "coerce")
            idx_dtype = getattr(idx_raw, "dtype", None)
            idx_is_numeric = bool(idx_dtype is not None and pd.api.types.is_numeric_dtype(idx_dtype))
            use_parsed_idx = isinstance(idx_raw, pd.DatetimeIndex) or ((not idx_is_numeric) and bool(pd.notna(idx_try).any()))

            if use_parsed_idx:
                bm_tmp.index = idx_try

            else:
                idx_src = test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index

                if len(bm_tmp) == len(test_data):
                    bm_tmp.index = pd.to_datetime(idx_src, errors = "coerce")

                elif (not idx_is_numeric) and bool(pd.notna(idx_try).any()):
                    bm_tmp.index = idx_try

            bm_tmp = bm_tmp.sort_index()
            bm_tmp = bm_tmp.reindex(nav_series.index)
            bm_valid = bm_tmp.dropna()

            if not bm_valid.empty:
                bm_first = float(bm_valid.iloc[0])

                if bm_first != 0.0:

                    bm_series_tmp = vbt_engine.initial_cash * (bm_tmp / bm_first)
                    bm_series_tmp.name = "BM_Buy_n_Hold"
                    bm_series = bm_series_tmp

        metric_map = BacktestEngineManager.performance_metrics(nav_series = nav_series, benchmark_series = bm_series, risk_free_rate = self.risk_free_rate)
        metric_map.update(vbt_engine.TradeAccount(portfolio = portfolio))

        for metric_key, metric_value in metric_map.items():
            setattr(vbt_engine, metric_key, metric_value)


        return nav_series.copy(), metric_map, portfolio





def trav_init_worker(args: Tuple[Any, dict]) -> None:
    """
    ### What It Does
    Initializes process-local state for traversal worker evaluation.

    #### Responsibility
    Loads shared configs and artifacts once per worker process for grid-search tasks.

    #### How To Use
    Use it as the multiprocessing initializer for traversal pools.

    #### Key Parameters In Practice
    - `args`
      - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
      - Expected shape/type: `Tuple[Any, dict]`.


    #### Usage Example
    `result = trav_init_worker(...)`

    ---

    ### Parameters
    - `args`: **Tuple[Any, dict]**.
    """

    try:
        log_queue, init_btmgr_args = args
        root, root.handlers = multi_log_set_top()
        handler = QueueHandler(log_queue)
        root.addHandler(handler)

        worker_args = dict(init_btmgr_args or {})
        BacktestEngineManager._btmgr = BacktestEngineManager(**worker_args)

    except Exception as e:
        sys.stderr.write(f"[WARNING] BTMGR Init Worker FAILED\n")
        sys.stderr.write(f"Exception: {e.__class__.__name__}: {e}\n")
        traceback.print_exc(file = sys.stderr)
        sys.stderr.flush()

        logging.error(f"[WARNING] BTMGR Init Worker FAILED: {e}\n", exc_info = True)

        raise


def trav_init_worker_persistent(args: Tuple[Any, Any]) -> None:
    """
    ### What It Does
    Initializer for the PERSISTENT traversal pool owned by window_processing.

    #### Responsibility
    Wires the shared log queue and stashes the shared window-state proxy, then
    defers per-window `_btmgr` construction to the worker's first task of each
    window (token-gated in `trav_worker`). This lets ONE pool serve every
    walk-forward window without respawning, so the Windows-spawn vectorbt/numba
    cold import is paid once per worker for the whole run instead of per window.

    ---

    ### Parameters
    - `args`: **Tuple[Any, Any]** -- `(log_queue, window_state_proxy)`.
    """

    try:
        log_queue, window_state = args
        root, root.handlers = multi_log_set_top()
        handler = QueueHandler(log_queue)
        root.addHandler(handler)

        BacktestEngineManager._trav_window_state = window_state
        BacktestEngineManager._trav_worker_token = None
        BacktestEngineManager._btmgr = None

    except Exception as e:
        sys.stderr.write(f"[WARNING] BTMGR Persistent Init Worker FAILED\n")
        sys.stderr.write(f"Exception: {e.__class__.__name__}: {e}\n")
        traceback.print_exc(file = sys.stderr)
        sys.stderr.flush()

        logging.error(f"[WARNING] BTMGR Persistent Init Worker FAILED: {e}\n", exc_info = True)

        raise


def trav_worker(factor_param_values: Sequence[Any], test_cycle: int, window_token: Optional[int] = None) -> Tuple[pd.Series, Optional[Dict[str, Any]], str, pd.DataFrame]:
    """
    ### What It Does
    Evaluates one traversal parameter combination inside a worker process.

    #### Responsibility
    Runs the configured backtest objective for a grid candidate and returns metrics.

    #### How To Use
    Use it as the multiprocessing worker function for traversal search.

    #### Key Parameters In Practice
    - `factor_param_values`
      - Concrete parameter values for a selected combo. Use this after optimizer selection, not as the original full search grid.
      - Expected shape/type: `List[float]`.
    - `test_cycle`
      - Cycle input payload. It represents the concrete data slice and parameter selection being evaluated.
      - Expected shape/type: `int`.


    #### Usage Example
    `result = trav_worker(...)`

    ---

    ### Parameters
    - `factor_param_values`: **List[float]**.
    - `test_cycle`: **int**.

    ---

    ### Returns
    - `result`: **Tuple[pd.Series, Optional[Dict[str, Any]], str, pd.DataFrame]**.
    """

    try:

        if window_token is not None and BacktestEngineManager._trav_worker_token != window_token:
            # Persistent pool: (re)build this worker's _btmgr for the current window the
            # first time it sees a new token. The per-window test_data arrives via the
            # shared proxy published by _traversal_backend before this window's submits.
            state = BacktestEngineManager._trav_window_state
            BacktestEngineManager._btmgr = BacktestEngineManager(**dict(state["args"]))
            BacktestEngineManager._trav_worker_token = window_token

        if BacktestEngineManager._btmgr is None:

            raise RuntimeError("[WARNING] BacktestEngineManager Not Initialized")

        mgr = BacktestEngineManager._btmgr
        factor_param_ranges: Dict[str, Any] = {}

        for i in range(len(factor_param_values)):
            raw_value = factor_param_values[i]
            clean_value = cast(Any, raw_value).item() if isinstance(raw_value, np.generic) else raw_value
            key = str(mgr.param_keys[i]) if i < len(mgr.param_keys) else f"p{i}"
            factor_param_ranges[key] = FactorEngine.AutoParam.clean_value(clean_value)

        return mgr.test_run(factor_param_ranges = factor_param_ranges, test_cycle = test_cycle)

    except Exception as e:

        if _is_signature_mismatch_type_error(e):

            raise

        sys.stderr.write(f"[WARNING] BTMGR Worker CRASHED\n")
        sys.stderr.write(f"Exception: {e.__class__.__name__}: {e}\n")
        traceback.print_exc(file = sys.stderr)
        sys.stderr.flush()

        logging.error(f"[WARNING] BTMGR Worker CRASHED: {e}\n", exc_info = True)

        return pd.Series(dtype = float), None, "ERROR", pd.DataFrame()
