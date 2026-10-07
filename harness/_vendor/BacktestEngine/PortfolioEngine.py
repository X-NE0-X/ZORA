from ENV_MGMT.imports import *
import FactorEngine

if TYPE_CHECKING:  # type-only import; runtime import is deferred inside methods to break circular dependency
    from .VBTEngine import BacktestEngine_VBT







# Portfolio Backtest Class
#----------------------------------------------------------------------------------------
class PortfolioEngine:
    """
    ### What It Does
    Provides the true shared-cash portfolio workflow used by portfolio-mode research and execution.

    #### Responsibility
    Splits wide `test_data` into per-underlying frames, constructs aligned target/price panels, applies portfolio-level masks, and runs shared-cash vectorbt execution with attribution payloads.

    #### How To Use
    Use `split_portfolio_test_data(...)` for sleeve extraction, `portfolio_inputs(...)` to build aligned panels and masks, then `portfolio_bt(...)` to run the shared-cash portfolio path.

    #### Usage Example
    `obj = PortfolioEngine(...)`
    """

    @staticmethod
    def TradeAccount(portfolio: Optional[Any] = None,
                     position_panel: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        ### What It Does
        Summarizes portfolio-level trade records.

        #### Responsibility
        Aggregates trade diagnostics for multi-asset VBT portfolio results.

        #### How To Use
        Call it after portfolio backtest execution when trade metrics are needed.

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

        closed_pnls: List[float] = []
        holding_hours: List[float] = []
        open_trades = 0
        cover_trades = 0
        eps = 1e-12

        if position_panel is not None and not pd.DataFrame(position_panel).empty:
            frame = pd.DataFrame(position_panel).copy()

            if "trade_date" in frame.columns:
                frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors = "coerce")

            else:
                frame["trade_date"] = pd.to_datetime(frame.index, errors = "coerce")

            for column in ("portfolio_weight_prev_close", "portfolio_weight_open", "portfolio_weight_close", "portfolio_contribution"):

                if column not in frame.columns:
                    frame[column] = 0.0

                frame[column] = pd.to_numeric(frame[column], errors = "coerce").fillna(0.0)

            if "symbol" not in frame.columns:
                frame["symbol"] = "__portfolio__"

            for _, group in frame.sort_values(["symbol", "trade_date"]).groupby("symbol", sort = False):

                entry_time = None
                active_side = 0.0
                trade_pnl = 0.0

                for _, row in group.iterrows():

                    trade_time = row["trade_date"]
                    prev_weight = float(row["portfolio_weight_prev_close"])
                    open_weight = float(row["portfolio_weight_open"])
                    close_weight = float(row["portfolio_weight_close"])
                    contrib = float(row["portfolio_contribution"])

                    prev_side = float(np.sign(prev_weight)) if abs(prev_weight) > eps else 0.0
                    open_side = float(np.sign(open_weight)) if abs(open_weight) > eps else 0.0
                    close_side = float(np.sign(close_weight)) if abs(close_weight) > eps else 0.0

                    if entry_time is None and (open_side != 0.0 or prev_side != 0.0):

                        entry_time = trade_time
                        active_side = open_side if open_side != 0.0 else prev_side
                        open_trades += 1

                    if entry_time is not None:
                        trade_pnl += contrib

                    exit_trade = (
                                    entry_time is not None
                                    and (close_side == 0.0 or (active_side != 0.0 and close_side != active_side))
                                 )

                    if exit_trade:

                        closed_pnls.append(float(trade_pnl))
                        cover_trades += 1

                        if pd.notna(entry_time) and pd.notna(trade_time):
                            holding_hours.append(abs((trade_time - entry_time).total_seconds()) / 3600.0)

                        if close_side != 0.0 and close_side != active_side:

                            entry_time = trade_time
                            active_side = close_side
                            trade_pnl = 0.0
                            open_trades += 1

                        else:
                            entry_time = None
                            active_side = 0.0
                            trade_pnl = 0.0

        elif portfolio is not None:

            try:
                trades = getattr(portfolio, "trades", None)
                records = getattr(trades, "records_readable", None)
                records = records() if callable(records) else records
                records_frame = pd.DataFrame() if records is None else pd.DataFrame(cast(Any, records))

                if not records_frame.empty:

                    open_trades = int(len(records_frame))
                    status_col = next((col for col in records_frame.columns if str(col).lower() == "status"), None)
                    closed_frame = records_frame if status_col is None else records_frame.loc[~records_frame[status_col].astype(str).str.lower().str.contains("open")].copy()
                    cover_trades = int(len(closed_frame))
                    pnl_col = next((col for col in records_frame.columns if str(col).lower() in {"pnl", "profit", "return"}), None)

                    if pnl_col is not None:
                        closed_pnls = pd.to_numeric(closed_frame[pnl_col], errors = "coerce").dropna().astype(float).tolist()

                    duration_col = next((col for col in records_frame.columns if "duration" in str(col).lower()), None)

                    if duration_col is not None:

                        durations = pd.to_timedelta(closed_frame[duration_col], errors = "coerce").dropna()
                        holding_hours = (durations.dt.total_seconds() / 3600.0).astype(float).tolist()

            except Exception:
                pass

        pnl_series = pd.Series(closed_pnls, dtype = float)
        hold_series = pd.Series(holding_hours, dtype = float)
        win_trades = int((pnl_series > 0.0).sum()) if not pnl_series.empty else 0
        gain_sum = float(pnl_series[pnl_series > 0.0].sum()) if not pnl_series.empty else 0.0
        loss_sum = float(abs(pnl_series[pnl_series < 0.0].sum())) if not pnl_series.empty else 0.0


        return {

                "open_trades": int(open_trades),
                "cover_trades": int(cover_trades),
                "win_trades": int(win_trades),
                "win_rate": float(win_trades / cover_trades * 100.0) if cover_trades > 0 else np.nan,
                "pl_ratio": float(gain_sum / loss_sum) if loss_sum > 0.0 else np.nan,
                "holding_period_h_mean": float(hold_series.mean()) if not hold_series.empty else np.nan,
                "holding_period_h_median": float(hold_series.median()) if not hold_series.empty else np.nan,
                "holding_period_h_max": float(hold_series.max()) if not hold_series.empty else np.nan,
                "holding_period_h_min": float(hold_series.min()) if not hold_series.empty else np.nan,
                "holding_period_d_mean": float(hold_series.mean() / 24.0) if not hold_series.empty else np.nan,
                "holding_period_d_median": float(hold_series.median() / 24.0) if not hold_series.empty else np.nan,
                "holding_period_d_max": float(hold_series.max() / 24.0) if not hold_series.empty else np.nan,
                "holding_period_d_min": float(hold_series.min() / 24.0) if not hold_series.empty else np.nan,
                "max_profit_trade": float(pnl_series[pnl_series > 0.0].max()) if (not pnl_series.empty and (pnl_series > 0.0).any()) else np.nan,
                "min_profit_trade": float(pnl_series[pnl_series > 0.0].min()) if (not pnl_series.empty and (pnl_series > 0.0).any()) else np.nan,
                "max_dd_trade": float(pnl_series[pnl_series < 0.0].min()) if (not pnl_series.empty and (pnl_series < 0.0).any()) else np.nan,
                "min_dd_trade": float(pnl_series[pnl_series < 0.0].max()) if (not pnl_series.empty and (pnl_series < 0.0).any()) else np.nan,
               }


    @staticmethod
    def resolve_portfolio_config(portfolio_config: Optional[Dict[str, Any]],
                                factor_param_ranges: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ### What It Does
        Merges user portfolio config with the engine defaults.

        #### Responsibility
        Merges the user portfolio config with defaults and resolves `ParamValue`/string parameter references
        into concrete values. It does not itself clip exposure: the gross ceiling is enforced downstream by
        `compute_weight_panel` (per-panel `gross_target`) and `portfolio_bt` (`max_gross_exposure`).

        #### How To Use
        Call it before computing signal-derived portfolio weights.

        #### Key Parameters In Practice
        - `portfolio_config`
          - Portfolio execution configuration. Put shared-cash, force-cover, prevent-open, sleeve, and report-epsilon behavior here.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Optional[Dict[str, Any]]`.

        #### Usage Example
        `result = resolve_portfolio_config(...)`

        ---

        ### Parameters
        - `portfolio_config`: **Optional[Dict[str, Any]]**.

        #### Optional Parameters
        - `factor_param_ranges`: **Optional[Dict[str, Any]]** = *None*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        resolved = dict(portfolio_config or {})

        if not factor_param_ranges:

            return resolved

        params: Dict[str, Any] = {}

        for key, raw_value in factor_param_ranges.items():

            value = raw_value.unwrap() if isinstance(raw_value, FactorEngine.AutoParam.ParamValue) else raw_value
            value = cast(Any, value).item() if isinstance(value, np.generic) else value
            params[str(key)] = value

        for key, raw_value in list(resolved.items()):

            if isinstance(raw_value, FactorEngine.AutoParam.ParamValue):

                ref_key = str(raw_value.key)
                resolved[key] = params[ref_key] if ref_key in params else raw_value.unwrap()

                continue

            if isinstance(raw_value, str):
                ref_key = raw_value.strip()

                if ref_key in params:
                    resolved[key] = params[ref_key]


        return resolved


    @staticmethod
    def weighted_return_nav(*,
                            member_nav: pd.DataFrame,
                            weights: np.ndarray,
                            initial_cash: float) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        ### What It Does
        Combines member NAV streams into a weighted portfolio NAV.

        #### Responsibility
        Converts constituent NAVs and static weights into member returns, portfolio return, and portfolio NAV.

        #### How To Use
        Call it when combining several backtest members outside the normal portfolio executor.

        #### Key Parameters In Practice
        - `member_nav`
          - NAV stream for one portfolio member. Use it for attribution or weighted portfolio recombination.
          - Expected shape/type: `pd.DataFrame`.
        - `weights`
          - Collection of computed weights or sleeve weights. Keep its shape consistent with downstream normalization and portfolio input expectations.
          - Expected shape/type: `np.ndarray`.
        - `initial_cash`
          - Starting capital for a run. It must be positive numeric because engines use it as the base for NAV, cash-sharing, and performance normalization.
          - Expected shape/type: `float`.

        #### Usage Example
        `result = weighted_return_nav(...)`

        ---

        ### Parameters
        - `member_nav`: **pd.DataFrame**.
        - `weights`: **np.ndarray**.
        - `initial_cash`: **float**.

        ---

        ### Returns
        - `result`: **Tuple[pd.DataFrame, pd.Series, pd.Series]**.
        """

        nav_frame = pd.DataFrame(member_nav).apply(pd.to_numeric, errors = "coerce").astype(float)
        member_ret = nav_frame.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        combo_ret = pd.Series(member_ret.to_numpy(dtype = float) @ np.asarray(weights, dtype = float), index = member_ret.index, dtype = float)
        combo_nav = (1.0 + combo_ret).cumprod() * float(initial_cash)


        return member_ret, combo_ret, combo_nav


    @staticmethod
    def split_portfolio_test_data(test_data: Optional[pd.DataFrame], cal_column: str = "Close") -> Dict[str, pd.DataFrame]:
        """
        ### What It Does
        Splits a portfolio test frame into per-underlying frames.

        #### Responsibility
        Creates asset-specific DataFrames while preserving calendar and column semantics for each member.

        #### How To Use
        Call it before running batch portfolio branches over individual underlyings.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `Optional[pd.DataFrame]`.
        - `cal_column`
          - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = split_portfolio_test_data(...)`

        ---

        ### Parameters
        - `test_data`: **Optional[pd.DataFrame]**.

        #### Optional Parameters
        - `cal_column`: **str** = *"Close"*.

        ---

        ### Returns
        - `result`: **Dict[str, pd.DataFrame]**.
        """

        def _extract_underlyings(test_data: pd.DataFrame, cal_column: str = "Close") -> List[str]:

            if test_data is None or not isinstance(test_data, pd.DataFrame) or test_data.empty:

                return []

            suffix = f"_{str(cal_column).strip()}".lower()
            underlyings: List[str] = []

            for col in test_data.columns:

                if not isinstance(col, str):

                    continue

                low = col.lower()

                if not low.endswith(suffix):

                    continue

                name = col[:-len(suffix)]

                if name:
                    underlyings.append(name)


            return sorted(set(underlyings))


        if test_data is None or not isinstance(test_data, pd.DataFrame) or test_data.empty:

            return {}

        resolved_cal_column = str(cal_column).strip() if isinstance(cal_column, str) else "Close"

        if not resolved_cal_column:
            resolved_cal_column = "Close"

        underlyings = _extract_underlyings(test_data, cal_column = resolved_cal_column)

        if len(underlyings) < 1 and resolved_cal_column.lower() != "close":
            underlyings = _extract_underlyings(test_data, cal_column = "Close")

        if len(underlyings) < 1:

            return {}

        prefixes = [f"{u}_" for u in underlyings]
        prefix_set = tuple(prefixes)
        shared_cols = [c for c in test_data.columns if not (isinstance(c, str) and c.startswith(prefix_set))]
        shared_frame = test_data[shared_cols].copy()
        shared_frame.attrs = {}

        if "__zora_row_id__" not in shared_frame.columns:
            shared_frame["__zora_row_id__"] = np.arange(len(test_data), dtype = np.int64)

        single_frames: Dict[str, pd.DataFrame] = {}

        for u in underlyings:

            prefix = f"{u}_"
            asset_cols = [c for c in test_data.columns if isinstance(c, str) and c.startswith(prefix)]
            frame = shared_frame.copy()
            frame.attrs = {}

            if asset_cols:
                asset_frame = test_data[asset_cols].copy()
                asset_frame.attrs = {}
                frame = pd.concat([frame, asset_frame], axis = 1)
                frame.attrs = {}

                rename_allow = {"Open", "High", "Low", "Close", "Volume", "target", "net", "Datetime"}
                alias_map: Dict[str, str] = {}

                for c in asset_cols:
                    raw_field = c[len(prefix):]

                    if raw_field in rename_allow:
                        alias_map[c] = raw_field

                    elif raw_field.lower() == resolved_cal_column.lower():
                        alias_map[c] = "Close"

                # Prefer underlying-scoped columns when name conflicts with shared/global columns.
                prefer_asset_cols = {"Open", "High", "Low", "Close", "Volume", "target", "net"}

                for src_col, dst_col in alias_map.items():

                    if dst_col not in frame.columns or dst_col in prefer_asset_cols:
                        frame[dst_col] = frame[src_col]

            if "Datetime" not in frame.columns:
                frame["Datetime"] = pd.to_datetime(test_data.index, errors = "coerce")

            valuation_column = resolved_cal_column if resolved_cal_column in frame.columns else "Close"

            if valuation_column not in frame.columns:

                # Portfolio split frame must retain at least one valuation column
                # (preferred cal_column, fallback Close) for downstream NAV/margin flow.

                continue

            # Portfolio merge may introduce sparse rows per asset (e.g., outer join).
            # Remove rows where the active valuation column is missing to prevent NAV/metric NaN propagation.
            frame[valuation_column] = pd.to_numeric(frame[valuation_column], errors = "coerce")
            frame = frame.loc[frame[valuation_column].notna()].copy()

            if frame.empty:

                continue

            if "Datetime" in frame.columns:

                frame["Datetime"] = pd.to_datetime(frame["Datetime"], errors = "coerce")
                frame = frame.loc[frame["Datetime"].notna()].copy()

                if frame.empty:

                    continue

                frame = (frame.assign(__zora_dt_sort__ = pd.to_datetime(frame["Datetime"], errors = "coerce")).sort_values("__zora_dt_sort__").drop(columns = ["__zora_dt_sort__"]))

            # Trim FactorEngine registry to current single-asset frame columns;
            # otherwise semantic/feature lookup may accidentally target other assets.
            registry_key = "_zora_factor_registry"
            registry = frame.attrs.get(registry_key)

            if isinstance(registry, dict):
                registry_cols = registry.get("columns")

                if isinstance(registry_cols, dict):

                    frame_col_set = {c for c in frame.columns if isinstance(c, str)}
                    filtered_cols = {k: v for k, v in registry_cols.items() if k in frame_col_set}

                    registry_copy = dict(registry)
                    registry_copy["columns"] = filtered_cols
                    frame.attrs[registry_key] = registry_copy

            feature_map_key = FactorEngine.AutoParam.FEATURE_MAP_KEY
            feature_map = frame.attrs.get(feature_map_key)

            if isinstance(feature_map, dict):
                remapped_refs: Dict[str, str] = {}

                for sem, ref in feature_map.items():
                    ref_text = str(ref)
                    new_ref = ref_text

                    for src in underlyings:

                        if src == u:

                            continue

                        if new_ref.startswith(f"{src}."):
                            new_ref = f"{u}.{new_ref[len(src) + 1:]}"

                        if new_ref.startswith(f"{src}_"):
                            new_ref = f"{u}_{new_ref[len(src) + 1:]}"

                        new_ref = new_ref.replace(f"{src}_", f"{u}_")

                    # split frame keeps generic OHLCV names; remap prefixed refs when applicable
                    prefixed_ohlcv = {
                                        f"{u}_Open": "Open",
                                        f"{u}_High": "High",
                                        f"{u}_Low": "Low",
                                        f"{u}_Close": "Close",
                                        f"{u}_Volume": "Volume",
                                        f"{u}_target": "target",
                                        f"{u}_net": "net",
                                        f"{u}_Datetime": "Datetime",
                                    }

                    if resolved_cal_column.lower() != "close":
                        prefixed_ohlcv[f"{u}_{resolved_cal_column}"] = "Close"

                    if new_ref in prefixed_ohlcv and prefixed_ohlcv[new_ref] in frame.columns:
                        new_ref = prefixed_ohlcv[new_ref]

                    remapped_refs[str(sem)] = new_ref

                frame.attrs[feature_map_key] = remapped_refs

            frame.attrs["__zora_ctx_source__"] = "portfolio_split"
            frame.attrs["__zora_ctx_underlyings__"] = [u]
            single_frames[u] = frame

        return single_frames


    @staticmethod
    def portfolio_inputs(test_data: pd.DataFrame,
                        weight_panel: pd.DataFrame,
                        *,
                        cal_column: str = "Close",
                        portfolio_frames: Optional[Dict[str, pd.DataFrame]] = None,
                        portfolio_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ### What It Does
        Builds aligned price, signal, and weight panels for portfolio execution.

        #### Responsibility
        Prepares close/open data, target panels, force-flat masks, and portfolio frames under one alignment contract.

        #### How To Use
        Call it before `portfolio_bt(...)` when inputs arrive as frames or signal panels.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `weight_panel`
          - Target-weight matrix. Rows must align to execution timestamps and columns to assets; portfolio paths require matching asset order.
          - Expected shape/type: `pd.DataFrame`.
        - `cal_column`
          - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
          - Expected shape/type: `str`.
        - `portfolio_frames`
          - Per-underlying split frames. They must retain enough datetime and price columns for independent batch or portfolio input construction.
          - Expected shape/type: `Optional[Dict[str, pd.DataFrame]]`.

        #### Usage Example
        `result = portfolio_inputs(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `weight_panel`: **pd.DataFrame**.

        #### Optional Parameters
        - `cal_column`: **str** = *"Close"*.
        - `portfolio_frames`: **Optional[Dict[str, pd.DataFrame]]** = *None*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        if portfolio_frames is None:
            portfolio_frames = {}

        calendar = pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce")
        calendar = pd.DatetimeIndex(calendar)
        weight_panel_df = pd.DataFrame(weight_panel).reindex(calendar)


        def _build_price_panels(test_data: pd.DataFrame,
                                underlyings: List[str],
                                cal_column: str = "Close") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

            calendar = pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce")

            calendar = pd.DatetimeIndex(calendar)
            resolved_cal_column = str(cal_column).strip() if isinstance(cal_column, str) else "Close"

            if not resolved_cal_column:
                resolved_cal_column = "Close"

            close_parts: Dict[str, pd.Series] = {}
            open_parts: Dict[str, pd.Series] = {}

            for underlying in underlyings:
                close_candidates = [f"{underlying}_{resolved_cal_column}"]

                if resolved_cal_column.lower() != "close":
                    close_candidates.append(f"{underlying}_Close")

                open_candidates = [f"{underlying}_Open"]
                close_series = None
                open_series = None

                for col in close_candidates:

                    if col in test_data.columns:
                        close_series = pd.Series(
                                                pd.to_numeric(test_data[col], errors = "coerce").to_numpy(dtype = float),
                                                index = calendar,
                                                name = underlying,
                                                )

                        break

                for col in open_candidates:

                    if col in test_data.columns:
                        open_series = pd.Series(
                                                pd.to_numeric(test_data[col], errors = "coerce").to_numpy(dtype = float),
                                                index = calendar,
                                                name = underlying,
                                                )

                        break

                if close_series is None:

                    continue

                if open_series is None:
                    open_series = close_series.copy()

                close_parts[underlying] = close_series
                open_parts[underlying] = open_series

            close_panel_raw = pd.DataFrame(close_parts, index = calendar, dtype = float).sort_index()
            open_panel_raw = pd.DataFrame(open_parts, index = calendar, dtype = float).sort_index()
            close_panel_raw = close_panel_raw.apply(pd.to_numeric, errors = "coerce")
            open_panel_raw = open_panel_raw.apply(pd.to_numeric, errors = "coerce")
            close_panel_raw = close_panel_raw.mask(lambda df: df <= 0.0)
            open_panel_raw = open_panel_raw.mask(lambda df: df <= 0.0)
            tradable_mask = close_panel_raw.notna()
            close_panel = close_panel_raw
            open_panel = open_panel_raw


            return open_panel, close_panel, tradable_mask


        panel_columns = [str(col) for col in weight_panel_df.columns]

        if len(panel_columns) <= 1:

            if len(portfolio_frames) <= 1:
                portfolio_frames = PortfolioEngine.split_portfolio_test_data(test_data = test_data, cal_column = cal_column)

            panel_columns = sorted(portfolio_frames.keys())

        open_panel, close_panel, tradable_mask = _build_price_panels(test_data = test_data, underlyings = panel_columns, cal_column = cal_column)

        if close_panel.empty:

            raise ValueError("[WARNING] portfolio mode requires at least one priced underlying.")

        aligned_weight_panel = weight_panel_df.reindex(index = close_panel.index, columns = close_panel.columns).fillna(0.0).astype(float)
        aligned_weight_panel = aligned_weight_panel.where(tradable_mask.reindex_like(aligned_weight_panel).fillna(False), np.nan)

        if portfolio_config:
            from .Position import Position  # deferred import: breaks circular dependency
            force_flat_arr, prevent_open_arr = Position.build_event_risk_masks(close_panel.index, portfolio_config)
            force_flat_mask = pd.Series(force_flat_arr, index = close_panel.index, dtype = bool)
            prevent_open_mask = pd.Series(prevent_open_arr, index = close_panel.index, dtype = bool)

        else:
            force_flat_mask = pd.Series(False, index = close_panel.index, dtype = bool)
            prevent_open_mask = pd.Series(False, index = close_panel.index, dtype = bool)


        return {

            "weight_panel": aligned_weight_panel,
            "open_panel": open_panel,
            "close_panel": close_panel,
            "portfolio_frames": portfolio_frames,
            "force_flat_mask": force_flat_mask,
            "prevent_open_mask": prevent_open_mask,
        }


    @staticmethod
    def portfolio_bt(*,
                    btengine: "BacktestEngine_VBT",
                    weight_panel: pd.DataFrame,
                    open_panel: pd.DataFrame,
                    close_panel: pd.DataFrame,
                    benchmark_series: Optional[pd.Series] = None,
                    risk_free_rate: float = 0.02,
                    force_flat_mask: Optional[pd.Series] = None,
                    prevent_open_mask: Optional[pd.Series] = None,
                    max_gross_exposure: Optional[float] = None,
                    close_delisted_at_last: bool = False) -> Dict[str, Any]:
        """
        ### What It Does
        Executes a portfolio-level vectorbt backtest from aligned panels.

        #### Responsibility
        Converts target weights and price panels into executable vectorbt orders and report fields.

        #### How To Use
        Call it after `portfolio_inputs(...)` has prepared aligned panels.

        #### Key Parameters In Practice
        - `btengine`
          - Backtest engine instance or class. It defines which execution backend actually consumes prices, signals, and weights.
          - Expected shape/type: `"BacktestEngine_VBT"`.
        - `weight_panel`
          - Target-weight matrix. Rows must align to execution timestamps and columns to assets; portfolio paths require matching asset order.
          - Expected shape/type: `pd.DataFrame`.
        - `open_panel`
          - Open-price matrix for execution. Use it when the strategy or backend executes at open rather than close.
          - Expected shape/type: `pd.DataFrame`.
        - `close_panel`
          - Close-price matrix for portfolio execution. It is the price authority used to build VBT orders and shared-cash NAV.
          - Expected shape/type: `pd.DataFrame`.
        - `benchmark_series`
          - Benchmark NAV/price series for relative reporting. Align it to the tested period instead of filling missing pre-history.
          - Expected shape/type: `Optional[pd.Series]`.
        - `risk_free_rate`
          - Annual risk-free assumption threaded into `performance_metrics` (Sharpe/Sortino excess returns). Pass the caller's configured rate; leaving it at the default silently pins Sharpe to a fixed 0.02.
          - Expected shape/type: `float`.
        - `force_flat_mask`
          - Boolean mask that forces flat exposure. Use it for holidays, macro dates, weekends, or other no-hold periods.
          - Expected shape/type: `Optional[pd.Series]`.
        - `prevent_open_mask`
          - Boolean mask that blocks new entries. Use it to encode dates/timestamps where opening risk is disallowed.
          - Expected shape/type: `Optional[pd.Series]`.
        - `max_gross_exposure`
          - Per-row gross (sum of |weight|) ceiling for the shared-cash book. `None` resolves to `1.0` (fully invested, no leverage); set it above 1.0 to authorize intended leverage. Rows exceeding it are scaled down and a warning is logged.
          - Expected shape/type: `Optional[float]`.

        #### Usage Example
        `result = portfolio_bt(...)`

        ---

        ### Parameters
        - `btengine`: **"BacktestEngine_VBT"**.
        - `weight_panel`: **pd.DataFrame**.
        - `open_panel`: **pd.DataFrame**.
        - `close_panel`: **pd.DataFrame**.

        #### Optional Parameters
        - `benchmark_series`: **Optional[pd.Series]** = *None*.
        - `risk_free_rate`: **float** = *0.02*.
        - `force_flat_mask`: **Optional[pd.Series]** = *None*.
        - `prevent_open_mask`: **Optional[pd.Series]** = *None*.
        - `max_gross_exposure`: **Optional[float]** = *None*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        from .Manager import BacktestEngineManager  # deferred import: breaks circular dependency

        if vbt is None:

            raise ImportError("[WARNING] vectorbt is unavailable; portfolio VBT path cannot run.")

        weight_panel = weight_panel.apply(pd.to_numeric, errors = "coerce").astype(float)

        # Leverage guard (universal sink): cap each row's gross (sum of |weight|) before it reaches
        # cash_sharing=True targetpercent. Whatever upstream path produced the panel -- single sleeve,
        # blended sleeves, feature-driven, external caller, or a stale cached artifact -- a row whose
        # gross exceeds the ceiling would be read as >100% of shared cash, silently over-allocating
        # (long-only concentration collapse) or levering (long/short). None -> 1.0 (no leverage).
        gross_cap = 1.0 if max_gross_exposure is None else float(max_gross_exposure)

        if np.isfinite(gross_cap) and gross_cap > 0.0:
            row_gross = weight_panel.abs().sum(axis = 1)
            over_budget = row_gross > (gross_cap + 1e-12)

            if bool(over_budget.any()):

                logging.warning(
                            "[WARNING] Portfolio target gross exceeded the %.4g cap on %d of %d rows "
                            "(max observed gross %.4g); scaling those rows down to the cap. "
                            "Set portfolio_config['max_gross_exposure'] to authorize intended leverage.",
                            gross_cap,
                            int(over_budget.sum()),
                            int(len(row_gross)),
                            float(row_gross.max()),
                            )
                row_scale = pd.Series(1.0, index = weight_panel.index, dtype = float)
                row_scale.loc[over_budget] = gross_cap / row_gross.loc[over_budget]
                weight_panel = weight_panel.mul(row_scale, axis = 0)

        effective_slippage = float(btengine.slippage) + (float(btengine.spread) / 2.0)

        open_panel = (open_panel.apply(pd.to_numeric, errors = "coerce").replace([np.inf, -np.inf], np.nan).mask(lambda df: df <= 0.0, np.nan).astype(float))
        close_panel = (close_panel.apply(pd.to_numeric, errors = "coerce").replace([np.inf, -np.inf], np.nan).mask(lambda df: df <= 0.0, np.nan).astype(float))
        # Causal missing-open settlement (opt-in). The old implementation scanned
        # the COMPLETE future open panel, found each column's final valid row and
        # rewrote that historical target to zero. Appending one future bar could
        # therefore change an already-reported trade. Here a missing open is acted
        # on only when that row is reached: any carried position is cash-settled
        # at the most recent close known strictly before the row. A temporary gap
        # may reopen later under a later target; no future permanence test exists.
        missing_open_mask = open_panel.isna()
        settlement_price_panel = close_panel.ffill().shift(1)
        settlement_mask = missing_open_mask & settlement_price_panel.notna()
        execution_open_panel = (
            open_panel.where(~settlement_mask, settlement_price_panel)
            if close_delisted_at_last else open_panel
        )
        daily_index = pd.DatetimeIndex(pd.to_datetime(close_panel.index, errors = "coerce"))

        force_flat_mask_arr = (
                                np.zeros(len(daily_index), dtype = bool)

                                if force_flat_mask is None

                                else np.asarray(pd.Series(force_flat_mask, index = daily_index).fillna(False).to_numpy(dtype = bool), dtype = bool)
                              )
        prevent_open_mask_arr = (
                                np.zeros(len(daily_index), dtype = bool)

                                if prevent_open_mask is None

                                else np.asarray(pd.Series(prevent_open_mask, index = daily_index).fillna(False).to_numpy(dtype = bool), dtype = bool)
                                )

        resolved_freq = BacktestEngineManager._infer_vbt_freq(close_panel.index)
        intraday_returns = close_panel.div(open_panel).sub(1.0).replace([np.inf, -np.inf], np.nan)
        overnight_returns = open_panel.div(close_panel.shift(1)).sub(1.0).replace([np.inf, -np.inf], np.nan)
        open_event_index = pd.DatetimeIndex(daily_index)
        close_event_index = pd.DatetimeIndex(daily_index + pd.to_timedelta(1, unit = "ns"))
        open_event_prices = execution_open_panel.copy()
        close_event_prices = close_panel.copy()
        open_event_prices.index = open_event_index
        close_event_prices.index = close_event_index
        event_price_panel = pd.concat([open_event_prices, close_event_prices], axis = 0).sort_index()
        event_price_nan_count = int(event_price_panel.isna().sum().sum())
        valuation_event_price_panel = event_price_panel.ffill()

        if event_price_nan_count > 0:

            logging.warning(
                        "[WARNING] Portfolio valuation price panel contains %s NaN values; fallback to mark_to_last for NAV valuation. "
                        "Execution prices remain strict and are not forward-filled.",
                        event_price_nan_count,
                        )

        # signal_start_t is a warmup cutoff (nav_series[:start_t] is blanked below). Zero the warmup rows of
        # weight_panel so BOTH the executed portfolio (event_weight_panel) and the reported weight frames stay
        # flat during warmup on caller-injected non-zero weights -- mirrors the VBTEngine guard so both engines
        # behave identically off the normal path, where Position already zeros these rows.
        warmup_cutoff = max(0, int(getattr(btengine, "signal_start_t", 2)))

        if warmup_cutoff > 0 and len(weight_panel) > 0:
            weight_panel = weight_panel.copy()
            weight_panel.iloc[: min(warmup_cutoff, len(weight_panel)), :] = 0.0

        open_event_targets = weight_panel.copy()
        open_event_targets.index = open_event_index

        if close_delisted_at_last:
            settlement_targets = settlement_mask.copy()
            settlement_targets.index = open_event_index
            open_event_targets = open_event_targets.mask(settlement_targets, 0.0)

        if np.any(prevent_open_mask_arr):
            open_event_targets.loc[open_event_index[prevent_open_mask_arr], :] = np.nan

        close_event_targets = pd.DataFrame(np.nan, index = close_event_index, columns = weight_panel.columns, dtype = float)

        if np.any(force_flat_mask_arr):
            close_event_targets.loc[close_event_index[force_flat_mask_arr], :] = 0.0

        event_weight_panel = pd.concat([open_event_targets, close_event_targets], axis = 0).sort_index()

        portfolio = vbt.Portfolio.from_orders(
                                                close = valuation_event_price_panel,
                                                size = event_weight_panel,
                                                size_type = "targetpercent",
                                                price = event_price_panel,
                                                val_price = valuation_event_price_panel,
                                                init_cash = float(btengine.initial_cash),
                                                cash_sharing = True,
                                                slippage = effective_slippage,
                                                fees = float(btengine.fees),
                                                freq = resolved_freq,
                                             )

        nav_raw = pd.to_numeric(pd.Series(portfolio.value()), errors = "coerce").reindex(valuation_event_price_panel.index)
        nav_open_series = pd.Series(nav_raw.reindex(open_event_index).to_numpy(dtype = float), index = daily_index, dtype = float)
        nav_close_series = pd.Series(nav_raw.reindex(close_event_index).to_numpy(dtype = float), index = daily_index, dtype = float)
        # Realized weights come from vectorbt's filled position state, not from
        # the requested targets. This captures rejected orders, missing-price
        # carry, partial/rounded fills, forced closes and mark-to-market drift.
        event_asset_value = pd.DataFrame(portfolio.asset_value(group_by = False))
        event_asset_value = event_asset_value.reindex(index = event_price_panel.index,
                                                      columns = weight_panel.columns)
        event_total_value = pd.to_numeric(
            pd.Series(portfolio.value(group_by = True)), errors = "coerce"
        ).reindex(event_price_panel.index)
        event_realized_weights = event_asset_value.div(
            event_total_value.replace(0.0, np.nan), axis = 0
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)

        weights_open_df = pd.DataFrame(
            event_realized_weights.reindex(open_event_index).to_numpy(dtype = float),
            index = daily_index, columns = weight_panel.columns, dtype = float,
        )
        weights_close_df = pd.DataFrame(
            event_realized_weights.reindex(close_event_index).to_numpy(dtype = float),
            index = daily_index, columns = weight_panel.columns, dtype = float,
        )
        weights_prev_close_df = weights_close_df.shift(1).fillna(0.0)

        # Actual order turnover: absolute filled notional divided by portfolio
        # value at the execution event, aggregated over open+close for each bar.
        # A constant TARGET can still rebalance after prices move, so target
        # delta is not an execution statistic.
        turnover_event_arr = np.zeros(len(event_price_panel.index), dtype = float)
        order_records = pd.DataFrame(portfolio.orders.records)

        if not order_records.empty:
            order_idx = pd.to_numeric(order_records["idx"], errors = "coerce").to_numpy(dtype = float)
            valid_order = np.isfinite(order_idx)
            order_pos = order_idx[valid_order].astype(int)
            valid_order &= (order_idx >= 0) & (order_idx < len(turnover_event_arr))
            order_pos = order_idx[valid_order].astype(int)
            order_notional = (
                pd.to_numeric(order_records.loc[valid_order, "size"], errors = "coerce").abs()
                * pd.to_numeric(order_records.loc[valid_order, "price"], errors = "coerce").abs()
            ).fillna(0.0).to_numpy(dtype = float)
            event_value_arr = event_total_value.abs().to_numpy(dtype = float)
            denom = event_value_arr[order_pos]
            ratios = np.divide(order_notional, denom,
                               out = np.zeros_like(order_notional, dtype = float),
                               where = np.isfinite(denom) & (denom > 1e-12))
            np.add.at(turnover_event_arr, order_pos, ratios)

        turnover_event = pd.Series(turnover_event_arr,
                                   index = event_price_panel.index, dtype = float)
        turnover_series = pd.Series(
            turnover_event.reindex(open_event_index).to_numpy(dtype = float)
            + turnover_event.reindex(close_event_index).to_numpy(dtype = float),
            index = daily_index, dtype = float,
        )
        nav_series = nav_close_series.copy()
        nav_series.iloc[: max(0, int(getattr(btengine, "signal_start_t", 2)))] = np.nan
        returns_series = nav_series.div(nav_series.shift(1)).sub(1.0)

        if not nav_series.empty:
            returns_series.iloc[0] = float(nav_series.iloc[0] / float(btengine.initial_cash)) - 1.0

        returns_series = pd.to_numeric(pd.Series(returns_series), errors = "coerce").fillna(0.0)
        returns_series.index = daily_index


        def _build_position_panel(*,
                                  weights_prev_close: pd.DataFrame,
                                  weights_open: pd.DataFrame,
                                  weights_close: pd.DataFrame,
                                  nav_close_series: pd.Series,
                                  overnight_returns: pd.DataFrame,
                                  intraday_returns: pd.DataFrame) -> pd.DataFrame:

            overnight_ret = overnight_returns.reindex(index = weights_open.index, columns = weights_open.columns).apply(pd.to_numeric, errors = "coerce")
            intraday_ret = intraday_returns.reindex(index = weights_open.index, columns = weights_open.columns).apply(pd.to_numeric, errors = "coerce")
            nav_prev_close = pd.to_numeric(pd.Series(nav_close_series), errors = "coerce").shift(1)

            if len(nav_prev_close) > 0:
                nav_prev_close.iloc[0] = float(btengine.initial_cash)

            nav_prev_close = nav_prev_close.fillna(float(btengine.initial_cash))
            overnight_pnl = weights_prev_close.mul(overnight_ret.fillna(0.0), fill_value = 0.0).mul(nav_prev_close, axis = 0)
            nav_open = pd.to_numeric(pd.Series(nav_open_series), errors = "coerce").reindex(weights_open.index).fillna(nav_prev_close)
            intraday_pnl = weights_open.mul(intraday_ret.fillna(0.0), fill_value = 0.0).mul(nav_open, axis = 0)
            contribution_overnight = overnight_pnl / float(btengine.initial_cash)
            contribution_intraday = intraday_pnl / float(btengine.initial_cash)
            contribution = contribution_overnight.add(contribution_intraday, fill_value = 0.0)

            panel_index = pd.MultiIndex.from_product([weights_open.index, weights_open.columns], names = ["trade_date", "symbol"])

            position_panel = (
                                pd.DataFrame(
                                    {
                                        "portfolio_weight_prev_close": weights_prev_close.to_numpy().reshape(-1),
                                        "portfolio_weight_open": weights_open.to_numpy().reshape(-1),
                                        "portfolio_weight_close": weights_close.to_numpy().reshape(-1),
                                        "portfolio_weight": weights_open.to_numpy().reshape(-1),
                                        "portfolio_contribution_overnight": contribution_overnight.to_numpy().reshape(-1),
                                        "portfolio_contribution_intraday": contribution_intraday.to_numpy().reshape(-1),
                                        "portfolio_contribution": contribution.to_numpy().reshape(-1),
                                    },
                                    index = panel_index,
                                )
                                .reset_index()
                            )

            position_panel["portfolio_weight_prev_close"] = pd.to_numeric(position_panel["portfolio_weight_prev_close"], errors = "coerce").fillna(0.0)
            position_panel["portfolio_weight_open"] = pd.to_numeric(position_panel["portfolio_weight_open"], errors = "coerce").fillna(0.0)
            position_panel["portfolio_weight_close"] = pd.to_numeric(position_panel["portfolio_weight_close"], errors = "coerce").fillna(0.0)
            position_panel["portfolio_weight"] = pd.to_numeric(position_panel["portfolio_weight"], errors = "coerce").fillna(0.0)
            position_panel["portfolio_contribution_overnight"] = pd.to_numeric(position_panel["portfolio_contribution_overnight"], errors = "coerce").fillna(0.0)
            position_panel["portfolio_contribution_intraday"] = pd.to_numeric(position_panel["portfolio_contribution_intraday"], errors = "coerce").fillna(0.0)
            position_panel["portfolio_contribution"] = pd.to_numeric(position_panel["portfolio_contribution"], errors = "coerce").fillna(0.0)

            active_mask = (
                            (position_panel["portfolio_weight_prev_close"].abs() > 1e-12)
                            | (position_panel["portfolio_weight_open"].abs() > 1e-12)
                            | (position_panel["portfolio_contribution"].abs() > 1e-12)
                          )

            position_panel = position_panel.loc[active_mask].copy()

            if position_panel.empty:

                return pd.DataFrame()

            weight_arr = pd.to_numeric(position_panel["portfolio_weight"], errors = "coerce").fillna(0.0).to_numpy(dtype = float)
            position_panel["side"] = np.where(weight_arr > 1e-12, "long", np.where(weight_arr < -1e-12, "short", "flat"))
            position_panel["portfolio_contribution_rate"] = position_panel["portfolio_contribution"] * 100.0


            return position_panel.reset_index(drop = True)


        position_panel = _build_position_panel(
                                                weights_prev_close = weights_prev_close_df,
                                                weights_open = weights_open_df,
                                                weights_close = weights_close_df,
                                                nav_close_series = nav_close_series,
                                                overnight_returns = overnight_returns,
                                                intraday_returns = intraday_returns,
                                              )

        metric_map = BacktestEngineManager.performance_metrics(nav_series = nav_series, benchmark_series = benchmark_series, risk_free_rate = risk_free_rate)
        trade_stats = PortfolioEngine.TradeAccount(portfolio = portfolio, position_panel = position_panel)

        metric_map.update(trade_stats)
        underlying_metrics: Dict[str, Dict[str, Any]] = {}
        nav_prev_close = nav_close_series.shift(1)

        if len(nav_prev_close) > 0:
            nav_prev_close.iloc[0] = float(btengine.initial_cash)

        nav_prev_close = nav_prev_close.fillna(float(btengine.initial_cash))
        contribution_overnight = weights_prev_close_df.mul(overnight_returns.fillna(0.0), fill_value = 0.0).mul(nav_prev_close, axis = 0) / float(btengine.initial_cash)
        nav_open = pd.to_numeric(pd.Series(nav_open_series), errors = "coerce").reindex(weights_open_df.index).fillna(nav_prev_close)
        contribution_intraday = weights_open_df.mul(intraday_returns.fillna(0.0), fill_value = 0.0).mul(nav_open, axis = 0) / float(btengine.initial_cash)
        contribution_total = contribution_overnight.add(contribution_intraday, fill_value = 0.0)
        member_return_frame = contribution_total.copy()

        for underlying in weight_panel.columns:

            open_weight_series = pd.to_numeric(weights_open_df[underlying], errors = "coerce").fillna(0.0)
            contrib_series = pd.to_numeric(contribution_total[underlying], errors = "coerce").fillna(0.0)
            active_days = int((open_weight_series.abs() > 1e-12).sum())

            if active_days == 0 and float(contrib_series.abs().sum()) <= 1e-12:

                continue

            contrib_nav = float(btengine.initial_cash) * (1.0 + contrib_series.cumsum())
            contrib_metrics = BacktestEngineManager.performance_metrics(nav_series = contrib_nav, risk_free_rate = risk_free_rate)

            underlying_metrics[str(underlying)] = {
                                                    "nav_rate": contrib_metrics.get("nav_rate", np.nan),
                                                    "maxdd_rate": contrib_metrics.get("maxdd_rate", np.nan),
                                                    "sharpe_ratio": contrib_metrics.get("sharpe_ratio", np.nan),
                                                    "active_days": active_days,
                                                    "avg_abs_weight": float(open_weight_series.abs().mean()),
                                                  }

        underlying_trade_rec: Dict[str, List[Dict[str, Any]]] = {}

        if not position_panel.empty:

            for underlying, group in position_panel.groupby("symbol", sort = True):
                underlying_trade_rec[str(underlying)] = [{str(key): value for key, value in row.items()} for row in group.to_dict(orient = "records")]

        btengine.vbt_portfolio = portfolio
        btengine.event_price_panel = event_price_panel
        btengine.event_weight_panel = event_weight_panel
        btengine.benchmark_series_override = benchmark_series
        btengine.nav_line = pd.DataFrame({"NAV": nav_series.values}, index = nav_series.index)
        btengine.nav = float(nav_series.dropna().iloc[-1]) if not nav_series.dropna().empty else np.nan

        for metric_key, metric_value in metric_map.items():
            setattr(btengine, metric_key, metric_value)


        return {

                    "nav": nav_series,
                    "nav_close": nav_close_series,
                    "nav_open": nav_open_series,
                    "event_nav": nav_raw,
                    "returns": returns_series,
                    "portfolio": portfolio,
                    "event_price_panel": event_price_panel,
                    "event_weight_panel": event_weight_panel,
                    "weight_panel": weight_panel,
                    "executed_weight_panel": weights_open_df,
                    "executed_weight_close_panel": weights_close_df,
                    "turnover": turnover_series,
                    "position_panel": position_panel,
                    "member_return_frame": member_return_frame,
                    "metric_map": metric_map,
                    "underlying_metrics": underlying_metrics,
                    "underlying_trade_rec": underlying_trade_rec,
                }
