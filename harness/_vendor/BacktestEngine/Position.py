from ENV_MGMT.imports import *
from .Calendars import default_holidays, default_macro_release_dates
import FactorEngine




# Position Class  (portfolio-weight construction, split out of Signal)
#----------------------------------------------------------------------------------------
class Position:
    """
    ### What It Does
    Position turns regularized signal-strength panels into executable portfolio weights.

    #### Responsibility
    Owns weight-mode construction (margin/softsign, equal, raw, rank), selection modes,
    gross budgeting, temporal hold/rebalance logic, and event-risk force-cover masks.
    It is stateless apart from three read-only sizing scalars supplied by the owning Signal.

    #### How To Use
    Build it with the same sizing scalars as the owning `Signal`, then call
    `compute_weight_panel(...)` or `compute_multi_sleeve_weights(...)`.

    #### Usage Example
    `obj = Position(...)`

    ---

    ### Parameters
    #### Optional Parameters
    - `signal_z_window`: **int** = *120*.
    - `signal_epsilon`: **float** = *1e-9*.
    - `signal_start_t`: **int** = *2*.
    """

    def __init__(self,
                 signal_z_window: int = 120,
                 signal_epsilon: float = 1e-9,
                 signal_start_t: int = 2) -> None:

        self.signal_z_window = int(cast(Any, signal_z_window))
        self.signal_start_t = max(0, int(cast(Any, signal_start_t)))
        self.signal_epsilon = float(cast(Any, signal_epsilon))


    @staticmethod
    @njit(cache = True)
    def _signal_margin(net_arr: np.ndarray, z_window: int, epsilon: float) -> np.ndarray:

        n = len(net_arr)
        out = np.zeros(n, dtype = np.float64)

        for i in range(n):

            if i == 0:
                out[i] = 0.0

                continue

            start = 0

            if i > z_window:
                start = i - z_window

            count = 0
            mean = 0.0
            m2 = 0.0

            for j in range(start, i):
                x = net_arr[j]

                if np.isnan(x):

                    continue

                count += 1
                delta = x - mean
                mean += delta / count
                m2 += delta * (x - mean)

            if count <= 1:
                out[i] = 0.0

                continue

            std = np.sqrt(m2 / count)

            if std == 0.0 or np.isnan(std):
                out[i] = 0.0

            else:
                out[i] = net_arr[i] / (std + epsilon)


        return out


    @staticmethod
    @njit(cache = True)
    def _softsign_nb(margin_arr: np.ndarray) -> np.ndarray:

        out = np.zeros(len(margin_arr), dtype = np.float64)

        for i in range(len(margin_arr)):
            value = margin_arr[i]

            if np.isnan(value):
                out[i] = 0.0

            else:
                out[i] = value / (1.0 + abs(value))


        return out


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
    def _resolve_weight_config(weight_config: Optional[Dict[str, Any]], params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:

        resolved = dict(weight_config or {})

        if not params:

            return resolved

        for key, raw_value in list(resolved.items()):

            if isinstance(raw_value, FactorEngine.AutoParam.ParamValue):
                ref_key = str(raw_value.key)

                if ref_key in params:
                    resolved[key] = params[ref_key]

                else:
                    resolved[key] = raw_value.unwrap()

                continue

            if isinstance(raw_value, str):
                ref_key = raw_value.strip()

                if ref_key in params:
                    resolved[key] = params[ref_key]


        return resolved


    @staticmethod
    def build_event_risk_masks(index: Any, config: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        ### What It Does
        Builds `(force_flat_mask, prevent_open_mask)` boolean arrays aligned to `index` from the
        `force_cover_weekend` / `force_cover_holidays` / `force_cover_macros` event-risk switches.

        #### Responsibility
        Single source of truth for event-risk dates so both the weight constructor and the portfolio
        execution layer flatten/block on identical calendars. `force_cover_*` means FORCE COVER: it both
        flattens existing exposure (force_flat) and blocks new opens (prevent_open); holidays additionally
        flatten on the pre-holiday session and block opens on the holiday itself.

        #### How To Use
        Call it with a datetime-like index and the portfolio/weight config dict.

        ---

        ### Parameters
        - `index`: **Any** (datetime-like, coerced to `pd.DatetimeIndex`).
        - `config`: **Optional[Dict[str, Any]]** = *None*.

        ---

        ### Returns
        - `result`: **Tuple[np.ndarray, np.ndarray]** = *(force_flat_mask, prevent_open_mask)*.
        """

        config = dict(config or {})
        idx = pd.DatetimeIndex(pd.to_datetime(index, errors = "coerce"))
        force_flat_mask = np.zeros(len(idx), dtype = bool)
        prevent_open_mask = np.zeros(len(idx), dtype = bool)

        if len(idx) == 0:

            return force_flat_mask, prevent_open_mask

        date_idx = idx.normalize()
        force_cover_weekend = config.get("force_cover_weekend", config.get("force_flat_weekdays", None))
        force_cover_holidays = config.get("force_cover_holidays", False)
        force_cover_macros = config.get("force_cover_macros", False)

        if isinstance(force_cover_weekend, (list, tuple, set, np.ndarray)):
            force_cover_weekdays = {int(day) for day in cast(Iterable[Any], force_cover_weekend) if pd.notna(day) and 0 <= int(day) <= 6}

        elif force_cover_weekend is None:
            force_cover_weekdays = set()

        elif isinstance(force_cover_weekend, (int, np.integer)) and not isinstance(force_cover_weekend, bool):

            raw_day = int(force_cover_weekend)
            force_cover_weekdays = {raw_day} if 0 <= raw_day <= 6 else set()

        else:
            force_cover_weekdays = {4} if bool(force_cover_weekend) else set()

        if force_cover_weekdays:
            weekend_hits = np.asarray([int(day) in force_cover_weekdays for day in date_idx.weekday], dtype = bool)
            prevent_open_mask |= weekend_hits
            force_flat_mask |= weekend_hits

        if force_cover_holidays is True:
            normalized_holidays = pd.DatetimeIndex(pd.to_datetime(default_holidays(), errors = "coerce")).normalize().dropna().unique()

        elif force_cover_holidays is False or force_cover_holidays is None:
            normalized_holidays = pd.DatetimeIndex([])

        else:
            normalized_holidays = pd.DatetimeIndex(pd.to_datetime(force_cover_holidays, errors = "coerce")).normalize().dropna().unique()

        if len(normalized_holidays) > 0:

            holiday_dates = set(pd.DatetimeIndex(normalized_holidays).date)
            pre_holiday_dates = set((pd.DatetimeIndex(normalized_holidays) - pd.offsets.BDay(1)).date)

            for t, current_date in enumerate(date_idx.date):

                if current_date in holiday_dates:
                    prevent_open_mask[t] = True

                if current_date in pre_holiday_dates:
                    force_flat_mask[t] = True

        if force_cover_macros:

            if force_cover_macros is True:
                normalized_macro = pd.DatetimeIndex(pd.to_datetime(default_macro_release_dates(), errors = "coerce")).normalize().dropna().unique()

            else:
                normalized_macro = pd.DatetimeIndex(pd.to_datetime(force_cover_macros, errors = "coerce")).normalize().dropna().unique()

            if len(normalized_macro) > 0:
                # Mirror the holiday branch so the book is flat THROUGH the release: force_flat on the close of
                # the prior business day (D-1) and block new opens on the release day (D) itself. The previous
                # code shifted BOTH masks to D-1 only, leaving the volatile release day fully tradeable.
                macro_release_dates = set(pd.DatetimeIndex(normalized_macro).date)
                pre_macro_dates = set((pd.DatetimeIndex(normalized_macro) - pd.offsets.BDay(1)).date)

                for t, current_date in enumerate(date_idx.date):

                    if current_date in macro_release_dates:
                        prevent_open_mask[t] = True

                    if current_date in pre_macro_dates:
                        force_flat_mask[t] = True


        return force_flat_mask, prevent_open_mask


    def compute_weight_panel(self,
                            *,
                            test_data: pd.DataFrame,
                            signal_strength_panel: pd.DataFrame,
                            weight_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        ### What It Does
        Converts signal-strength panels into normalized portfolio weights.

        #### Responsibility
        Applies portfolio configuration, exposure constraints, and calendar alignment to produce executable weights.

        #### How To Use
        Call it before VBT or for-loop backtests need target weights.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `signal_strength_panel`
          - Aligned signal-strength matrix. Rows must match the trading calendar and columns must match assets or sleeves expected by the weighting path.
          - Expected shape/type: `pd.DataFrame`.
        - `weight_config`
          - Signal-to-weight configuration. Use it to choose weighting mode, clipping, normalization, and other portfolio weight construction behavior.
          - Expected shape/type: `Optional[Dict[str, Any]]`.

        #### Usage Example
        `result = compute_weight_panel(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `signal_strength_panel`: **pd.DataFrame**.

        #### Optional Parameters
        - `weight_config`: **Optional[Dict[str, Any]]** = *None*.

        ---

        ### Returns
        - `result`: **pd.DataFrame**.
        """

        signal_strength_df = pd.DataFrame(signal_strength_panel).apply(pd.to_numeric, errors = "coerce").astype(float)
        config = dict(weight_config or {})
        weight_mode = str(config.get("weight_mode", "margin_softsign")).strip().lower() or "margin_softsign"

        if weight_mode == "equal":
            weight_source_panel = np.sign(signal_strength_df).astype(float)

        elif weight_mode in {"raw_strength", "rank"}:
            weight_source_panel = signal_strength_df.astype(float)

        elif weight_mode == "margin_softsign":
            weight_source_panel = pd.DataFrame(0.0, index = signal_strength_df.index, columns = signal_strength_df.columns, dtype = float)

            for column in signal_strength_df.columns:
                margin_arr = self._signal_margin(signal_strength_df[column].to_numpy(dtype = float), int(self.signal_z_window), float(self.signal_epsilon))
                weight_source_panel[column] = self._softsign_nb(margin_arr)

            weight_source_panel = weight_source_panel.astype(float)

        else:

            raise ValueError(f"[WARNING] unsupported weight_mode: {weight_mode!r}")

        selection_mode = str(config.get("selection_mode", "none")).strip().lower() or "none"
        side_mode = str(config.get("side_mode", "neutral")).strip().lower() or "neutral"
        require_signal_sign = bool(config.get("require_signal_sign", True))
        allow_partial_book = bool(config.get("allow_partial_book", True))
        gross_target = float(config.get("gross_target", 1.0))
        long_k = config.get("long_k")
        short_k = config.get("short_k")
        long_q = float(config.get("long_q", config.get("top_q", 0.20)))
        short_q = float(config.get("short_q", config.get("top_q", 0.20)))
        long_percentile = float(config.get("long_percentile", config.get("percentile", 0.20)))
        short_percentile = float(config.get("short_percentile", config.get("percentile", 0.20)))
        selection_sign_epsilon = max(float(config.get("selection_sign_epsilon", 1e-12)), 0.0)
        weight_df = pd.DataFrame(weight_source_panel).reindex(index = signal_strength_df.index, columns = signal_strength_df.columns).fillna(0.0).astype(float)

        def _coerce_selection_mask(raw_mask: Any) -> Optional[np.ndarray]:

            if raw_mask is None:

                return None

            mask_df = pd.DataFrame(raw_mask).reindex(index = signal_strength_df.index, columns = signal_strength_df.columns)
            stacked = mask_df.stack()
            non_null = stacked.dropna()

            # Classify boolean-ness from the NON-NULL cells only. A bool eligibility panel reindexed
            # to wider labels upcasts to object/float + NaN, and pandas 3.x stack() keeps those NaN;
            # testing the raw stacked values would mislabel it as non-bool and fall through to notna(),
            # which maps explicit False (banned/halted/unlisted) to True (eligible).
            stacked_is_bool = (not non_null.empty) and bool(np.asarray(non_null.map(lambda value: isinstance(value, (bool, np.bool_))), dtype = bool).all())

            if stacked_is_bool:

                return mask_df.fillna(False).astype(bool).to_numpy(dtype = bool, copy = False)

            return mask_df.notna().to_numpy(dtype = bool, copy = False)


        long_eligibility_arr = _coerce_selection_mask(config.get("long_eligibility_panel"))
        short_eligibility_arr = _coerce_selection_mask(config.get("short_eligibility_panel"))

        if side_mode not in {"neutral", "long_only", "short_only"}:

            raise ValueError(f"[WARNING] unsupported side_mode: {side_mode!r}")

        if selection_mode not in {"none", "top_k", "top_q", "percentile"}:

            raise ValueError(f"[WARNING] unsupported selection_mode: {selection_mode!r}")

        if selection_mode == "none":
            scaled_panel = (weight_df * gross_target).astype(float)

            # Cap each row's gross (sum of |weight|) at gross_target. Without this, a row with K
            # signalled assets carries gross ~K*gross_target, which a cash-shared targetpercent book
            # reads as >100% of capital -- silently collapsing onto the first-filled columns (long)
            # or levering up (long/short). Scale down only over-budget rows so weak-conviction rows
            # keep their signal-proportional sizing unchanged.
            if gross_target > 0.0:
                row_gross = scaled_panel.abs().sum(axis = 1)
                over_budget = row_gross > (gross_target + 1e-12)

                if bool(over_budget.any()):
                    row_scale = pd.Series(1.0, index = scaled_panel.index, dtype = float)
                    row_scale.loc[over_budget] = gross_target / row_gross.loc[over_budget]
                    scaled_panel = scaled_panel.mul(row_scale, axis = 0)

            selected_weight_panel = scaled_panel.astype(float)

        else:
            signal_arr = signal_strength_df.to_numpy(dtype = float, copy = False)
            weight_arr = weight_df.to_numpy(dtype = float, copy = False)
            entry_weights = np.zeros_like(weight_arr, dtype = float)

            for t in range(signal_arr.shape[0]):

                row = signal_arr[t]
                weight_row = weight_arr[t]
                valid = np.isfinite(row)
                long_allowed = valid.copy()
                short_allowed = valid.copy()

                if long_eligibility_arr is not None:
                    long_allowed &= long_eligibility_arr[t]

                if short_eligibility_arr is not None:
                    short_allowed &= short_eligibility_arr[t]

                if require_signal_sign:

                    long_allowed &= row > selection_sign_epsilon
                    short_allowed &= row < -selection_sign_epsilon

                long_candidates = np.flatnonzero(long_allowed) if side_mode != "short_only" else np.array([], dtype = int)
                short_candidates = np.flatnonzero(short_allowed) if side_mode != "long_only" else np.array([], dtype = int)
                selected_long = np.array([], dtype = int)
                selected_short = np.array([], dtype = int)

                if selection_mode == "top_k":

                    long_count = 0 if long_k is None else max(0, min(len(long_candidates), int(long_k)))
                    short_count = 0 if short_k is None else max(0, min(len(short_candidates), int(short_k)))

                    if long_count > 0:

                        selected_long = long_candidates[np.argsort(row[long_candidates])[-long_count:]]
                        selected_long = selected_long[np.argsort(row[selected_long])[::-1]]

                    if short_count > 0:

                        selected_short = short_candidates[np.argsort(row[short_candidates])[:short_count]]
                        selected_short = selected_short[np.argsort(row[selected_short])]

                elif selection_mode == "top_q":

                    if len(long_candidates) > 0 and long_q > 0.0:

                        long_count = max(1, int(np.floor(len(long_candidates) * long_q)))
                        long_count = min(long_count, len(long_candidates))
                        selected_long = long_candidates[np.argsort(row[long_candidates])[-long_count:]]
                        selected_long = selected_long[np.argsort(row[selected_long])[::-1]]

                    if len(short_candidates) > 0 and short_q > 0.0:

                        short_count = max(1, int(np.floor(len(short_candidates) * short_q)))
                        short_count = min(short_count, len(short_candidates))
                        selected_short = short_candidates[np.argsort(row[short_candidates])[:short_count]]
                        selected_short = selected_short[np.argsort(row[selected_short])]

                else:

                    if len(long_candidates) > 0 and long_percentile > 0.0:

                        threshold = np.nanpercentile(row[long_candidates], max(0.0, 100.0 * (1.0 - long_percentile)))
                        selected_long = long_candidates[row[long_candidates] >= threshold]
                        selected_long = selected_long[np.argsort(row[selected_long])[::-1]]

                    if len(short_candidates) > 0 and short_percentile > 0.0:

                        threshold = np.nanpercentile(row[short_candidates], min(100.0, 100.0 * short_percentile))
                        selected_short = short_candidates[row[short_candidates] <= threshold]
                        selected_short = selected_short[np.argsort(row[selected_short])]

                if selected_long.size > 0 and selected_short.size > 0:
                    overlap_mask = np.isin(selected_long, selected_short)

                    if overlap_mask.any():
                        # With require_signal_sign=False the same asset can rank into BOTH books; without
                        # this the later short assignment overwrites the long weight, under-filling the long
                        # leg and skewing net exposure. Net the ambiguous asset flat (drop from both sides).
                        dropped = selected_long[overlap_mask]
                        selected_long = selected_long[~overlap_mask]
                        selected_short = selected_short[~np.isin(selected_short, dropped)]

                long_gross = 0.0
                short_gross = 0.0

                if side_mode == "long_only":
                    long_gross = gross_target if len(selected_long) > 0 else 0.0

                elif side_mode == "short_only":
                    short_gross = gross_target if len(selected_short) > 0 else 0.0

                else:

                    if len(selected_long) > 0 and len(selected_short) > 0:

                        long_gross = gross_target / 2.0
                        short_gross = gross_target / 2.0

                    elif allow_partial_book and len(selected_long) > 0:
                        long_gross = gross_target

                    elif allow_partial_book and len(selected_short) > 0:
                        short_gross = gross_target

                if len(selected_long) > 0 and long_gross > 0.0:

                    if weight_mode == "equal":
                        long_scale = np.ones(len(selected_long), dtype = float)

                    elif weight_mode == "rank":
                        long_scale = np.arange(len(selected_long), 0, -1, dtype = float)

                    else:
                        long_scale = np.abs(weight_row[selected_long].astype(float))

                    long_total = float(long_scale.sum())
                    long_weights = np.full(len(selected_long), 1.0 / float(len(selected_long)), dtype = float) if long_total <= 0.0 else long_scale / long_total
                    entry_weights[t, selected_long] = long_weights * long_gross

                if len(selected_short) > 0 and short_gross > 0.0:

                    if weight_mode == "equal":
                        short_scale = np.ones(len(selected_short), dtype = float)

                    elif weight_mode == "rank":
                        short_scale = np.arange(len(selected_short), 0, -1, dtype = float)

                    else:
                        short_scale = np.abs(weight_row[selected_short].astype(float))

                    short_total = float(short_scale.sum())
                    short_weights = np.full(len(selected_short), 1.0 / float(len(selected_short)), dtype = float) if short_total <= 0.0 else short_scale / short_total
                    entry_weights[t, selected_short] = -short_weights * short_gross

            selected_weight_panel = pd.DataFrame(entry_weights, index = signal_strength_df.index, columns = signal_strength_df.columns, dtype = float)

        hold_every = max(1, int(config.get("hold_every", 1)))
        rebalance_every = max(1, int(config.get("rebalance_every", 1)))
        idx = pd.DatetimeIndex(pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce"))

        if bool(pd.isna(idx).any()):

            raise ValueError("[WARNING] temporal weight constructor received invalid Datetime values.")

        force_flat_mask, prevent_open_mask = Position.build_event_risk_masks(idx, config)

        reindexed_weight_panel = pd.DataFrame(selected_weight_panel).reindex(idx)

        if len(pd.DataFrame(selected_weight_panel).index) > 0 and bool(reindexed_weight_panel.isna().all().all()):

            logging.warning(
                            "[WARNING] signal/weight panel index does not overlap the test calendar; "
                            "result is forced all-flat (source rows=%d).",
                            len(pd.DataFrame(selected_weight_panel).index),
                            )

        elif len(pd.DataFrame(selected_weight_panel).index) > 0:

            # Partial overlap: calendar rows with no aligned source row reindex to all-NaN and are
            # about to be zeroed (forced flat) by the fillna below. The all-flat branch above only
            # catches ZERO overlap; without this, a calendar/timezone mismatch that drops SOME rows
            # is silently swallowed.
            _missing_row_mask = reindexed_weight_panel.isna().all(axis = 1)

            if bool(_missing_row_mask.any()):

                logging.warning(
                                "[WARNING] signal/weight panel only partially overlaps the test calendar; "
                                "%d of %d result rows have no aligned source weights and are being zeroed "
                                "(forced flat) -- check for a calendar/timezone mismatch. source rows=%d.",
                                int(_missing_row_mask.sum()), len(idx), len(pd.DataFrame(selected_weight_panel).index),
                                )

        entry_df = reindexed_weight_panel.fillna(0.0).astype(float)
        entry_arr = entry_df.to_numpy(dtype = float, copy = True)
        entry_arr[:min(len(entry_arr), max(0, int(self.signal_start_t))), :] = 0.0

        for t in range(entry_arr.shape[0]):

            if prevent_open_mask[t] or (t % rebalance_every != 0):
                entry_arr[t, :] = 0.0

        final_arr = np.zeros_like(entry_arr, dtype = float)
        active_weight_sum = np.zeros(entry_arr.shape[1], dtype = float)
        entry_live = (np.abs(entry_arr).sum(axis = 1) > 1e-12).astype(float)
        active_count = 0.0

        for t in range(entry_arr.shape[0]):
            active_count += entry_live[t]

            if entry_live[t] > 0.0:
                active_weight_sum += entry_arr[t]

            drop_idx = t - hold_every

            if drop_idx >= 0:
                active_count -= entry_live[drop_idx]

                if entry_live[drop_idx] > 0.0:
                    active_weight_sum -= entry_arr[drop_idx]

            if active_count > 0.0:
                final_arr[t] = active_weight_sum / active_count

            if force_flat_mask[t]:
                final_arr[t] = 0.0
                start_clear = max(0, t - hold_every + 1)

                for s in range(start_clear, t + 1):
                    entry_live[s] = 0.0

                active_weight_sum[:] = 0.0
                active_count = 0.0


        return pd.DataFrame(final_arr, index = idx, columns = entry_df.columns, dtype = float)


    def compute_multi_sleeve_weights(self,
                                    *,
                                    test_data: pd.DataFrame,
                                    sleeve_specs: List[Dict[str, Any]],
                                    asset_keys: List[str],
                                    factor_param_ranges: Optional[Dict[str, Any]] = None,
                                    dtype: str = "float32") -> np.ndarray:
        """
        ### What It Does
        Computes blended weights for multiple one-dimensional signal sleeves.

        #### Responsibility
        Combines sleeve-level signals and normalization rules for single-asset workflows.

        #### How To Use
        Call it when a strategy has several signal sources for one asset.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `sleeve_specs`
          - Multi-sleeve signal specification. Each sleeve should define its factor source, rule, and weight behavior clearly enough for independent construction.
          - Expected shape/type: `List[Dict[str, Any]]`.
        - `asset_keys`
          - Canonical asset-axis labels. Preserve this order when moving between tensor, panel, and portfolio execution code.
          - Expected shape/type: `List[str]`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `dtype`
          - Numerical dtype for stored arrays. Choose it to balance precision and memory; artifact metadata must match the actual array dtype.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = compute_multi_sleeve_weights(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `sleeve_specs`: **List[Dict[str, Any]]**.
        - `asset_keys`: **List[str]**.

        #### Optional Parameters
        - `factor_param_ranges`: **Optional[Dict[str, Any]]** = *None*.
        - `dtype`: **str** = *"float32"*.

        ---

        ### Returns
        - `result`: **np.ndarray**.
        """

        from .Signal import Signal

        if len(sleeve_specs) == 0:

            raise ValueError("[WARNING] compute_multi_sleeve_weights requires at least one sleeve specification.")

        reference_tensor = sleeve_specs[0].get("signal_strength")

        if reference_tensor is None:

            raise ValueError("[WARNING] each sleeve requires signal_strength.")

        reference_arr = np.asarray(reference_tensor)

        if reference_arr.ndim != 3:

            raise ValueError("[WARNING] compute_multi_sleeve_weights requires 3D sleeve tensors.")

        time_indexer: Optional[Signal.ParamIndexer] = None

        if factor_param_ranges:
            normalized_factor_param_ranges: Dict[str, List[Any]] = {}

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

                cleaned: List[Any] = []

                for item in values:

                    current = item.unwrap() if isinstance(item, FactorEngine.AutoParam.ParamValue) else item
                    current = cast(Any, current).item() if isinstance(current, np.generic) else current
                    cleaned.append(FactorEngine.AutoParam.clean_value(current))

                normalized_factor_param_ranges[str(key)] = cleaned

            time_indexer = Signal.ParamIndexer(
                                                            param_keys = list(normalized_factor_param_ranges.keys()),
                                                            param_values = [normalized_factor_param_ranges[key] for key in normalized_factor_param_ranges.keys()],
                                                            asset_keys = asset_keys,
                                                            )

        idx = pd.DatetimeIndex(pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce"))
        weight = np.zeros(reference_arr.shape, dtype = np.dtype(dtype))

        for combo_idx in range(reference_arr.shape[0]):

            params = None if time_indexer is None else {str(key): value for key, value in zip(time_indexer.param_keys, time_indexer.index_to_combo(combo_idx))}
            panel_specs: List[Tuple[pd.DataFrame, float]] = []

            for sleeve_spec in sleeve_specs:
                sleeve_tensor = sleeve_spec.get("signal_strength")

                if sleeve_tensor is None:

                    raise ValueError("[WARNING] each sleeve requires signal_strength.")

                sleeve_arr = np.asarray(sleeve_tensor)

                if sleeve_arr.shape != reference_arr.shape:

                    raise ValueError("[WARNING] all sleeve tensors must share the same shape.")

                weight_config = self._resolve_weight_config(sleeve_spec.get("weight_config"), params = params)

                if "gross_target" in sleeve_spec and "gross_target" not in weight_config:
                    weight_config["gross_target"] = sleeve_spec["gross_target"]

                signal_strength_panel = pd.DataFrame(
                                                    np.asarray(sleeve_arr[combo_idx, :, :], dtype = float),
                                                    index = idx,
                                                    columns = asset_keys,
                                                    dtype = float,
                                                    )

                sleeve_weight_panel = self.compute_weight_panel(
                                                                test_data = test_data,
                                                                signal_strength_panel = signal_strength_panel,
                                                                weight_config = weight_config,
                                                                )

                panel_specs.append((sleeve_weight_panel, 1.0))

            combined_weight_panel = self._weighted_panel_sum(panel_specs = panel_specs, index = idx, columns = asset_keys)
            weight[combo_idx, :, :] = combined_weight_panel.to_numpy(dtype = np.dtype(dtype), copy = True)


        return weight
