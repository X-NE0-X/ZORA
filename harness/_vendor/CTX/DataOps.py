from ENV_MGMT.imports import *







#----------------------------------------------------------------------------------------
class _DataOps:
    """
    ### What It Does
    Provides low-level frequency normalization and OHLCV resampling helpers for `CTX` and factor workflows.

    #### Responsibility
    Normalizes frequency rules, validates aggregation maps, and resamples market data under the engine's timing semantics.

    #### How To Use
    Use it through `CTX` unless you need direct access to data-level resample helpers.

    #### Usage Example
    `obj = _DataOps(...)`
    """

    ROLE_ALIAS_MAP = {
                        "o": "open",
                        "open": "open",
                        "h": "high",
                        "high": "high",
                        "l": "low",
                        "low": "low",
                        "c": "close",
                        "close": "close",
                        "adjclose": "close",
                        "adj_close": "close",
                        "volume": "volume",
                        "vol": "volume",
                        "turnover": "turnover",
                        "vwap": "vwap",
                        "dividend": "dividends",
                        "dividends": "dividends",
                        "split": "splits",
                        "splits": "splits",
                     }

    VALID_AGG = {"first", "last", "min", "max", "sum", "mean", "median"}

    @staticmethod
    def _col_key_to_text(col: Any) -> str:
        """
        ### What It Does
        Converts a column key into the text form used for column matching.

        #### Responsibility
        Normalizes strings, tuples, and other column labels before OHLCV role detection.

        #### How To Use
        Call it when data code needs a comparable column-name token.

        #### Key Parameters In Practice
        - `col`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `Any`.

        #### Usage Example
        `result = _col_key_to_text(...)`

        ---

        ### Parameters
        - `col`: **Any**.

        ---

        ### Returns
        - `result`: **str**.
        """

        if isinstance(col, tuple):

            return "__".join(str(x) for x in col)


        return str(col)


    @staticmethod
    def normalise_freq_rule(freq: Any) -> str:
        """
        ### What It Does
        Normalizes a pandas frequency rule into the canonical spelling used by CTX.

        #### Responsibility
        Keeps alias handling consistent when callers pass minute, hourly, daily, or monthly rules.

        #### How To Use
        Pass the requested target frequency before resample or interval comparisons.

        #### Key Parameters In Practice
        - `freq`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `Any`.

        #### Usage Example
        `result = normalise_freq_rule(...)`

        ---

        ### Parameters
        - `freq`: **Any**.

        ---

        ### Returns
        - `result`: **str**.
        """

        raw = str(freq).strip()

        if not raw:

            raise ValueError("[CTX WARNING] frequency cannot be empty")

        # Split an optional leading integer multiplier from the unit token so that
        # bare ("H") and multiplied ("4H", "30T") forms share one normalisation path.
        # pandas >= 3.0 rejects the deprecated uppercase intraday codes ("H", "T", "S")
        # and the bare month code ("M"); map the unit to its canonical spelling before
        # handing off to to_offset (which would otherwise raise "Invalid frequency").
        split_at = 0

        while split_at < len(raw) and raw[split_at].isdigit():
            split_at += 1

        multiplier = raw[:split_at]
        unit = raw[split_at:].strip()
        unit_alias = {
                    "MIN": "min",
                    "T": "min",
                    "S": "s",
                    "H": "h",
                    "D": "D",
                    "B": "B",
                    "W": "W",
                    "M": "ME",
                    "MONTH": "ME",
                    "MONTHLY": "ME",
                    }
        norm = f"{multiplier}{unit_alias.get(unit.upper(), unit)}" if unit else raw
        offset = pd.tseries.frequencies.to_offset(norm)


        return str(offset.freqstr)


    @classmethod
    def normalize_agg_map(cls, agg_map: Optional[Mapping[str, Any]]) -> Dict[str, str]:
        """
        ### What It Does
        Builds a resample aggregation map from detected OHLCV roles.

        #### Responsibility
        Applies first/max/min/last/sum semantics to open, high, low, close, and volume columns.

        #### How To Use
        Call it before resampling market bars with mixed price and volume fields.

        #### Key Parameters In Practice
        - `agg_map`
          - Mapping object used for lookup/resolution. Keys should match the semantic or column names used by downstream code.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.

        #### Usage Example
        `result = normalize_agg_map(...)`

        ---

        ### Parameters
        - `agg_map`: **Optional[Mapping[str, Any]]**.

        ---

        ### Returns
        - `result`: **Dict[str, str]**.
        """

        if not agg_map:

            return {}

        normalized: Dict[str, str] = {}

        for key, agg in dict(agg_map).items():
            k = str(key).strip()

            if not k:

                continue

            agg_name = str(agg).strip().lower()

            if agg_name not in cls.VALID_AGG:

                raise ValueError(
                                f"[CTX WARNING] unsupported aggregation rule '{agg}'. "
                                f"allowed: {sorted(cls.VALID_AGG)}"
                                )

            normalized[k] = agg_name


        return normalized


    @classmethod
    def freq_seconds(cls, freq: Any) -> float:
        """
        ### What It Does
        Converts a pandas frequency rule into an approximate number of seconds.

        #### Responsibility
        Lets resampling code compare source and target intervals without duplicating offset parsing.

        #### How To Use
        Call it with a normalized rule when deciding whether market data needs resampling.

        #### Key Parameters In Practice
        - `freq`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `Any`.

        #### Usage Example
        `result = freq_seconds(...)`

        ---

        ### Parameters
        - `freq`: **Any**.

        ---

        ### Returns
        - `result`: **float**.
        """

        rule = cls.normalise_freq_rule(freq)
        offset = pd.tseries.frequencies.to_offset(rule)

        try:
            nanos = float(offset.nanos)

        except ValueError:
            rule_upper = str(offset.freqstr).upper()

            if rule_upper in {"ME", "M"}:
                nanos = float(31 * 24 * 60 * 60 * 1_000_000_000)

            elif rule_upper == "MS":
                nanos = float(31 * 24 * 60 * 60 * 1_000_000_000)

            else:

                raise

        if nanos <= 0:

            raise ValueError(f"[CTX WARNING] invalid frequency interval: {freq}")


        return nanos / 1_000_000_000.0


    @staticmethod
    def _parse_session_clock(value: Any) -> int:

        text = str(value).strip()

        if not text:

            raise ValueError("[CTX WARNING] empty session clock value")

        parts = text.split(":")

        if len(parts) < 2:

            raise ValueError(f"[CTX WARNING] session clock must use HH:MM format: {value}")

        hour = int(parts[0])
        minute = int(parts[1])

        if hour == 24 and minute == 0:

            return 24 * 60

        if hour < 0 or hour > 23 or minute < 0 or minute > 59:

            raise ValueError(f"[CTX WARNING] invalid session clock value: {value}")


        return hour * 60 + minute


    @classmethod
    def _intraday_session_groups(cls,
                                indexed: pd.DataFrame,
                                session_profile: Optional[Mapping[str, Any]],
                               ) -> List[Tuple[pd.Timestamp, Optional[pd.Timestamp], pd.DataFrame]]:

        if not isinstance(session_profile, Mapping):

            return []

        raw_sessions = session_profile.get("regular_sessions")

        if not isinstance(raw_sessions, list):

            return []

        windows: List[Tuple[int, int]] = []

        for raw_window in raw_sessions:

            if not isinstance(raw_window, (list, tuple)) or len(raw_window) != 2:

                continue

            start_minute = cls._parse_session_clock(raw_window[0])
            end_minute = cls._parse_session_clock(raw_window[1])

            if start_minute == end_minute:

                continue

            windows.append((start_minute, end_minute))

        if not windows:

            return []

        dt_index = pd.DatetimeIndex(indexed.index)
        base_index = dt_index.normalize()
        candidate_bases = sorted(set(base_index) | {base - pd.Timedelta(days = 1) for base in base_index})
        groups: List[Tuple[pd.Timestamp, Optional[pd.Timestamp], pd.DataFrame]] = []

        for base in candidate_bases:

            for start_minute, end_minute in windows:
                start_ts = base + pd.Timedelta(minutes = start_minute)
                end_ts = base + pd.Timedelta(minutes = end_minute)

                if end_ts <= start_ts:
                    end_ts = end_ts + pd.Timedelta(days = 1)

                mask = (dt_index >= start_ts) & (dt_index < end_ts)

                if not mask.any():

                    continue

                session_data = indexed.loc[mask].copy()

                if not session_data.empty:
                    groups.append((start_ts, end_ts, session_data))


        return groups


    @classmethod
    def resample_price_data(cls,
                            test_data: pd.DataFrame,
                            rule: str,
                            cal_columns: tuple = ("Open", "High", "Low", "Close"),
                            sum_columns: tuple = ("Volume", "Turnover"),
                            mean_columns: tuple = ("VWAP",),
                            custom_agg_map: Optional[Mapping[str, str]] = None,
                            strict_custom_agg: bool = True,
                            session_profile: Optional[Mapping[str, Any]] = None,) -> pd.DataFrame:
        """
        ### What It Does
        Resamples OHLCV-like price data to a target frequency.

        #### Responsibility
        Preserves price semantics by applying role-aware aggregation and dropping empty bars.

        #### How To Use
        Call it with raw market data, a datetime column if needed, and the target frequency.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `rule`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `str`.
        - `cal_columns`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `tuple`.
        - `sum_columns`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `tuple`.
        - `mean_columns`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `tuple`.
        - `custom_agg_map`
          - Mapping object used for lookup/resolution. Keys should match the semantic or column names used by downstream code.
          - Expected shape/type: `Optional[Mapping[str, str]]`.
        - `strict_custom_agg`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `bool`.

        #### Usage Example
        `result = resample_price_data(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `rule`: **str**.

        #### Optional Parameters
        - `cal_columns`: **tuple** = *("Open", "High", "Low", "Close")*.
        - `sum_columns`: **tuple** = *("Volume", "Turnover")*.
        - `mean_columns`: **tuple** = *("VWAP",)*.
        - `custom_agg_map`: **Optional[Mapping[str, str]]** = *None*.
        - `strict_custom_agg`: **bool** = *True*.

        ---

        ### Returns
        - `result`: **pd.DataFrame**.
        """

        df = test_data.copy()

        if "Datetime" not in df.columns:

            raise KeyError(f"[CTX WARNING] resample requires 'Datetime' column")

        df["Datetime"] = pd.to_datetime(df["Datetime"], errors = "coerce")
        df = df.dropna(subset = ["Datetime"]).sort_values("Datetime", kind = "stable")
        dedup_subset = ["Datetime"]

        if "Symbol" in df.columns:
            dedup_subset = ["Symbol", "Datetime"]

        df = df.loc[~df.duplicated(subset = dedup_subset, keep = "last")]

        if df.empty:

            return df.reset_index(drop = True)

        aggregation: Dict[Any, str] = {}
        vwap_weight_links: Dict[Any, Tuple[Any, str]] = {}
        missing_custom_cols: List[str] = []
        cal_columns_lc = {str(col).lower() for col in cal_columns}
        sum_columns_lc = {str(col).lower() for col in sum_columns}
        mean_columns_lc = {str(col).lower() for col in mean_columns}
        custom_agg_map = cls.normalize_agg_map(custom_agg_map)
        wildcard_aggs = [
                        (pattern, agg_name)
                        for pattern, agg_name in custom_agg_map.items()

                        if any(ch in pattern for ch in "*?[]")
                        ]

        for column in df.columns:

            if column in {"Datetime", "Symbol"}:

                continue

            role: Optional[str] = None

            if isinstance(column, tuple):
                for part in reversed(column):
                    role = cls.ROLE_ALIAS_MAP.get(str(part).strip().lower())

                    if role is not None:

                        break

            if role is None:
                col_text = cls._col_key_to_text(column).lower()

                for token in reversed(re.split(r"[^a-z0-9]+", col_text)):

                    if not token:

                        continue

                    role = cls.ROLE_ALIAS_MAP.get(token)

                    if role is not None:

                        break

            if role != "vwap":

                continue

            weight_match = None
            weight_kind = ""

            if isinstance(column, tuple) and len(column) >= 2:
                prefix = tuple(column[:-1])

                for candidate_suffix in ("Volume", "Turnover"):
                    candidate = prefix + (candidate_suffix,)

                    if candidate in df.columns:
                        weight_match = candidate
                        weight_kind = str(candidate_suffix).strip().lower()

                        break

            else:
                column_text = str(column)
                column_text_lc = column_text.lower()
                column_lookup = {str(candidate).lower(): candidate for candidate in df.columns}

                for src, dst in (("vwap", "volume"), ("vwap", "turnover")):
                    candidate_text = column_text_lc.replace(src, dst)
                    candidate = column_lookup.get(candidate_text)

                    if candidate is not None and candidate != column:
                        weight_match = candidate
                        weight_kind = "volume" if dst.lower() == "volume" else "turnover"

                        break

            if weight_match is not None:
                vwap_weight_links[column] = (weight_match, weight_kind)

        for column in df.columns:

            if column in {"Datetime", "Symbol"}:

                continue

            role: Optional[str] = None

            if isinstance(column, tuple):
                for part in reversed(column):
                    role = cls.ROLE_ALIAS_MAP.get(str(part).strip().lower())

                    if role is not None:

                        break

            if role is None:
                col_text = cls._col_key_to_text(column).lower()
                for token in reversed(re.split(r"[^a-z0-9]+", col_text)):

                    if not token:

                        continue

                    role = cls.ROLE_ALIAS_MAP.get(token)

                    if role is not None:

                        break

            if role == "open" and "interest" in re.split(r"[^a-z0-9]+", cls._col_key_to_text(column).lower()):
                # 'Open Interest'/'Open_Interest' is an end-of-period level, not the OHLC 'open'. The
                # reversed-token scan matches its leading 'open' token -> aggregates 'first' (period-open)
                # and, because a role matched, silently bypasses the strict_custom_agg guard. Clear the
                # role so it requires an explicit aggregation rule instead of being mis-aggregated.
                role = None

            col_lc = str(column).lower()

            if role == "open" or col_lc in cal_columns_lc and col_lc.endswith("open"):
                aggregation[column] = "first"

            elif role == "high" or col_lc in cal_columns_lc and col_lc.endswith("high"):
                aggregation[column] = "max"

            elif role == "low" or col_lc in cal_columns_lc and col_lc.endswith("low"):
                aggregation[column] = "min"

            elif role == "close" or col_lc in cal_columns_lc and col_lc.endswith("close"):
                aggregation[column] = "last"

            elif role in {"volume", "turnover"} or col_lc in sum_columns_lc:
                aggregation[column] = "sum"

            elif role == "vwap" or col_lc in mean_columns_lc:
                aggregation[column] = "last" if column in vwap_weight_links else "mean"

            elif role == "dividends":
                aggregation[column] = "sum"

            elif role == "splits":
                aggregation[column] = "last"

            elif col_lc in {"shares outstanding", "shares_outstanding", "source"}:
                aggregation[column] = "last"

            else:
                col_text = cls._col_key_to_text(column)
                custom_agg = custom_agg_map.get(col_text)

                if custom_agg is None:
                    col_text_lc = col_text.lower()

                    for pattern, agg_name in custom_agg_map.items():

                        if any(ch in pattern for ch in "*?[]"):

                            continue

                        pattern_lc = str(pattern).lower()

                        if col_text_lc.endswith(f"_{pattern_lc}"):
                            custom_agg = agg_name

                            break

                if custom_agg is None and wildcard_aggs:
                    matches: List[Tuple[int, str]] = []

                    for pattern, agg_name in wildcard_aggs:

                        if fnmatch.fnmatch(col_text, pattern):
                            matches.append((len(pattern), agg_name))

                    if matches:
                        matches.sort(reverse = True)
                        custom_agg = matches[0][1]

                if custom_agg is None:
                    missing_custom_cols.append(cls._col_key_to_text(column))

                    continue

                aggregation[column] = custom_agg

        if missing_custom_cols and strict_custom_agg:
            missing_sorted = sorted(set(missing_custom_cols))

            raise ValueError(
                            "[CTX WARNING] resample requires explicit aggregation rules for non-OHLC/custom columns: "
                            f"{missing_sorted}. Please set `resample_rules`."
                            )

        if not aggregation:

            return df.reset_index(drop = True)

        if "Symbol" in df.columns:
            symbol_results: List[pd.DataFrame] = []
            child_custom_agg_map = {
                                    key: value
                                    for key, value in custom_agg_map.items()

                                    if str(key).strip().lower() != "symbol"
                                   }

            for symbol, symbol_data in df.groupby("Symbol", sort = True):
                child_data = symbol_data.drop(columns = ["Symbol"]).copy()
                child_result = cls.resample_price_data(
                                                        child_data,
                                                        rule,
                                                        cal_columns = cal_columns,
                                                        sum_columns = sum_columns,
                                                        mean_columns = mean_columns,
                                                        custom_agg_map = child_custom_agg_map,
                                                        strict_custom_agg = strict_custom_agg,
                                                        session_profile = session_profile,
                                                       )

                if child_result.empty:

                    continue

                child_result["Symbol"] = str(symbol)
                symbol_results.append(child_result)

            if not symbol_results:

                return pd.DataFrame(columns = list(df.columns)).reset_index(drop = True)

            symbol_result = pd.concat(symbol_results, ignore_index = True)
            symbol_result = symbol_result.sort_values(["Datetime", "Symbol"]).reset_index(drop = True)

            return symbol_result

        indexed = df.set_index("Datetime")
        offset = pd.tseries.frequencies.to_offset(rule)

        try:
            is_intraday = offset.nanos < 24 * 60 * 60 * 1_000_000_000

        except Exception:
            is_intraday = False

        resampled_data: List[pd.DataFrame] = []
        presence_columns = [column for column, agg_name in aggregation.items() if agg_name != "sum"]

        if is_intraday:
            session_groups = cls._intraday_session_groups(indexed, session_profile)

            if not session_groups:

                raise ValueError(
                                "[CTX WARNING] intraday resample requires a valid session_profile; "
                                "natural-day fallback is disabled."
                                )

            for anchor, session_end, day_data in session_groups:

                if day_data.empty:

                    continue

                session_resampled = (
                                day_data
                                .resample(rule, origin = anchor, label = "left", closed = "left")
                                .agg(aggregation)
                                .dropna(how = "all")
                                )

                if presence_columns:
                    existing_presence = [column for column in presence_columns if column in session_resampled.columns]

                    if existing_presence:
                        session_resampled = session_resampled.dropna(subset = existing_presence, how = "all")

                for vwap_column, (weight_column, weight_kind) in vwap_weight_links.items():

                    if vwap_column not in day_data.columns or weight_column not in day_data.columns:

                        continue

                    if weight_kind == "volume":

                        weighted_value = (day_data[vwap_column] * day_data[weight_column]).resample(rule, origin = anchor, label = "left", closed = "left").sum(min_count = 1)
                        volume_total = day_data[weight_column].resample(rule, origin = anchor, label = "left", closed = "left").sum(min_count = 1)
                        session_resampled[vwap_column] = weighted_value.divide(volume_total.replace(0, np.nan)).reindex(session_resampled.index)

                    else:
                        turnover_total = day_data[weight_column].resample(rule, origin = anchor, label = "left", closed = "left").sum(min_count = 1)
                        implied_volume = day_data[weight_column].divide(day_data[vwap_column].replace(0, np.nan))
                        volume_total = implied_volume.resample(rule, origin = anchor, label = "left", closed = "left").sum(min_count = 1)
                        session_resampled[vwap_column] = turnover_total.divide(volume_total.replace(0, np.nan)).reindex(session_resampled.index)

                if session_resampled.empty:

                    continue

                # Use bar-end timestamps so one fully aggregated intraday bar
                # cannot be consumed as if it were known at bar start.
                session_resampled.index = pd.DatetimeIndex(session_resampled.index + offset)

                if session_end is not None:
                    bar_end_index = pd.DatetimeIndex(session_resampled.index)
                    bar_start_index = pd.DatetimeIndex(bar_end_index - offset)
                    valid_mask = bar_start_index < session_end
                    clipped_index = pd.DatetimeIndex(
                                                        [
                                                            min(ts, session_end)
                                                            for ts in bar_end_index[valid_mask]
                                                        ]
                                                      )
                    session_resampled = session_resampled.loc[valid_mask]
                    session_resampled.index = clipped_index
                    session_resampled = session_resampled.loc[~session_resampled.index.duplicated(keep = "last")]

                resampled_data.append(session_resampled)

        else:
            day_resampled = (indexed.resample(rule, label = "right", closed = "right").agg(aggregation).dropna(how = "all"))

            if presence_columns:
                existing_presence = [column for column in presence_columns if column in day_resampled.columns]

                if existing_presence:
                    day_resampled = day_resampled.dropna(subset = existing_presence, how = "all")

            for vwap_column, (weight_column, weight_kind) in vwap_weight_links.items():

                if vwap_column not in indexed.columns or weight_column not in indexed.columns:

                    continue

                if weight_kind == "volume":
                    weighted_value = (indexed[vwap_column] * indexed[weight_column]).resample(rule, label = "right", closed = "right").sum(min_count = 1)
                    volume_total = indexed[weight_column].resample(rule, label = "right", closed = "right").sum(min_count = 1)
                    day_resampled[vwap_column] = weighted_value.divide(volume_total.replace(0, np.nan)).reindex(day_resampled.index)

                else:
                    turnover_total = indexed[weight_column].resample(rule, label = "right", closed = "right").sum(min_count = 1)
                    implied_volume = indexed[weight_column].divide(indexed[vwap_column].replace(0, np.nan))
                    volume_total = implied_volume.resample(rule, label = "right", closed = "right").sum(min_count = 1)
                    day_resampled[vwap_column] = turnover_total.divide(volume_total.replace(0, np.nan)).reindex(day_resampled.index)

            if not day_resampled.empty:
                resampled_data.append(day_resampled)

        if resampled_data:
            result = pd.concat(resampled_data).sort_index()
            result = result.loc[~result.index.duplicated(keep = "last")]

        else:
            result = pd.DataFrame(columns = [c for c in df.columns if c != "Datetime"])
            result.index = pd.DatetimeIndex([], name = "Datetime")

        result = result.reset_index().rename(columns = {"index": "Datetime"})
        result = result.sort_values("Datetime")
        result = result.loc[~result["Datetime"].duplicated(keep = "last")]
        result = result.reset_index(drop = True)


        return result
