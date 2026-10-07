from ENV_MGMT.imports import *
from .Config import (
    DEFAULT_DATA_CONFIG,
    DEFAULT_FACTOR_CONFIG, DEFAULT_EXECUTION_CONFIG, DEFAULT_SIGNAL_CONFIG,
    DEFAULT_TRAVERSAL_CONFIG, DEFAULT_PARQUET_CATALOG,
    DEFAULT_TIME_PROFILES, SESSION_SIGNATURE_FIELDS, DEFAULT_PARQUET_PATTERNS,
    DATA_ARCHIVES_ROOT, _deep_config_merge,
)
from .DataOps import _DataOps
from .MarketData import MarketData







# CTX Class
#----------------------------------------------------------------------------------------
@dataclass(frozen = True)
class CTX:
    """
    ### What It Does
    Defines the research context object that carries dataset paths, runtime metadata, and prepared market data.

    #### Responsibility
    Keeps loading rules, universe configuration, notebook metadata, and engine payload wiring in one reusable container.

    #### How To Use
    Create it through `ctx_single(...)` or `ctx_portfolio(...)`, then read data through `get_data`, attribute access, or payload helpers.

    #### Usage Example
    `obj = CTX(...)`
    """

    data_paths: Dict[str, pathlib.Path]
    data_load_rules: Dict[str, Dict[str, Any]]
    notebook_dir: pathlib.Path
    frequency: Optional[str] = None
    resample: bool = True
    resample_rules: Optional[Dict[str, str]] = None
    UNDERLYING: Optional[str] = None
    universe: Optional[Dict[str, Any]] = None
    strategy: Optional[Dict[str, Any]] = None
    snapshot_id: Optional[str] = None
    seed: Optional[int] = None
    logger_name: Optional[str] = None
    log_dir: Optional[pathlib.Path] = None
    payload: Optional[Dict[str, Any]] = None
    payload_path: Optional[pathlib.Path] = None
    start_date: Optional[Any] = None
    end_date: Optional[Any] = None
    factor_param_ranges: Optional[Dict[str, Any]] = None


    @staticmethod
    def _normalize_underlyings(raw_underlying: Any, *, allow_empty: bool = False, allow_all: bool = False) -> List[str]:

        if raw_underlying is None:

            if allow_empty:

                return []

            raise ValueError("[CTX WARNING] UNDERLYING is required and must contain actual ticker(s).")

        if isinstance(raw_underlying, str):
            text = raw_underlying.strip()

            if not text:

                if allow_empty:

                    return []

                raise ValueError("[CTX WARNING] UNDERLYING cannot be empty.")

            normalized = [text.upper()]

        elif isinstance(raw_underlying, Iterable):
            normalized = [str(x).strip().upper() for x in raw_underlying if str(x).strip()]

        else:

            raise TypeError("[CTX WARNING] UNDERLYING must be a string or a list/tuple/set of tickers.")

        deduped = sorted(set(normalized))

        if not deduped and not allow_empty:

            raise ValueError("[CTX WARNING] UNDERLYING cannot be empty.")

        if any(x == "PORTFOLIO" for x in deduped):

            raise ValueError("[CTX WARNING] UNDERLYING must contain explicit tickers; placeholder 'Portfolio' is no longer supported.")

        if not allow_all and any(x == "ALL" for x in deduped):

            raise ValueError("[CTX WARNING] UNDERLYING = 'ALL' is only supported in ctx_portfolio.")


        return deduped


    @staticmethod
    def _coerce_parquet_sources(raw_value: Any, *, freq_key: str) -> List[pathlib.Path]:

        if isinstance(raw_value, Mapping):

            raise TypeError(
                f"[CTX WARNING] parquet backend expects dataset path(s) for freq='{freq_key}', "
                "not per-underlying maps."
            )

        if raw_value is None:
            freq_upper = str(freq_key).strip().upper()
            patterns = DEFAULT_PARQUET_PATTERNS.get(freq_upper, ())
            values: List[pathlib.Path] = []

            for pattern in patterns:
                values.extend(sorted(DATA_ARCHIVES_ROOT.glob(f"Data_*/{pattern}")))

        elif isinstance(raw_value, (str, pathlib.Path)):
            values = [pathlib.Path(raw_value)]

        elif isinstance(raw_value, Iterable):
            values = [pathlib.Path(value) for value in raw_value]

        else:

            raise TypeError(
                            f"[CTX WARNING] parquet source for freq='{freq_key}' must be a path or a list of paths."
                           )

        output: List[pathlib.Path] = []
        seen: Set[str] = set()

        for value in values:
            path = pathlib.Path(value).expanduser().resolve()

            if path.suffix.lower() != ".parquet":

                raise ValueError(f"[CTX WARNING] parquet source must end with '.parquet': {path}")

            if not path.exists():

                raise FileNotFoundError(f"[CTX WARNING] parquet source missing: {path}")

            resolved = str(path)

            if resolved in seen:

                continue

            seen.add(resolved)
            output.append(path)


        return output


    @staticmethod
    def _catalog_dataset_to_symbols(freq_key: str) -> Dict[str, Set[str]]:

        raw_catalog = DEFAULT_PARQUET_CATALOG.get(str(freq_key).strip().upper(), {})

        if not isinstance(raw_catalog, dict):

            return {}

        catalog: Dict[str, Set[str]] = {}

        for dataset_name, raw_symbols in raw_catalog.items():

            if isinstance(raw_symbols, (list, tuple, set)):
                catalog[str(dataset_name)] = {
                                                str(symbol).strip().upper()
                                                for symbol in raw_symbols

                                                if str(symbol).strip()
                                             }


        return catalog


    @staticmethod
    def _infer_parquet_session_profile(path: pathlib.Path) -> Dict[str, Any]:

        dataset_path = pathlib.Path(path)
        asset = dataset_path.parent.name
        region = ""
        stem_parts = dataset_path.stem.split("_")

        if len(stem_parts) >= 2 and stem_parts[0] == "Data":
            asset = "_".join(stem_parts[:2])

        if len(stem_parts) >= 4 and stem_parts[0] == "Data":
            region = stem_parts[2].upper()

        raw_regions = DEFAULT_TIME_PROFILES.get("regions")
        raw_otc = DEFAULT_TIME_PROFILES.get("otc")
        region_profiles = raw_regions if isinstance(raw_regions, Mapping) else {}
        otc_profiles = raw_otc if isinstance(raw_otc, Mapping) else {}

        if asset in {"Data_EQT", "Data_ETF", "Data_INDEX"} and region in region_profiles:

            profile = copy.deepcopy(region_profiles[region])
            profile["asset"] = asset
            profile["region"] = region

            return cast(Dict[str, Any], profile)

        if asset in otc_profiles:

            profile = copy.deepcopy(otc_profiles[asset])
            profile["asset"] = asset
            profile["region"] = region

            return cast(Dict[str, Any], profile)


        return {}


    @staticmethod
    def _session_profile_signature(profile: Mapping[str, Any]) -> str:

        if not isinstance(profile, Mapping) or not profile:

            return "__NO_SESSION_PROFILE__"

        payload = {
                    field: copy.deepcopy(profile.get(field))
                    for field in SESSION_SIGNATURE_FIELDS

                    if field in profile
                  }

        if not payload:

            return "__NO_SESSION_PROFILE__"


        return json.dumps(payload, sort_keys = True, default = str)


    @staticmethod
    def _build_symbol_key_map(freq_key: str,
                              dataset_keys: List[str],
                              data_paths: Mapping[str, pathlib.Path],
                              selected_underlyings: List[str]) -> Dict[str, List[str]]:

        catalog = CTX._catalog_dataset_to_symbols(freq_key)
        symbol_map: Dict[str, List[str]] = {symbol: [] for symbol in selected_underlyings}

        if not catalog:

            return symbol_map

        for dataset_key in dataset_keys:
            dataset_path = pathlib.Path(data_paths[dataset_key])
            dataset_name = dataset_path.name
            symbols = catalog.get(dataset_name, set())

            if not symbols:

                continue

            for symbol in selected_underlyings:

                if symbol in symbols:
                    symbol_map.setdefault(symbol, []).append(dataset_key)


        return symbol_map


    @staticmethod
    def _coerce_datetime_bound(raw_value: Any, *, label: str) -> Optional[datetime.datetime]:

        if raw_value is None:

            return None

        if isinstance(raw_value, str) and not raw_value.strip():

            return None

        try:
            ts = pd.Timestamp(raw_value)

        except Exception as exc:

            raise ValueError(f"[CTX WARNING] failed to parse {label}: {raw_value!r}") from exc

        if pd.isna(ts):

            return None

        if ts.tzinfo is not None:
            # Drop the tz keeping the wall-clock (NOT tz_convert, which shifts to UTC): the data axis is
            # tz-naive session-local wall-clock, so a tz-aware bound must be compared on the same clock.
            ts = ts.tz_localize(None)


        return ts.to_pydatetime()


    @staticmethod
    def ctx_single(*,
                    data_min: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_h: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_d: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_m: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_alter: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    UNDERLYING: Optional[str] = None,

                    strategy_name: Optional[str] | None = None,
                    frequency: str = "H",
                    resample: bool = True,
                    resample_rules: Optional[Mapping[str, str]] = None,
                    freq_min: str = "1min",
                    freq_h: str = "1h",
                    freq_d: str = "1d",
                    freq_m: str = "ME",
                    normalize_daily: bool = True,
                    timezone_regex: str = r"([+-][0-9]{2}:?[0-9]{2}|Z)$",
                    start_date: Optional[Any] = None,
                    end_date: Optional[Any] = None,

                    **ctx_kwargs: Any,) -> "CTX":
        """
        ### What It Does
        Builds a `CTX` context for single-asset research from parquet sources and runtime overrides.

        #### Responsibility
        Normalizes source paths, dataset rules, date filters, and runtime metadata into a ready-to-use context object.

        #### How To Use
        Pass the available source paths plus the target symbol or universe, then use the returned context to access prepared data and payload helpers.

        #### Key Parameters In Practice
        - `data_min`
          - Minute-level parquet/source path. Provide it when CTX should load or resample from minute data.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `data_h`
          - Hourly parquet/source path. Provide it when hourly data is the source or intermediate clock.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `data_d`
          - Daily parquet/source path. Provide it when daily data is the source or target clock.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `data_alter`
          - Alternative-data source path. Use it for non-price features that should be loaded alongside market data.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `UNDERLYING`
          - Asset selector. In single workflows this is one ticker; in portfolio workflows it should identify the explicit universe/list being loaded and must match available data columns/artifacts.
          - Expected shape/type: `Optional[str]`.
        - `strategy_name`
          - Strategy/run label stored in CTX metadata and later payloads. Set it to a stable notebook strategy name so run logs and artifacts remain traceable.
          - Expected shape/type: `Optional[str] | None`.
        - `frequency`
          - Data frequency label used by CTX, logs, and runtime metadata. It describes the target working clock; it is not itself a resampling instruction unless CTX is asked to resample.
          - Expected shape/type: `str`.
        - `resample`
          - Resampling switch. Enable it when source frequency should be converted into the requested working frequency.
          - Expected shape/type: `bool`.
        - `resample_rules`
          - Custom aggregation map. Use it for non-OHLC columns so CTX knows whether to take last, sum, mean, etc.
          - Expected shape/type: `Optional[Mapping[str, str]]`.
        - `freq_min`
          - Target minute frequency rule. Use pandas-compatible spelling such as `1min` when resampling minute data.
          - Expected shape/type: `str`.
        - `freq_h`
          - Target hourly frequency rule. Use pandas-compatible spelling such as `1h` for hourly CTX outputs.
          - Expected shape/type: `str`.
        - `freq_d`
          - Target daily frequency rule. Use pandas-compatible spelling such as `1d` for daily CTX outputs.
          - Expected shape/type: `str`.
        - `normalize_daily`
          - Daily timestamp normalization switch. Use it to standardize date labels after CTX loading/resampling.
          - Expected shape/type: `bool`.
        - `timezone_regex`
          - Timezone suffix pattern. Use it to strip or normalize timezone text embedded in raw datetime strings.
          - Expected shape/type: `str`.
        - `start_date`
          - Left boundary for loading, slicing, or protocol windows. Use it to restrict available history before factor/SR computation.
          - Expected shape/type: `Optional[Any]`.
        - `end_date`
          - Right boundary for loading, slicing, or protocol windows. Use it to stop context preparation and OOS replay at the intended cutoff.
          - Expected shape/type: `Optional[Any]`.
        - `**ctx_kwargs`
          - Additional CTX dataclass fields or payload metadata overrides. Use sparingly for explicit runtime metadata such as seed, logger name, payload path, or custom run-output paths.
          - Expected shape/type: `Any`.

        #### Usage Example
        `result = ctx_single(...)`

        ---

        ### Parameters
        - `**ctx_kwargs`: **Any**.

        #### Optional Parameters
        - `data_min`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `data_h`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `data_d`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `data_alter`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `UNDERLYING`: **Optional[str]** = *None*.
        - `strategy_name`: **Optional[str] | None** = *None*.
        - `frequency`: **str** = *"H"*.
        - `resample`: **bool** = *True*.
        - `resample_rules`: **Optional[Mapping[str, str]]** = *None*.
        - `freq_min`: **str** = *"1min"*.
        - `freq_h`: **str** = *"1h"*.
        - `freq_d`: **str** = *"1d"*.
        - `normalize_daily`: **bool** = *True*.
        - `timezone_regex`: **str** = *r"([+-][0-9]{2}:?[0-9]{2}|Z)$"*.
        - `start_date`: **Optional[Any]** = *None*.
        - `end_date`: **Optional[Any]** = *None*.

        ---

        ### Returns
        - `result`: **"CTX"**.
        """

        rename_map = {
                        "Datetime": "Datetime",
                        "Open": "Open",
                        "High": "High",
                        "Low": "Low",
                        "Close": "Close",
                        "Volume": "Volume",
                        "Dividends": "Dividends",
                        "Stock Splits": "Stock Splits",
                    }

        data_paths: Dict[str, pathlib.Path] = {}
        data_load_rules: Dict[str, Dict[str, Any]] = {}
        use_default_price_sources = all(value is None for value in (data_min, data_h, data_d, data_m))
        min_paths = CTX._coerce_parquet_sources(data_min, freq_key = "MIN") if use_default_price_sources or data_min is not None else []
        h_paths = CTX._coerce_parquet_sources(data_h, freq_key = "H") if use_default_price_sources or data_h is not None else []
        d_paths = CTX._coerce_parquet_sources(data_d, freq_key = "D") if use_default_price_sources or data_d is not None else []
        m_paths = CTX._coerce_parquet_sources(data_m, freq_key = "M") if use_default_price_sources or data_m is not None else []
        alter_paths = CTX._coerce_parquet_sources(data_alter, freq_key = "ALTER")

        dataset_keys_min: List[str] = []
        dataset_keys_h: List[str] = []
        dataset_keys_d: List[str] = []
        dataset_keys_m: List[str] = []
        dataset_keys_alter: List[str] = []

        for idx, path in enumerate(min_paths):

            key_name = f"MIN::{idx}"
            data_paths[key_name] = path
            data_load_rules[key_name] = {"schema": "ohlc", "rename": rename_map, "timezone_suffix_regex": timezone_regex, "session_profile": CTX._infer_parquet_session_profile(path), "to_datetime": True, "format": "parquet"}
            dataset_keys_min.append(key_name)

        for idx, path in enumerate(h_paths):

            key_name = f"H::{idx}"
            data_paths[key_name] = path
            data_load_rules[key_name] = {"schema": "ohlc", "rename": rename_map, "timezone_suffix_regex": timezone_regex, "session_profile": CTX._infer_parquet_session_profile(path), "to_datetime": True, "format": "parquet"}
            dataset_keys_h.append(key_name)

        for idx, path in enumerate(d_paths):

            key_name = f"D::{idx}"
            data_paths[key_name] = path

            data_load_rules[key_name] = {
                                        "schema": "ohlc",
                                        "rename": rename_map,
                                        "timezone_suffix_regex": timezone_regex,
                                        "session_profile": CTX._infer_parquet_session_profile(path),
                                        "to_datetime": True,
                                        "normalize_dates": normalize_daily,
                                        "format": "parquet",
                                        }

            dataset_keys_d.append(key_name)

        for idx, path in enumerate(m_paths):

            key_name = f"M::{idx}"
            data_paths[key_name] = path

            data_load_rules[key_name] = {
                                        "schema": "ohlc",
                                        "rename": rename_map,
                                        "timezone_suffix_regex": timezone_regex,
                                        "session_profile": CTX._infer_parquet_session_profile(path),
                                        "to_datetime": True,
                                        "normalize_dates": normalize_daily,
                                        "format": "parquet",
                                        }

            dataset_keys_m.append(key_name)

        for idx, path in enumerate(alter_paths):

            key_name = f"ALTER::{idx}"
            data_paths[key_name] = path
            data_load_rules[key_name] = {"schema": "alter", "format": "parquet",}
            dataset_keys_alter.append(key_name)

        if not min_paths and not h_paths and not d_paths and not m_paths:

            raise ValueError("[CTX WARNING] ctx_single requires at least one parquet dataset in data_min/data_h/data_d/data_m.")

        ctx_kwargs = dict(ctx_kwargs)
        ctx_kwargs.setdefault("notebook_dir", pathlib.Path.cwd())

        if "underlying" in ctx_kwargs or "underlyings" in ctx_kwargs:

            raise TypeError("[CTX WARNING] ctx_single uses `UNDERLYING`; legacy kwargs `underlying/underlyings` are not supported")

        normalized_underlyings = CTX._normalize_underlyings(UNDERLYING, allow_empty = False)

        if len(normalized_underlyings) != 1:

            raise ValueError("[CTX WARNING] ctx_single requires exactly one ticker in UNDERLYING.")

        underlying = normalized_underlyings[0]

        universe = {
                    "underlyings": [underlying],
                    "price_underlyings": [underlying],
                    "alter_underlyings": [underlying] if dataset_keys_alter else [],
                    "primary_underlying": underlying,
                    "frequency": {"MIN": freq_min, "H": freq_h, "D": freq_d, "M": freq_m},
                    "price_data_keys": {"MIN": dataset_keys_min, "H": dataset_keys_h, "D": dataset_keys_d, "M": dataset_keys_m},
                    "alter_data_keys": dataset_keys_alter,
                    "price_symbol_data_keys": {
                                                "MIN": CTX._build_symbol_key_map("MIN", dataset_keys_min, data_paths, [underlying]),
                                                "H": CTX._build_symbol_key_map("H", dataset_keys_h, data_paths, [underlying]),
                                                "D": CTX._build_symbol_key_map("D", dataset_keys_d, data_paths, [underlying]),
                                                "M": CTX._build_symbol_key_map("M", dataset_keys_m, data_paths, [underlying]),
                                              },
                    "alter_symbol_data_keys": CTX._build_symbol_key_map("ALTER", dataset_keys_alter, data_paths, [underlying]),
                    "ctx_mode": "single",
                    }

        strategy = {
                    "name": strategy_name,
                    "frequency": frequency,
                    "UNDERLYING": underlying,
                    "ctx_mode": "single",
                    }


        return CTX(

            frequency = frequency,
            resample = resample,
            resample_rules = dict(resample_rules) if resample_rules is not None else None,
            UNDERLYING = underlying,
            data_paths = data_paths,
            data_load_rules = data_load_rules,
            universe = universe,
            strategy = strategy,
            start_date = start_date,
            end_date = end_date,
            **ctx_kwargs,
        )


    @staticmethod
    def ctx_portfolio(*,
                    data_min: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_h: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_d: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_m: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    data_alter: Optional[str | pathlib.Path | Sequence[str | pathlib.Path]] = None,
                    UNDERLYING: Sequence[str] | str = (),
                    strategy_name: Optional[str] | None = None,
                    frequency: str = "H",
                    resample: bool = True,
                    resample_rules: Optional[Mapping[str, str]] = None,
                    freq_min: str = "1min",
                    freq_h: str = "1h",
                    freq_d: str = "1d",
                    freq_m: str = "ME",
                    normalize_daily: bool = True,
                    timezone_regex: str = r"([+-][0-9]{2}:?[0-9]{2}|Z)$",
                    start_date: Optional[Any] = None,
                    end_date: Optional[Any] = None,
                    **ctx_kwargs: Any,) -> "CTX":
        """
        ### What It Does
        Builds a `CTX` context for portfolio research from parquet sources and runtime overrides.

        #### Responsibility
        Normalizes source paths, dataset rules, date filters, and runtime metadata into a ready-to-use context object.

        #### How To Use
        Pass the available source paths plus the target symbol or universe, then use the returned context to access prepared data and payload helpers.

        #### Key Parameters In Practice
        - `data_min`
          - Minute-level parquet/source path. Provide it when CTX should load or resample from minute data.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `data_h`
          - Hourly parquet/source path. Provide it when hourly data is the source or intermediate clock.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `data_d`
          - Daily parquet/source path. Provide it when daily data is the source or target clock.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `data_alter`
          - Alternative-data source path. Use it for non-price features that should be loaded alongside market data.
          - Expected shape/type: `Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]`.
        - `UNDERLYING`
          - Asset selector. In single workflows this is one ticker; in portfolio workflows it should identify the explicit universe/list being loaded and must match available data columns/artifacts.
          - Expected shape/type: `Sequence[str] | str`.
        - `strategy_name`
          - Strategy/run label stored in CTX metadata and later payloads. Set it to a stable notebook strategy name so run logs and artifacts remain traceable.
          - Expected shape/type: `Optional[str] | None`.
        - `frequency`
          - Data frequency label used by CTX, logs, and runtime metadata. It describes the target working clock; it is not itself a resampling instruction unless CTX is asked to resample.
          - Expected shape/type: `str`.
        - `resample`
          - Resampling switch. Enable it when source frequency should be converted into the requested working frequency.
          - Expected shape/type: `bool`.
        - `resample_rules`
          - Custom aggregation map. Use it for non-OHLC columns so CTX knows whether to take last, sum, mean, etc.
          - Expected shape/type: `Optional[Mapping[str, str]]`.
        - `freq_min`
          - Target minute frequency rule. Use pandas-compatible spelling such as `1min` when resampling minute data.
          - Expected shape/type: `str`.
        - `freq_h`
          - Target hourly frequency rule. Use pandas-compatible spelling such as `1h` for hourly CTX outputs.
          - Expected shape/type: `str`.
        - `freq_d`
          - Target daily frequency rule. Use pandas-compatible spelling such as `1d` for daily CTX outputs.
          - Expected shape/type: `str`.
        - `normalize_daily`
          - Daily timestamp normalization switch. Use it to standardize date labels after CTX loading/resampling.
          - Expected shape/type: `bool`.
        - `timezone_regex`
          - Timezone suffix pattern. Use it to strip or normalize timezone text embedded in raw datetime strings.
          - Expected shape/type: `str`.
        - `start_date`
          - Left boundary for loading, slicing, or protocol windows. Use it to restrict available history before factor/SR computation.
          - Expected shape/type: `Optional[Any]`.
        - `end_date`
          - Right boundary for loading, slicing, or protocol windows. Use it to stop context preparation and OOS replay at the intended cutoff.
          - Expected shape/type: `Optional[Any]`.
        - `**ctx_kwargs`
          - Additional CTX dataclass fields or payload metadata overrides. Use sparingly for explicit runtime metadata such as seed, logger name, payload path, or custom run-output paths.
          - Expected shape/type: `Any`.

        #### Usage Example
        `result = ctx_portfolio(...)`

        ---

        ### Parameters
        - `**ctx_kwargs`: **Any**.

        #### Optional Parameters
        - `data_min`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `data_h`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `data_d`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `data_alter`: **Optional[str | pathlib.Path | Sequence[str | pathlib.Path]]** = *None*.
        - `UNDERLYING`: **Sequence[str] | str** = *()*.
        - `strategy_name`: **Optional[str] | None** = *None*.
        - `frequency`: **str** = *"H"*.
        - `resample`: **bool** = *True*.
        - `resample_rules`: **Optional[Mapping[str, str]]** = *None*.
        - `freq_min`: **str** = *"1min"*.
        - `freq_h`: **str** = *"1h"*.
        - `freq_d`: **str** = *"1d"*.
        - `normalize_daily`: **bool** = *True*.
        - `timezone_regex`: **str** = *r"([+-][0-9]{2}:?[0-9]{2}|Z)$"*.
        - `start_date`: **Optional[Any]** = *None*.
        - `end_date`: **Optional[Any]** = *None*.

        ---

        ### Returns
        - `result`: **"CTX"**.
        """

        rename_map = {
            "Datetime": "Datetime",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
            "Dividends": "Dividends",
            "Stock Splits": "Stock Splits",
        }

        ctx_kwargs = dict(ctx_kwargs)
        ctx_kwargs.setdefault("notebook_dir", pathlib.Path.cwd())

        if "underlying" in ctx_kwargs or "underlyings" in ctx_kwargs:

            raise TypeError("[CTX WARNING] ctx_portfolio uses `UNDERLYING`; legacy kwargs `underlying/underlyings` are not supported")

        if "data_min_map" in ctx_kwargs or "data_h_map" in ctx_kwargs or "data_d_map" in ctx_kwargs or "data_alter_map" in ctx_kwargs:

            raise TypeError("[CTX WARNING] parquet backend no longer accepts per-underlying *_map arguments; pass parquet dataset path(s) via data_min/data_h/data_d/data_alter.")

        portfolio_underlying = "Portfolio"

        use_default_price_sources = all(value is None for value in (data_min, data_h, data_d, data_m))
        min_paths = CTX._coerce_parquet_sources(data_min, freq_key = "MIN") if use_default_price_sources or data_min is not None else []
        h_paths = CTX._coerce_parquet_sources(data_h, freq_key = "H") if use_default_price_sources or data_h is not None else []
        d_paths = CTX._coerce_parquet_sources(data_d, freq_key = "D") if use_default_price_sources or data_d is not None else []
        m_paths = CTX._coerce_parquet_sources(data_m, freq_key = "M") if use_default_price_sources or data_m is not None else []
        alter_paths = CTX._coerce_parquet_sources(data_alter, freq_key = "ALTER")

        if not min_paths and not h_paths and not d_paths and not m_paths:

            raise ValueError("[CTX WARNING] ctx_portfolio requires at least one parquet dataset in data_min/data_h/data_d/data_m.")

        data_paths: Dict[str, pathlib.Path] = {}
        data_load_rules: Dict[str, Dict[str, Any]] = {}
        dataset_keys_min: List[str] = []
        dataset_keys_h: List[str] = []
        dataset_keys_d: List[str] = []
        dataset_keys_m: List[str] = []
        dataset_keys_alter: List[str] = []

        for idx, path in enumerate(min_paths):

            key_name = f"MIN::{idx}"
            data_paths[key_name] = path

            data_load_rules[key_name] = {
                                        "schema": "ohlc",
                                        "rename": rename_map,
                                        "timezone_suffix_regex": timezone_regex,
                                        "session_profile": CTX._infer_parquet_session_profile(path),
                                        "to_datetime": True,
                                        "format": "parquet",
                                        }

            dataset_keys_min.append(key_name)

        for idx, path in enumerate(h_paths):

            key_name = f"H::{idx}"
            data_paths[key_name] = path

            data_load_rules[key_name] = {
                                        "schema": "ohlc",
                                        "rename": rename_map,
                                        "timezone_suffix_regex": timezone_regex,
                                        "session_profile": CTX._infer_parquet_session_profile(path),
                                        "to_datetime": True,
                                        "format": "parquet",
                                        }

            dataset_keys_h.append(key_name)

        for idx, path in enumerate(d_paths):

            key_name = f"D::{idx}"
            data_paths[key_name] = path

            data_load_rules[key_name] = {
                                        "schema": "ohlc",
                                        "rename": rename_map,
                                        "timezone_suffix_regex": timezone_regex,
                                        "session_profile": CTX._infer_parquet_session_profile(path),
                                        "to_datetime": True,
                                        "normalize_dates": normalize_daily,
                                        "format": "parquet",
                                        }

            dataset_keys_d.append(key_name)

        for idx, path in enumerate(m_paths):

            key_name = f"M::{idx}"
            data_paths[key_name] = path

            data_load_rules[key_name] = {
                                        "schema": "ohlc",
                                        "rename": rename_map,
                                        "timezone_suffix_regex": timezone_regex,
                                        "session_profile": CTX._infer_parquet_session_profile(path),
                                        "to_datetime": True,
                                        "normalize_dates": normalize_daily,
                                        "format": "parquet",
                                        }

            dataset_keys_m.append(key_name)

        for idx, path in enumerate(alter_paths):

            key_name = f"ALTER::{idx}"
            data_paths[key_name] = path
            data_load_rules[key_name] = {
                                        "schema": "alter",
                                        "format": "parquet",
                                        }
            dataset_keys_alter.append(key_name)

        if not data_paths:

            raise ValueError("[CTX WARNING] ctx_portfolio found no valid parquet datasets for requested underlyings")

        requested_underlyings = CTX._normalize_underlyings(UNDERLYING, allow_empty = False, allow_all = True)

        if "ALL" in requested_underlyings:
            expanded_underlyings: Set[str] = {symbol for symbol in requested_underlyings if symbol != "ALL"}

            for freq_key, dataset_keys in (("MIN", dataset_keys_min), ("H", dataset_keys_h), ("D", dataset_keys_d), ("M", dataset_keys_m)):

                if dataset_keys:
                    catalog = CTX._catalog_dataset_to_symbols(freq_key)
                    unresolved_paths: List[pathlib.Path] = []

                    for dataset_key in dataset_keys:
                        dataset_path = pathlib.Path(data_paths[dataset_key])
                        dataset_name = dataset_path.name
                        known_symbols = catalog.get(dataset_name)

                        if known_symbols:
                            expanded_underlyings.update(known_symbols)

                        else:
                            unresolved_paths.append(dataset_path)

                    if unresolved_paths:

                        path_sql = ", ".join(
                                            "'" + str(path.as_posix()).replace("'", "''") + "'"
                                            for path in unresolved_paths
                                            )

                        relation_sql = f"read_parquet([{path_sql}])"
                        conn = duckdb.connect(database = ":memory:")
                        conn.execute("PRAGMA disable_progress_bar")

                        try:
                            rows = conn.execute(
                                                f"SELECT DISTINCT upper(Symbol) AS symbol FROM {relation_sql} "
                                                "WHERE Symbol IS NOT NULL ORDER BY 1"
                                               ).fetchall()
                        finally:
                            conn.close()

                        expanded_underlyings.update(
                                                    str(row[0]).strip().upper()
                                                    for row in rows

                                                    if row and str(row[0]).strip()
                                                    )

            if not expanded_underlyings:

                raise ValueError("[CTX WARNING] UNDERLYING = 'ALL' could not resolve any ticker from parquet datasets.")

            price_underlyings = sorted(expanded_underlyings)

        else:
            price_underlyings = requested_underlyings

        alter_underlyings = list(price_underlyings)

        primary_underlying = price_underlyings[0]

        symbol_keys_min = CTX._build_symbol_key_map("MIN", dataset_keys_min, data_paths, price_underlyings)
        symbol_keys_h = CTX._build_symbol_key_map("H", dataset_keys_h, data_paths, price_underlyings)
        symbol_keys_d = CTX._build_symbol_key_map("D", dataset_keys_d, data_paths, price_underlyings)
        symbol_keys_m = CTX._build_symbol_key_map("M", dataset_keys_m, data_paths, price_underlyings)
        symbol_keys_alter = CTX._build_symbol_key_map("ALTER", dataset_keys_alter, data_paths, alter_underlyings)

        min_catalog = CTX._catalog_dataset_to_symbols("MIN")
        h_catalog = CTX._catalog_dataset_to_symbols("H")
        d_catalog = CTX._catalog_dataset_to_symbols("D")
        m_catalog = CTX._catalog_dataset_to_symbols("M")

        def _catalog_covers_all(dataset_keys: List[str], catalog: Dict[str, Set[str]]) -> bool:

            return bool(dataset_keys) and all(pathlib.Path(data_paths[key]).name in catalog for key in dataset_keys)


        if _catalog_covers_all(dataset_keys_min, min_catalog) and any(not symbol_keys_min.get(symbol) for symbol in price_underlyings):
            missing = sorted(symbol for symbol in price_underlyings if not symbol_keys_min.get(symbol))

            raise ValueError(f"[CTX WARNING] requested UNDERLYING not present in MIN parquet catalog: {missing}")

        if _catalog_covers_all(dataset_keys_h, h_catalog) and any(not symbol_keys_h.get(symbol) for symbol in price_underlyings):
            missing = sorted(symbol for symbol in price_underlyings if not symbol_keys_h.get(symbol))

            raise ValueError(f"[CTX WARNING] requested UNDERLYING not present in H parquet catalog: {missing}")

        if _catalog_covers_all(dataset_keys_d, d_catalog) and any(not symbol_keys_d.get(symbol) for symbol in price_underlyings):
            missing = sorted(symbol for symbol in price_underlyings if not symbol_keys_d.get(symbol))

            raise ValueError(f"[CTX WARNING] requested UNDERLYING not present in D parquet catalog: {missing}")

        if _catalog_covers_all(dataset_keys_m, m_catalog) and any(not symbol_keys_m.get(symbol) for symbol in price_underlyings):
            missing = sorted(symbol for symbol in price_underlyings if not symbol_keys_m.get(symbol))

            raise ValueError(f"[CTX WARNING] requested UNDERLYING not present in M parquet catalog: {missing}")

        universe = {
                    "underlyings": price_underlyings,
                    "price_underlyings": price_underlyings,
                    "alter_underlyings": alter_underlyings,
                    "primary_underlying": primary_underlying,
                    "frequency": {"MIN": freq_min, "H": freq_h, "D": freq_d, "M": freq_m},
                    "price_data_keys": {"MIN": dataset_keys_min, "H": dataset_keys_h, "D": dataset_keys_d, "M": dataset_keys_m},
                    "alter_data_keys": dataset_keys_alter,
                    "price_symbol_data_keys": {"MIN": symbol_keys_min, "H": symbol_keys_h, "D": symbol_keys_d, "M": symbol_keys_m},
                    "alter_symbol_data_keys": symbol_keys_alter,
                    "ctx_mode": "portfolio",
                    "requested_all": "ALL" in requested_underlyings,
                    }

        strategy = {
                    "name": strategy_name,
                    "frequency": frequency,
                    "UNDERLYING": portfolio_underlying,
                    "ctx_mode": "portfolio",
                    }


        return CTX(

                    frequency = frequency,
                    resample = resample,
                    resample_rules = dict(resample_rules) if resample_rules is not None else None,
                    UNDERLYING = portfolio_underlying,
                    data_paths = data_paths,
                    data_load_rules = data_load_rules,
                    universe = universe,
                    strategy = strategy,
                    start_date = start_date,
                    end_date = end_date,
                    **ctx_kwargs,
                  )


    def __post_init__(self) -> None:

        object.__setattr__(self, "data_paths", {k: pathlib.Path(v) for k, v in self.data_paths.items()})

        notebook_dir = pathlib.Path(self.notebook_dir)
        payload_path = self.payload_path
        resample_flag = bool(self.resample)
        agg_rules = _DataOps.normalize_agg_map(self.resample_rules)

        if payload_path is None:
            payload_path = notebook_dir / "ArchiveDeck" / "TransmissionLog" / "payload.pkl"

        else:
            payload_path = pathlib.Path(payload_path)

        start_ts = self._coerce_datetime_bound(self.start_date, label = "start_date")
        end_ts = self._coerce_datetime_bound(self.end_date, label = "end_date")

        object.__setattr__(self, "notebook_dir", notebook_dir)
        object.__setattr__(self, "payload_path", payload_path)
        object.__setattr__(self, "resample", resample_flag)
        object.__setattr__(self, "resample_rules", agg_rules)
        object.__setattr__(self, "start_date", start_ts)
        object.__setattr__(self, "end_date", end_ts)

        object.__setattr__(self, "_single_data", {})
        object.__setattr__(self, "_portfolio_data", {})
        object.__setattr__(self, "_underlying_data", {})
        object.__setattr__(self, "_alter_data", {})
        object.__setattr__(self, "_target_freq_rule", _DataOps.normalise_freq_rule(self.frequency or "H"))

        STRATEGY_NAME = "Strategy"
        FREQUENCY = str(self.frequency) if self.frequency is not None else "Freq"
        UNDERLYING = str(self.UNDERLYING) if self.UNDERLYING is not None else "Underlying"

        if self.strategy:

            STRATEGY_NAME = self.strategy.get('name') if self.strategy.get('name') else 'Strategy'
            FREQUENCY = self.strategy.get('frequency') if self.strategy.get('frequency') else 'Freq'
            UNDERLYING = self.strategy.get('UNDERLYING') if self.strategy.get('UNDERLYING') else 'Underlying'

        log_dir = pathlib.Path(self.log_dir) if self.log_dir is not None else pathlib.Path(self.notebook_dir) / "ArchiveDeck" / "ComputerLog"
        log_dir.mkdir(parents = True, exist_ok = True)
        logger_name = self.logger_name or f"{STRATEGY_NAME}_{FREQUENCY}_CTX_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}_{UNDERLYING}"
        log_file = log_dir / f"{logger_name.replace(':', '_').replace('/', '_')}.log"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        handler: Optional[logging.Handler] = None
        owns_handler = False

        for existing in logger.handlers:

            if getattr(existing, "_ctx_handler", False):
                handler = existing

                break

        if handler is None:

            handler = logging.FileHandler(log_file, encoding = "utf-8")
            setattr(handler, "_ctx_handler", True)
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
            logger.addHandler(handler)
            owns_handler = True

        logger.propagate = False
        object.__setattr__(self, "_logger", logger)
        object.__setattr__(self, "_log_handler", handler)
        object.__setattr__(self, "_owns_log_handler", owns_handler)
        object.__setattr__(self, "_log_path", log_file)

        self._ctx_log(
                            f"[CTX INIT] ctx_mode = {self._ctx_mode()} frequency = {self.frequency} "
                            f"resample = {self.resample} agg_rules = {self.resample_rules} "
                            f"paths = {list(self.data_paths.keys())}"
                            )

        self._ctx_log(f"[CTX INIT] log_path = {getattr(self, '_log_path', None)}")

        freq_map = {
                    "MIN": "1min",
                    "H": "1h",
                    "D": "1d",
                    "M": "ME",
                   }

        if isinstance(self.universe, dict):
            raw_freq = self.universe.get("frequency")

            if isinstance(raw_freq, dict):
                for key, value in raw_freq.items():
                    key_upper = str(key).strip().upper()

                    if key_upper in {"MIN", "H", "D", "M"} and value is not None:
                        freq_map[key_upper] = _DataOps.normalise_freq_rule(value)

        target_rule = _DataOps.normalise_freq_rule(self.frequency or "H")
        ctx_mode = self._ctx_mode()
        available_price_keys: List[str] = []

        if isinstance(self.universe, dict):
            raw_price_keys = self.universe.get("price_data_keys")

            if isinstance(raw_price_keys, dict):
                for freq_key in ("MIN", "H", "D", "M"):
                    raw_keys = raw_price_keys.get(freq_key)

                    if isinstance(raw_keys, list) and any(str(key) in self.data_paths for key in raw_keys):
                        available_price_keys.append(freq_key)

        if not available_price_keys:

            for freq_key in ("MIN", "H", "D", "M"):

                if freq_key in self.data_paths or any(str(key).startswith(f"{freq_key}::") for key in self.data_paths):
                    available_price_keys.append(freq_key)

        self._ctx_log(f"[CTX BOOTSTRAP] ctx_mode={ctx_mode} target_rule={target_rule} freq_map={freq_map}")

        if ctx_mode == "portfolio":
            per_alter: Dict[str, pd.DataFrame] = {}
            price_underlyings = self._portfolio_price_underlyings()
            alter_underlyings = self._portfolio_alter_underlyings()
            has_alter_data = False

            if isinstance(self.universe, dict):
                raw_alter_keys = self.universe.get("alter_data_keys")

                if isinstance(raw_alter_keys, list) and any(str(key) in self.data_paths for key in raw_alter_keys):
                    has_alter_data = True

            self._ctx_log(
                                f"[CTX BOOTSTRAP] price_underlyings={price_underlyings} "
                                f"alter_underlyings={alter_underlyings}"
                                )

            profile_groups: Dict[str, Dict[str, Any]] = {}
            profile_errors: List[str] = []

            for freq_key in available_price_keys:

                try:
                    dataset_keys = self._resolve_data_keys(freq_key)
                    freq_key_upper = str(freq_key).strip().upper()

                    for dataset_key in dataset_keys:

                        if str(dataset_key) not in self.data_paths:

                            continue

                        raw_rule = self.data_load_rules.get(str(dataset_key), {})
                        raw_profile = raw_rule.get("session_profile") if isinstance(raw_rule, Mapping) else None
                        session_profile = raw_profile if isinstance(raw_profile, dict) else {}

                        if not session_profile:
                            session_profile = CTX._infer_parquet_session_profile(self.data_paths[str(dataset_key)])

                        if not session_profile:
                            profile_errors.append(
                                                f"{freq_key_upper}:{self.data_paths[str(dataset_key)]}"
                                                )

                            continue

                        profile_signature = CTX._session_profile_signature(session_profile)
                        group_entry = profile_groups.setdefault(
                                                            profile_signature,
                                                            {"profile": copy.deepcopy(session_profile), "loaded": {}},
                                                            )
                        loaded_by_freq = group_entry["loaded"]

                        if not isinstance(loaded_by_freq, dict):

                            raise TypeError("[CTX WARNING] invalid portfolio session-profile grouping state")

                        loaded_by_freq.setdefault(freq_key_upper, []).append(str(dataset_key))

                except Exception as exc:
                    self._ctx_log(
                                        f"[CTX BOOTSTRAP] profile group miss freq_key={freq_key} reason={exc}",
                                        level = logging.WARNING,
                                        )

                    continue

            if profile_errors:

                raise ValueError(
                                "[CTX WARNING] ctx_portfolio price parquet requires region/session-profiled parquet names; "
                                f"unprofiled datasets={profile_errors}"
                                )

            if not profile_groups:

                raise ValueError("[CTX WARNING] ctx_portfolio failed to resolve any valid session-profile data groups")

            if len(profile_groups) > 1:
                profile_detail = {
                                profile_signature: {
                                                    str(freq_key): [
                                                                    str(self.data_paths[str(dataset_key)])
                                                                    for dataset_key in dataset_keys

                                                                    if str(dataset_key) in self.data_paths
                                                                  ]
                                                    for freq_key, dataset_keys in cast(Dict[str, List[str]], profile_entry.get("loaded", {})).items()
                                                   }
                                for profile_signature, profile_entry in profile_groups.items()
                               }

                raise ValueError(
                                "[CTX WARNING] ctx_portfolio refuses to build one portfolio data clock from multiple session profiles. "
                                "Create one CTX per region/session for factor construction, then combine signals at portfolio/backtest level. "
                                f"profiles={profile_detail}"
                                )

            portfolio_data: Dict[str, pd.DataFrame] = {}

            for profile_signature, profile_entry in profile_groups.items():

                try:
                    group_profile = profile_entry.get("profile")
                    session_profile = group_profile if isinstance(group_profile, Mapping) and group_profile else None
                    raw_loaded_by_freq = profile_entry.get("loaded")

                    if not isinstance(raw_loaded_by_freq, dict):

                        continue

                    loaded_price_data: Dict[str, pd.DataFrame] = {}

                    for freq_key, dataset_keys in raw_loaded_by_freq.items():

                        dataset_keys = [str(key) for key in dataset_keys if str(key) in self.data_paths]

                        if not dataset_keys:

                            continue

                        rule = self.data_load_rules.get(dataset_keys[0], {})
                        freq_key_upper = str(freq_key).strip().upper()
                        selected_symbols = None if self._portfolio_requested_all() else self._portfolio_price_underlyings()
                        dataset_paths = [pathlib.Path(self.data_paths[k]) for k in dataset_keys]

                        self._ctx_log(
                                            f"[CTX BULK LOAD] key={freq_key_upper} profile={profile_signature} "
                                            f"requested_all={self._portfolio_requested_all()} "
                                            f"symbols={None if selected_symbols is None else len(selected_symbols)} "
                                            f"datasets={[str(path) for path in dataset_paths]}"
                                            )

                        long_data = self._load_ohlc_data_from_datasets(
                                                                        key = freq_key_upper,
                                                                        dataset_keys = dataset_keys,
                                                                        rule = rule,
                                                                        selected_symbols = selected_symbols,
                                                                        keep_symbol = True,
                                                                      )

                        if long_data.empty:
                            portfolio_data_item = pd.DataFrame(columns = ["Datetime"])

                        else:

                            if "Datetime" not in long_data.columns or "Symbol" not in long_data.columns:

                                raise KeyError("[CTX WARNING] portfolio bulk pivot requires both 'Datetime' and 'Symbol' columns")

                            value_columns = [col for col in long_data.columns if col not in {"Datetime", "Symbol"}]

                            if not value_columns:
                                portfolio_data_item = long_data[["Datetime"]].drop_duplicates().sort_values("Datetime").reset_index(drop = True)

                            else:
                                wide_data = long_data.copy()
                                wide_data["Datetime"] = pd.to_datetime(wide_data["Datetime"], errors = "coerce")
                                wide_data = wide_data.dropna(subset = ["Datetime", "Symbol"])
                                wide_data["Symbol"] = wide_data["Symbol"].astype(str).str.strip().str.upper()
                                wide_data = wide_data.loc[wide_data["Symbol"].ne("")]
                                wide_data = wide_data.loc[~wide_data.duplicated(subset = ["Symbol", "Datetime"], keep = "last")]
                                wide_data = wide_data.sort_values(["Datetime", "Symbol"])

                                portfolio_data_item = cast(pd.DataFrame, wide_data.set_index(["Datetime", "Symbol"])[value_columns].unstack("Symbol"))
                                portfolio_data_item.columns = [f"{symbol}_{field}" for field, symbol in portfolio_data_item.columns]
                                portfolio_data_item = portfolio_data_item.sort_index()
                                portfolio_data_item = portfolio_data_item.loc[~portfolio_data_item.index.duplicated(keep = "last")]
                                portfolio_data_item = portfolio_data_item.reset_index()
                                portfolio_data_item["Datetime"] = pd.to_datetime(portfolio_data_item["Datetime"], errors = "coerce")
                                portfolio_data_item = portfolio_data_item.dropna(subset = ["Datetime"]).sort_values("Datetime").reset_index(drop = True)

                        loaded_price_data[freq_key_upper] = portfolio_data_item
                        self._ctx_log(
                                            f"[CTX BOOTSTRAP] bulk portfolio profile={profile_signature} "
                                            f"freq={freq_key_upper} rows={len(loaded_price_data[freq_key_upper])}"
                                            )

                    if not loaded_price_data:

                        continue

                    group_data = self._derive_standard_data(loaded_price_data, freq_map, session_profile = session_profile)
                    group_data["TARGET"] = self._build_data_for_target(group_data, target_rule, freq_map, session_profile = session_profile)

                    for data_key, data_item in group_data.items():

                        if data_item is None or data_item.empty:

                            continue

                        data_item = data_item.copy()

                        if "Datetime" not in data_item.columns:

                            raise KeyError(f"[CTX WARNING] portfolio data key '{data_key}' missing 'Datetime' after profile resample")

                        data_item["Datetime"] = pd.to_datetime(data_item["Datetime"], errors = "coerce")
                        data_item = data_item.dropna(subset = ["Datetime"]).sort_values("Datetime")
                        data_item = data_item.loc[~data_item["Datetime"].duplicated(keep = "last")]

                        if data_key not in portfolio_data:
                            portfolio_data[data_key] = data_item.reset_index(drop = True)

                            continue

                        left_data = portfolio_data[data_key].copy()
                        left_data["Datetime"] = pd.to_datetime(left_data["Datetime"], errors = "coerce")
                        left_data = left_data.dropna(subset = ["Datetime"]).sort_values("Datetime")
                        left_data = left_data.loc[~left_data["Datetime"].duplicated(keep = "last")].set_index("Datetime")
                        right_data = data_item.set_index("Datetime")
                        duplicate_columns = sorted(set(left_data.columns).intersection(set(right_data.columns)))

                        if duplicate_columns:

                            raise ValueError(
                                            f"[CTX WARNING] portfolio profile merge found duplicate columns for key='{data_key}': "
                                            f"{duplicate_columns}"
                                            )

                        merged_data = left_data.join(right_data, how = "outer").sort_index()
                        merged_data = merged_data.loc[~merged_data.index.duplicated(keep = "last")]
                        portfolio_data[data_key] = merged_data.reset_index().sort_values("Datetime").reset_index(drop = True)

                except Exception as exc:
                    self._ctx_log(
                                        f"[CTX BOOTSTRAP] profile group miss profile={profile_signature} reason={exc}",
                                        level = logging.WARNING,
                                        )

                    continue

            if not portfolio_data:

                raise ValueError("[CTX WARNING] ctx_portfolio failed to load any valid data")

            if "TARGET" not in portfolio_data:
                portfolio_data["TARGET"] = self._build_data_for_target(portfolio_data, target_rule, freq_map)

            self._ctx_log(
                                f"[CTX BOOTSTRAP] portfolio data={sorted(portfolio_data.keys())} "
                                f"target_rows={len(portfolio_data['TARGET'])}"
                                )

            for alter_key in alter_underlyings if has_alter_data else []:

                try:
                    alter_data_item = self.load_price_data("ALTER", UNDERLYING = alter_key)
                    per_alter[alter_key] = alter_data_item
                    self._ctx_log(f"[CTX BOOTSTRAP] alter_key={alter_key} alter_rows={len(alter_data_item)}")

                except Exception as exc:
                    self._ctx_log(
                                        f"[CTX BOOTSTRAP] load miss alter_key={alter_key} reason={exc}",
                                        level = logging.WARNING,
                                        )

            if per_alter:
                mergeable_alter = {k: v for k, v in per_alter.items() if "Datetime" in v.columns}

                if mergeable_alter:
                    merged: Optional[pd.DataFrame] = None

                    for underlying, data_item in mergeable_alter.items():
                        part = data_item.copy()

                        if "Datetime" not in part.columns:

                            raise KeyError(f"[CTX WARNING] 'Datetime' column missing for underlying '{underlying}'")

                        part["Datetime"] = pd.to_datetime(part["Datetime"], errors = "coerce")
                        part = part.dropna(subset = ["Datetime"]).sort_values("Datetime")
                        part = part.loc[~part["Datetime"].duplicated(keep = "last")]
                        part = part.set_index("Datetime")
                        part = part.rename(columns = {col: f"{underlying}_{col}" for col in part.columns})

                        if merged is None:
                            merged = part

                        else:
                            merged = merged.join(part, how = "outer")

                    merged_data = merged

                    if merged_data is None:
                        portfolio_data["ALTER"] = pd.DataFrame(columns = ["Datetime"])

                    else:
                        merged_data = merged_data.sort_index()
                        merged_data = merged_data.loc[~merged_data.index.duplicated(keep = "last")]
                        merged_data = merged_data.reset_index()
                        merged_data["Datetime"] = pd.to_datetime(merged_data["Datetime"], errors = "coerce")
                        merged_data = merged_data.dropna(subset = ["Datetime"]).sort_values("Datetime")
                        portfolio_data["ALTER"] = merged_data.reset_index(drop = True)

                    self._ctx_log(f"[CTX BOOTSTRAP] portfolio freq=ALTER rows={len(portfolio_data['ALTER'])}")

                else:
                    self._ctx_log(
                                        "[CTX BOOTSTRAP] skip ALTER merge: no alter data contain 'Datetime' column",
                                        level = logging.WARNING,
                                        )

            object.__setattr__(self, "_underlying_data", {})
            object.__setattr__(self, "_alter_data", per_alter)
            object.__setattr__(self, "_portfolio_data", portfolio_data)
            object.__setattr__(self, "_single_data", {})

        else:
            loaded_data: Dict[str, pd.DataFrame] = {}
            has_alter_data = False

            if isinstance(self.universe, dict):
                raw_alter_keys = self.universe.get("alter_data_keys")

                if isinstance(raw_alter_keys, list) and any(str(key) in self.data_paths for key in raw_alter_keys):
                    has_alter_data = True

            for freq_key in available_price_keys:

                try:
                    loaded_data[freq_key] = self.load_price_data(freq_key, UNDERLYING = self.UNDERLYING)

                except Exception as exc:
                    self._ctx_log(f"[CTX BOOTSTRAP] load miss freq_key={freq_key} reason={exc}", level = logging.WARNING)

                    continue

            if not loaded_data:

                raise ValueError("[CTX WARNING] ctx_single failed to load any valid data")

            standard_data = self._derive_standard_data(loaded_data, freq_map)
            standard_data["TARGET"] = self._build_data_for_target(standard_data, target_rule, freq_map)

            if has_alter_data:

                try:
                    standard_data["ALTER"] = self.load_price_data("ALTER")
                    self._ctx_log(f"[CTX BOOTSTRAP] single alter_rows={len(standard_data['ALTER'])}")

                except Exception as exc:
                    self._ctx_log(f"[CTX BOOTSTRAP] load miss freq_key=ALTER reason={exc}", level = logging.WARNING)

            self._ctx_log(f"[CTX BOOTSTRAP] single data={sorted(standard_data.keys())} target_rows={len(standard_data['TARGET'])}")

            object.__setattr__(self, "_single_data", standard_data)
            object.__setattr__(self, "_portfolio_data", {})
            object.__setattr__(self, "_underlying_data", {})
            object.__setattr__(self, "_alter_data", {})

        self._ctx_log(f"[CTX BOOTSTRAP] completed ctx_mode={ctx_mode}")


    def __del__(self) -> None:
        """
        ### What It Does
        Implements Python protocol method `__del__` for `CTX`.

        #### Responsibility
        Allows `CTX` instances to work with the corresponding Python operation.

        #### How To Use
        Use it implicitly through Python syntax rather than calling `CTX.__del__(...)` directly.

        #### Usage Example
        `result = __del__(...)`
        """

        try:
            self.close_logger()

        except Exception:
            pass


    def _ctx_log(self, message: str, level: int = logging.INFO) -> None:

        print(message)
        logger = getattr(self, "_logger", None)

        if logger is not None:
            logger.log(level, message)


    def _ctx_mode(self) -> str:

        strategy_ctx_mode = None

        if isinstance(self.strategy, dict):
            strategy_ctx_mode = self.strategy.get("ctx_mode")

        if isinstance(strategy_ctx_mode, str) and strategy_ctx_mode.strip().lower() == "portfolio":

            return "portfolio"


        return "single"


    def _resolve_data_keys(self, key: str) -> List[str]:

        raw_key = str(key).strip()

        if not raw_key:

            raise KeyError(f"[CTX WARNING] data key is empty")

        if raw_key in self.data_paths:

            return [raw_key]

        freq_key = raw_key.upper()

        if isinstance(self.universe, dict):

            if freq_key == "ALTER":
                raw_keys = self.universe.get("alter_data_keys")

            else:
                price_data_keys = self.universe.get("price_data_keys")

                if isinstance(price_data_keys, dict):
                    raw_keys = price_data_keys.get(freq_key)

                else:
                    raw_keys = None

            if isinstance(raw_keys, list):
                resolved = [str(k) for k in raw_keys if str(k) in self.data_paths]

                if resolved:

                    return resolved


        raise KeyError(f"[CTX WARNING] data path '{key}' missing in ctx")


    def _portfolio_price_underlyings(self) -> List[str]:

        raw_list: Optional[List[Any]] = None

        if isinstance(self.universe, dict):
            raw_list = self.universe.get("price_underlyings")

            if not isinstance(raw_list, list):
                raw_list = self.universe.get("underlyings")

        if not isinstance(raw_list, list):

            return []


        return [str(x).strip().upper() for x in raw_list if str(x).strip()]


    def _portfolio_alter_underlyings(self) -> List[str]:

        raw_list: Optional[List[Any]] = None

        if isinstance(self.universe, dict):
            raw_list = self.universe.get("alter_underlyings")

        if not isinstance(raw_list, list):

            return []


        return [str(x).strip().upper() for x in raw_list if str(x).strip()]


    def _portfolio_requested_all(self) -> bool:

        if not isinstance(self.universe, dict):

            return False


        return bool(self.universe.get("requested_all"))


    def _session_profile_for_price_key(self, freq_key: str) -> Dict[str, Any]:

        if not isinstance(self.universe, dict):

            return {}

        raw_keys_map = self.universe.get("price_data_keys")

        if not isinstance(raw_keys_map, Mapping):

            return {}

        raw_keys = raw_keys_map.get(str(freq_key).strip().upper())

        if not isinstance(raw_keys, list):

            return {}

        dataset_keys = [str(key) for key in raw_keys if str(key) in self.data_paths]

        if not dataset_keys:

            return {}

        profiles: List[Dict[str, Any]] = []

        for dataset_key in dataset_keys:

            raw_rule = self.data_load_rules.get(str(dataset_key), {})
            raw_profile = raw_rule.get("session_profile") if isinstance(raw_rule, Mapping) else None
            profile = raw_profile if isinstance(raw_profile, dict) else {}

            if not profile and str(dataset_key) in self.data_paths:
                profile = CTX._infer_parquet_session_profile(self.data_paths[str(dataset_key)])

            if profile:
                profiles.append(profile)

        if not profiles or len(profiles) != len(dataset_keys):

            return {}

        signatures = {CTX._session_profile_signature(profile) for profile in profiles}

        if len(signatures) != 1:

            return {}


        return copy.deepcopy(profiles[0])


    def _load_ohlc_data_from_datasets(self,
                                        *,
                                        key: str,
                                        dataset_keys: List[str],
                                        rule: Mapping[str, Any],
                                        selected_symbols: Optional[List[str]],
                                        keep_symbol: bool,
                                      ) -> pd.DataFrame:

        dataset_paths = [pathlib.Path(self.data_paths[k]) for k in dataset_keys]

        path_sql = ", ".join(
                            "'" + str(path.as_posix()).replace("'", "''") + "'"
                            for path in dataset_paths
                            )

        relation_sql = f"read_parquet([{path_sql}])"
        where_clauses = ["Symbol IS NOT NULL"]
        query_params: List[Any] = []

        if selected_symbols:

            symbol_sql = ", ".join(
                                    "'" + str(value).replace("'", "''") + "'"
                                    for value in selected_symbols
                                  )

            where_clauses.append(f"upper(Symbol) IN ({symbol_sql})")

        conn = duckdb.connect(database = ":memory:")
        conn.execute("PRAGMA disable_progress_bar")

        try:
            schema_probe = conn.execute(f"DESCRIBE SELECT * FROM {relation_sql}").fetch_df()
            source_columns = [str(name) for name in schema_probe["column_name"]]
            source_lookup = {str(col).lower(): str(col) for col in source_columns}
            source_types = {str(name).lower(): str(dtype).upper() for name, dtype in zip(schema_probe["column_name"], schema_probe["column_type"])}

            def _quote_identifier(name: str) -> str:

                return '"' + str(name).replace('"', '""') + '"'


            def _select_optional(name: str, alias: Optional[str] = None) -> Optional[str]:

                resolved = source_lookup.get(str(name).lower())
                output_name = alias or name

                if resolved is None:

                    return None

                if output_name == resolved:

                    return _quote_identifier(resolved)


                return f"{_quote_identifier(resolved)} AS {_quote_identifier(output_name)}"


            def _select_required(name: str) -> str:

                resolved = str(required_lookup[name])

                # Alias the resolved source column to its canonical name so downstream (which references
                # "Datetime"/"Open"/... by exact case) always sees canonical labels; byte-identical SQL when
                # the source is already canonically cased.
                if name == resolved:

                    return _quote_identifier(resolved)


                return f"{_quote_identifier(resolved)} AS {_quote_identifier(name)}"


            required_lookup = {name: source_lookup.get(name.lower()) for name in ("Datetime", "Symbol", "Open", "High", "Low", "Close")}
            missing_required = [name for name, resolved in required_lookup.items() if resolved is None]

            if missing_required:

                raise KeyError(f"[CTX WARNING] parquet data missing required columns for '{key}': {missing_required}; available columns: {source_columns}")

            symbol_column = str(required_lookup["Symbol"])

            # Datetime window pushdown, resolved AFTER the schema probe so the predicate can adapt to the
            # stored column type. The start/end bounds are tz-naive: a plain TIMESTAMP column compares
            # correctly (and keeps parquet row-group statistics pushdown), but a VARCHAR Datetime carrying a
            # tz offset (the exact shape the downstream strip handles) cannot be compared to a TIMESTAMP
            # literal and raises a binder error. Normalize only that case with the same tz-suffix strip the
            # loader applies downstream, so the filtered window matches the returned wall-clock.
            datetime_resolved = str(required_lookup["Datetime"])
            datetime_quoted = _quote_identifier(datetime_resolved)

            if "VARCHAR" in source_types.get(datetime_resolved.lower(), ""):
                datetime_tz_regex = str(rule.get("timezone_suffix_regex") or "").replace("'", "''")

                if datetime_tz_regex:
                    datetime_pred = f"TRY_CAST(regexp_replace({datetime_quoted}, '{datetime_tz_regex}', '') AS TIMESTAMP)"

                else:
                    datetime_pred = f"TRY_CAST({datetime_quoted} AS TIMESTAMP)"

            else:
                datetime_pred = datetime_quoted

            if self.start_date is not None:
                where_clauses.append(f"{datetime_pred} >= ?")
                query_params.append(self.start_date)

            if self.end_date is not None:
                # start_date is inclusive from midnight (>= 00:00 admits the whole first day). A date-only
                # end_date sits at midnight too, so a symmetric window must include the WHOLE last day --
                # otherwise every intraday bar (09:30..16:00 > 00:00) on the final day is silently dropped
                # while a normalized daily bar survives. Use half-open < (end + 1 day) for the date-only
                # case; honor an explicit intraday cutoff time verbatim.
                if isinstance(self.end_date, datetime.datetime) and self.end_date.time() == datetime.time(0, 0):
                    where_clauses.append(f"{datetime_pred} < ?")
                    query_params.append(self.end_date + datetime.timedelta(days = 1))

                else:
                    where_clauses.append(f"{datetime_pred} <= ?")
                    query_params.append(self.end_date)

            where_sql = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

            select_exprs = [
                            _select_required("Datetime"),
                            f"upper({_quote_identifier(symbol_column)}) AS Symbol",
                            _select_required("Open"),
                            _select_required("High"),
                            _select_required("Low"),
                            _select_required("Close"),
                           ]
            select_exprs.extend(
                                expr for expr in (
                                                  _select_optional("Volume"),
                                                  _select_optional("OpenInterest"),
                                                  _select_optional("Vwap"),
                                                  _select_optional("Dividends"),
                                                  _select_optional("Stock_Splits", "Stock Splits"),
                                                  _select_optional("Shares_Outstanding", "Shares Outstanding"),
                                                  _select_optional("Source"),
                                                 )

                                if expr is not None
                               )

            sql = f"SELECT {', '.join(select_exprs)} FROM {relation_sql}{where_sql} ORDER BY Datetime, Symbol"
            df = conn.execute(sql, query_params).fetch_df()

        finally:
            conn.close()

        rename_map = rule.get("rename")

        if rename_map:
            df = df.rename(columns = rename_map)

        if "Datetime" not in df.columns:

            raise KeyError(f"[CTX WARNING] column 'Datetime' missing for '{key}' after rename; available columns: {list(df.columns)}")

        required_ohlc = tuple(rule.get("ohlc_required", ("Open", "High", "Low", "Close")))
        missing_ohlc = [col for col in required_ohlc if col not in df.columns]

        if missing_ohlc:

            raise KeyError(
                            f"[CTX WARNING] OHLC data required for '{key}', missing columns: {missing_ohlc}; "
                            f"available columns: {list(df.columns)}"
                          )

        tz_pattern = rule.get("timezone_suffix_regex")

        if tz_pattern:
            df["Datetime"] = df["Datetime"].astype(str).str.replace(tz_pattern, "", regex = True)

        if rule.get("to_datetime", True):
            df["Datetime"] = pd.to_datetime(df["Datetime"], utc = False, errors = "coerce")

            if isinstance(df["Datetime"].dtype, pd.DatetimeTZDtype):
                # Single naive-wall-clock contract: any tz that survives here (a non-US offset the strip
                # regex missed, or a native TIMESTAMPTZ source column) is dropped -- this is the one axis
                # every backtest engine reads. tz_localize(None) keeps the session-local wall-clock (does
                # NOT shift to UTC), matching how producers localize->strip before writing parquet.
                df["Datetime"] = df["Datetime"].dt.tz_localize(None)

        df = df.dropna(subset = ["Datetime"])

        if "Symbol" in df.columns:
            df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
            df = df.loc[df["Symbol"].ne("")]

        if rule.get("normalize_dates"):
            # Normalize BEFORE dedup: daily/monthly data stored with intraday timestamps has distinct
            # raw Datetimes on the same day, so deduping on the raw Datetime keeps both rows and the
            # later normalize collapses them to the same date -> duplicate same-date bars, double-counted.
            df["Datetime"] = df["Datetime"].dt.normalize()

        dedup_subset = ["Datetime"]

        if "Symbol" in df.columns:
            dedup_subset = ["Symbol", "Datetime"]

        self._ctx_log(f"[CTX LOAD] dedup_subset={dedup_subset}")
        df = df.loc[~df.duplicated(subset = dedup_subset, keep = "last")]
        sort_cols = [c for c in dedup_subset if c != "Datetime"] + ["Datetime"]
        df = df.sort_values(sort_cols)

        if not keep_symbol and "Symbol" in df.columns:
            df = df.drop(columns = ["Symbol"])

        if rule.get("reset_index", True):
            df = df.reset_index(drop = True)

        self._ctx_log(
                            f"[CTX LOAD] done key={key} rows={len(df)} cols={len(df.columns)} "
                            f"datetime_min={df['Datetime'].min() if 'Datetime' in df.columns and not df.empty else None} "
                            f"datetime_max={df['Datetime'].max() if 'Datetime' in df.columns and not df.empty else None}"
                            )


        return df


    def _build_data_for_target(self,
                                available_data: Dict[str, pd.DataFrame],
                                target_rule: str,
                                freq_map: Dict[str, str],
                                session_profile: Optional[Mapping[str, Any]] = None,
                                ) -> pd.DataFrame:

        if not available_data:

            raise ValueError("[CTX WARNING] no source data available for target frequency")

        target_rule = _DataOps.normalise_freq_rule(target_rule)
        target_sec = _DataOps.freq_seconds(target_rule)

        self._ctx_log(
                            f"[CTX BUILD TARGET] target_rule={target_rule} available={sorted(available_data.keys())} "
                            f"resample={self.resample}"
                            )

        candidates: List[Tuple[float, str]] = []

        for src_key, src_data in available_data.items():

            if src_data is None or src_data.empty:

                continue

            if src_key not in freq_map:

                continue

            src_rule = _DataOps.normalise_freq_rule(freq_map[src_key])
            src_sec = _DataOps.freq_seconds(src_rule)

            if src_sec <= target_sec + 1e-9:
                candidates.append((src_sec, src_key))

        if not candidates:
            available_keys = sorted(available_data.keys())

            raise ValueError(
                            f"[CTX WARNING] frequency '{target_rule}' not covered by available data keys {available_keys}; "
                            "please provide finer-grained data_min/data_h/data_d."
                            )

        candidates.sort(key = lambda x: x[0], reverse = True)
        _, src_key = candidates[0]
        src_data = available_data[src_key].copy()
        src_rule = _DataOps.normalise_freq_rule(freq_map[src_key])
        active_session_profile = session_profile

        if not active_session_profile:
            active_session_profile = self._session_profile_for_price_key(src_key)

        if not active_session_profile:
            profile_candidates: List[Tuple[float, Dict[str, Any]]] = []
            max_seconds = _DataOps.freq_seconds(src_rule)

            for profile_freq_key, profile_freq_rule in freq_map.items():

                try:
                    source_seconds = _DataOps.freq_seconds(profile_freq_rule)

                except Exception:

                    continue

                if source_seconds > max_seconds + 1e-9:

                    continue

                candidate_profile = self._session_profile_for_price_key(str(profile_freq_key))

                if candidate_profile:
                    profile_candidates.append((source_seconds, candidate_profile))

            if profile_candidates:
                profile_candidates.sort(key = lambda item: item[0], reverse = True)
                active_session_profile = copy.deepcopy(profile_candidates[0][1])

        if abs(_DataOps.freq_seconds(src_rule) - target_sec) <= 1e-9:
            self._ctx_log(f"[CTX BUILD TARGET] direct_use src_key={src_key} src_rule={src_rule} rows={len(src_data)}")

            return src_data.reset_index(drop = True)

        if not self.resample:

            raise ValueError(
                            f"[CTX WARNING] resample disabled: target '{target_rule}' requires converting from '{src_rule}' "
                            f"(src_key={src_key})."
                            )

        self._ctx_log(
                            f"[CTX BUILD TARGET] resampling src_key={src_key} src_rule={src_rule} -> target_rule={target_rule} "
                            f"agg_rules={self.resample_rules}"
                            )


        return _DataOps.resample_price_data(

                                            src_data,
                                            target_rule,
                                            custom_agg_map = self.resample_rules,
                                            strict_custom_agg = True,
                                            session_profile = active_session_profile,
                                            )


    def _derive_standard_data(self,
                              loaded_data: Dict[str, pd.DataFrame],
                              freq_map: Dict[str, str],
                              session_profile: Optional[Mapping[str, Any]] = None,
                             ) -> Dict[str, pd.DataFrame]:

        result: Dict[str, pd.DataFrame] = {}

        for key, data in loaded_data.items():

            if data is None or data.empty:

                continue

            result[str(key).upper()] = data.reset_index(drop = True)

        for freq_key in ("MIN", "H", "D", "M"):

            if freq_key in result:

                continue

            if freq_key not in freq_map:

                continue

            target_rule = _DataOps.normalise_freq_rule(freq_map[freq_key])
            target_sec = _DataOps.freq_seconds(target_rule)

            has_candidate = any(
                                src_key in freq_map and _DataOps.freq_seconds(freq_map[src_key]) <= target_sec + 1e-9
                                for src_key in result
                                )

            if not has_candidate:

                continue

            try:
                result[freq_key] = self._build_data_for_target(
                                                                result,
                                                                target_rule,
                                                                freq_map,
                                                                session_profile = session_profile or self._session_profile_for_price_key(freq_key),
                                                              )
                self._ctx_log(f"[CTX DERIVE] generated freq_key={freq_key} target_rule={target_rule} rows={len(result[freq_key])}")

            except Exception as exc:
                self._ctx_log(
                                    f"[CTX DERIVE] skip freq_key={freq_key} target_rule={target_rule} reason={exc}",
                                    level = logging.WARNING,
                                    )

                continue


        return result


    def __getattr__(self, name: str) -> Any:
        """
        ### What It Does
        Implements Python protocol method `__getattr__` for `CTX`.

        #### Responsibility
        Allows `CTX` instances to work with the corresponding Python operation.

        #### How To Use
        Use it implicitly through Python syntax rather than calling `CTX.__getattr__(...)` directly.

        #### Key Parameters In Practice
        - `name`
          - Workflow input for this operation. Set it according to the current data shape and execution path; do not treat the default as correct unless it matches the run contract.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = __getattr__(...)`

        ---

        ### Parameters
        - `name`: **str**.

        ---

        ### Returns
        - `result`: **Any**.
        """

        attr = str(name).strip()
        attr_upper = attr.upper()

        if attr_upper in {"MIN", "H", "D", "M"}:

            try:
                freq_map = object.__getattribute__(self, "freq_map")

            except AttributeError:
                freq_map = {}

            if isinstance(freq_map, dict) and attr_upper in freq_map:

                return freq_map[attr_upper]

        if attr_upper in {"MIN", "H", "D", "M", "ALTER", "TARGET", "FREQUENCY"}:

            return self.get_data("TARGET" if attr_upper == "FREQUENCY" else attr_upper)

        if self._ctx_mode() == "portfolio":

            if attr_upper in self._portfolio_price_underlyings():

                return self.get_data("TARGET", UNDERLYING = attr_upper)

            alter_map = getattr(self, "_alter_data", {})

            if attr_upper in alter_map or attr_upper in self._portfolio_alter_underlyings():

                return self.get_data("ALTER", UNDERLYING = attr_upper)


        raise AttributeError(f"[CTX WARNING] {self.__class__.__name__} has no attribute '{name}'")


    def __getitem__(self, key: str) -> pd.DataFrame:
        """
        ### What It Does
        Implements Python protocol method `__getitem__` for `CTX`.

        #### Responsibility
        Allows `CTX` instances to work with the corresponding Python operation.

        #### How To Use
        Use it implicitly through Python syntax rather than calling `CTX.__getitem__(...)` directly.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = __getitem__(...)`

        ---

        ### Parameters
        - `key`: **str**.

        ---

        ### Returns
        - `result`: **pd.DataFrame**.
        """

        key_upper = str(key).strip().upper()

        if key_upper in {"MIN", "H", "D", "M"}:

            try:
                freq_map = object.__getattribute__(self, "freq_map")

            except AttributeError:
                freq_map = {}

            if isinstance(freq_map, dict) and key_upper in freq_map:

                return freq_map[key_upper]

        if key_upper in {"MIN", "H", "D", "M", "ALTER", "TARGET", "FREQUENCY"}:

            return self.get_data("TARGET" if key_upper == "FREQUENCY" else key_upper)

        if self._ctx_mode() == "portfolio":

            if key_upper in self._portfolio_price_underlyings():

                return self.get_data("TARGET", UNDERLYING = key_upper)

            alter_map = getattr(self, "_alter_data", {})

            if key_upper in alter_map or key_upper in self._portfolio_alter_underlyings():

                return self.get_data("ALTER", UNDERLYING = key_upper)

        if self.UNDERLYING and key_upper == str(self.UNDERLYING).strip().upper():

            return self.get_data("TARGET")


        raise KeyError(f"[CTX WARNING] ctx key '{key}' not found")


    def close_logger(self) -> None:
        """
        ### What It Does
        Closes the CTX-owned logger and releases its file handler.

        #### Responsibility
        Prevents duplicated log writes and locked files after a context finishes building payloads.

        #### How To Use
        Call it when a notebook or run script is done using a CTX instance.

        #### Usage Example
        `result = close_logger(...)`
        """

        logger = getattr(self, "_logger", None)
        handler = getattr(self, "_log_handler", None)
        owns_handler = bool(getattr(self, "_owns_log_handler", False))

        if logger is None or handler is None or not owns_handler:

            return

        try:
            logger.removeHandler(handler)

        except Exception:
            pass

        try:
            handler.flush()

        except Exception:
            pass

        try:
            handler.close()

        except Exception:
            pass

        object.__setattr__(self, "_owns_log_handler", False)


    def load_price_data(self, key: str, UNDERLYING: Optional[str] = None) -> pd.DataFrame:
        """
        ### What It Does
        Returns context-managed market data.

        #### Responsibility
        Provides one stable entry point for retrieving raw or prepared data from the active context.

        #### How To Use
        Use it when you need a specific data family or one underlying slice from a portfolio context.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.
        - `UNDERLYING`
          - Asset selector. In single workflows this is one ticker; in portfolio workflows it should identify the explicit universe/list being loaded and must match available data columns/artifacts.
          - Expected shape/type: `Optional[str]`.

        #### Usage Example
        `result = load_price_data(...)`

        ---

        ### Parameters
        - `key`: **str**.

        #### Optional Parameters
        - `UNDERLYING`: **Optional[str]** = *None*.

        ---

        ### Returns
        - `result`: **pd.DataFrame**.
        """

        dataset_keys = self._resolve_data_keys(key)
        rule = self.data_load_rules.get(dataset_keys[0], {})
        schema = str(rule.get("schema", "ohlc")).strip().lower()
        key_upper = str(key).strip().upper()
        resolved_underlying = str(UNDERLYING).strip().upper() if UNDERLYING is not None and str(UNDERLYING).strip() else None

        if resolved_underlying is None:
            raw_underlyings: Any = None

            if isinstance(self.universe, dict):

                if key_upper == "ALTER":
                    raw_underlyings = self.universe.get("alter_underlyings")

                else:
                    raw_underlyings = self.universe.get("price_underlyings")

                    if not isinstance(raw_underlyings, list):
                        raw_underlyings = self.universe.get("underlyings")

            if isinstance(raw_underlyings, list) and raw_underlyings:
                resolved_underlying = str(raw_underlyings[0]).strip().upper()

        if schema == "ohlc" and not resolved_underlying:

            raise ValueError(f"[CTX WARNING] OHLC parquet load requires underlying selection for key='{key}'.")

        if schema == "ohlc" and isinstance(self.universe, dict):
            raw_symbol_map = self.universe.get("alter_symbol_data_keys") if key_upper == "ALTER" else (
                self.universe.get("price_symbol_data_keys", {}).get(key_upper) if isinstance(self.universe.get("price_symbol_data_keys"), dict) else None
            )

            if isinstance(raw_symbol_map, dict) and resolved_underlying in raw_symbol_map and raw_symbol_map[resolved_underlying]:
                dataset_keys = [str(k) for k in raw_symbol_map[resolved_underlying] if str(k) in self.data_paths]

        dataset_paths = [pathlib.Path(self.data_paths[k]) for k in dataset_keys]
        self._ctx_log(
                            f"[CTX LOAD] key={key_upper} UNDERLYING = {resolved_underlying} datasets={[str(path) for path in dataset_paths]}"
                            )

        if schema == "alter":

            path_sql = ", ".join(
                                "'" + str(path.as_posix()).replace("'", "''") + "'"
                                for path in dataset_paths
                                )

            relation_sql = f"read_parquet([{path_sql}])"
            where_clauses: List[str] = []
            query_params: List[Any] = []

        else:

            return self._load_ohlc_data_from_datasets(

                                                    key = key_upper,
                                                    dataset_keys = dataset_keys,
                                                    rule = rule,
                                                    selected_symbols = [resolved_underlying] if resolved_underlying else None,
                                                    keep_symbol = False,
                                                    )

        conn = duckdb.connect(database = ":memory:")
        conn.execute("PRAGMA disable_progress_bar")

        try:
            # clip ALTER alt-data to the context window BEFORE it can be joined into any factor,
            # mirroring the OHLC path — otherwise rows dated after end_date leak future information.
            if self.start_date is not None or self.end_date is not None:
                alter_schema_probe = conn.execute(f"DESCRIBE SELECT * FROM {relation_sql}").fetch_df()
                alter_columns = [str(name) for name in alter_schema_probe["column_name"]]
                alter_types = {str(name).lower(): str(dtype).upper() for name, dtype in zip(alter_schema_probe["column_name"], alter_schema_probe["column_type"])}
                datetime_col = next((col for col in alter_columns if str(col).lower() == "datetime"), None)

                if datetime_col is not None:
                    quoted_dt = '"' + str(datetime_col).replace('"', '""') + '"'

                    # Mirror the OHLC pushdown: a VARCHAR Datetime carrying a tz offset cannot compare to the
                    # tz-naive bound literal, so normalize that case with the same strip; plain TIMESTAMP stays raw.
                    if "VARCHAR" in alter_types.get(str(datetime_col).lower(), ""):
                        alter_tz_regex = str(rule.get("timezone_suffix_regex") or "").replace("'", "''")

                        if alter_tz_regex:
                            datetime_pred = f"TRY_CAST(regexp_replace({quoted_dt}, '{alter_tz_regex}', '') AS TIMESTAMP)"

                        else:
                            datetime_pred = f"TRY_CAST({quoted_dt} AS TIMESTAMP)"

                    else:
                        datetime_pred = quoted_dt

                    if self.start_date is not None:
                        where_clauses.append(f"{datetime_pred} >= ?")
                        query_params.append(self.start_date)

                    if self.end_date is not None:
                        # Mirror the OHLC pushdown: a date-only (midnight) end_date is inclusive of the whole
                        # last day via half-open < (end + 1 day), so intraday alt-data on the final day is not
                        # silently dropped; an explicit intraday cutoff time is honored verbatim.
                        if isinstance(self.end_date, datetime.datetime) and self.end_date.time() == datetime.time(0, 0):
                            where_clauses.append(f"{datetime_pred} < ?")
                            query_params.append(self.end_date + datetime.timedelta(days = 1))

                        else:
                            where_clauses.append(f"{datetime_pred} <= ?")
                            query_params.append(self.end_date)

            sql = f"SELECT * FROM {relation_sql}" + (f" WHERE {' AND '.join(where_clauses)}" if where_clauses else "")
            df = conn.execute(sql, query_params).fetch_df()

        finally:
            conn.close()

        if schema == "alter":

            if resolved_underlying and "Symbol" in df.columns:
                symbol_series = df["Symbol"].astype(str).str.upper()
                df = df.loc[symbol_series == resolved_underlying].copy()

            if rule.get("reset_index", True):
                df = df.reset_index(drop = True)

            self._ctx_log(
                f"[CTX LOAD] done key={key} schema=alter rows={len(df)} cols={len(df.columns)}"
            )

            return df

        if schema != "ohlc":

            raise ValueError(f"[CTX WARNING] unsupported data schema '{schema}' for key='{key}'")


        return df


    def get_data(self, key: str = "TARGET", UNDERLYING: Optional[str] = None) -> pd.DataFrame:
        """
        ### What It Does
        Returns context-managed market data.

        #### Responsibility
        Provides one stable entry point for retrieving raw or prepared data from the active context.

        #### How To Use
        Use it when you need a specific data family or one underlying slice from a portfolio context.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.
        - `UNDERLYING`
          - Asset selector. In single workflows this is one ticker; in portfolio workflows it should identify the explicit universe/list being loaded and must match available data columns/artifacts.
          - Expected shape/type: `Optional[str]`.

        #### Usage Example
        `result = get_data(...)`

        ---

        #### Optional Parameters
        - `key`: **str** = *"TARGET"*.
        - `UNDERLYING`: **Optional[str]** = *None*.

        ---

        ### Returns
        - `result`: **pd.DataFrame**.
        """

        def _attach_ctx_attrs(data: pd.DataFrame, *, source: str, underlyings: Optional[Sequence[str]] = None) -> MarketData:

            out = data.copy() if isinstance(data, MarketData) else MarketData(data.copy())
            attrs = dict(getattr(out, "attrs", {}) or {})
            raw_underlyings: Optional[List[Any]] = None

            if isinstance(self.universe, dict):
                raw_underlyings = self.universe.get("price_underlyings")

                if not isinstance(raw_underlyings, list):
                    raw_underlyings = self.universe.get("underlyings")

            normalized_underlyings: List[str] = []

            if underlyings is not None:
                normalized_underlyings = [str(x).strip().upper() for x in underlyings if str(x).strip()]

            elif isinstance(raw_underlyings, list):
                normalized_underlyings = [str(x).strip().upper() for x in raw_underlyings if str(x).strip()]

            if not normalized_underlyings and isinstance(self.UNDERLYING, str) and self.UNDERLYING.strip():
                normalized_underlyings = [self.UNDERLYING.strip().upper()]

            attrs["__zora_ctx_source__"] = str(source)
            attrs["__zora_ctx_underlyings__"] = normalized_underlyings
            attrs["__zora_ctx_notebook_dir__"] = str(self.notebook_dir)
            out.attrs = attrs


            return out


        key_upper = str(key).strip().upper()

        if not key_upper:
            key_upper = "TARGET"

        if key_upper == "FREQUENCY":
            key_upper = "TARGET"

        ctx_mode = self._ctx_mode()
        self._ctx_log(f"[CTX GET] ctx_mode={ctx_mode} key={key_upper} UNDERLYING = {UNDERLYING}")

        if ctx_mode == "portfolio":

            if key_upper == "ALTER":
                alter_map = getattr(self, "_alter_data", {})

                if UNDERLYING is not None:
                    alter_key = str(UNDERLYING).strip().upper()

                    if alter_key in alter_map:
                        data = alter_map[alter_key].copy()
                        self._ctx_log(f"[CTX GET] resolved alter_key={alter_key} rows={len(data)}")

                        return _attach_ctx_attrs(data, source = f"portfolio:ALTER:{alter_key}", underlyings = [alter_key])

                    raise KeyError(f"[CTX WARNING] portfolio alter data missing for key='{UNDERLYING}'")

                port_map = getattr(self, "_portfolio_data", {})

                if key_upper in port_map:
                    data = port_map[key_upper].copy()
                    self._ctx_log(f"[CTX GET] resolved portfolio key={key_upper} rows={len(data)}")

                    return _attach_ctx_attrs(data, source = f"portfolio:{key_upper}")

                raise KeyError(f"[CTX WARNING] portfolio data 'ALTER' not available")

            if UNDERLYING is not None:
                u_key = str(UNDERLYING).strip().upper()
                under_map = getattr(self, "_underlying_data", {})

                if u_key in under_map and key_upper in under_map[u_key]:
                    data = under_map[u_key][key_upper].copy()
                    self._ctx_log(f"[CTX GET] resolved UNDERLYING = {u_key} key={key_upper} rows={len(data)}")

                    return _attach_ctx_attrs(data, source = f"portfolio:{u_key}:{key_upper}", underlyings = [u_key])

                port_map = getattr(self, "_portfolio_data", {})

                if key_upper in port_map:

                    source_data = port_map[key_upper]
                    prefix = f"{u_key}_"
                    portfolio_symbols = {str(sym).strip().upper() for sym in self._portfolio_price_underlyings()}

                    def _resolve_column_owner(col):
                        owners = [sym for sym in portfolio_symbols if col == sym or col.startswith(f"{sym}_")]

                        return max(owners, key = len) if owners else None

                    selected_cols = [
                                    col for col in source_data.columns
                                    if isinstance(col, str) and col != "Datetime" and _resolve_column_owner(col) == u_key
                                    ]

                    if not selected_cols:

                        raise KeyError(f"[CTX WARNING] portfolio data missing for UNDERLYING = '{u_key}'")

                    data = source_data[["Datetime", *selected_cols]].copy()
                    data = data.rename(columns = {col: col[len(prefix):] for col in selected_cols})
                    data["Datetime"] = pd.to_datetime(data["Datetime"], errors = "coerce")
                    data = data.dropna(subset = ["Datetime"]).sort_values("Datetime").reset_index(drop = True)
                    under_map.setdefault(u_key, {})[key_upper] = data
                    self._ctx_log(f"[CTX GET] sliced UNDERLYING = {u_key} key={key_upper} rows={len(data)}")

                    return _attach_ctx_attrs(data, source = f"portfolio:{u_key}:{key_upper}", underlyings = [u_key])

                raise KeyError(f"[CTX WARNING] portfolio data missing for UNDERLYING = '{UNDERLYING}', key='{key_upper}'")

            port_map = getattr(self, "_portfolio_data", {})

            if key_upper in port_map:
                data = port_map[key_upper].copy()
                self._ctx_log(f"[CTX GET] resolved portfolio key={key_upper} rows={len(data)}")

                return _attach_ctx_attrs(data, source = f"portfolio:{key_upper}")

            raise KeyError(f"[CTX WARNING] portfolio data '{key_upper}' not available")

        single_map = getattr(self, "_single_data", {})

        if key_upper in single_map:
            data = single_map[key_upper].copy()
            self._ctx_log(f"[CTX GET] resolved single key={key_upper} rows={len(data)}")

            return _attach_ctx_attrs(data, source = f"single:{key_upper}")


        raise KeyError(f"[CTX WARNING] single data '{key_upper}' not available")


    def to_meta(self) -> Dict[str, Any]:
        """
        ### What It Does
        Serializes the current object into a plain metadata dictionary.

        #### Responsibility
        Captures the runtime configuration in a transport-friendly structure for logging, persistence, or payload export.

        #### How To Use
        Use it before saving context state or embedding metadata into downstream payloads.

        #### Usage Example
        `result = to_meta(...)`

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        meta = asdict(self)
        meta["data_paths"] = {k: str(pathlib.Path(v)) for k, v in self.data_paths.items()}
        meta["notebook_dir"] = str(self.notebook_dir)
        meta["payload_path"] = str(self.payload_path)


        return meta


    def build_payload(self,
                    *,
                    backtest_strategy: Optional[Callable[..., Any]] = None,
                    Data_Config: Dict[str, Any],
                    Factor_Config: Dict[str, Any],
                    factor_param_ranges: Dict[str, Any],
                    Execution_Config: Dict[str, Any],
                    Signal_Config: Optional[Dict[str, Any]] = None,
                    Traversal_Config: Optional[Dict[str, Any]] = None,
                    payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        ### What It Does
        Builds a runtime payload dictionary for the next execution layer.

        #### Responsibility
        Combines the current context metadata with runtime overrides such as strategy callables, factor ranges, and engine settings.

        #### How To Use
        Pass the active `test_data` and any workflow-specific overrides, then forward the result into the target engine.

        #### Key Parameters In Practice
        - `backtest_strategy`
          - For-loop strategy callable. Supply it only when `vbt` is false or when a custom Python entry/exit path is required; VBT signal/weight paths normally do not need it.
          - Expected shape/type: `Optional[Callable[..., Any]]`.
        - `Data_Config`
          - Top-level data-domain contract. Put dataset identity, prepared `test_data`, benchmark inputs, date bounds, and universe labels here; do not put signal rules or optimizer domains into it.
          - Expected shape/type: `Dict[str, Any]`.
        - `Factor_Config`
          - Factor-computation contract. Use it to pass `cal_column`, the active `FactorManager`/`FactorLibrary`, and factor-local constants that factor code actually consumes.
          - Expected shape/type: `Dict[str, Any]`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Dict[str, Any]`.
        - `Execution_Config`
          - Execution-domain contract. Use it for capital, backend selection, runtime worker count, logging paths, and rolling protocol shape.
          - Expected shape/type: `Dict[str, Any]`.
        - `Signal_Config`
          - Signal-construction contract. Use it for SR thresholds, signal gates, signal artifact paths, temporal cover/prevent-open rules, and portfolio weighting rules.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `Traversal_Config`
          - Traversal optimizer contract. Use it for the fixed-grid objective and traversal-specific selection settings.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `payload`
          - Runtime artifact container. Use it for precomputed signal/weight arrays, metadata paths, calendar/indexer/asset keys, and other objects that should stay outside config domains.
          - Expected shape/type: `Optional[Dict[str, Any]]`.

        #### Usage Example
        `result = build_payload(...)`

        ---

        ### Parameters
        - `Data_Config`: **Dict[str, Any]**.
        - `Factor_Config`: **Dict[str, Any]**.
        - `factor_param_ranges`: **Dict[str, Any]**.
        - `Execution_Config`: **Dict[str, Any]**.

        #### Optional Parameters
        - `backtest_strategy`: **Optional[Callable[..., Any]]** = *None*.
        - `Signal_Config`: **Optional[Dict[str, Any]]** = *None*.
        - `Traversal_Config`: **Optional[Dict[str, Any]]** = *None*.
        - `payload`: **Optional[Dict[str, Any]]** = *None*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        data_config = _deep_config_merge(DEFAULT_DATA_CONFIG, Data_Config)
        factor_config = _deep_config_merge(DEFAULT_FACTOR_CONFIG, Factor_Config)
        execution_config = _deep_config_merge(DEFAULT_EXECUTION_CONFIG, Execution_Config)
        signal_config = _deep_config_merge(DEFAULT_SIGNAL_CONFIG, Signal_Config)
        traversal_config = _deep_config_merge(DEFAULT_TRAVERSAL_CONFIG, Traversal_Config)
        runtime_payload = dict(payload or {})

        if not factor_param_ranges:

            raise ValueError("[CTX WARNING] factor_param_ranges missing for payload")

        if data_config.get("test_data") is None:

            raise ValueError("[CTX WARNING] Data_Config['test_data'] missing for payload")

        strategy_name = data_config.get("strategy_name")
        frequency = data_config.get("frequency")
        UNDERLYING = data_config.get("UNDERLYING")

        if strategy_name is None and self.strategy:
            strategy_name = self.strategy.get("name")

        if frequency is None and self.strategy:
            frequency = self.strategy.get("frequency")

        if isinstance(frequency, list):

            if len(frequency) == 1:
                frequency = frequency[0]

            else:

                raise ValueError("[CTX WARNING] frequency missing for payload")

        if UNDERLYING is None:
            UNDERLYING = self.UNDERLYING if self.UNDERLYING else ((self.strategy or {}).get("UNDERLYING") if isinstance(self.strategy, dict) else None)

        underlyings: Optional[List[str]] = None

        if isinstance(self.universe, dict):
            u_list = self.universe.get("underlyings")

            if isinstance(u_list, list):
                underlyings = [str(x) for x in u_list]

        if underlyings and UNDERLYING is None:
            UNDERLYING = underlyings[0]

        data_config["strategy_name"] = strategy_name
        data_config["frequency"] = frequency
        data_config["UNDERLYING"] = UNDERLYING
        data_config["underlyings"] = underlyings

        payload = {
                    'factor_param_ranges': factor_param_ranges,
                    'notebook_dir': self.notebook_dir,

                    'Data_Config': data_config,
                    'Factor_Config': factor_config,
                    'Execution_Config': execution_config,
                    'Signal_Config': signal_config,
                    'Traversal_Config': traversal_config,
                    'payload': runtime_payload,
                  }

        if backtest_strategy is not None:
            payload['backtest_strategy'] = backtest_strategy


        return payload
