from ENV_MGMT.imports import *
from .Calendars import default_holidays, default_macro_release_dates
import FactorEngine
from .Position import Position







# Signal Regularization Class
#----------------------------------------------------------------------------------------
class Signal:
    """
    ### What It Does
    Signal turns factor-rule output and precomputed signal artifacts into regularized signal-strength panels.

    #### Responsibility
    It owns feature tensor loading, rule scoring, signal confirmation/gating, cooldown handling, macro blocking, and disk-cache metadata validation. Portfolio-weight construction lives on the `Position` layer.

    #### How To Use
    Use `feature_tensor_loader(...)` or `disk_cache(...)` to load/build wide artifacts, call `compute_signal_strength(...)` for confirmed signals, then hand the signal-strength panels to `Position.compute_weight_panel(...)` to produce executable weights.

    #### Usage Example
    `obj = Signal(...)`

    ---

    ### Parameters
    #### Optional Parameters
    - `signal_z_window`: **int** = *120*.
    - `signal_start_t`: **int** = *2*.
    - `signal_cooldown`: **int** = *1*.
    - `signal_epsilon`: **float** = *1e-9*.
    - `signal_confirm_mode`: **str** = *"mean"*.
    - `signal_gate_mode`: **str** = *"mean"*.
    - `signal_gate_style`: **str** = *"soft"*.
    - `signal_gate_threshold`: **float** = *0.0*.
    - `Signal_Config`: **Optional[Dict[str, Any]]** = *None*.
    """

    @dataclass
    class ParamIndexer:
        """
        ### What It Does
        Maintains the state and behavior represented by `ParamIndexer`.

        #### Responsibility
        Groups the data, validation, and operations that share `ParamIndexer` lifecycle state.

        #### How To Use
        Instantiate `ParamIndexer` when the surrounding workflow needs that stateful component.

        #### Usage Example
        `obj = ParamIndexer(...)`
        """

        param_keys: List[str]
        param_values: List[List[Any]]
        asset_keys: Optional[List[str]] = None

        def __post_init__(self) -> None:

            self.lengths = [len(v) for v in self.param_values]
            self.multipliers = []

            total = 1

            for length in reversed(self.lengths):
                self.multipliers.append(total)
                total *= max(1, length)

            self.multipliers = list(reversed(self.multipliers))
            self.param_total = total
            self.value_to_index = []

            for values in self.param_values:
                self.value_to_index.append({v: i for i, v in enumerate(values)})

            cleaned_asset_keys: List[str] = []

            if self.asset_keys:

                for asset in self.asset_keys:
                    asset_text = str(asset).strip()

                    if asset_text:
                        cleaned_asset_keys.append(asset_text)

            self.asset_keys = list(dict.fromkeys(cleaned_asset_keys))
            self.asset_count = len(self.asset_keys) if self.asset_keys else 1
            self.asset_to_index = {asset: i for i, asset in enumerate(self.asset_keys)}
            self.total = int(self.param_total * self.asset_count)


        def index_to_combo(self, idx: int) -> List[Any]:
            """
            ### What It Does
            Maps a tensor column index back to its parameter combination.

            #### Responsibility
            Provides diagnostic and reporting access to the params behind a computed column.

            #### How To Use
            Call it when interpreting artifact columns or optimizer output.

            #### Key Parameters In Practice
            - `idx`
              - Index selector. It should refer to the intended parameter, asset, or row position in the current artifact context.
              - Expected shape/type: `int`.

            #### Usage Example
            `result = index_to_combo(...)`

            ---

            ### Parameters
            - `idx`: **int**.

            ---

            ### Returns
            - `result`: **List[Any]**.
            """

            if idx < 0 or idx >= self.total:

                raise IndexError("[WARNING] Parameter index out of range.")

            combo = []
            remaining = int(idx)

            if self.asset_keys and self.param_total > 0:
                remaining = remaining % self.param_total

            for i, values in enumerate(self.param_values):
                length = self.lengths[i]

                if length == 0:
                    combo.append(None)

                    continue

                base = self.multipliers[i]
                pos = (remaining // base) % length
                combo.append(values[pos])


            return combo


    @dataclass
    class FactorSignalContext:
        """
        ### What It Does
        Maintains the state and behavior represented by `FactorSignalContext`.

        #### Responsibility
        Groups the data, validation, and operations that share `FactorSignalContext` lifecycle state.

        #### How To Use
        Instantiate `FactorSignalContext` when the surrounding workflow needs that stateful component.

        #### Usage Example
        `obj = FactorSignalContext(...)`
        """

        calendar: pd.DatetimeIndex
        asset_keys: List[str]
        factor_manager: FactorEngine.FactorManager

        def __post_init__(self) -> None:

            self.calendar = pd.DatetimeIndex(pd.to_datetime(self.calendar, errors = "coerce"))
            self.asset_keys = [str(asset).strip() for asset in self.asset_keys if str(asset).strip()]

            if len(self.asset_keys) == 0:

                raise ValueError("[WARNING] FactorSignalContext requires non-empty asset_keys.")

            if not isinstance(self.factor_manager, FactorEngine.FactorManager):

                raise TypeError("[WARNING] FactorSignalContext requires a FactorEngine.FactorManager instance.")

            if list(self.factor_manager.asset_keys) != list(self.asset_keys):

                raise ValueError("[WARNING] FactorSignalContext asset_keys must match FactorEngine.FactorManager.asset_keys.")

            if len(self.factor_manager.calendar) != len(self.calendar) or not self.factor_manager.calendar.equals(self.calendar):

                raise ValueError("[WARNING] FactorSignalContext calendar must match FactorEngine.FactorManager.calendar.")


    @staticmethod
    def feature_tensor_loader(feature_name: str) -> Callable[[List[Any], "Signal.FactorSignalContext"], np.ndarray]:
        """
        ### What It Does
        Loads a feature tensor artifact and its metadata from disk.

        #### Responsibility
        Rehydrates precomputed factor features with calendar and asset labels intact.

        #### How To Use
        Call it when signal construction should reuse a saved factor cube.

        #### Key Parameters In Practice
        - `feature_name`
          - Human/semantic feature name. Use it when notebook code should refer to a registered factor by meaning rather than generated id.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = feature_tensor_loader(...)`

        ---

        ### Parameters
        - `feature_name`: **str**.

        ---

        ### Returns
        - `result`: **Callable[[List[Any], "Signal.FactorSignalContext"], np.ndarray]**.
        """

        feature_id = str(feature_name).strip()

        if not feature_id:

            raise ValueError("[WARNING] feature_tensor_loader requires a non-empty feature name.")

        def _loader(combo: List[Any], factor_ctx: "Signal.FactorSignalContext") -> np.ndarray:

            del combo

            return np.asarray(

                factor_ctx.factor_manager.get_cube(feature_id),
                dtype = np.float32,
            )


        return _loader


    @staticmethod
    @njit(cache = True)
    def _rolling_std_series_nb(arr: np.ndarray, window: int) -> np.ndarray:

        n = len(arr)
        out = np.empty(n, dtype = np.float64)

        for i in range(n):
            start = 0

            if i + 1 > window:
                start = i + 1 - window

            count = 0
            mean = 0.0
            m2 = 0.0

            for j in range(start, i + 1):
                x = arr[j]

                if np.isnan(x):

                    continue

                count += 1
                delta = x - mean
                mean += delta / count
                m2 += delta * (x - mean)

            if count == 0:
                out[i] = np.nan

            elif count == 1:
                out[i] = 0.0

            else:
                out[i] = np.sqrt(m2 / count)


        return out


    @staticmethod
    def _aggregate_tensor(score_tensor: np.ndarray, weights: np.ndarray, mode_code: int) -> np.ndarray:

        if score_tensor.ndim != 3:

            raise ValueError("[WARNING] score_tensor must be 3D (rule, time, asset).")

        n_rules, rows, asset_count = score_tensor.shape

        if n_rules == 0:

            return np.zeros((rows, asset_count), dtype = np.float64)

        out = np.zeros((rows, asset_count), dtype = np.float64)

        if mode_code == 0:
            positive = np.where(score_tensor > 0.0, score_tensor, 0.0)
            weight = weights.reshape(-1, 1, 1) * (positive > 0.0)
            denom = weight.sum(axis = 0)
            numer = (positive * weights.reshape(-1, 1, 1)).sum(axis = 0)

            with np.errstate(invalid = "ignore", divide = "ignore"):
                out = np.divide(numer, denom, out = np.zeros_like(numer), where = denom > 0.0)

            return out

        if mode_code == 1:
            positive_mask = np.all(score_tensor > 0.0, axis = 0)
            minv = np.min(np.where(np.isnan(score_tensor), 0.0, score_tensor), axis = 0)
            out = np.where(positive_mask, minv, 0.0)

            return out

        maxv = np.max(np.where(score_tensor > 0.0, score_tensor, 0.0), axis = 0)


        return np.where(maxv > 0.0, maxv, 0.0)


    @staticmethod
    def slice_panel(*,
                    test_data: pd.DataFrame,
                    values: np.ndarray | np.memmap,
                    indexer: "Signal.ParamIndexer",
                    factor_param_ranges: Dict[str, Any],
                    calendar: pd.DatetimeIndex,
                    asset_keys: List[str]) -> pd.DataFrame:
        """
        ### What It Does
        Slices a signal or weight panel by date bounds.

        #### Responsibility
        Keeps calendar filtering centralized for protocol windows and replay periods.

        #### How To Use
        Call it before passing panel artifacts into a windowed backtest.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `values`
          - Parameter domain or array values. For parameter lists these become iterable candidate values; for artifacts they are the numeric payload.
          - Expected shape/type: `np.ndarray | np.memmap`.
        - `indexer`
          - Parameter/asset index mapper for artifact tensors. It lets selected optimizer combos map back to the correct array row or column.
          - Expected shape/type: `"Signal.ParamIndexer"`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Dict[str, Any]`.
        - `calendar`
          - Canonical row-axis timestamps for an artifact or factor cube. It must match the arrays being sliced or loaded, especially for memmap artifacts.
          - Expected shape/type: `pd.DatetimeIndex`.
        - `asset_keys`
          - Canonical asset-axis labels. Preserve this order when moving between tensor, panel, and portfolio execution code.
          - Expected shape/type: `List[str]`.

        #### Usage Example
        `result = slice_panel(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `values`: **np.ndarray | np.memmap**.
        - `indexer`: **"Signal.ParamIndexer"**.
        - `factor_param_ranges`: **Dict[str, Any]**.
        - `calendar`: **pd.DatetimeIndex**.
        - `asset_keys`: **List[str]**.

        ---

        ### Returns
        - `result`: **pd.DataFrame**.
        """

        combo = [factor_param_ranges.get(str(key)) for key in indexer.param_keys]

        if any(value is None for value in combo):

            raise ValueError("[WARNING] factor_param_ranges missing required key(s) for artifact alignment.")

        combo_idx = 0

        for i, value in enumerate(combo):

            if value not in indexer.value_to_index[i]:

                raise KeyError(f"[WARNING] Value {value} not found in param_values[{i}]")

            combo_idx += indexer.value_to_index[i][value] * indexer.multipliers[i]

        combo_idx = int(combo_idx)
        values_arr = np.asarray(values)

        if values_arr.ndim != 3:

            raise ValueError("[WARNING] slice_panel requires a 3D artifact.")

        panel_arr = np.asarray(values_arr[combo_idx, :, :], dtype = float)
        dt_index = pd.DatetimeIndex(pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce"))

        if "__zora_row_id__" in test_data.columns:
            row_id = pd.to_numeric(test_data["__zora_row_id__"], errors = "coerce")

            if bool(row_id.isna().any()):

                raise ValueError("[WARNING] test_data contains invalid __zora_row_id__ values for artifact alignment.")

            row_idx = row_id.to_numpy(dtype = np.int64)

        else:
            calendar_index = pd.DatetimeIndex(pd.to_datetime(calendar, errors = "coerce"))

            if calendar_index.has_duplicates:

                raise ValueError("[WARNING] Artifact calendar contains duplicate timestamps; __zora_row_id__ is required for safe alignment.")

            lookup = {pd.Timestamp(ts): idx for idx, ts in enumerate(calendar_index)}
            row_idx = np.asarray([lookup.get(pd.Timestamp(ts), -1) for ts in dt_index], dtype = np.int64)

        valid = (row_idx >= 0) & (row_idx < panel_arr.shape[0])
        out = np.full((len(test_data), len(asset_keys)), np.nan, dtype = float)
        out[valid, :] = panel_arr[row_idx[valid], :]


        return pd.DataFrame(out, index = dt_index, columns = asset_keys, dtype = float)


    def __init__(self,
                 signal_z_window: int | Dict[str, Any] = 120,
                 signal_start_t: int = 2,
                 signal_cooldown: int = 1,
                 signal_epsilon: float = 1e-9,
                 signal_confirm_mode: str = "mean",
                 signal_gate_mode: str = "mean",
                 signal_gate_style: str = "soft",
                 signal_gate_threshold: float = 0.0,
                 Signal_Config: Optional[Dict[str, Any]] = None) -> None:

        if isinstance(signal_z_window, dict) and Signal_Config is None:
            Signal_Config = signal_z_window

        if isinstance(Signal_Config, dict):
            signal_z_window = Signal_Config.get("signal_z_window", 120)
            signal_start_t = Signal_Config.get("signal_start_t", signal_start_t)
            signal_cooldown = Signal_Config.get("signal_cooldown", signal_cooldown)
            signal_epsilon = Signal_Config.get("signal_epsilon", signal_epsilon)
            signal_confirm_mode = Signal_Config.get("signal_confirm_mode", signal_confirm_mode)
            signal_gate_mode = Signal_Config.get("signal_gate_mode", signal_gate_mode)
            signal_gate_style = Signal_Config.get("signal_gate_style", signal_gate_style)
            signal_gate_threshold = Signal_Config.get("signal_gate_threshold", signal_gate_threshold)

        self.signal_z_window = int(cast(Any, signal_z_window))
        self.signal_start_t = max(0, int(cast(Any, signal_start_t)))
        self.signal_cooldown = max(0, int(cast(Any, signal_cooldown)))
        self.signal_epsilon = float(cast(Any, signal_epsilon))
        self.signal_confirm_mode = signal_confirm_mode
        self.signal_gate_mode = signal_gate_mode
        self.signal_gate_style = signal_gate_style
        self.signal_gate_threshold = float(cast(Any, signal_gate_threshold))
        self._signal_data_context: Optional[pd.DataFrame] = None
        self._signal_param_context: Dict[str, Any] = {}
        self._signal_feature_refs: Dict[str, str] = {}
        self._signal_manifest: Optional[pd.DataFrame] = None
        self._signal_feature_cache: Dict[Tuple[str, str], str] = {}
        self._signal_param_signature: str = ""
        self._factor_signal_context: Optional["Signal.FactorSignalContext"] = None
        self._signal_combo_idx: int = 0


    def _set_context(self,
                    *,
                    test_data: pd.DataFrame,
                    factor_manager: Optional[FactorEngine.FactorManager] = None,
                    factor_param_ranges: Optional[Dict[str, Any]] = None,
                    feature_ref_map: Optional[Dict[str, str]] = None) -> "Signal.FactorSignalContext":

        self._signal_data_context = test_data
        manifest_rows: List[Dict[str, Any]] = []
        manifest_attrs = dict(getattr(test_data, "attrs", {}) or {})
        manifest_registry = manifest_attrs.get(FactorEngine.AutoParam.REGISTRY_KEY, {})

        if isinstance(manifest_registry, dict):

            for column_name, meta_raw in dict(manifest_registry.get("columns", {}) or {}).items():
                meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}

                manifest_rows.append(
                                    {
                                     "column_name": str(column_name),
                                     "feature_id": str(meta.get("feature_id", column_name)),
                                     "semantic_id": str(meta.get("semantic_id", meta.get("feature_id", column_name))),
                                     "factor": str(meta.get("factor", "")).upper(),
                                     "target_freq": meta.get("target_freq"),
                                     "cal_column": meta.get("cal_column"),
                                     "output_idx": int(meta.get("output_idx", 0)),
                                     "timing_semantics": str(meta.get("timing_semantics", "")).strip().lower(),
                                     "prev_idx": bool(meta.get("prev_idx", False)),
                                     "params": FactorEngine.AutoParam.clean_value(meta.get("params", {})),
                                    }
                                   )

        manifest_known = {str(row.get("column_name", "")) for row in manifest_rows}

        for column_name in test_data.columns:
            column_text = str(column_name)

            if column_text in manifest_known:

                continue

            manifest_rows.append(
                                {
                                 "column_name": column_text,
                                 "feature_id": column_text,
                                 "semantic_id": column_text,
                                 "factor": "",
                                 "target_freq": None,
                                 "cal_column": None,
                                 "output_idx": 0,
                                 "timing_semantics": "",
                                 "prev_idx": False,
                                 "params": {},
                                }
                               )

        self._signal_manifest = pd.DataFrame(manifest_rows)
        default_refs: Dict[str, str] = {}
        attrs = getattr(test_data, "attrs", None)

        if isinstance(attrs, dict):
            raw_refs = attrs.get(FactorEngine.AutoParam.FEATURE_MAP_KEY, {})

            if isinstance(raw_refs, dict):
                for k, v in raw_refs.items():
                    key = str(k).strip()
                    value = str(v).strip()

                    if key and value:
                        default_refs[key] = value

        merged_refs = dict(default_refs)

        if isinstance(feature_ref_map, dict):
            for k, v in feature_ref_map.items():
                key = str(k).strip()
                value = str(v).strip()

                if key and value:
                    merged_refs[key] = value

        self._signal_feature_refs = merged_refs
        self._signal_feature_cache.clear()

        runtime_param_map: Dict[str, Any] = {}

        if factor_param_ranges is not None:

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

                if len(cleaned) == 0:

                    raise ValueError("[WARNING] runtime factor_param_ranges cannot contain empty parameter ranges.")

                if len(cleaned) != 1:

                    raise ValueError("[WARNING] runtime factor_param_ranges requires exactly one value per key.")

                runtime_param_map[str(key)] = cleaned[0]

        self._signal_param_context = {str(k): FactorEngine.AutoParam.clean_value(v) for k, v in runtime_param_map.items()}
        self._signal_param_signature = json.dumps(self._signal_param_context, sort_keys = True, default = str)
        store = factor_manager if isinstance(factor_manager, FactorEngine.FactorManager) else FactorEngine.FactorManager.get_attached_store(test_data)

        if store is None:

            raise ValueError("[WARNING] factor runtime context requires a FactorEngine.FactorManager.")

        calendar = pd.DatetimeIndex(pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce"))

        if bool(pd.isna(calendar).any()):

            raise ValueError("[WARNING] factor runtime context test_data contains invalid Datetime values.")

        if len(calendar) != len(store.calendar) or not store.calendar.equals(calendar):

            raise ValueError("[WARNING] FactorEngine.FactorManager calendar must match signal test_data calendar.")

        if "Datetime" in test_data.columns:
            candidate_columns = [col for col in test_data.columns if col != "Datetime"]

        else:
            candidate_columns = list(test_data.columns)

        if not store.asset_keys and candidate_columns:
            store.asset_keys = [str(col) for col in candidate_columns]

        self._factor_signal_context = Signal.FactorSignalContext(calendar = calendar, asset_keys = list(store.asset_keys), factor_manager = store)


        return cast("Signal.FactorSignalContext", self._factor_signal_context)


    def _coerce_tensor_input(self, value: Any, rows: int, asset_count: int) -> Optional[np.ndarray]:

        if value is None:

            return None

        if isinstance(value, FactorEngine.AutoParam.ParamValue):
            key = value.key

            if key in self._signal_param_context:
                value = self._signal_param_context[key]

            else:
                value = value.unwrap()

        if isinstance(value, str):
            ref_text = value.strip()

            if not ref_text:

                return None

            if ref_text in self._signal_param_context:
                value = self._signal_param_context[ref_text]

            else:
                factor_ctx = self._factor_signal_context

                if factor_ctx is None:

                    raise ValueError("[WARNING] Tensor runtime context is not set; cannot resolve signal reference.")

                cube = np.asarray(factor_ctx.factor_manager.get_cube(ref_text), dtype = np.float64)

                if cube.ndim != 3:

                    raise ValueError(f"[WARNING] FactorEngine.FactorManager cube must be 3D, got {cube.ndim}D for {ref_text}.")

                combo_idx = int(getattr(self, "_signal_combo_idx", 0))

                if cube.shape[0] == 1:
                    combo_idx = 0

                if combo_idx < 0 or combo_idx >= cube.shape[0]:

                    raise IndexError(f"[WARNING] cube combo index out of range for {ref_text}: {combo_idx}")

                return np.ascontiguousarray(cube[combo_idx])

        if np.isscalar(value):

            return np.full((rows, asset_count), float(cast(Any, value)), dtype = np.float64)

        if isinstance(value, pd.Series):
            series = pd.Series(pd.to_numeric(value, errors = "coerce"), index = value.index, dtype = float)
            factor_ctx = self._factor_signal_context

            if factor_ctx is None:

                raise ValueError("[WARNING] Tensor runtime context is not set.")

            series.index = pd.DatetimeIndex(pd.to_datetime(series.index, errors = "coerce"))
            arr = series.reindex(factor_ctx.calendar).to_numpy(dtype = np.float64, copy = True).reshape(-1, 1)

            return np.repeat(arr, asset_count, axis = 1)

        if isinstance(value, pd.DataFrame):
            factor_ctx = self._factor_signal_context

            if factor_ctx is None:

                raise ValueError("[WARNING] Tensor runtime context is not set.")

            frame = value.copy()
            frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index, errors = "coerce"))
            frame = frame.reindex(index = factor_ctx.calendar)

            if frame.shape[1] == 1:
                arr = frame.to_numpy(dtype = np.float64, copy = True)

                return np.repeat(arr, asset_count, axis = 1)

            frame = frame.reindex(columns = factor_ctx.asset_keys)

            return frame.to_numpy(dtype = np.float64, copy = True)

        arr = np.asarray(value, dtype = np.float64)

        if arr.ndim == 1:

            if arr.shape[0] != rows:

                raise ValueError("[WARNING] Tensor series length mismatch.")

            return np.repeat(arr.reshape(-1, 1), asset_count, axis = 1)

        if arr.ndim == 3:
            combo_idx = int(getattr(self, "_signal_combo_idx", 0))

            if arr.shape[0] == 1:
                combo_idx = 0

            if combo_idx < 0 or combo_idx >= arr.shape[0]:

                raise IndexError(f"[WARNING] tensor combo index out of range: {combo_idx}")

            arr = arr[combo_idx]

        if arr.ndim != 2:

            raise ValueError(f"[WARNING] Tensor input must be 1D/2D, got {arr.ndim}D.")

        if arr.shape == (rows, asset_count):

            return np.ascontiguousarray(arr)

        if arr.shape == (rows, 1):

            return np.repeat(arr, asset_count, axis = 1)


        raise ValueError(f"[WARNING] Tensor shape mismatch: expected {(rows, asset_count)} or {(rows, 1)}, got {arr.shape}.")


    def _rolling_std_tensor(self, arr: Any) -> Optional[np.ndarray]:

        tensor_arr = self._coerce_tensor_input(arr, len(self._factor_signal_context.calendar), len(self._factor_signal_context.asset_keys)) if self._factor_signal_context is not None else None

        if tensor_arr is None:

            return None

        out = np.full(tensor_arr.shape, np.nan, dtype = np.float64)

        for asset_idx in range(tensor_arr.shape[1]):
            out[:, asset_idx] = self._rolling_std_series_nb(tensor_arr[:, asset_idx].astype(np.float64), int(self.signal_z_window))


        return out


    def single_net_tensor(self,
                        buy_signals: List[Tuple],
                        sell_signals: List[Tuple],
                        buy_gate_signals: Optional[List[Tuple]] = None,
                        sell_gate_signals: Optional[List[Tuple]] = None) -> np.ndarray:
        """
        ### What It Does
        Builds one net-signal tensor for a single parameter combination.

        #### Responsibility
        Applies buy/sell rules across the aligned feature context and returns the resulting signal grid.

        #### How To Use
        Call it when constructing artifacts for one combo outside the full grid path.

        #### Key Parameters In Practice
        - `buy_signals`
          - Long-entry signal series/panel. It should already reflect factor rules and confirmation logic before execution.
          - Expected shape/type: `List[Tuple]`.
        - `sell_signals`
          - Short-entry or exit signal series/panel. Its semantics must match the chosen strategy direction convention.
          - Expected shape/type: `List[Tuple]`.
        - `buy_gate_signals`
          - Confirmation gate for long entries. Use it to require additional evidence before opening long exposure.
          - Expected shape/type: `Optional[List[Tuple]]`.
        - `sell_gate_signals`
          - Confirmation gate for short entries. Use it to require additional evidence before opening short exposure.
          - Expected shape/type: `Optional[List[Tuple]]`.

        #### Usage Example
        `result = single_net_tensor(...)`

        ---

        ### Parameters
        - `buy_signals`: **List[Tuple]**.
        - `sell_signals`: **List[Tuple]**.

        #### Optional Parameters
        - `buy_gate_signals`: **Optional[List[Tuple]]** = *None*.
        - `sell_gate_signals`: **Optional[List[Tuple]]** = *None*.

        ---

        ### Returns
        - `result`: **np.ndarray**.
        """

        factor_ctx = self._factor_signal_context

        if factor_ctx is None:

            raise ValueError("[WARNING] single_net_tensor requires factor runtime context.")

        rows = len(factor_ctx.calendar)
        asset_count = len(factor_ctx.asset_keys)
        confirm_mode = (self.signal_confirm_mode or "mean").strip().lower()
        gate_mode = (self.signal_gate_mode or "mean").strip().lower()
        gate_style = (self.signal_gate_style or "soft").strip().lower()
        confirm_mode_code = 0 if confirm_mode == "mean" else (1 if confirm_mode == "min" else 2)
        gate_mode_code = 0 if gate_mode == "mean" else (1 if gate_mode == "min" else 2)
        gate_style_code = 1 if gate_style == "hard" else 0

        def _pack_tensor_signals(signals: Optional[List[Tuple]]) -> Tuple[np.ndarray, np.ndarray]:

            if not signals:

                return np.zeros((0, rows, asset_count), dtype = np.float64), np.zeros(0, dtype = np.float64)

            score_list: List[np.ndarray] = []
            weight_list: List[float] = []

            for signal in signals:

                if signal is None or len(signal) < 4:

                    continue

                mode = str(signal[0]).strip().lower()

                if mode == "gt":
                    _, x, threshold, arr_ref = signal[:4]
                    weight = float(signal[4] if len(signal) >= 5 and signal[4] is not None else 1.0)
                    x_arr = self._coerce_tensor_input(x, rows, asset_count)
                    thr_arr = self._coerce_tensor_input(threshold, rows, asset_count)
                    std_arr = self._rolling_std_tensor(x if arr_ref is None else arr_ref)

                    if x_arr is None or thr_arr is None or std_arr is None:

                        continue

                    with np.errstate(invalid = "ignore", divide = "ignore"):

                        score = np.divide(
                                            x_arr - thr_arr,
                                            std_arr + float(self.signal_epsilon),
                                            out = np.zeros_like(x_arr, dtype = np.float64),
                                            where = np.isfinite(std_arr) & (std_arr > 0.0),
                                         )

                    score[~np.isfinite(score)] = 0.0
                    score_list.append(score)
                    weight_list.append(weight)

                    continue

                if mode == "lt":

                    _, x, threshold, arr_ref = signal[:4]
                    weight = float(signal[4] if len(signal) >= 5 and signal[4] is not None else 1.0)
                    x_arr = self._coerce_tensor_input(x, rows, asset_count)
                    thr_arr = self._coerce_tensor_input(threshold, rows, asset_count)
                    std_arr = self._rolling_std_tensor(x if arr_ref is None else arr_ref)

                    if x_arr is None or thr_arr is None or std_arr is None:

                        continue

                    with np.errstate(invalid = "ignore", divide = "ignore"):

                        score = np.divide(
                                            thr_arr - x_arr,
                                            std_arr + float(self.signal_epsilon),
                                            out = np.zeros_like(x_arr, dtype = np.float64),
                                            where = np.isfinite(std_arr) & (std_arr > 0.0),
                                         )

                    score[~np.isfinite(score)] = 0.0
                    score_list.append(score)
                    weight_list.append(weight)

                    continue

                if mode == "band":

                    _, x, low, high, arr_ref = signal[:5]
                    weight = float(signal[5] if len(signal) >= 6 and signal[5] is not None else 1.0)
                    x_arr = self._coerce_tensor_input(x, rows, asset_count)
                    low_arr = self._coerce_tensor_input(low, rows, asset_count)
                    high_arr = self._coerce_tensor_input(high, rows, asset_count)
                    std_arr = self._rolling_std_tensor(x if arr_ref is None else arr_ref)

                    if x_arr is None or low_arr is None or high_arr is None or std_arr is None:

                        continue

                    lo = np.minimum(low_arr, high_arr)
                    hi = np.maximum(low_arr, high_arr)
                    margin = np.minimum(x_arr - lo, hi - x_arr)

                    with np.errstate(invalid = "ignore", divide = "ignore"):

                        score = np.divide(
                                            margin,
                                            std_arr + float(self.signal_epsilon),
                                            out = np.zeros_like(x_arr, dtype = np.float64),
                                            where = np.isfinite(std_arr) & (std_arr > 0.0),
                                         )

                    score[~np.isfinite(score)] = 0.0
                    score_list.append(score)
                    weight_list.append(weight)

                    continue

                raise ValueError(f"[WARNING] Unsupported signal mode: {mode}")

            if len(score_list) == 0:

                return np.zeros((0, rows, asset_count), dtype = np.float64), np.zeros(0, dtype = np.float64)

            return np.stack(score_list, axis = 0), np.asarray(weight_list, dtype = np.float64)


        confirm_long_scores, confirm_long_weights = _pack_tensor_signals(buy_signals)
        confirm_short_scores, confirm_short_weights = _pack_tensor_signals(sell_signals)
        gate_long_scores, gate_long_weights = _pack_tensor_signals(buy_gate_signals)
        gate_short_scores, gate_short_weights = _pack_tensor_signals(sell_gate_signals)

        confirm_long = self._aggregate_tensor(confirm_long_scores, confirm_long_weights, confirm_mode_code)
        confirm_short = self._aggregate_tensor(confirm_short_scores, confirm_short_weights, confirm_mode_code)
        gate_long = np.ones((rows, asset_count), dtype = np.float64) if gate_long_scores.shape[0] == 0 else self._aggregate_tensor(gate_long_scores, gate_long_weights, gate_mode_code)
        gate_short = np.ones((rows, asset_count), dtype = np.float64) if gate_short_scores.shape[0] == 0 else self._aggregate_tensor(gate_short_scores, gate_short_weights, gate_mode_code)

        if gate_style_code == 1:
            long_strength = np.where(gate_long > float(self.signal_gate_threshold), confirm_long, 0.0)
            short_strength = np.where(gate_short > float(self.signal_gate_threshold), confirm_short, 0.0)
            net = long_strength - short_strength

        else:
            net = (confirm_long * gate_long) - (confirm_short * gate_short)
        start_t = max(0, int(self.signal_start_t))

        if start_t > 0 and net.shape[0] > 0:
            net[:min(start_t, net.shape[0]), :] = 0.0


        return net.astype(np.float32, copy = False)


    def signal_pass_on(self, factor_score: Any) -> Any:
        """
        ### What It Does
        Passes factor scores through as signal strength without thresholding or gating.

        #### Responsibility
        Makes the identity SR rule explicit for cross-sectional factors whose scores already define signal strength.

        #### How To Use
        Pass a factor-score cube artifact and receive the same values as signal strength.

        #### Key Parameters In Practice
        - `factor_score`
          - Numerical factor panel or series being transformed into signals. It should already be aligned to the intended calendar and asset axis.
          - Expected shape/type: `Any`.

        #### Usage Example
        `base_signal_strength = signal.signal_pass_on(base_factor_artifact)`

        ---

        ### Parameters
        - `factor_score`: **Any**.

        ---

        ### Returns
        - `result`: **Any**.
        """

        return factor_score


    def signal_greater_than(self,
                        x: Any,
                        threshold: Any,
                        arr: Optional[Any] = None,

                        weight: Optional[float] = 1.0) -> Tuple:
        """
        ### What It Does
        Builds rule tuple for condition x > threshold (mode gt).

        #### Responsibility
        Encodes upper-threshold logic as reusable rule metadata.

        #### How To Use
        Returns ('gt', x, threshold, arr_ref) or ('gt', x, threshold, arr_ref, weight).

        #### Key Parameters In Practice
        - `x`
          - Factor value, column reference, or expression being compared against the threshold.
          - Expected shape/type: `Any`.
        - `threshold`
          - Activation cutoff. Use the factor scale after preprocessing; string values can be resolved from the current optimizer combo when intended.
          - Expected shape/type: `Any`.
        - `arr`
          - Optional array/reference payload carried with the rule tuple for later signal resolution.
          - Expected shape/type: `Optional[Any]`.
        - `weight`
          - Optional rule multiplier used when this condition is blended with other SR rules.
          - Expected shape/type: `Optional[float]`.

        #### Usage Example
        `result = greater_than(...)`

        ---

        ### Parameters
        - `x`: **Any**.
        - `threshold`: **Any**.

        #### Optional Parameters
        - `arr`: **Optional[Any]** = *None*.
        - `weight`: **Optional[float]** = *1.0*.

        ---

        ### Returns
        - `result`: **Tuple**.
        """

        arr_ref = x if arr is None else arr

        if weight is None:

            return ("gt", x, threshold, arr_ref)


        return ("gt", x, threshold, arr_ref, weight)


    def signal_less_than(self,
                     x: Any,
                     threshold: Any,
                     arr: Optional[Any] = None,

                     weight: Optional[float] = 1.0) -> Tuple:
        """
        ### What It Does
        Builds rule tuple for condition x < threshold (mode lt).

        #### Responsibility
        Encodes lower-threshold logic as reusable rule metadata.

        #### How To Use
        Returns ('lt', x, threshold, arr_ref) or ('lt', x, threshold, arr_ref, weight).

        #### Key Parameters In Practice
        - `x`
          - Factor value, column reference, or expression being compared against the threshold.
          - Expected shape/type: `Any`.
        - `threshold`
          - Activation cutoff. Use the factor scale after preprocessing; string values can be resolved from the current optimizer combo when intended.
          - Expected shape/type: `Any`.
        - `arr`
          - Optional array/reference payload carried with the rule tuple for later signal resolution.
          - Expected shape/type: `Optional[Any]`.
        - `weight`
          - Optional rule multiplier used when this condition is blended with other SR rules.
          - Expected shape/type: `Optional[float]`.

        #### Usage Example
        `result = less_than(...)`

        ---

        ### Parameters
        - `x`: **Any**.
        - `threshold`: **Any**.

        #### Optional Parameters
        - `arr`: **Optional[Any]** = *None*.
        - `weight`: **Optional[float]** = *1.0*.

        ---

        ### Returns
        - `result`: **Tuple**.
        """

        arr_ref = x if arr is None else arr

        if weight is None:

            return ("lt", x, threshold, arr_ref)


        return ("lt", x, threshold, arr_ref, weight)


    def signal_band(self,
                x: Any,
                low: Any,
                high: Any,
                arr: Optional[Any] = None,

                weight: Optional[float] = 1.0) -> Tuple:
        """
        ### What It Does
        Builds rule tuple for condition low <= x <= high (mode band).

        #### Responsibility
        Encodes bounded-range logic as reusable rule metadata.

        #### How To Use
        Returns ('band', x, low, high, arr_ref) or ('band', x, low, high, arr_ref, weight).

        #### Key Parameters In Practice
        - `x`
          - Factor value, column reference, or expression being tested by the band rule. This is the signal input that must fall between `low` and `high`.
          - Expected shape/type: `Any`.
        - `low`
          - Lower activation bound for the band rule. Set it in the same scale as `x` after any factor normalization.
          - Expected shape/type: `Any`.
        - `high`
          - Upper activation bound for the band rule. It should be greater than or equal to `low` and use the same factor scale.
          - Expected shape/type: `Any`.
        - `arr`
          - Optional array/reference payload carried with the rule tuple. Use it when the rule needs to remember the source array or semantic reference for later resolution.
          - Expected shape/type: `Optional[Any]`.
        - `weight`
          - Optional rule multiplier. Use it to make this condition contribute more or less when multiple SR rules are combined.
          - Expected shape/type: `Optional[float]`.

        #### Usage Example
        `result = band(...)`

        ---

        ### Parameters
        - `x`: **Any**.
        - `low`: **Any**.
        - `high`: **Any**.

        #### Optional Parameters
        - `arr`: **Optional[Any]** = *None*.
        - `weight`: **Optional[float]** = *1.0*.

        ---

        ### Returns
        - `result`: **Tuple**.
        """

        arr_ref = x if arr is None else arr

        if weight is None:

            return ("band", x, low, high, arr_ref)


        return ("band", x, low, high, arr_ref, weight)


    @staticmethod
    def _close_memmap(obj: Any) -> None:
        """
        Release the OS handle behind a memmap. On Windows a live mapping locks the backing file, so an
        unclosed artifact handle blocks recompute/overwrite of the same path and temp-dir cleanup
        (WinError 32). Accepts a memmap, a plain ndarray (no-op), or a dict of them; best-effort:
        flush only if writable, close the underlying mmap, and swallow teardown errors.
        """
        if isinstance(obj, dict):

            for value in obj.values():
                Signal._close_memmap(value)

            return

        if not isinstance(obj, np.memmap):

            return

        try:

            if str(getattr(obj, "mode", "r")) != "r":
                obj.flush()

        except Exception:
            pass

        try:
            underlying = getattr(obj, "_mmap", None)

            if underlying is not None:
                underlying.close()

        except Exception:
            pass


    def save_metadata(self,
                    meta_path: str | pathlib.Path,
                    indexer: "Signal.ParamIndexer",
                    artifact_path: str | pathlib.Path,
                    shape: Tuple[int, int, int],
                    dtype: str,
                    asset_keys: List[str],
                    calendar: Optional[pd.Index] = None) -> None:
        """
        ### What It Does
        Writes artifact metadata next to a saved signal artifact.

        #### Responsibility
        Persists calendar, asset, parameter, and schema information required for later validation.

        #### How To Use
        Call it after writing signal or weight arrays to disk.

        #### Key Parameters In Practice
        - `meta_path`
          - Path to artifact metadata JSON. Use it as the contract source for shape, dtype, param axis, calendar, asset keys, and artifact file location.
          - Expected shape/type: `str | pathlib.Path`.
        - `indexer`
          - Parameter/asset index mapper for artifact tensors. It lets selected optimizer combos map back to the correct array row or column.
          - Expected shape/type: `"Signal.ParamIndexer"`.
        - `artifact_path`
          - Path to the numeric artifact array. It must point to the actual `.npy`/memmap file referenced by metadata.
          - Expected shape/type: `str | pathlib.Path`.
        - `shape`
          - Expected artifact/tensor shape. Use it to validate array compatibility before attaching or slicing values.
          - Expected shape/type: `Tuple[int, int, int]`.
        - `dtype`
          - Numerical dtype for stored arrays. Choose it to balance precision and memory; artifact metadata must match the actual array dtype.
          - Expected shape/type: `str`.
        - `asset_keys`
          - Canonical asset-axis labels. Preserve this order when moving between tensor, panel, and portfolio execution code.
          - Expected shape/type: `List[str]`.
        - `calendar`
          - Canonical row-axis timestamps for an artifact or factor cube. It must match the arrays being sliced or loaded, especially for memmap artifacts.
          - Expected shape/type: `Optional[pd.Index]`.

        #### Usage Example
        `result = save_metadata(...)`

        ---

        ### Parameters
        - `meta_path`: **str | pathlib.Path**.
        - `indexer`: **"Signal.ParamIndexer"**.
        - `artifact_path`: **str | pathlib.Path**.
        - `shape`: **Tuple[int, int, int]**.
        - `dtype`: **str**.
        - `asset_keys`: **List[str]**.

        #### Optional Parameters
        - `calendar`: **Optional[pd.Index]** = *None*.
        """

        param_values = []

        for values in indexer.param_values:
            cleaned = []

            for v in values:
                cleaned.append(cast(Any, v).item() if isinstance(v, np.generic) else v)

            param_values.append(cleaned)

        meta = {
                "param_keys": indexer.param_keys,
                "param_values": param_values,
                "asset_keys": [str(asset) for asset in asset_keys],
                "shape": [int(shape[0]), int(shape[1]), int(shape[2])],
                "dtype": str(dtype),
                "artifact_path": str(artifact_path),
                }

        if calendar is None:

            raise ValueError("[WARNING] Artifact metadata requires row-axis calendar.")

        calendar_index = pd.DatetimeIndex(pd.to_datetime(calendar, errors = "coerce"))

        if len(calendar_index) != int(shape[1]):

            raise ValueError("[WARNING] Artifact metadata calendar length must match row axis.")

        if bool(pd.isna(calendar_index).any()):

            raise ValueError("[WARNING] Cube metadata calendar contains invalid timestamps.")

        # Persist tz-naive: a reloaded tz-aware calendar would zero-match the tz-naive test_data axis. The
        # naive-wall-clock contract holds on disk too, not just in memory. No-op when already naive.
        if calendar_index.tz is not None:
            calendar_index = calendar_index.tz_localize(None)

        meta["calendar"] = [ts.isoformat() for ts in calendar_index]

        meta_path = pathlib.Path(meta_path)
        meta_path.parent.mkdir(parents = True, exist_ok = True)

        with meta_path.open("w", encoding = "utf-8") as f:
            json.dump(meta, f, indent = 2)


    def compute_signal_strength(self,
                                *,
                                mode: str = "batch",
                                test_data: pd.DataFrame,
                                factor_param_ranges: Dict[str, Any],
                                factor_manager: Optional[FactorEngine.FactorManager] = None,
                                buy_signals: Optional[List[Tuple]] = None,
                                sell_signals: Optional[List[Tuple]] = None,
                                buy_gate_signals: Optional[List[Tuple]] = None,
                                sell_gate_signals: Optional[List[Tuple]] = None,
                                signal_strength_func: Optional[Callable[[List[Any], "Signal.FactorSignalContext"], Any]] = None,
                                signal_strength_path: Optional[str | pathlib.Path] = None,
                                meta_path: Optional[str | pathlib.Path] = None,
                                dtype: str = "float32") -> Tuple[np.ndarray | np.memmap, "Signal.ParamIndexer", "Signal.FactorSignalContext"]:
        """
        ### What It Does
        Converts net signal panels into regularized signal-strength panels.

        #### Responsibility
        Applies confirmation, gating, z-score, cooldown, and macro-blocking rules before portfolio weighting.

        #### How To Use
        Call it after raw buy/sell net signals are available.

        #### Key Parameters In Practice
        - `mode`
          - Rule or execution mode selector. Set it to the branch whose semantics you actually want; do not rely on default mode when the function supports multiple interpretations.
          - Expected shape/type: `str`.
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Dict[str, Any]`.
        - `factor_manager`
          - Active factor registry/cache. Pass the `FactorManager` or `FactorLibrary` that owns registered factor cubes so SR can resolve semantic factor names into actual panels.
          - Expected shape/type: `Optional[FactorEngine.FactorManager]`.
        - `buy_signals`
          - Long-entry signal series/panel. It should already reflect factor rules and confirmation logic before execution.
          - Expected shape/type: `Optional[List[Tuple]]`.
        - `sell_signals`
          - Short-entry or exit signal series/panel. Its semantics must match the chosen strategy direction convention.
          - Expected shape/type: `Optional[List[Tuple]]`.
        - `buy_gate_signals`
          - Confirmation gate for long entries. Use it to require additional evidence before opening long exposure.
          - Expected shape/type: `Optional[List[Tuple]]`.
        - `sell_gate_signals`
          - Confirmation gate for short entries. Use it to require additional evidence before opening short exposure.
          - Expected shape/type: `Optional[List[Tuple]]`.
        - `signal_strength_func`
          - Callable that creates signal strength from factor data. It must return values aligned to the active calendar/asset shape.
          - Expected shape/type: `Optional[Callable[[List[Any], "Signal.FactorSignalContext"], Any]]`.
        - `signal_strength_path`
          - Disk path for a saved signal-strength artifact. Use it only when loading/saving artifact arrays outside memory.
          - Expected shape/type: `Optional[str | pathlib.Path]`.
        - `meta_path`
          - Path to artifact metadata JSON. Use it as the contract source for shape, dtype, param axis, calendar, asset keys, and artifact file location.
          - Expected shape/type: `Optional[str | pathlib.Path]`.
        - `dtype`
          - Numerical dtype for stored arrays. Choose it to balance precision and memory; artifact metadata must match the actual array dtype.
          - Expected shape/type: `str`.

        #### Usage Example
        `result = compute_signal_strength(...)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `factor_param_ranges`: **Dict[str, Any]**.

        #### Optional Parameters
        - `mode`: **str** = *"batch"*.
        - `factor_manager`: **Optional[FactorEngine.FactorManager]** = *None*.
        - `buy_signals`: **Optional[List[Tuple]]** = *None*.
        - `sell_signals`: **Optional[List[Tuple]]** = *None*.
        - `buy_gate_signals`: **Optional[List[Tuple]]** = *None*.
        - `sell_gate_signals`: **Optional[List[Tuple]]** = *None*.
        - `signal_strength_func`: **Optional[Callable[[List[Any], "Signal.FactorSignalContext"], Any]]** = *None*.
        - `signal_strength_path`: **Optional[str | pathlib.Path]** = *None*.
        - `meta_path`: **Optional[str | pathlib.Path]** = *None*.
        - `dtype`: **str** = *"float32"*.

        ---

        ### Returns
        - `result`: **Tuple[np.ndarray | np.memmap, "Signal.ParamIndexer", "Signal.FactorSignalContext"]**.
        """

        if test_data is None or not isinstance(test_data, pd.DataFrame) or test_data.empty:

            raise ValueError("[WARNING] compute_signal_strength requires non-empty test_data.")

        normalized_factor_param_ranges: Dict[str, List[Any]] = {}

        for key, raw_value in (factor_param_ranges or {}).items():
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

            if len(cleaned) == 0:

                raise ValueError("[WARNING] factor_param_ranges cannot contain empty parameter ranges.")

            normalized_factor_param_ranges[str(key)] = cleaned

        param_keys = list(normalized_factor_param_ranges.keys())
        param_values = [normalized_factor_param_ranges[key] for key in param_keys]
        indexer = Signal.ParamIndexer(param_keys = param_keys, param_values = param_values, asset_keys = None)

        context = self._set_context(
                                    test_data = test_data,
                                    factor_manager = factor_manager,
                                    factor_param_ranges = {key: values[0] for key, values in normalized_factor_param_ranges.items()} if mode == "single" else None,
                                    )

        rows = len(context.calendar)
        asset_count = len(context.asset_keys)
        shape = (int(indexer.total), int(rows), int(asset_count))
        dtype_obj = np.dtype(dtype)

        if signal_strength_path is not None:
            resolved_signal_strength_path = pathlib.Path(signal_strength_path)
            resolved_signal_strength_path.parent.mkdir(parents = True, exist_ok = True)
            signal_strength = np.memmap(resolved_signal_strength_path, dtype = dtype_obj, mode = "w+", shape = shape)

            if meta_path is not None:

                self.save_metadata(
                                    meta_path = meta_path,
                                    indexer = indexer,
                                    artifact_path = resolved_signal_strength_path,
                                    shape = shape,
                                    dtype = dtype,
                                    asset_keys = context.asset_keys,
                                    calendar = context.calendar,
                                  )

        else:
            signal_strength = np.full(shape, np.nan, dtype = dtype_obj)

        for combo_idx in range(indexer.total):

            combo = indexer.index_to_combo(combo_idx)
            runtime_param_map = {str(key): combo[pos] for pos, key in enumerate(indexer.param_keys)}
            self._signal_combo_idx = int(combo_idx)
            self._set_context(test_data = test_data, factor_manager = context.factor_manager, factor_param_ranges = runtime_param_map)

            if callable(signal_strength_func):

                resolved = signal_strength_func(combo, cast("Signal.FactorSignalContext", self._factor_signal_context))
                combo_signal_strength = self._coerce_tensor_input(resolved, rows, asset_count)

            else:

                if buy_signals is None or sell_signals is None:

                    raise ValueError("[WARNING] compute_signal_strength requires buy_signals and sell_signals when signal_strength_func is not provided.")

                combo_signal_strength = self.single_net_tensor(
                                                                buy_signals = buy_signals,
                                                                sell_signals = sell_signals,
                                                                buy_gate_signals = buy_gate_signals,
                                                                sell_gate_signals = sell_gate_signals,
                                                              )

            if combo_signal_strength is None:

                raise ValueError(f"[WARNING] compute_signal_strength produced empty artifact for combo index {combo_idx}.")

            signal_strength[combo_idx, :, :] = np.asarray(combo_signal_strength, dtype = dtype_obj)

        if isinstance(signal_strength, np.memmap):
            # Release the writable ('w+') handle the instant the artifact is durable on disk. A live w+
            # mapping locks the file on Windows, so a later recompute of the same path -- or temp cleanup
            # -- would fail with WinError 32. Reopen read-only: the data stays disk-backed with no write
            # lock, and the read handle is owned/released by the consuming manager (see Manager.close()).
            signal_strength.flush()
            Signal._close_memmap(signal_strength)
            signal_strength = np.memmap(resolved_signal_strength_path, dtype = dtype_obj, mode = "r", shape = shape)


        return signal_strength, indexer, context


    def get_signal(self,
                *,
                mode: str,
                test_data: pd.DataFrame,
                factor_param_ranges: Dict[str, Any],
                factor_manager: Optional[FactorEngine.FactorManager] = None,
                factor_name: Optional[Mapping[str, str]] = None,
                signal_strength: Optional[Mapping[str, Any]] = None,
                portfolio_config: Optional[Dict[str, Any]] = None,
                combined_signal_weights: Optional[Mapping[str, float]] = None,
                ondisk_cache: bool = False,
                artifact_dir: Optional[str | pathlib.Path] = None,
                artifact_prefix: Optional[str] = None,
                dtype: str = "float32") -> Dict[str, Any]:
        """
        ### What It Does
        Builds feature-level weights from feature values and weighting config.

        #### Responsibility
        Turns feature scores into normalized allocation or importance weights.

        #### How To Use
        Call it when features, rather than raw signal panels, drive exposure sizing.

        #### Key Parameters In Practice
        - `mode`
          - Rule or execution mode selector. Set it to the branch whose semantics you actually want; do not rely on default mode when the function supports multiple interpretations.
          - Expected shape/type: `str`.
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Dict[str, Any]`.
        - `factor_manager`
          - Active factor registry/cache. Pass the `FactorManager` or `FactorLibrary` that owns registered factor cubes so SR can resolve semantic factor names into actual panels.
          - Expected shape/type: `Optional[FactorEngine.FactorManager]`.
        - `factor_name`
          - Factor lookup key. Use a TA-Lib name, predefined FactorEngine name, or registered semantic/custom factor name owned by the active manager.
          - Expected shape/type: `Optional[Mapping[str, str]]`.
        - `signal_strength`
          - Precomputed signal-strength artifact or panel. Use it when SR has already been run and backtest should consume the artifact directly.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `portfolio_config`
          - Portfolio execution configuration. Put shared-cash, force-cover, prevent-open, sleeve, and report-epsilon behavior here.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `combined_signal_weights`
          - Blend weights for multiple signal sleeves. Use it to control contribution by sleeve before final portfolio normalization.
          - Expected shape/type: `Optional[Mapping[str, float]]`.

        #### Usage Example
        `result = get_signal(...)`

        ---

        ### Parameters
        - `mode`: **str**.
        - `test_data`: **pd.DataFrame**.
        - `factor_param_ranges`: **Dict[str, Any]**.
        - `factor_manager`: **Optional[FactorEngine.FactorManager]** = *None*.
        - `signal_strength`: **Optional[Mapping[str, Any]]** = *None*.

        #### Optional Parameters
        - `factor_name`: **Optional[Mapping[str, str]]** = *None*.
        - `portfolio_config`: **Optional[Dict[str, Any]]** = *None*.
        - `combined_signal_weights`: **Optional[Mapping[str, float]]** = *None*.
        - `ondisk_cache`: **bool** = *False*.
        - `artifact_dir`: **Optional[str | pathlib.Path]** = *None*.
        - `artifact_prefix`: **Optional[str]** = *None*.
        - `dtype`: **str** = *"float32"*.

        ---

        ### Returns
        - `result`: **Dict[str, Any]**.
        """

        signal_strengths: Dict[str, np.ndarray | np.memmap] = {}
        reference_indexer: Optional[Signal.ParamIndexer] = None
        reference_context: Optional[Signal.FactorSignalContext] = None
        raw_signal_strengths = dict(signal_strength or {})

        if raw_signal_strengths:
            normalized_factor_param_ranges: Dict[str, List[Any]] = {}

            for key, raw_value in (factor_param_ranges or {}).items():
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

                cleaned = []

                for item in values:
                    current = item.unwrap() if isinstance(item, FactorEngine.AutoParam.ParamValue) else item
                    current = cast(Any, current).item() if isinstance(current, np.generic) else current
                    cleaned.append(FactorEngine.AutoParam.clean_value(current))

                if len(cleaned) == 0:

                    raise ValueError("[WARNING] factor_param_ranges cannot contain empty parameter ranges.")

                normalized_factor_param_ranges[str(key)] = cleaned

            param_keys = list(normalized_factor_param_ranges.keys())
            param_values = [normalized_factor_param_ranges[key] for key in param_keys]
            reference_indexer = Signal.ParamIndexer(param_keys = param_keys, param_values = param_values, asset_keys = None)
            dt_index = pd.DatetimeIndex(pd.to_datetime(test_data["Datetime"] if "Datetime" in test_data.columns else test_data.index, errors = "coerce"))

            if bool(pd.isna(dt_index).any()):

                raise ValueError("[WARNING] get_signal test_data contains invalid Datetime values.")

            asset_keys: Optional[List[str]] = None

            for signal_key, raw_signal_panel in raw_signal_strengths.items():

                if isinstance(raw_signal_panel, FactorEngine.FactorManager.CubeArtifact):
                    artifact_asset_keys = [str(asset) for asset in raw_signal_panel.asset_keys]

                    if asset_keys is None:
                        asset_keys = artifact_asset_keys

                    elif asset_keys != artifact_asset_keys:

                        raise ValueError("[WARNING] get_signal signal_strength CubeArtifact assets do not match prior signal assets.")

                    artifact_calendar = pd.DatetimeIndex(pd.to_datetime(raw_signal_panel.calendar, errors = "coerce"))

                    if bool(pd.isna(artifact_calendar).any()):

                        raise ValueError("[WARNING] get_signal signal_strength CubeArtifact contains invalid calendar values.")

                    source_values = np.asarray(raw_signal_panel.values, dtype = np.float32)

                    if source_values.ndim != 3:

                        raise ValueError(f"[WARNING] get_signal signal_strength CubeArtifact must be P x T x N, got {source_values.ndim}D.")

                    if source_values.shape[2] != len(asset_keys):

                        raise ValueError("[WARNING] get_signal signal_strength CubeArtifact asset axis does not match asset metadata.")

                    if source_values.shape[0] == 1 and int(reference_indexer.total) != 1:
                        source_values = np.repeat(source_values, int(reference_indexer.total), axis = 0)

                    elif source_values.shape[0] != int(reference_indexer.total):

                        raise ValueError("[WARNING] get_signal signal_strength CubeArtifact parameter axis does not match factor_param_ranges.")

                    row_indexer = artifact_calendar.get_indexer(dt_index)
                    aligned_values = np.full((int(reference_indexer.total), len(dt_index), len(asset_keys)), np.nan, dtype = np.float32)
                    valid_rows = row_indexer >= 0

                    if bool(valid_rows.any()):
                        aligned_values[:, valid_rows, :] = source_values[:, row_indexer[valid_rows], :]

                    signal_strengths[str(signal_key)] = aligned_values

                    continue

                raise TypeError("[WARNING] get_signal signal_strength accepts only FactorEngine.FactorManager.CubeArtifact inputs; register 2D factor panels with FactorEngine first.")

            if asset_keys is None:

                raise ValueError("[WARNING] get_signal requires non-empty signal_strength.")

            context_store = FactorEngine.FactorManager(calendar = dt_index, asset_keys = asset_keys)
            reference_context = Signal.FactorSignalContext(calendar = dt_index, asset_keys = asset_keys, factor_manager = context_store)

        else:
            signal_sources = {str(signal_key): Signal.feature_tensor_loader(str(raw_name)) for signal_key, raw_name in dict(factor_name or {}).items()}

            if not signal_sources:

                raise ValueError("[WARNING] get_signal requires signal_strength or factor_name.")

            if not isinstance(factor_manager, FactorEngine.FactorManager):

                raise ValueError("[WARNING] get_signal requires factor_manager when signal_strength is not provided.")

            for signal_key, signal_strength_func in signal_sources.items():

                signal_values, signal_indexer, factor_context = self.compute_signal_strength(
                                                                                            mode = mode,
                                                                                            test_data = test_data,
                                                                                            factor_param_ranges = factor_param_ranges,
                                                                                            factor_manager = factor_manager,
                                                                                            signal_strength_func = signal_strength_func,
                                                                                          )

                if reference_indexer is None:

                    reference_indexer = signal_indexer
                    reference_context = factor_context

                elif (reference_indexer.param_keys != signal_indexer.param_keys or reference_indexer.param_values != signal_indexer.param_values):

                    raise RuntimeError("[WARNING] feature weight bundle requires identical indexers across all signal artifacts.")

                signal_strengths[str(signal_key)] = signal_values

        if reference_indexer is None or reference_context is None:

            raise ValueError("[WARNING] get_signal requires at least one signal.")

        resolved_sleeves: List[Dict[str, Any]] = []
        resolved_portfolio_config = dict(portfolio_config or {})
        signal_configs = resolved_portfolio_config.get("signals")

        if not isinstance(signal_configs, dict) or len(signal_configs) == 0:

            raise ValueError("[WARNING] portfolio_config['signals'] is required for feature weight construction.")

        shared_weight_keys = ("force_cover_weekend", "force_cover_holidays", "force_cover_macros", "report_epsilon")

        for signal_key, raw_signal_config in dict(signal_configs).items():
            signal_key = str(signal_key).strip()

            if not signal_key or signal_key not in signal_strengths:

                raise KeyError(f"[WARNING] unknown sleeve signal_key: {signal_key!r}")

            weight_config = dict(raw_signal_config or {})

            for shared_key in shared_weight_keys:

                if shared_key in resolved_portfolio_config and shared_key not in weight_config:
                    weight_config[shared_key] = resolved_portfolio_config[shared_key]

            resolved_sleeves.append(
                                    {
                                        "signal_key": signal_key,
                                        "weight_config": weight_config,
                                        "signal_strength": signal_strengths[signal_key],
                                    }
                                    )

        weight = Position(signal_z_window = self.signal_z_window,
                          signal_epsilon = self.signal_epsilon,
                          signal_start_t = self.signal_start_t).compute_multi_sleeve_weights(
                                                    test_data = test_data,
                                                    sleeve_specs = resolved_sleeves,
                                                    asset_keys = list(reference_context.asset_keys),
                                                    factor_param_ranges = factor_param_ranges,
                                                  )

        if combined_signal_weights:

            for signal_key in dict(combined_signal_weights):

                if signal_key not in signal_strengths:

                    raise KeyError(f"[WARNING] unknown combined signal key: {signal_key!r}")

        signal_strength_meta_paths: Dict[str, str] = {}
        weight_meta_path: Optional[str] = None
        cached_signal_strengths: Dict[str, np.ndarray | np.memmap] = signal_strengths
        cached_weight: np.ndarray | np.memmap = weight

        if ondisk_cache:

            if artifact_dir is None:

                raise ValueError("[WARNING] get_signal ondisk_cache requires artifact_dir.")

            resolved_artifact_dir = pathlib.Path(artifact_dir)
            resolved_artifact_dir.mkdir(parents = True, exist_ok = True)
            safe_prefix = str(artifact_prefix or "sr").strip() or "sr"
            cached_signal_strengths = {}

            for signal_key, signal_values in signal_strengths.items():
                safe_key = re.sub(r"[^0-9A-Za-z]+", "_", str(signal_key)).strip("_") or "signal"
                signal_meta_path = resolved_artifact_dir / f"{safe_prefix}_signal_strength_{safe_key}.json"
                signal_artifact_path = resolved_artifact_dir / f"{safe_prefix}_signal_strength_{safe_key}.dat"
                cached_signal_strengths[str(signal_key)] = self.disk_cache(
                                                                        signal_values,
                                                                        artifact_path = signal_artifact_path,
                                                                        meta_path = signal_meta_path,
                                                                        indexer = reference_indexer,
                                                                        asset_keys = list(reference_context.asset_keys),
                                                                        dtype = dtype,
                                                                        calendar = reference_context.calendar,
                                                                        )
                signal_strength_meta_paths[str(signal_key)] = str(signal_meta_path)

            resolved_weight_meta_path = resolved_artifact_dir / f"{safe_prefix}_weight.json"
            resolved_weight_artifact_path = resolved_artifact_dir / f"{safe_prefix}_weight.dat"
            cached_weight = self.disk_cache(
                                            weight,
                                            artifact_path = resolved_weight_artifact_path,
                                            meta_path = resolved_weight_meta_path,
                                            indexer = reference_indexer,
                                            asset_keys = list(reference_context.asset_keys),
                                            dtype = dtype,
                                            calendar = reference_context.calendar,
                                            )
            weight_meta_path = str(resolved_weight_meta_path)

        self._factor_signal_context = reference_context

        return {

                "signal_strengths": cached_signal_strengths,
                "signal_strength": None,
                "signal_strength_meta_paths": signal_strength_meta_paths,
                "weight": cached_weight,
                "weight_meta_path": weight_meta_path,
                "indexer": reference_indexer,
                "factor_context": reference_context,
                }


    def disk_cache(self,
                    values: np.ndarray | np.memmap,
                    *,
                    artifact_path: str | pathlib.Path,
                    meta_path: str | pathlib.Path,
                    indexer: "Signal.ParamIndexer",
                    asset_keys: List[str],
                    dtype: str = "float32",
                    calendar: Optional[pd.Index] = None) -> np.memmap:
        """
        ### What It Does
        Builds or loads disk-backed signal regularization artifacts.

        #### Responsibility
        Coordinates feature tensor loading, signal tensor generation, metadata validation, and cache reuse.

        #### How To Use
        Call it from runner or research workflows that need reusable signal artifacts.

        #### Key Parameters In Practice
        - `values`
          - Parameter domain or array values. For parameter lists these become iterable candidate values; for artifacts they are the numeric payload.
          - Expected shape/type: `np.ndarray | np.memmap`.
        - `artifact_path`
          - Path to the numeric artifact array. It must point to the actual `.npy`/memmap file referenced by metadata.
          - Expected shape/type: `str | pathlib.Path`.
        - `meta_path`
          - Path to artifact metadata JSON. Use it as the contract source for shape, dtype, param axis, calendar, asset keys, and artifact file location.
          - Expected shape/type: `str | pathlib.Path`.
        - `indexer`
          - Parameter/asset index mapper for artifact tensors. It lets selected optimizer combos map back to the correct array row or column.
          - Expected shape/type: `"Signal.ParamIndexer"`.
        - `asset_keys`
          - Canonical asset-axis labels. Preserve this order when moving between tensor, panel, and portfolio execution code.
          - Expected shape/type: `List[str]`.
        - `dtype`
          - Numerical dtype for stored arrays. Choose it to balance precision and memory; artifact metadata must match the actual array dtype.
          - Expected shape/type: `str`.
        - `calendar`
          - Canonical row-axis timestamps for an artifact or factor cube. It must match the arrays being sliced or loaded, especially for memmap artifacts.
          - Expected shape/type: `Optional[pd.Index]`.

        #### Usage Example
        `result = disk_cache(...)`

        ---

        ### Parameters
        - `values`: **np.ndarray | np.memmap**.
        - `artifact_path`: **str | pathlib.Path**.
        - `meta_path`: **str | pathlib.Path**.
        - `indexer`: **"Signal.ParamIndexer"**.
        - `asset_keys`: **List[str]**.

        #### Optional Parameters
        - `dtype`: **str** = *"float32"*.
        - `calendar`: **Optional[pd.Index]** = *None*.

        ---

        ### Returns
        - `result`: **np.memmap**.
        """

        values_arr = np.asarray(values, dtype = np.dtype(dtype))
        resolved_path = pathlib.Path(artifact_path)
        resolved_meta_path = pathlib.Path(meta_path)
        resolved_path.parent.mkdir(parents = True, exist_ok = True)
        resolved_meta_path.parent.mkdir(parents = True, exist_ok = True)
        memmap_values = np.memmap(resolved_path, dtype = np.dtype(dtype), mode = "w+", shape = values_arr.shape)
        memmap_values[:] = values_arr[:]
        memmap_values.flush()

        self.save_metadata(
                            meta_path = resolved_meta_path,
                            indexer = indexer,
                            artifact_path = resolved_path,
                            shape = values_arr.shape,
                            dtype = dtype,
                            asset_keys = list(asset_keys),
                            calendar = calendar if calendar is not None else (self._factor_signal_context.calendar if self._factor_signal_context is not None else None),
                          )

        # Release the writable handle now that the artifact + metadata are durable (see
        # compute_signal_strength): a live 'w+' mapping locks the file on Windows against recompute
        # and cleanup. Reopen read-only for the caller; the consuming manager releases it (Manager.close()).
        Signal._close_memmap(memmap_values)
        memmap_values = np.memmap(resolved_path, dtype = np.dtype(dtype), mode = "r", shape = values_arr.shape)


        return memmap_values
