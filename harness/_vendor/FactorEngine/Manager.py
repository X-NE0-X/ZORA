from ENV_MGMT.imports import *
from .Specs import INDICATOR_SPECS
from .Params import clean_value
from .Library import FactorLibrary  # predefined-factor kernel namespace consumed by _compute_customized



def _factor_worker(payload_bytes: bytes) -> bytes:

    payload = cloudpickle.loads(payload_bytes)

    manager = FactorManager(
                            payload["test_data"],
                            data_freq = payload.get("data_freq"),
                            memory_cache = False,
                            ondisk_cache = False,
                           )

    artifact = manager._get(
                            payload["factor_name"],
                            params = payload.get("params"),
                            cal_column = payload.get("cal_column", "Close"),
                            pair_column = payload.get("pair_column"),
                            timing_semantics = payload.get("timing_semantics"),
                            output_name = payload.get("output_name"),
                            semantic_name = payload.get("semantic_name"),
                            store = False,
                            logical_processors = 1,
                            _param_combos = payload["param_combos"],
                            _param_keys = payload["param_keys"],
                            **dict(payload.get("kwargs", {})),
                           )


    return cloudpickle.dumps(

                             {
                              "values": artifact.values,
                              "param_indexer": artifact.param_indexer,
                              "output_name": artifact.output_name,
                              "metadata": artifact.metadata,
                             }
                            )



class FactorManager:
    """
    ### What It Does
    Manages aligned factor cubes for one market-data panel.

    #### Responsibility
    Normalizes input data into a calendar and asset universe, computes TA-Lib/predefined/custom factor artifacts, caches results, and exposes retrieval by feature id or semantic name.

    #### How To Use
    Instantiate it with wide market data, then call `get.single(...)`, `get.batch(...)`, `register(...)`, or `register_vbt_factor(...)` to build factors.

    #### Key Parameters In Practice
    - `test_data`
      - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
      - Expected shape/type: `Optional[pd.DataFrame]`.
    - `calendar`
      - Canonical row-axis timestamps for an artifact or factor cube. It must match the arrays being sliced or loaded, especially for memmap artifacts.
      - Expected shape/type: `Optional[Iterable[Any]]`.
    - `asset_keys`
      - Canonical asset-axis labels. Preserve this order when moving between tensor, panel, and portfolio execution code.
      - Expected shape/type: `Optional[Iterable[str]]`.
    - `data_freq`
      - Source data frequency metadata. It helps factor and cache logic understand the working clock without guessing from timestamps.
      - Expected shape/type: `str | None`.
    - `memory_cache`
      - In-memory cache switch. Enable it when repeated factor calls should reuse artifacts during one process lifetime.
      - Expected shape/type: `bool`.
    - `ondisk_cache`
      - Disk cache switch. Enable it when factor artifacts should survive process boundaries or notebook restarts.
      - Expected shape/type: `bool`.
    - `max_factor_cache_bytes`
      - Memory budget for factor artifacts. Lower it when large cubes risk exhausting RAM; higher values reduce recomputation.
      - Expected shape/type: `int`.
    - `max_resample_cache_bytes`
      - Memory budget for resampled panels. Tune it separately from factor cache when resampling dominates repeated work.
      - Expected shape/type: `int`.
    - `ondisk_cache_dir`
      - Disk cache directory. Point it at a stable artifact folder, not a transient notebook working directory.
      - Expected shape/type: `Optional[str | pathlib.Path]`.


    #### Usage Example
    `manager = FactorManager(test_data).get.batch("RSI", {"period": [14, 28]})`

    ---

    ### Parameters
    #### Optional Parameters
    - `test_data`: **Optional[pd.DataFrame]** = *None*.
    - `calendar`: **Optional[Iterable[Any]]** = *None*.
    - `asset_keys`: **Optional[Iterable[str]]** = *None*.
    - `data_freq`: **str | None** = *None*.
    - `memory_cache`: **bool** = *False*.
    - `ondisk_cache`: **bool** = *False*.
    - `max_factor_cache_bytes`: **int** = *268435456*.
    - `max_resample_cache_bytes`: **int** = *268435456*.
    - `ondisk_cache_dir`: **Optional[str | pathlib.Path]** = *None*.
    """

    STORE_ATTR_KEY = "__zora_factor_manager__"

    def __deepcopy__(self, memo: Dict[int, Any]) -> "FactorManager":

        # A FactorManager is a live registry/cache service, not value data. `attach`
        # stores it in `test_data.attrs[STORE_ATTR_KEY]`, so the manager and its
        # market-data frame reference each other. pandas' NDFrame.__finalize__
        # deep-copies `.attrs` WITHOUT threading the deepcopy memo, so that cycle is
        # invisible to normal memoization and copy.deepcopy recurses without bound
        # (e.g. CTX.build_payload deep-merging Data_Config['test_data'], whose .attrs
        # carries this manager).
        # Sharing the manager by identity is the intended -- and only sound -- copy
        # semantics: its caches/artifacts must never be silently cloned.
        memo[id(self)] = self

        return self


    class FactorCache:
        """
        ### What It Does
        Caches computed factor artifacts in memory and optionally on disk.

        #### Responsibility
        Builds stable cache keys from factor parameters and source panels, manages memory eviction, and persists cube metadata plus `.npy` values.

        #### How To Use
        Use it through `FactorManager`; direct construction is only needed when testing cache behavior.

        #### Key Parameters In Practice
        - `memory_cache`
          - In-memory cache switch. Enable it when repeated factor calls should reuse artifacts during one process lifetime.
          - Expected shape/type: `bool`.
        - `ondisk_cache`
          - Disk cache switch. Enable it when factor artifacts should survive process boundaries or notebook restarts.
          - Expected shape/type: `bool`.
        - `max_factor_cache_bytes`
          - Memory budget for factor artifacts. Lower it when large cubes risk exhausting RAM; higher values reduce recomputation.
          - Expected shape/type: `int`.
        - `ondisk_cache_dir`
          - Disk cache directory. Point it at a stable artifact folder, not a transient notebook working directory.
          - Expected shape/type: `pathlib.Path`.

        #### Usage Example
        `cache = FactorManager.FactorCache(memory_cache=True, ondisk_cache=False, max_factor_cache_bytes=1024, ondisk_cache_dir=path)`

        ---

        ### Parameters
        - `memory_cache`: **bool**.
        - `ondisk_cache`: **bool**.
        - `max_factor_cache_bytes`: **int**.
        - `ondisk_cache_dir`: **pathlib.Path**.
        """

        @staticmethod
        def _stable_json(value: Any) -> str:

            return json.dumps(clean_value(value), sort_keys = True, separators = (",", ":"), default = str)


        @staticmethod
        def _hash_dataframe(frame: pd.DataFrame) -> str:

            hashed = pd.util.hash_pandas_object(frame, index = True).to_numpy(dtype = np.uint64, copy = False)
            digest = hashlib.blake2b(digest_size = 16)
            digest.update(np.ascontiguousarray(hashed).view(np.uint8))
            digest.update(FactorManager.FactorCache._stable_json([str(col) for col in frame.columns]).encode("utf-8"))


            return digest.hexdigest()


        def __init__(self,
                     *,
                     memory_cache: bool,
                     ondisk_cache: bool,
                     max_factor_cache_bytes: int,
                     ondisk_cache_dir: pathlib.Path) -> None:

            self.memory_cache = bool(memory_cache)
            self.ondisk_cache = bool(ondisk_cache)
            self.max_factor_cache_bytes = int(max_factor_cache_bytes)
            self.ondisk_cache_dir = pathlib.Path(ondisk_cache_dir)
            self._factor_cache: Dict[str, FactorManager.CubeArtifact] = {}
            self._factor_cache_order: List[str] = []


        def key(self,
                factor_upper: str,
                params: Sequence[Mapping[str, Any]],
                cal_column: Any,
                output_name: Optional[str],
                panel: pd.DataFrame,
                pair_panel: Optional[pd.DataFrame],
                vbt_version: str,
                timing_semantics: Optional[str] = None,
                extra_panels: Optional[Sequence[pd.DataFrame]] = None) -> str:
            """
            ### What It Does
            Builds a deterministic cache key for one factor computation.

            #### Responsibility
            Hashes factor identity, parameter grid, selected input columns, optional pair panel, output selection, and vectorbt version into a compact key.

            #### How To Use
            Call it before cache lookup or storage when implementing a new factor execution path.

            #### Key Parameters In Practice
            - `factor_upper`
              - Upper factor threshold/bound. Use it for band or greater-than rules when the long/short rule needs an explicit ceiling.
              - Expected shape/type: `str`.
            - `params`
              - Selected factor parameter values for one computation. Use scalar values for `single` calls and keep grid domains in `factor_param_ranges` for batch/search paths.
              - Expected shape/type: `Sequence[Mapping[str, Any]]`.
            - `cal_column`
              - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
              - Expected shape/type: `Any`.
            - `output_name`
              - Specific output selector for multi-output indicators. Use it for factors such as MACD where one call returns multiple arrays.
              - Expected shape/type: `Optional[str]`.
            - `panel`
              - Input data panel. It should be pre-aligned to the intended calendar and asset labels before conversion or slicing.
              - Expected shape/type: `pd.DataFrame`.
            - `pair_panel`
              - Secondary comparison panel. Use it for pair/factor calculations requiring another aligned price or feature matrix.
              - Expected shape/type: `Optional[pd.DataFrame]`.
            - `vbt_version`
              - vectorbt version metadata. Use it to document/reproduce backend behavior across environment changes.
              - Expected shape/type: `str`.

            #### Usage Example
            `cache_key = cache.key("RSI", combos, "Close", None, panel, None, version)`

            ---

            ### Parameters
            - `factor_upper`: **str**.
            - `params`: **Sequence[Mapping[str, Any]]**.
            - `cal_column`: **Any**.
            - `output_name`: **Optional[str]**.
            - `panel`: **pd.DataFrame**.
            - `pair_panel`: **Optional[pd.DataFrame]**.
            - `vbt_version`: **str**.

            ---

            ### Returns
            - `cache_key`: **str**.
            """

            payload = {
                        "version": 3,
                        "backend": "FactorEngine",
                        "factor": factor_upper,
                        "params": list(params),
                        "cal_column": str(cal_column),
                        "output_name": output_name,
                        "panel_hash": self._hash_dataframe(panel),
                        "pair_hash": None if pair_panel is None else self._hash_dataframe(pair_panel),
                        "vbt_version": vbt_version,
                        "timing_semantics": str(timing_semantics or "").strip().lower(),
                      }

            # N-field (multi-input) factors declare extra_panels as a list (possibly empty); legacy callers pass
            # None and the key never gains this member -> byte-identical legacy digest. `is not None` (NOT truthiness)
            # so the payload shape is a pure function of the spec: a 1-field N-field factor still carries an empty
            # `extra_hashes` and can never collide with a legacy single-panel factor sharing the same anchor.
            if extra_panels is not None:
                payload["extra_hashes"] = [self._hash_dataframe(frame) for frame in extra_panels]


            return hashlib.blake2b(self._stable_json(payload).encode("utf-8"), digest_size = 20).hexdigest()


        def get(self, key: str) -> Optional["FactorManager.CubeArtifact"]:
            """
            ### What It Does
            Retrieves a cached factor artifact by key.

            #### Responsibility
            Checks the in-memory LRU cache first, then reads disk metadata and values when on-disk caching is enabled.

            #### How To Use
            Call it before computing a factor so repeated grids can reuse existing cube artifacts.

            #### Key Parameters In Practice
            - `key`
              - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
              - Expected shape/type: `str`.

            #### Usage Example
            `artifact = cache.get(cache_key)`

            ---

            ### Parameters
            - `key`: **str**.

            ---

            ### Returns
            - `artifact`: **Optional["FactorManager.CubeArtifact"]**.
            """

            if self.memory_cache and key in self._factor_cache:
                artifact = self._factor_cache[key]
                self._factor_cache_order = [item for item in self._factor_cache_order if item != key] + [key]

                return artifact

            meta_path = self.ondisk_cache_dir / f"{key}.json"
            values_path = self.ondisk_cache_dir / f"{key}.npy"

            if self.ondisk_cache and meta_path.exists() and values_path.exists():

                meta = json.loads(meta_path.read_text(encoding = "utf-8"))
                values = np.load(values_path)

                artifact = FactorManager.CubeArtifact(
                                                     feature_id = str(meta["feature_id"]),
                                                     values = values,
                                                     calendar = pd.DatetimeIndex(pd.to_datetime(list(meta["calendar"]))),
                                                     asset_keys = list(meta["asset_keys"]),
                                                     param_indexer = list(meta["param_indexer"]),
                                                     timing_semantics = str(meta["timing_semantics"]),
                                                     semantic_name = meta.get("semantic_name"),
                                                     factor_name = meta.get("factor_name"),
                                                     output_name = meta.get("output_name"),
                                                     metadata = dict(meta.get("metadata", {})),
                                                    )

                self.put(key, artifact, disk = False)

                return artifact


            return None


        def put(self, key: str, artifact: "FactorManager.CubeArtifact", *, disk: bool = True) -> None:
            """
            ### What It Does
            Stores a factor artifact in the configured caches.

            #### Responsibility
            Updates the memory cache with eviction by byte budget and writes disk metadata/value files when enabled.

            #### How To Use
            Call it after computing a factor artifact that should be reusable by later requests.

            #### Key Parameters In Practice
            - `key`
              - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
              - Expected shape/type: `str`.
            - `artifact`
              - Registered factor artifact. It should include values, calendar, asset keys, parameter indexer, and metadata before being stored or reused.
              - Expected shape/type: `"FactorManager.CubeArtifact"`.
            - `disk`
              - Disk persistence switch/path. Use it only when the function should read/write artifacts instead of keeping arrays in memory.
              - Expected shape/type: `bool`.

            #### Usage Example
            `cache.put(cache_key, artifact)`

            ---

            ### Parameters
            - `key`: **str**.
            - `artifact`: **"FactorManager.CubeArtifact"**.

            #### Optional Parameters
            - `disk`: **bool** = *True*.
            """

            if self.memory_cache:
                self._factor_cache[key] = artifact
                self._factor_cache_order = [item for item in self._factor_cache_order if item != key] + [key]

                while sum(int(self._factor_cache[item].values.nbytes) for item in self._factor_cache_order) > self.max_factor_cache_bytes and self._factor_cache_order:
                    evict_key = self._factor_cache_order.pop(0)
                    self._factor_cache.pop(evict_key, None)

            if self.ondisk_cache and disk:

                self.ondisk_cache_dir.mkdir(parents = True, exist_ok = True)
                values_path = self.ondisk_cache_dir / f"{key}.npy"
                meta_path = self.ondisk_cache_dir / f"{key}.json"
                np.save(values_path, artifact.values)

                # Persist tz-naive: a reloaded tz-aware calendar would zero-match the tz-naive test_data
                # axis. Keep the naive-wall-clock contract on disk. No-op when already naive.
                _persist_cal = pd.DatetimeIndex(pd.to_datetime(list(artifact.calendar)))

                if _persist_cal.tz is not None:
                    _persist_cal = _persist_cal.tz_localize(None)

                meta = {
                        "feature_id": artifact.feature_id,
                        "calendar": [str(ts) for ts in _persist_cal],
                        "asset_keys": artifact.asset_keys,
                        "param_indexer": clean_value(artifact.param_indexer),
                        "timing_semantics": artifact.timing_semantics,
                        "semantic_name": artifact.semantic_name,
                        "factor_name": artifact.factor_name,
                        "output_name": artifact.output_name,
                        "metadata": clean_value({**artifact.metadata, "cached_at": time.time()}),
                        }

                meta_path.write_text(json.dumps(meta, ensure_ascii = False, indent = 2), encoding = "utf-8")


    @dataclass
    class CubeArtifact:
        """
        ### What It Does
        Stores one computed factor cube with its calendar, assets, parameters, and metadata.

        #### Responsibility
        Validates the canonical `P x T x N` cube shape, aligns parameter indexers with cube slices, and normalizes timing and naming metadata.

        #### How To Use
        Use artifacts returned by `FactorManager.get`, `register`, or `register_vbt_factor`; pass them to `add(...)` when manually storing.

        #### Key Parameters In Practice
        - `feature_id`
          - Stable feature identifier. Use it for exact artifact lookup when semantic aliases are not enough.
          - Expected shape/type: `str`.
        - `values`
          - Parameter domain or array values. For parameter lists these become iterable candidate values; for artifacts they are the numeric payload.
          - Expected shape/type: `np.ndarray`.
        - `calendar`
          - Canonical row-axis timestamps for an artifact or factor cube. It must match the arrays being sliced or loaded, especially for memmap artifacts.
          - Expected shape/type: `pd.DatetimeIndex`.
        - `asset_keys`
          - Canonical asset-axis labels. Preserve this order when moving between tensor, panel, and portfolio execution code.
          - Expected shape/type: `List[str]`.
        - `param_indexer`
          - Parameter index mapper attached to signal/factor artifacts. Keep it synchronized with `factor_param_ranges` and artifact metadata.
          - Expected shape/type: `List[Dict[str, Any]]`.
        - `timing_semantics`
          - Data timing declaration. Use it to distinguish close-known, next-bar, or custom timing assumptions before execution consumes the factor.
          - Expected shape/type: `str`.
        - `semantic_name`
          - Optional semantic alias. Set it when downstream signal configs should refer to this artifact by a stable business name.
          - Expected shape/type: `Optional[str]`.
        - `factor_name`
          - Factor lookup key. Use a TA-Lib name, predefined FactorEngine name, or registered semantic/custom factor name owned by the active manager.
          - Expected shape/type: `Optional[str]`.
        - `output_name`
          - Specific output selector for multi-output indicators. Use it for factors such as MACD where one call returns multiple arrays.
          - Expected shape/type: `Optional[str]`.
        - `metadata`
          - Structured metadata dictionary. Use it to preserve feature id, semantic name, output name, calendar, asset keys, and parameter values across cache boundaries.
          - Expected shape/type: `Dict[str, Any]`.

        #### Usage Example
        `artifact = manager.get.single("PRICE")`

        ---

        ### Parameters
        - `feature_id`: **str**.
        - `values`: **np.ndarray**.
        - `calendar`: **pd.DatetimeIndex**.
        - `asset_keys`: **List[str]**.
        - `param_indexer`: **List[Dict[str, Any]]**.

        #### Optional Parameters
        - `timing_semantics`: **str** = *"t_close"*.
        - `semantic_name`: **Optional[str]** = *None*.
        - `factor_name`: **Optional[str]** = *None*.
        - `output_name`: **Optional[str]** = *None*.
        - `metadata`: **Dict[str, Any]** = *field(default_factory=dict)*.
        """

        feature_id: str
        values: np.ndarray
        calendar: pd.DatetimeIndex
        asset_keys: List[str]
        param_indexer: List[Dict[str, Any]]
        timing_semantics: str = "t_close"
        semantic_name: Optional[str] = None
        factor_name: Optional[str] = None
        output_name: Optional[str] = None
        metadata: Dict[str, Any] = field(default_factory = dict)


        def __post_init__(self) -> None:

            arr = np.asarray(self.values, dtype = np.float32)

            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]

            if arr.ndim != 3:

                raise ValueError(f"[WARNING] CubeArtifact values must be P x T x N, got {arr.ndim}D.")

            calendar = pd.DatetimeIndex(pd.to_datetime(self.calendar, errors = "coerce"))

            if bool(pd.isna(calendar).any()):

                raise ValueError("[WARNING] CubeArtifact calendar contains invalid timestamps.")

            asset_keys = [str(asset) for asset in self.asset_keys]
            param_indexer = [dict(item or {}) for item in self.param_indexer]

            if arr.shape[1] != len(calendar):

                raise ValueError(f"[WARNING] CubeArtifact calendar mismatch: values T={arr.shape[1]}, calendar={len(calendar)}.")

            if arr.shape[2] != len(asset_keys):

                raise ValueError(f"[WARNING] CubeArtifact asset mismatch: values N={arr.shape[2]}, assets={len(asset_keys)}.")

            if arr.shape[0] != len(param_indexer):

                raise ValueError(f"[WARNING] CubeArtifact parameter mismatch: values P={arr.shape[0]}, params={len(param_indexer)}.")

            timing = str(self.timing_semantics or "t_close").strip().lower()
            self.values = np.ascontiguousarray(arr)
            self.calendar = calendar
            self.asset_keys = asset_keys
            self.param_indexer = [clean_value(item) for item in param_indexer]
            self.timing_semantics = "t_close" if timing in {"", "none"} else timing
            self.semantic_name = str(self.semantic_name or self.feature_id)
            self.factor_name = None if self.factor_name is None else str(self.factor_name).upper()
            self.output_name = None if self.output_name is None else str(self.output_name)
            self.metadata = dict(self.metadata or {})


        @property
        def shape(self) -> Tuple[int, int, int]:
            """
            ### What It Does
            Returns the canonical cube dimensions.

            #### Responsibility
            Exposes `(parameter_count, time_count, asset_count)` without callers reaching into `values.shape` directly.

            #### How To Use
            Call it when validating artifact compatibility or printing factor manifests.

            #### Key Parameters In Practice
            - No external parameters.

            #### Usage Example
            `p_count, t_count, n_count = artifact.shape`

            ---

            ### Returns
            - `shape`: **Tuple[int, int, int]**.
            """

            return self.values.shape


    @staticmethod
    def _asset_panel(test_data: pd.DataFrame, cal_column: Any = "Close") -> Tuple[pd.DataFrame, List[str]]:

        data = test_data.copy()

        if "Datetime" in data.columns:
            data.index = pd.DatetimeIndex(pd.to_datetime(data["Datetime"], errors = "coerce"))

        if isinstance(data.columns, pd.MultiIndex):
            panel_parts: Dict[str, pd.Series] = {}

            for column in data.columns:

                if len(column) < 2:

                    continue

                asset_key = str(column[0]).strip()
                field_key = str(column[-1]).strip()

                if asset_key and field_key == str(cal_column):
                    panel_parts[asset_key] = pd.to_numeric(data[column], errors = "coerce")

            if panel_parts:
                asset_keys = sorted(panel_parts.keys())

                return pd.DataFrame({asset: panel_parts[asset] for asset in asset_keys}, index = data.index, dtype = float), asset_keys

        suffix = f"_{cal_column}"
        panel_parts: Dict[str, pd.Series] = {}

        for column in data.columns:
            text = str(column)

            if text.endswith(suffix):
                panel_parts[text[: -len(suffix)]] = pd.to_numeric(data[column], errors = "coerce")

        if panel_parts:
            asset_keys = sorted(panel_parts.keys())

            return pd.DataFrame({asset: panel_parts[asset] for asset in asset_keys}, index = data.index, dtype = float), asset_keys

        if str(cal_column) in data.columns:

            raw_underlyings = getattr(test_data, "attrs", {}).get("__zora_ctx_underlyings__")
            asset_name = str(raw_underlyings[0]) if isinstance(raw_underlyings, list) and raw_underlyings else "SINGLE"

            return pd.DataFrame({asset_name: pd.to_numeric(data[str(cal_column)], errors = "coerce")}, index = data.index, dtype = float), [asset_name]


        raise ValueError(f"[WARNING] FactorEngine cannot resolve cal_column: {cal_column!r}.")


    @staticmethod
    def _assert_field_coverage(field: Any, panel: pd.DataFrame) -> None:

        # Fail-loud symmetry: _asset_panel already raises when a field matches ZERO columns (structural absence);
        # this closes the present-but-empty hole (a column that exists but is entirely NaN, e.g. a buggy all-NaN
        # OI field), which would otherwise silently produce an all-NaN ADV / exposure basis.
        if not bool(np.isfinite(panel.to_numpy(dtype = np.float64, copy = False)).any()):

            raise ValueError(f"[WARNING] FactorEngine input_field {field!r} resolved to an all-NaN panel (present-but-empty); a declared input field must carry at least one finite value.")


    def _resolve_field_panels(self,
                              spec: Mapping[str, Any],
                              cal_column: Any,
                              pair_column: Optional[Any]) -> Tuple[Optional[Dict[str, pd.DataFrame]], pd.DataFrame, List[str], Optional[pd.DataFrame], Optional[List[pd.DataFrame]]]:

        # The single input-materialization point, shared by _get (serial) and _parallel_get (aggregation) so their
        # cache keys can never diverge. Returns (field_panels, anchor_panel, asset_keys, pair_panel, extra_panels).
        #   * N-field spec (input_fields): ordered named dict; anchor = input_fields[0] defines the asset universe
        #     (columns), every other field reindexed onto the anchor's (index, columns); a declared field absent OR
        #     all-NaN is fail-loud; pair_panel retired (=None); extra_panels = the non-anchor panels (may be []).
        #   * legacy spec (no input_fields): the historical primary(+pair) materialization; extra_panels = None ->
        #     cache.key omits extra_hashes -> byte-identical to the pre-N-field digest.
        input_fields = list(spec.get("input_fields") or [])

        if input_fields:
            anchor_panel, asset_keys = self._asset_panel(self.test_data, cal_column = input_fields[0])
            self._assert_field_coverage(input_fields[0], anchor_panel)
            field_panels: Dict[str, pd.DataFrame] = {input_fields[0]: anchor_panel}

            for field in input_fields[1:]:
                raw_panel = self._asset_panel(self.test_data, cal_column = field)[0]
                self._assert_field_coverage(field, raw_panel)
                field_panels[field] = raw_panel.reindex(index = anchor_panel.index, columns = anchor_panel.columns)

            extra_panels = [field_panels[field] for field in input_fields[1:]]


            return field_panels, anchor_panel, asset_keys, None, extra_panels

        anchor_panel, asset_keys = self._asset_panel(self.test_data, cal_column = cal_column)
        pair_panel = self._asset_panel(self.test_data, cal_column = pair_column)[0] if pair_column is not None else None


        return None, anchor_panel, asset_keys, pair_panel, None


    def _factor_key_and_panels(self,
                               factor_upper: str,
                               spec: Mapping[str, Any],
                               params: Sequence[Mapping[str, Any]],
                               cal_column: Any,
                               pair_column: Optional[Any],
                               output_name: Optional[str],
                               resolved_timing: str) -> Tuple[str, Optional[Dict[str, pd.DataFrame]], pd.DataFrame, List[str], Optional[pd.DataFrame]]:

        # THE cache-key construction chokepoint. Serial (_get) and parallel-aggregation (_parallel_get) both route
        # through here, so extra_hashes can never be present on one path and absent on the other (serial==parallel).
        field_panels, anchor_panel, asset_keys, pair_panel, extra_panels = self._resolve_field_panels(spec, cal_column, pair_column)
        cache_key = self.cache.key(factor_upper, params, cal_column, output_name, anchor_panel, pair_panel, self._vbt_version(), resolved_timing, extra_panels)


        return cache_key, field_panels, anchor_panel, asset_keys, pair_panel


    @staticmethod
    def _slug(value: Any) -> str:

        text = str(value).strip()
        text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


        return text or "value"


    @staticmethod
    def _normalize_params(factor_upper: str,
                          params: Optional[Mapping[str, Any]],
                          factor_param_ranges: Optional[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:

        spec = INDICATOR_SPECS.get(factor_upper, {})
        defaults = {key: value for key, value in dict(spec.get("defaults", {}) or {}).items() if value is not None}
        raw_params = {**defaults, **dict(params or {})}
        alias_map = dict(spec.get("alias_map", {}) or {})
        param_names = list(spec.get("param_names", []) or [])

        if factor_param_ranges is None:
            raw_grid = {key: [value] for key, value in raw_params.items()}

        else:
            raw_grid = {}

            for key, raw in {**raw_params, **dict(factor_param_ranges)}.items():

                if isinstance(raw, range):
                    items = list(raw)

                elif isinstance(raw, np.ndarray):
                    items = raw.tolist()

                elif isinstance(raw, pd.Index):
                    items = raw.tolist()

                elif isinstance(raw, (list, tuple, set)):
                    items = sorted(raw, key = lambda item: (type(item).__name__, str(item))) if isinstance(raw, set) else list(raw)

                else:
                    items = [raw]

                if len(items) == 0:

                    raise ValueError(f"[WARNING] FactorEngine parameter range is empty: {key}")

                raw_grid[str(key)] = [clean_value(item) for item in items]

        param_grid: Dict[str, List[Any]] = {}

        for param_name in param_names:
            source_name = str(alias_map.get(param_name, param_name))

            if source_name in raw_grid:
                param_grid[str(param_name)] = list(raw_grid[source_name])

            elif param_name in raw_grid:
                param_grid[str(param_name)] = list(raw_grid[param_name])

        if not param_grid and factor_upper == "RSI":
            param_grid = {"timeperiod": [14]}

        if not param_grid and param_names:
            missing = ", ".join(param_names)

            raise ValueError(f"[WARNING] missing parameters for {factor_upper}: {missing}")

        keys = list(param_grid.keys())

        combos = [
                  {key: clean_value(value) for key, value in zip(keys, values)}
                  for values in itertools.product(*[param_grid[key] for key in keys])
                 ] if keys else [{}]


        return combos, keys


    @staticmethod
    def _select_array_output(factor_upper: str, arr: np.ndarray, output_name: Optional[str], labels: Sequence[str] = ()) -> Tuple[np.ndarray, str]:

        labels = list(labels or [])

        if arr.ndim == 2:

            return arr.astype(np.float32, copy = False), str(output_name or "value")

        if arr.ndim != 3:

            raise ValueError(f"[WARNING] factor output must be 2D or channel x T x N, got {arr.ndim}D.")

        selected = str(output_name or (labels[0] if labels else "0"))
        label_lookup = {str(label).lower(): idx for idx, label in enumerate(labels)}
        alias_lookup = {"signal": "macdsignal", "histogram": "hist"}
        normalized = alias_lookup.get(selected.lower(), selected.lower())

        if normalized in label_lookup:
            idx = label_lookup[normalized]

        else:

            try:
                idx = int(selected)

            except ValueError as exc:

                raise KeyError(f"[WARNING] unknown output {selected!r} for {factor_upper}. Available: {labels}") from exc

        if idx < 0 or idx >= arr.shape[0]:

            raise IndexError(f"[WARNING] output index out of range for {factor_upper}: {idx}")


        return arr[idx].astype(np.float32, copy = False), selected


    @staticmethod
    def _pack_kernel_params(combo: Mapping[str, Any], param_names: Sequence[str], param_cast: Sequence[str], asset_count: int) -> List[Any]:

        # Cast each per-combo param to the concrete kernel argument type declared by the spec's `param_cast`,
        # aligned positionally with `param_names`. Tokens: `slope_mode` (str "slope" -> 1 else int), `<dtype>_array`
        # (per-asset broadcast ndarray), `int*` -> int(v), `float*` -> float(v); anything else passes through.
        casts = list(param_cast or [])
        packed: List[Any] = []

        for idx, name in enumerate(param_names):
            cast = str(casts[idx]) if idx < len(casts) else ""
            value = combo[name]

            if cast == "slope_mode":
                packed.append(1 if isinstance(value, str) and value.lower() == "slope" else int(value))

            elif cast.endswith("_array"):
                packed.append(FactorLibrary._as_asset_param_array(value, asset_count, np.dtype(cast[:-len("_array")])))

            elif cast.startswith("int"):
                packed.append(int(value))

            elif cast.startswith("float"):
                packed.append(float(value))

            else:
                packed.append(value)


        return packed


    @staticmethod
    def _select_output(indicator: Any, spec: Mapping[str, Any], output_name: Optional[str]) -> Tuple[pd.DataFrame, str]:

        output_names = [str(name) for name in list(spec.get("output_names", []) or [])]

        if output_name is None:
            selected = output_names[0] if output_names else "real"

        else:
            selected = str(output_name)

        attr_aliases = {
                        "signal": "macdsignal",
                        "hist": "macdhist",
                        "histogram": "macdhist",
                       }

        attr = attr_aliases.get(selected.lower(), selected)

        if not hasattr(indicator, attr) and selected.lower() in {"rsi", "value"}:
            attr = "real"

        if not hasattr(indicator, attr):
            available = [name for name in output_names if hasattr(indicator, name)]

            raise AttributeError(f"[WARNING] VBT output {selected!r} not found. Available outputs: {available}")


        return pd.DataFrame(getattr(indicator, attr)).copy(), selected


    @staticmethod
    def _slice_combo(result: pd.DataFrame,
                     combo: Mapping[str, Any],
                     param_keys: List[str],
                     *,
                     index: pd.Index,
                     columns: List[str]) -> pd.DataFrame:

        if not isinstance(result.columns, pd.MultiIndex):

            return result.reindex(index = index, columns = columns)

        current = result

        for key in param_keys:
            value = combo[key]
            level = key if key in current.columns.names else None

            if level is None:

                for candidate in current.columns.names:

                    if str(candidate).lower().endswith("_" + str(key).lower()):
                        level = candidate

                        break

            if level is None:
                level = 0

            current = pd.DataFrame(current.xs(value, level = level, axis = 1))

        if isinstance(current.columns, pd.MultiIndex):
            current.columns = current.columns.get_level_values(-1)


        return current.reindex(index = index, columns = columns)


    @staticmethod
    def _vbt_version() -> str:

        try:

            return str(getattr(vbt, "__version__", "unknown"))

        except Exception:

            return "unavailable"


    @staticmethod
    def feature_id(factor_name: str,
                   params: Optional[Mapping[str, Any]] = None,
                   *,
                   output_name: Optional[str] = None,
                   semantic_name: Optional[str] = None) -> str:
        """
        ### What It Does
        Builds a stable feature id from factor name, parameters, output name, or semantic override.

        #### Responsibility
        Creates readable artifact identifiers that remain deterministic across equivalent parameter dictionaries.

        #### How To Use
        Call it when constructing custom artifacts or when you need to predict the id generated for a factor grid.

        #### Key Parameters In Practice
        - `factor_name`
          - Factor lookup key. Use a TA-Lib name, predefined FactorEngine name, or registered semantic/custom factor name owned by the active manager.
          - Expected shape/type: `str`.
        - `params`
          - Selected factor parameter values for one computation. Use scalar values for `single` calls and keep grid domains in `factor_param_ranges` for batch/search paths.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `output_name`
          - Specific output selector for multi-output indicators. Use it for factors such as MACD where one call returns multiple arrays.
          - Expected shape/type: `Optional[str]`.
        - `semantic_name`
          - Optional semantic alias. Set it when downstream signal configs should refer to this artifact by a stable business name.
          - Expected shape/type: `Optional[str]`.

        #### Usage Example
        `feature_id = FactorManager.feature_id("RSI", {"period": 14})`

        ---

        ### Parameters
        - `factor_name`: **str**.

        #### Optional Parameters
        - `params`: **Optional[Mapping[str, Any]]** = *None*.
        - `output_name`: **Optional[str]** = *None*.
        - `semantic_name`: **Optional[str]** = *None*.

        ---

        ### Returns
        - `feature_id`: **str**.
        """

        if semantic_name:

            return FactorManager._slug(semantic_name)

        parts = [FactorManager._slug(str(factor_name).upper())]

        for key, value in sorted(dict(params or {}).items()):
            parts.append(f"{FactorManager._slug(key)}_{FactorManager._slug(value)}")

        if output_name:
            parts.append(FactorManager._slug(output_name))


        return "__".join(parts)


    def __init__(self,
                 test_data: Optional[pd.DataFrame] = None,
                 *,
                 calendar: Optional[Iterable[Any]] = None,
                 asset_keys: Optional[Iterable[str]] = None,
                 data_freq: str | None = None,
                 memory_cache: bool = False,
                 ondisk_cache: bool = False,
                 max_factor_cache_bytes: int = 268435456,
                 max_resample_cache_bytes: int = 268435456,
                 ondisk_cache_dir: Optional[str | pathlib.Path] = None) -> None:

        from .Proxy import GetProxy  # deferred import: breaks circular dependency
        if test_data is None:

            if calendar is None or asset_keys is None:

                raise ValueError("[WARNING] FactorEngine requires test_data or explicit calendar/asset_keys.")

            resolved_calendar = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors = "coerce"))
            resolved_assets = [str(asset) for asset in asset_keys if str(asset).strip()]

            if bool(pd.isna(resolved_calendar).any()):

                raise ValueError("[WARNING] FactorEngine calendar contains invalid timestamps.")

            if not resolved_assets:

                raise ValueError("[WARNING] FactorEngine requires non-empty asset_keys.")

            data = pd.DataFrame(
                                {f"{asset}_Close": np.nan for asset in resolved_assets},
                                index = resolved_calendar,
                                dtype = float,
                               )
            data["Datetime"] = resolved_calendar

        elif not isinstance(test_data, pd.DataFrame) or test_data.empty:

            raise ValueError("[WARNING] FactorEngine requires non-empty test_data.")

        else:
            data = test_data.copy()

        if not isinstance(data.index, pd.DatetimeIndex):

            if "Datetime" not in data.columns:

                raise ValueError("[WARNING] FactorEngine test_data requires a DatetimeIndex or Datetime column.")

            data.index = pd.DatetimeIndex(pd.to_datetime(data["Datetime"], errors = "coerce"))

        else:
            data.index = pd.DatetimeIndex(pd.to_datetime(data.index, errors = "coerce"))

        if bool(pd.isna(data.index).any()):

            raise ValueError("[WARNING] FactorEngine test_data contains invalid timestamps.")

        if "Datetime" not in data.columns:
            data["Datetime"] = data.index

        self.test_data = data
        self.calendar = pd.DatetimeIndex(data.index)
        # pd.infer_freq raises ("Need at least 3 dates to infer frequency") on a <3-row calendar,
        # which would crash construction; only infer when a freq wasn't supplied and there are
        # enough rows, and treat any residual inference failure as "unknown" ("").
        if data_freq:
            _inferred_freq = data_freq

        elif len(self.calendar) >= 3:

            try:
                _inferred_freq = pd.infer_freq(self.calendar)

            except (ValueError, TypeError):
                _inferred_freq = None

        else:
            _inferred_freq = None

        self.data_freq = str(_inferred_freq or "").upper()
        self.memory_cache = bool(memory_cache)
        self.ondisk_cache = bool(ondisk_cache)
        self.max_factor_cache_bytes = int(max_factor_cache_bytes)
        self.max_resample_cache_bytes = int(max_resample_cache_bytes)

        self.ondisk_cache_dir = pathlib.Path(ondisk_cache_dir) if ondisk_cache_dir is not None else (
            pathlib.Path(__file__).resolve().parents[3] / "ArchiveDeck" / "FactorEngineCache"
        )

        self._base_panel, inferred_asset_keys = self._asset_panel(data, cal_column = "Close")
        # Canonicalize to sorted order: _asset_panel always builds the panel/cube N-axis via sorted(keys),
        # so self.asset_keys must be sorted too or an explicit non-sorted asset_keys kwarg spuriously fails
        # the artifact-vs-manager consistency check (asset_keys mismatch) on every compute-and-store call.
        self.asset_keys = sorted(str(asset) for asset in (asset_keys if asset_keys is not None else inferred_asset_keys))
        self._artifacts: Dict[str, FactorManager.CubeArtifact] = {}
        self._semantic_index: Dict[str, str] = {}

        self.cache = self.FactorCache(
                                      memory_cache = self.memory_cache,
                                      ondisk_cache = self.ondisk_cache,
                                      max_factor_cache_bytes = self.max_factor_cache_bytes,
                                      ondisk_cache_dir = self.ondisk_cache_dir,
                                     )

        self.get = GetProxy(self)


    @classmethod
    def from_test_data(cls, test_data: pd.DataFrame, *, attach: bool = True, **kwargs: Any) -> "FactorManager":
        """
        ### What It Does
        Constructs a factor manager from market data and optionally attaches it to that DataFrame.

        #### Responsibility
        Provides the standard factory path for workflows that want later code to recover the manager from `test_data.attrs`.

        #### How To Use
        Call it after preparing test data and before passing that frame into signal or backtest code that expects an attached manager.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `attach`
          - DataFrame attachment switch. Enable it when the manager should be discoverable from the source `test_data` later.
          - Expected shape/type: `bool`.
        - `**kwargs`
          - Constructor options forwarded to `FactorManager`, for example cache switches, source frequency, cache directory, and cache-size budgets.
          - Expected shape/type: `Any`.

        #### Usage Example
        `manager = FactorManager.from_test_data(test_data, memory_cache=True)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.

        #### Optional Parameters
        - `attach`: **bool** = *True*.
        - `**kwargs`: **Any**.

        ---

        ### Returns
        - `manager`: **"FactorManager"**.
        """

        manager = cls(test_data, **kwargs)

        if attach:
            cls.attach(test_data, manager)

        return manager


    @classmethod
    def attach(cls, test_data: pd.DataFrame, manager: "FactorManager") -> "FactorManager":
        """
        ### What It Does
        Attaches an existing factor manager to a DataFrame.

        #### Responsibility
        Stores the manager in `test_data.attrs` so downstream components can discover the same factor universe without another constructor argument.

        #### How To Use
        Call it after building a manager when the DataFrame will be handed to BacktestEngine or signal-generation utilities.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `manager`
          - Owning factor manager. Pass the exact manager instance whose registry/cache should be used.
          - Expected shape/type: `"FactorManager"`.

        #### Usage Example
        `FactorManager.attach(test_data, manager)`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `manager`: **"FactorManager"**.

        ---

        ### Returns
        - `manager`: **"FactorManager"**.
        """

        if not isinstance(manager, FactorManager):

            raise TypeError("[WARNING] FactorManager.attach requires a FactorManager instance.")

        attrs = dict(getattr(test_data, "attrs", {}) or {})
        attrs[cls.STORE_ATTR_KEY] = manager
        test_data.attrs = attrs


        return manager


    @classmethod
    def get_attached_store(cls, test_data: Optional[pd.DataFrame]) -> Optional["FactorManager"]:
        """
        ### What It Does
        Reads a factor manager previously attached to a DataFrame.

        #### Responsibility
        Provides a safe optional lookup that returns `None` when no compatible manager is present.

        #### How To Use
        Call it in downstream code that receives only `test_data` but can reuse an attached factor manager if one exists.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `Optional[pd.DataFrame]`.

        #### Usage Example
        `manager = FactorManager.get_attached_store(test_data)`

        ---

        ### Parameters
        - `test_data`: **Optional[pd.DataFrame]**.

        ---

        ### Returns
        - `manager`: **Optional["FactorManager"]**.
        """

        if test_data is None:

            return None

        attrs = getattr(test_data, "attrs", None)

        if not isinstance(attrs, dict):

            return None

        manager = attrs.get(cls.STORE_ATTR_KEY)


        return manager if isinstance(manager, FactorManager) else None


    def _compute_customized(self,
                            factor_upper: str,
                            spec: Mapping[str, Any],
                            panel: pd.DataFrame,
                            pair_panel: Optional[pd.DataFrame],
                            combos: List[Dict[str, Any]],
                            output_name: Optional[str],
                            field_panels: Optional[Dict[str, pd.DataFrame]] = None) -> Tuple[np.ndarray, str, Dict[str, Any]]:

        arr_mat = panel.to_numpy(dtype = np.float32, copy = True)
        cube = np.full((len(combos), len(panel.index), len(panel.columns)), np.nan, dtype = np.float32)
        resolved_output = str(output_name or "value")
        custom = FactorLibrary
        asset_count = arr_mat.shape[1]

        # ---- spec-driven dispatch (no hardcoded factor names) --------------------------------------------
        # The spec declares HOW to call a factor's kernel via abstract "call protocols", so adding/retiring a
        # factor is a JSON edit, not an engine edit:
        #   * identity        (kind == "identity")            -> pass the panel through unchanged
        #   * per_column      (input_mode == "price1d")       -> apply a 1-D kernel column-by-column
        #   * matrix          (input_mode == "price2d")       -> apply a T x N-matrix kernel once
        #   * matrix_pair     (input_mode == "price2d_pair")  -> matrix kernel + an aligned reference panel
        #   * namespaced      (spec has "kernel_class")       -> getattr(FactorLibrary.<class>, calc_name); the
        #                                                         kernel receives (panels_dict, combo, kernel_config),
        #                                                         one named T x N panel per declared input_fields entry
        kind         = str(spec.get("kind", ""))
        param_names  = list(spec.get("param_names", []) or [])
        param_cast   = list(spec.get("param_cast", []) or [])
        labels       = list((spec.get("writeback") or {}).get("labels", []) or [])
        kernel_class = spec.get("kernel_class")
        input_mode   = str(spec.get("input_mode", ""))

        if kind == "identity":
            protocol = "identity"
            kernel = None

        elif kernel_class:
            protocol = "namespaced"
            kernel = getattr(getattr(custom, str(kernel_class)), str(spec["calc_name"]))

        elif input_mode == "price1d":
            protocol = "per_column"
            kernel = getattr(custom, str(spec["calc_name"]))

        elif input_mode == "price2d":
            protocol = "matrix"
            kernel = getattr(custom, str(spec["calc_name"]))

        elif input_mode == "price2d_pair":
            protocol = "matrix_pair"
            kernel = getattr(custom, str(spec["calc_name"]))

        else:

            raise NotImplementedError(f"[WARNING] unsupported FactorEngine customized factor: {factor_upper}")

        kernel_config = (spec.get("kernel_config", {}) or {}) if protocol == "namespaced" else {}
        panels_mat: Optional[Dict[str, np.ndarray]] = None

        if protocol == "namespaced":

            if field_panels is None:

                raise ValueError(f"[WARNING] {factor_upper} (namespaced kernel) requires materialized input_fields panels.")

            panels_mat = {name: frame.reindex(index = panel.index, columns = panel.columns).to_numpy(dtype = np.float64, copy = True) for name, frame in field_panels.items()}

        if protocol == "matrix_pair" and pair_panel is None:

            raise ValueError(f"[WARNING] {factor_upper} requires pair_column or pair_panel.")

        for combo_idx, combo in enumerate(combos):
            args = self._pack_kernel_params(combo, param_names, param_cast, asset_count)

            if protocol == "identity":
                out = arr_mat

            elif protocol == "per_column":
                out = np.column_stack([kernel(arr_mat[:, col_idx], *args) for col_idx in range(asset_count)]).astype(np.float32)

            elif protocol == "matrix":
                out = kernel(arr_mat, *args)

            elif protocol == "matrix_pair":
                out = kernel(arr_mat, pair_panel.to_numpy(dtype = np.float32, copy = True), *args)

            else:
                out = kernel(panels_mat, combo, kernel_config)

            out2d, resolved_output = self._select_array_output(factor_upper, np.asarray(out), output_name, labels)
            cube[combo_idx, :, :] = np.asarray(out2d, dtype = np.float32)


        return cube, resolved_output, {"backend": "customized", "kind": kind}


    def _compute_talib(self,
                       factor_upper: str,
                       spec: Mapping[str, Any],
                       panel: pd.DataFrame,
                       combos: List[Dict[str, Any]],
                       param_keys: List[str],
                       output_name: Optional[str]) -> Tuple[np.ndarray, str, Dict[str, Any]]:

        run_kwargs = {key: [combo[key] for combo in combos] for key in param_keys}
        indicator = vbt.talib(str(spec.get("talib_name", factor_upper))).run(panel.astype(np.float64), param_product = False, **run_kwargs)
        result, resolved_output = self._select_output(indicator, spec, output_name)
        cube = np.full((len(combos), len(panel.index), len(panel.columns)), np.nan, dtype = np.float32)

        for combo_idx, combo in enumerate(combos):
            combo_panel = self._slice_combo(result, combo, param_keys, index = panel.index, columns = list(panel.columns))
            cube[combo_idx, :, :] = combo_panel.to_numpy(dtype = np.float32, copy = True)


        return cube, resolved_output, {"backend": "vbt_talib", "talib_name": str(spec.get("talib_name", factor_upper))}


    def _parallel_get(self,
                      *,
                      factor_name: str,
                      params: Optional[Mapping[str, Any]],
                      cal_column: Any,
                      pair_column: Optional[Any],
                      combos: List[Dict[str, Any]],
                      param_keys: List[str],
                      timing_semantics: Optional[str],
                      output_name: Optional[str],
                      semantic_name: Optional[str],
                      kwargs: Mapping[str, Any],
                      logical_processors: int,
                      use_process: bool,
                      prefer_threads: bool,
                      store: bool) -> "FactorManager.CubeArtifact":

        # Normalize factor_name to match the serial _get path (which does .upper().strip()); otherwise a
        # trailing space ('RSI ') KeyErrors on the INDICATOR_SPECS lookup + cache key in this parallel
        # branch while the serial branch silently succeeds -> serial/parallel divergence.
        factor_name = str(factor_name).strip()
        worker_count = max(1, min(int(logical_processors), len(combos)))
        chunk_size = int(math.ceil(len(combos) / worker_count))
        chunks = [combos[idx: idx + chunk_size] for idx in range(0, len(combos), chunk_size)]
        payloads = [
                    cloudpickle.dumps(
                                      {
                                       "test_data": self.test_data,
                                       "data_freq": self.data_freq,
                                       "factor_name": factor_name,
                                       "params": dict(params or {}),
                                       "cal_column": cal_column,
                                       "pair_column": pair_column,
                                       "timing_semantics": timing_semantics,
                                       "output_name": output_name,
                                       "semantic_name": semantic_name,
                                       "kwargs": dict(kwargs),
                                       "param_combos": chunk,
                                       "param_keys": param_keys,
                                      }
                                     )
                    for chunk in chunks
                   ]

        if use_process:

            with ProcessPoolExecutor(max_workers = worker_count) as executor:
                parts = list(executor.map(_factor_worker, payloads))

        elif prefer_threads:

            with ThreadPoolExecutor(max_workers = worker_count) as executor:
                parts = list(executor.map(_factor_worker, payloads))

        else:
            parts = [_factor_worker(payload) for payload in payloads]

        parts = [cloudpickle.loads(part) for part in parts]

        values = np.concatenate([np.asarray(part["values"], dtype = np.float32) for part in parts], axis = 0)
        param_indexer = [dict(item) for part in parts for item in list(part["param_indexer"])]
        first = parts[0]
        factor_upper = str(factor_name).upper().strip()
        spec = INDICATOR_SPECS[factor_upper]
        resolved_output = first.get("output_name")
        resolved_timing = timing_semantics or str(spec.get("timing_semantics", "t_close"))
        # Route the aggregated cache key through the same chokepoint the serial path uses, so an N-field factor's
        # extra_hashes can never be present on one path and absent on the other (serial-key == parallel-key). For
        # legacy factors extra_panels is None -> no extra_hashes -> byte-identical to the historical parallel key.
        cache_key, _field_panels, panel, asset_keys, pair_panel = self._factor_key_and_panels(factor_upper, spec, param_indexer, cal_column, pair_column, output_name, resolved_timing)
        grid_sig = hashlib.blake2b(self.FactorCache._stable_json(param_indexer).encode("utf-8"), digest_size = 6).hexdigest()
        feature_id = self.feature_id(
                                     str(factor_name).upper(),
                                     {"field": str(cal_column), "grid": len(param_indexer), "grid_sig": grid_sig},
                                     output_name = resolved_output,
                                     semantic_name = semantic_name,
                                    )

        artifact = self.CubeArtifact(
                                     feature_id = feature_id,
                                     values = values,
                                     calendar = pd.DatetimeIndex(panel.index),
                                     asset_keys = asset_keys,
                                     param_indexer = param_indexer,
                                     timing_semantics = resolved_timing,
                                     semantic_name = semantic_name or feature_id,
                                     factor_name = str(factor_name).upper(),
                                     output_name = None if resolved_output is None else str(resolved_output),
                                     metadata = {**dict(first.get("metadata", {})), "cache_key": cache_key, "parallel_workers": worker_count, "parallel_chunks": len(payloads), "parallel_backend": "process" if use_process else ("thread" if prefer_threads else "serial_chunks")},
                                    )
        self.cache.put(cache_key, artifact)

        if store:
            self.add(artifact, overwrite = True)


        return artifact


    def _get(self,
             factor_name: str,
             params: Optional[Mapping[str, Any]] = None,
             cal_column: Any = "Close",
             pair_column: Optional[Any] = None,
             factor_param_ranges: Optional[Mapping[str, Any]] = None,
             timing_semantics: Optional[str] = None,
             output_name: Optional[str] = None,
             semantic_name: Optional[str] = None,
             store: bool = True,
             logical_processors: Optional[int] = 1,
             use_process: bool = False,
             prefer_threads: bool = False,
             _param_combos: Optional[List[Dict[str, Any]]] = None,
             _param_keys: Optional[List[str]] = None,
             **kwargs: Any) -> "FactorManager.CubeArtifact":

        factor_upper = str(factor_name).upper().strip()
        spec = INDICATOR_SPECS.get(factor_upper)

        if not isinstance(spec, dict):

            raise NotImplementedError(f"[WARNING] unknown FactorEngine factor: {factor_name!r}")

        if cal_column == "Close" and factor_upper in {"VOLUME", "TURNOVER"}:
            cal_column = factor_upper.title()

        # N-field kernels declare their ordered input_fields; the anchor (input_fields[0]) drives the asset universe
        # and pair_column is retired (all inputs travel by name through _resolve_field_panels). Idempotent for the
        # parallel worker (already receives cal_column == anchor). Fail loud if a caller wired an explicit foreign field.
        input_fields = list(spec.get("input_fields") or [])

        if input_fields:

            if cal_column not in ("Close", input_fields[0]) or pair_column is not None:

                raise ValueError(f"[WARNING] {factor_upper} owns its inputs {input_fields}; it rejects explicit cal_column / pair_column overrides.")

            cal_column = input_fields[0]
            pair_column = None

        if _param_combos is None:
            combos, param_keys = self._normalize_params(factor_upper, {**dict(params or {}), **dict(kwargs or {})}, factor_param_ranges)

        else:
            combos = [dict(item) for item in _param_combos]
            param_keys = list(_param_keys or (list(combos[0].keys()) if combos else []))

        # resolve timing once and feed BOTH the cache key and the artifact, so two different
        # timing declarations can never collide on the same key (stale-timing look-ahead).
        resolved_timing = timing_semantics or str(spec.get("timing_semantics", "t_close"))
        cache_key, field_panels, panel, asset_keys, pair_panel = self._factor_key_and_panels(factor_upper, spec, combos, cal_column, pair_column, output_name, resolved_timing)
        cached = self.cache.get(cache_key)

        if cached is not None:

            if store:
                self.add(cached, overwrite = True)

            return cached

        worker_count = max(1, int(logical_processors or 1))

        if factor_param_ranges is not None and worker_count > 1 and len(combos) > 1 and _param_combos is None:

            return self._parallel_get(

                                      factor_name = factor_name,
                                      params = params,
                                      cal_column = cal_column,
                                      pair_column = pair_column,
                                      combos = combos,
                                      param_keys = param_keys,
                                      timing_semantics = timing_semantics,
                                      output_name = output_name,
                                      semantic_name = semantic_name,
                                      kwargs = kwargs,
                                      logical_processors = worker_count,
                                      use_process = use_process,
                                      prefer_threads = prefer_threads,
                                      store = store,
                                     )

        kind = str(spec.get("kind", "")).lower()

        if kind == "talib":
            cube, resolved_output, metadata = self._compute_talib(factor_upper, spec, panel, combos, param_keys, output_name)

        elif kind in {"identity", "numba"}:
            cube, resolved_output, metadata = self._compute_customized(factor_upper, spec, panel, pair_panel, combos, output_name, field_panels)

        else:

            raise NotImplementedError(f"[WARNING] unsupported FactorEngine factor kind for {factor_upper}: {kind!r}")

        feature_params = {"field": str(cal_column), **combos[0]} if len(combos) == 1 else {
            "field": str(cal_column),
            "grid": len(combos),
            "grid_sig": hashlib.blake2b(self.FactorCache._stable_json(combos).encode("utf-8"), digest_size = 6).hexdigest(),
        }
        feature_id = self.feature_id(factor_upper, feature_params, output_name = resolved_output, semantic_name = semantic_name)

        artifact = self.CubeArtifact(
                                     feature_id = feature_id,
                                     values = cube,
                                     calendar = pd.DatetimeIndex(panel.index),
                                     asset_keys = asset_keys,
                                     param_indexer = combos,
                                     timing_semantics = resolved_timing,
                                     semantic_name = semantic_name or feature_id,
                                     factor_name = factor_upper,
                                     output_name = resolved_output,
                                     metadata = {**metadata, "cache_key": cache_key},
                                    )

        self.cache.put(cache_key, artifact)

        if store:
            self.add(artifact, overwrite = True)


        return artifact


    def add(self, artifact: "FactorManager.CubeArtifact", *, overwrite: bool = True) -> "FactorManager.CubeArtifact":
        """
        ### What It Does
        Stores a validated artifact in the manager registry.

        #### Responsibility
        Checks calendar and asset compatibility, applies overwrite rules, and updates both feature-id and semantic-name indexes.

        #### How To Use
        Call it when you have a `CubeArtifact` that should become retrievable through `get_feature`, or `get_cube`.

        #### Key Parameters In Practice
        - `artifact`
          - Registered factor artifact. It should include values, calendar, asset keys, parameter indexer, and metadata before being stored or reused.
          - Expected shape/type: `"FactorManager.CubeArtifact"`.
        - `overwrite`
          - Registry overwrite switch. Keep it false when accidental duplicate feature ids should fail instead of silently replacing artifacts.
          - Expected shape/type: `bool`.

        #### Usage Example
        `manager.add(artifact, overwrite=True)`

        ---

        ### Parameters
        - `artifact`: **"FactorManager.CubeArtifact"**.

        #### Optional Parameters
        - `overwrite`: **bool** = *True*.

        ---

        ### Returns
        - `artifact`: **"FactorManager.CubeArtifact"**.
        """

        if not artifact.calendar.equals(self.calendar):

            raise ValueError("[WARNING] CubeArtifact calendar does not match FactorManager calendar.")

        if artifact.asset_keys != self.asset_keys:

            raise ValueError("[WARNING] CubeArtifact assets do not match FactorManager assets.")

        if not overwrite and artifact.feature_id in self._artifacts:

            raise KeyError(f"[WARNING] feature already exists: {artifact.feature_id}")

        self._artifacts[artifact.feature_id] = artifact
        self._semantic_index[str(artifact.semantic_name)] = artifact.feature_id


        return artifact


    def get_feature(self, key: str) -> "FactorManager.CubeArtifact":
        """
        ### What It Does
        Returns a registered factor artifact by feature id or semantic name.

        #### Responsibility
        Provides the public artifact retrieval alias used by factor and signal workflows.

        #### How To Use
        Call it when you need the full artifact object rather than only its values.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.

        #### Usage Example
        `artifact = manager.get_feature("base_long")`

        ---

        ### Parameters
        - `key`: **str**.

        ---

        ### Returns
        - `artifact`: **"FactorManager.CubeArtifact"**.
        """

        return self.get_registered(key)


    def get_cube(self, key: str) -> np.ndarray:
        """
        ### What It Does
        Returns the raw `P x T x N` values for a registered feature.

        #### Responsibility
        Provides numpy access to stored factor cubes while preserving the manager's lookup rules.

        #### How To Use
        Call it from vectorized signal construction when the full parameter cube is needed.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.

        #### Usage Example
        `cube = manager.get_cube("RSI__period_14")`

        ---

        ### Parameters
        - `key`: **str**.

        ---

        ### Returns
        - `cube`: **np.ndarray**.
        """

        return np.asarray(self.get_registered(key).values, dtype = np.float32)


    def get_feature_cube(self, key: str) -> np.ndarray:
        """
        ### What It Does
        Returns the raw factor cube for a registered feature.

        #### Responsibility
        Maintains compatibility with callers that use the older feature-cube terminology.

        #### How To Use
        Call it as an alias for `get_cube(...)` when code reads more clearly with feature-cube naming.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.

        #### Usage Example
        `cube = manager.get_feature_cube("base_long")`

        ---

        ### Parameters
        - `key`: **str**.

        ---

        ### Returns
        - `cube`: **np.ndarray**.
        """

        return self.get_cube(key)


    def get_frame(self, key: str, combo: int = 0) -> pd.DataFrame:
        """
        ### What It Does
        Projects one parameter-combo slice of a registered `P x T x N` cube into a `T x N` DataFrame.

        #### Responsibility
        Provides a lightweight preview/inspection view of a stored factor without recomputing it.

        #### How To Use
        Call it in notebooks to eyeball a single combo of a factor cube as a time x asset table; it is not part of the signal/backtest execution path.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.
        - `combo`
          - Parameter-combo index along the cube's `P` axis. Use `0` for single-combo factors.
          - Expected shape/type: `int`.

        #### Usage Example
        `frame = manager.get_frame("RSI__period_14", combo = 0)`

        ---

        ### Parameters
        - `key`: **str**.

        #### Optional Parameters
        - `combo`: **int** = *0*.

        ---

        ### Returns
        - `frame`: **pd.DataFrame**.
        """

        artifact = self.get_registered(key)
        cube = np.asarray(artifact.values)
        n_combos = cube.shape[0]

        if not -n_combos <= combo < n_combos:

            raise IndexError(f"[WARNING] combo {combo} out of range for feature '{key}' with {n_combos} combo(s).")


        return pd.DataFrame(cube[combo], index = artifact.calendar, columns = artifact.asset_keys)


    def get_tensor(self, key: str, broadcast: bool = True) -> np.ndarray:
        """
        ### What It Does
        Returns a registered feature cube in tensor form.

        #### Responsibility
        Preserves the public tensor-style API while routing to the canonical cube storage.

        #### How To Use
        Call it from signal code that expects a tensor accessor; `broadcast` is accepted for API compatibility.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.
        - `broadcast`
          - Broadcast behavior switch. Use it when scalar/one-asset values should expand across the full asset axis.
          - Expected shape/type: `bool`.

        #### Usage Example
        `tensor = manager.get_tensor("base_long")`

        ---

        ### Parameters
        - `key`: **str**.

        #### Optional Parameters
        - `broadcast`: **bool** = *True*.

        ---

        ### Returns
        - `tensor`: **np.ndarray**.
        """

        del broadcast

        return self.get_cube(key)


    def get_registered(self, key: str) -> "FactorManager.CubeArtifact":
        """
        ### What It Does
        Resolves and returns a stored artifact from the manager registry.

        #### Responsibility
        Searches direct feature ids first, then semantic names, and raises when the key is unknown.

        #### How To Use
        Use it when implementing other retrieval methods or when strict artifact lookup is desired.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.

        #### Usage Example
        `artifact = manager.get_registered("base_long")`

        ---

        ### Parameters
        - `key`: **str**.

        ---

        ### Returns
        - `artifact`: **"FactorManager.CubeArtifact"**.
        """

        text = str(key)
        feature_id = text if text in self._artifacts else self._semantic_index.get(text)

        if feature_id is None or feature_id not in self._artifacts:

            raise KeyError(f"[WARNING] unknown feature: {key}")


        return self._artifacts[feature_id]


    def manifest(self) -> pd.DataFrame:
        """
        ### What It Does
        Builds a tabular summary of registered factor artifacts.

        #### Responsibility
        Collects feature ids, semantic names, factor names, output names, timing semantics, shapes, parameters, and metadata into a DataFrame.

        #### How To Use
        Call it when auditing which artifacts are present in a manager or exporting factor metadata for a notebook.

        #### Key Parameters In Practice
        - No external parameters.

        #### Usage Example
        `manifest_df = manager.manifest()`

        ---

        ### Returns
        - `manifest`: **pd.DataFrame**.
        """

        rows = []

        for artifact in self._artifacts.values():

            rows.append(
                        {
                         "feature_id": artifact.feature_id,
                         "semantic_name": artifact.semantic_name,
                         "factor_name": artifact.factor_name,
                         "output_name": artifact.output_name,
                         "timing_semantics": artifact.timing_semantics,
                         "shape": tuple(int(dim) for dim in artifact.values.shape),
                         "params": clean_value(artifact.param_indexer),
                         "metadata": clean_value(artifact.metadata),
                        }
                       )


        return pd.DataFrame(rows)


    def register(self,
                 feature_id: str,
                 values: Any,
                 *,
                 timing_semantics: str = "t_close",
                 params: Optional[Mapping[str, Any]] = None,
                 asset_keys: Optional[Iterable[str]] = None,
                 calendar: Optional[Iterable[Any]] = None,
                 metadata: Optional[Mapping[str, Any]] = None,
                 semantic_name: Optional[str] = None,
                 overwrite: bool = True,
                 store: bool = True,
                 **_: Any) -> "FactorManager.CubeArtifact":
        """
        ### What It Does
        Registers an already-materialized custom factor panel or cube.

        #### Responsibility
        Converts DataFrame or ndarray values into a validated `CubeArtifact`, builds parameter metadata, and optionally stores it in the manager registry.

        #### How To Use
        Call it when a custom factor has already been computed outside FactorEngine but should participate in the same artifact and signal pipeline.

        #### Key Parameters In Practice
        - `feature_id`
          - Stable feature identifier. Use it for exact artifact lookup when semantic aliases are not enough.
          - Expected shape/type: `str`.
        - `values`
          - Parameter domain or array values. For parameter lists these become iterable candidate values; for artifacts they are the numeric payload.
          - Expected shape/type: `Any`.
        - `timing_semantics`
          - Data timing declaration. Use it to distinguish close-known, next-bar, or custom timing assumptions before execution consumes the factor.
          - Expected shape/type: `str`.
        - `params`
          - Selected factor parameter values for one computation. Use scalar values for `single` calls and keep grid domains in `factor_param_ranges` for batch/search paths.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `asset_keys`
          - Canonical asset-axis labels. Preserve this order when moving between tensor, panel, and portfolio execution code.
          - Expected shape/type: `Optional[Iterable[str]]`.
        - `calendar`
          - Canonical row-axis timestamps for an artifact or factor cube. It must match the arrays being sliced or loaded, especially for memmap artifacts.
          - Expected shape/type: `Optional[Iterable[Any]]`.
        - `metadata`
          - Structured metadata dictionary. Use it to preserve feature id, semantic name, output name, calendar, asset keys, and parameter values across cache boundaries.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `semantic_name`
          - Optional semantic alias. Set it when downstream signal configs should refer to this artifact by a stable business name.
          - Expected shape/type: `Optional[str]`.
        - `overwrite`
          - Registry overwrite switch. Keep it false when accidental duplicate feature ids should fail instead of silently replacing artifacts.
          - Expected shape/type: `bool`.
        - `store`
          - Registry storage switch. Disable it for one-off calculations; enable it when downstream SR or manifest lookup needs the artifact.
          - Expected shape/type: `bool`.
        - `**_`
          - Ignored compatibility keywords. New code should put meaningful fields into explicit parameters or `metadata` instead.
          - Expected shape/type: `Any`.

        #### Usage Example
        `artifact = manager.register("base_long", base_long_panel, semantic_name="base_long")`

        ---

        ### Parameters
        - `feature_id`: **str**.
        - `values`: **Any**.

        #### Optional Parameters
        - `timing_semantics`: **str** = *"t_close"*.
        - `params`: **Optional[Mapping[str, Any]]** = *None*.
        - `asset_keys`: **Optional[Iterable[str]]** = *None*.
        - `calendar`: **Optional[Iterable[Any]]** = *None*.
        - `metadata`: **Optional[Mapping[str, Any]]** = *None*.
        - `semantic_name`: **Optional[str]** = *None*.
        - `overwrite`: **bool** = *True*.
        - `store`: **bool** = *True*.
        - `**_`: **Any**.

        ---

        ### Returns
        - `artifact`: **"FactorManager.CubeArtifact"**.
        """

        if isinstance(values, pd.DataFrame):
            calendar_source = list(calendar) if calendar is not None else list(self.calendar)
            resolved_calendar = pd.DatetimeIndex(pd.to_datetime(calendar_source, errors = "coerce"))
            resolved_assets = [str(asset) for asset in (asset_keys if asset_keys is not None else self.asset_keys)]
            panel = values.reindex(index = resolved_calendar, columns = resolved_assets)

            if not values.empty and bool(values.notna().to_numpy().any()) and not bool(panel.notna().to_numpy().any()):
                # reindex silently fills NaN when labels don't line up; asset_keys are then force-aligned so
                # the store-time shape guard can't tell an all-NaN factor from a real one. Surface the
                # label mismatch (tz-aware vs naive calendar, or asset-key spelling) instead of storing NaN.
                raise ValueError(
                                "[WARNING] register(): reindex to (calendar, asset_keys) produced an all-NaN panel "
                                f"while the input DataFrame had data. Index/column labels do not match "
                                f"(input columns={list(values.columns)[:8]}, expected asset_keys={resolved_assets[:8]}, "
                                f"input index tz={getattr(values.index, 'tz', None)}, calendar tz={getattr(resolved_calendar, 'tz', None)})."
                                )
            arr = panel.to_numpy(dtype = np.float32, copy = True)[np.newaxis, :, :]

        else:
            arr = np.asarray(values, dtype = np.float32)

            if arr.ndim == 2:
                arr = arr[np.newaxis, :, :]

            if calendar is None or asset_keys is None:

                raise ValueError("[WARNING] ndarray custom factors require explicit calendar and asset_keys.")

            resolved_calendar = pd.DatetimeIndex(pd.to_datetime(list(calendar), errors = "coerce"))
            resolved_assets = [str(asset) for asset in asset_keys]

        if isinstance(params, (list, tuple)) and all(isinstance(item, Mapping) for item in params):
            param_indexer = [dict(item) for item in params]

        elif isinstance(params, Mapping) and arr.shape[0] > 1 and all(isinstance(value, (list, tuple, np.ndarray, pd.Index, range)) for value in params.values()):
            param_keys = list(params.keys())
            param_values = [
                            list(value.tolist() if isinstance(value, (np.ndarray, pd.Index)) else value)
                            for value in params.values()
                           ]
            param_indexer = [
                            {str(key): clean_value(value) for key, value in zip(param_keys, values)}
                            for values in itertools.product(*param_values)
                            ]

        else:
            param_indexer = [dict(params or {})]

        metadata = dict(metadata or {})

        artifact = self.CubeArtifact(
                                     feature_id = str(feature_id),
                                     values = arr,
                                     calendar = resolved_calendar,
                                     asset_keys = resolved_assets,
                                     param_indexer = param_indexer,
                                     timing_semantics = timing_semantics,
                                     semantic_name = semantic_name or str(feature_id),
                                     factor_name = "CUSTOM",
                                     output_name = "value",
                                     metadata = metadata,
                                    )

        if store:

            if not artifact.calendar.equals(self.calendar) or artifact.asset_keys != self.asset_keys:

                raise ValueError(
                                "[WARNING] custom factor registration calendar/assets mismatch. "
                                "Align the custom panel to FactorManager.calendar and FactorManager.asset_keys before storing."
                                )

            self.add(artifact, overwrite = overwrite)


        return artifact


    def register_vbt_factor(self,
                            feature_id: str,
                            apply_func: Callable[..., Any],
                            *,
                            cal_column: Any = "Close",
                            params: Optional[Mapping[str, Any]] = None,
                            factor_param_ranges: Optional[Mapping[str, Any]] = None,
                            output_name: str = "value",
                            timing_semantics: str = "t_close",
                            semantic_name: Optional[str] = None,
                            metadata: Optional[Mapping[str, Any]] = None,
                            overwrite: bool = True,
                            store: bool = True,
                            **kwargs: Any) -> "FactorManager.CubeArtifact":
        """
        ### What It Does
        Registers a custom vectorbt `IndicatorFactory` factor from a callable.

        #### Responsibility
        Runs the callable across the manager's asset panel and parameter grid, slices vectorbt output back into `P x T x N`, and stores the resulting custom artifact.

        #### How To Use
        Call it when factor logic should remain as an executable function rather than a pre-materialized panel.

        #### Key Parameters In Practice
        - `feature_id`
          - Stable feature identifier. Use it for exact artifact lookup when semantic aliases are not enough.
          - Expected shape/type: `str`.
        - `apply_func`
          - Custom factor callable. It must accept the arrays/params supplied by FactorEngine and return output aligned to the source calendar/assets.
          - Expected shape/type: `Callable[..., Any]`.
        - `cal_column`
          - Primary calculation column. For price-based factors this is usually `Close`; for custom factors pass the exact source column or columns the factor expects.
          - Expected shape/type: `Any`.
        - `params`
          - Selected factor parameter values for one computation. Use scalar values for `single` calls and keep grid domains in `factor_param_ranges` for batch/search paths.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, GA treats it as gene domains, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `output_name`
          - Specific output selector for multi-output indicators. Use it for factors such as MACD where one call returns multiple arrays.
          - Expected shape/type: `str`.
        - `timing_semantics`
          - Data timing declaration. Use it to distinguish close-known, next-bar, or custom timing assumptions before execution consumes the factor.
          - Expected shape/type: `str`.
        - `semantic_name`
          - Optional semantic alias. Set it when downstream signal configs should refer to this artifact by a stable business name.
          - Expected shape/type: `Optional[str]`.
        - `metadata`
          - Structured metadata dictionary. Use it to preserve feature id, semantic name, output name, calendar, asset keys, and parameter values across cache boundaries.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `overwrite`
          - Registry overwrite switch. Keep it false when accidental duplicate feature ids should fail instead of silently replacing artifacts.
          - Expected shape/type: `bool`.
        - `store`
          - Registry storage switch. Disable it for one-off calculations; enable it when downstream SR or manifest lookup needs the artifact.
          - Expected shape/type: `bool`.
        - `**kwargs`
          - Extra keyword arguments forwarded into the custom VBT factor call; keep these deterministic because they become part of the produced artifact semantics.
          - Expected shape/type: `Any`.

        #### Usage Example
        `artifact = manager.register_vbt_factor("custom_ma", apply_func, factor_param_ranges={"window": [5, 10]})`

        ---

        ### Parameters
        - `feature_id`: **str**.
        - `apply_func`: **Callable[..., Any]**.

        #### Optional Parameters
        - `cal_column`: **Any** = *"Close"*.
        - `params`: **Optional[Mapping[str, Any]]** = *None*.
        - `factor_param_ranges`: **Optional[Mapping[str, Any]]** = *None*.
        - `output_name`: **str** = *"value"*.
        - `timing_semantics`: **str** = *"t_close"*.
        - `semantic_name`: **Optional[str]** = *None*.
        - `metadata`: **Optional[Mapping[str, Any]]** = *None*.
        - `overwrite`: **bool** = *True*.
        - `store`: **bool** = *True*.
        - `**kwargs`: **Any**.

        ---

        ### Returns
        - `artifact`: **"FactorManager.CubeArtifact"**.
        """

        panel, asset_keys = self._asset_panel(self.test_data, cal_column = cal_column)
        defaults = {**dict(params or {}), **dict(kwargs or {})}

        if factor_param_ranges is None:
            param_grid = {str(key): [value] for key, value in defaults.items()}

        else:
            param_grid = {}

            for key, raw in {**defaults, **dict(factor_param_ranges)}.items():

                if isinstance(raw, range):
                    items = list(raw)

                elif isinstance(raw, np.ndarray):
                    items = raw.tolist()

                elif isinstance(raw, pd.Index):
                    items = raw.tolist()

                elif isinstance(raw, (list, tuple, set)):
                    items = sorted(raw, key = lambda item: (type(item).__name__, str(item))) if isinstance(raw, set) else list(raw)

                else:
                    items = [raw]

                if len(items) == 0:

                    raise ValueError(f"[WARNING] FactorEngine VBT custom parameter range is empty: {key}")

                param_grid[str(key)] = [clean_value(item) for item in items]

        param_keys = list(param_grid.keys())
        combos = [
                  {key: clean_value(value) for key, value in zip(param_keys, values)}
                  for values in itertools.product(*[param_grid[key] for key in param_keys])
                 ] if param_keys else [{}]
        run_kwargs = {key: [combo[key] for combo in combos] for key in param_keys}
        indicator_cls = vbt.IndicatorFactory(
                                            input_names = ["data"],
                                            param_names = param_keys,
                                            output_names = [str(output_name)],
                                           ).from_apply_func(apply_func)
        indicator = indicator_cls.run(panel.astype(np.float64), param_product = False, **run_kwargs)
        result = pd.DataFrame(getattr(indicator, str(output_name))).copy()
        cube = np.full((len(combos), len(panel.index), len(panel.columns)), np.nan, dtype = np.float32)

        for combo_idx, combo in enumerate(combos):
            combo_panel = self._slice_combo(result, combo, param_keys, index = panel.index, columns = list(panel.columns))
            cube[combo_idx, :, :] = combo_panel.to_numpy(dtype = np.float32, copy = True)

        artifact = self.CubeArtifact(
                                     feature_id = str(feature_id),
                                     values = cube,
                                     calendar = pd.DatetimeIndex(panel.index),
                                     asset_keys = asset_keys,
                                     param_indexer = combos,
                                     timing_semantics = timing_semantics,
                                     semantic_name = semantic_name or str(feature_id),
                                     factor_name = "VBT_CUSTOM",
                                     output_name = str(output_name),
                                     metadata = {**dict(metadata or {}), "backend": "vbt_custom", "cal_column": str(cal_column)},
                                    )

        if store:
            self.add(artifact, overwrite = overwrite)

        return artifact
