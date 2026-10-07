from ENV_MGMT.imports import *



def clean_value(value: Any) -> Any:

    # Canonicalize identically to AutoParam.clean_value so cache keys / grid signatures are stable across
    # equivalent parameter representations: unwrap ParamValue, collapse numpy scalars, and round floats to
    # FLOAT_ROUND. Without this, 0.1+0.2 vs 0.3 (or a ParamValue vs its raw scalar) serialize to different
    # keys and miss the cache. The pathlib.Path/Mapping/set branches are retained because this serializer
    # is also used by the on-disk metadata writer (Manager.py) which has no json default=str fallback.
    if isinstance(value, AutoParam.ParamValue):
        value = value.unwrap()

    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, float):

        return round(float(value), AutoParam.FLOAT_ROUND)

    if isinstance(value, pathlib.Path):

        return str(value)

    if isinstance(value, Mapping):

        return {str(key): clean_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):

        return [clean_value(item) for item in value]


    return value


class AutoParam:
    """
    ### What It Does
    Provides small parameter wrapper utilities used by factor grids and feature-reference resolution.

    #### Responsibility
    Keeps parameter values serializable, rounds floating point keys consistently, and stores feature-reference metadata on pandas DataFrames.

    #### How To Use
    Use `param_ranges(...)` for normal workflow setup; use `AutoParam.clean_value(...)` and reference-map helpers when building lower-level factor metadata.

    #### Key Parameters In Practice
    - No constructor parameters.


    #### Usage Example
    `cleaned = AutoParam.clean_value(value)`
    """

    FLOAT_ROUND = 10
    REGISTRY_KEY = "_zora_factor_registry"
    FEATURE_MAP_KEY = "_zora_feature_refs"


    class ParamValue:
        """
        ### What It Does
        Wraps one named parameter value while preserving the parameter key that produced it.

        #### Responsibility
        Lets scalar and iterable parameter values behave like ordinary Python values while still carrying their original search-space key.

        #### How To Use
        Use it indirectly through AutoParam helpers or when a runtime parameter needs to remember which search key it came from.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.
        - `value`
          - Single parameter or feature value. Use it when wrapping, cleaning, or serializing one value.
          - Expected shape/type: `Any`.

        #### Usage Example
        `wrapped = AutoParam.ParamValue("q", 0.2)`

        ---

        ### Parameters
        - `key`: **str**.
        - `value`: **Any**.
        """

        def __init__(self, key: str, value: Any) -> None:

            self.key = str(key)
            self.value = value


        def unwrap(self) -> Any:
            """
            ### What It Does
            Returns the raw Python value stored inside a parameter wrapper.

            #### Responsibility
            Recursively unwraps nested `ParamValue` objects and converts numpy scalar values to native Python scalars.

            #### How To Use
            Call it before serialization, hashing, or passing a wrapped parameter into numerical code that expects a raw scalar.

            #### Key Parameters In Practice
            - No external parameters.

            #### Usage Example
            `raw_value = wrapped.unwrap()`

            ---

            ### Returns
            - `value`: **Any**.
            """

            value = self.value.unwrap() if isinstance(self.value, AutoParam.ParamValue) else self.value

            return value.item() if isinstance(value, np.generic) else value


        def __iter__(self) -> Iterator["AutoParam.ParamValue"]:

            value = self.unwrap()

            if isinstance(value, np.ndarray):
                value = value.tolist()

            if isinstance(value, range):
                value = list(value)

            if isinstance(value, (AutoParam.ParamList, list, tuple, set)):
                for item in value:
                    yield AutoParam.ParamValue(self.key, item)

                return

            yield AutoParam.ParamValue(self.key, value)


        def __len__(self) -> int:

            value = self.unwrap()

            if isinstance(value, np.ndarray):

                return int(value.size)

            if isinstance(value, (AutoParam.ParamList, list, tuple, set, range)):

                return len(value)

            return 1


        def __getitem__(self, idx: Any) -> "AutoParam.ParamValue":

            value = self.unwrap()

            if isinstance(value, np.ndarray):
                value = value.tolist()

            if isinstance(value, range):
                value = list(value)

            if isinstance(value, (AutoParam.ParamList, list, tuple)):

                return AutoParam.ParamValue(self.key, value[idx])

            if idx == 0:

                return AutoParam.ParamValue(self.key, value)

            raise TypeError(f"[WARNING] ParamValue({self.key}) is scalar and not subscriptable with idx={idx!r}")


        def __float__(self) -> float:

            return float(self.unwrap())


        def __int__(self) -> int:

            return int(self.unwrap())


        def __index__(self) -> int:

            return int(self.unwrap())


        def __bool__(self) -> bool:

            return bool(self.unwrap())


        def __array__(self, dtype: Any = None) -> np.ndarray:

            return np.asarray(self.unwrap(), dtype = dtype)


        def __str__(self) -> str:

            return str(self.unwrap())


        def __repr__(self) -> str:

            return f"ParamValue({self.key}={self.value!r})"


    class ParamList(list):
        """
        ### What It Does
        Stores an iterable parameter domain while returning `ParamValue` objects when indexed.

        #### Responsibility
        Preserves the search-space key across list indexing and slicing so selected values still carry their parameter identity.

        #### How To Use
        Use it indirectly through AutoParam-style range construction when indexed values need to remain key-aware.

        #### Key Parameters In Practice
        - `key`
          - Lookup key. It should match the registry, payload, context data family, or cache key being requested.
          - Expected shape/type: `str`.
        - `values`
          - Parameter domain or array values. For parameter lists these become iterable candidate values; for artifacts they are the numeric payload.
          - Expected shape/type: `Iterable[Any]`.

        #### Usage Example
        `domain = AutoParam.ParamList("q", [0.1, 0.2])`

        ---

        ### Parameters
        - `key`: **str**.
        - `values`: **Iterable[Any]**.
        """

        def __init__(self, key: str, values: Iterable[Any]) -> None:

            super().__init__(values)
            self._key = str(key)


        def __getitem__(self, idx: Any) -> Any:

            value = super().__getitem__(idx)

            if isinstance(idx, slice):

                return AutoParam.ParamList(self._key, value)


            return AutoParam.ParamValue(self._key, value)


    @staticmethod
    def clean_value(value: Any) -> Any:
        """
        ### What It Does
        Converts parameter values into stable, JSON-friendly Python objects.

        #### Responsibility
        Unwraps AutoParam values, converts numpy scalar/array types, rounds floats, and recursively cleans list-like and dict-like values.

        #### How To Use
        Call it before building cache keys, metadata payloads, feature ids, or parameter manifests.

        #### Key Parameters In Practice
        - `value`
          - Single parameter or feature value. Use it when wrapping, cleaning, or serializing one value.
          - Expected shape/type: `Any`.

        #### Usage Example
        `cleaned = AutoParam.clean_value(raw_value)`

        ---

        ### Parameters
        - `value`: **Any**.

        ---

        ### Returns
        - `cleaned`: **Any**.
        """

        if isinstance(value, AutoParam.ParamValue):
            value = value.unwrap()

        if isinstance(value, np.generic):
            value = value.item()

        if isinstance(value, float):

            return round(float(value), AutoParam.FLOAT_ROUND)

        if isinstance(value, np.ndarray):

            return [AutoParam.clean_value(item) for item in value.tolist()]

        if isinstance(value, (AutoParam.ParamList, list, tuple)):

            return [AutoParam.clean_value(item) for item in list(value)]

        if isinstance(value, dict):

            return {str(key): AutoParam.clean_value(item) for key, item in value.items()}


        return value


    @staticmethod
    def set_feature_ref_map(test_data: pd.DataFrame, feature_ref_map: Dict[str, str], merge: bool = True) -> Dict[str, str]:
        """
        ### What It Does
        Stores semantic feature-reference aliases on a DataFrame.

        #### Responsibility
        Merges or replaces the DataFrame-level feature reference map used later by signal construction and factor lookup code.

        #### How To Use
        Call it after creating factor columns when a short semantic name should resolve to a concrete column or feature id.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `feature_ref_map`
          - Feature alias map stored on a DataFrame. It lets semantic names resolve to concrete columns later.
          - Expected shape/type: `Dict[str, str]`.
        - `merge`
          - Merge behavior switch. Use it to decide whether new semantic maps extend or replace existing mappings.
          - Expected shape/type: `bool`.

        #### Usage Example
        `refs = AutoParam.set_feature_ref_map(test_data, {"base": "RSI__period_14"})`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `feature_ref_map`: **Dict[str, str]**.

        #### Optional Parameters
        - `merge`: **bool** = *True*.

        ---

        ### Returns
        - `resolved`: **Dict[str, str]**.
        """

        attrs = dict(getattr(test_data, "attrs", {}) or {})
        previous = attrs.get(AutoParam.FEATURE_MAP_KEY, {})
        resolved: Dict[str, str] = {}

        if merge and isinstance(previous, dict):
            resolved.update({str(key): str(value) for key, value in previous.items()})

        resolved.update({str(key): str(value) for key, value in dict(feature_ref_map or {}).items()})
        attrs[AutoParam.FEATURE_MAP_KEY] = resolved
        test_data.attrs = attrs


        return resolved


    @staticmethod
    def resolve_feature_references(test_data: pd.DataFrame,
                                    semantic_to_ref: Dict[str, str],
                                    create_columns: bool = False,
                                    strict: bool = True,
                                    param_map: Optional[Dict[str, Any]] = None,
                                    manifest_df: Optional[pd.DataFrame] = None) -> Dict[str, str]:
        """
        ### What It Does
        Resolves semantic feature names to concrete DataFrame columns.

        #### Responsibility
        Builds or consumes a feature manifest, checks direct column names, feature ids, and semantic ids, and optionally materializes resolved aliases as new columns.

        #### How To Use
        Call it inside signal code when user-facing names must be mapped back to generated factor columns.

        #### Key Parameters In Practice
        - `test_data`
          - Canonical working market-data frame. It must carry a usable datetime axis and the columns required by factors, SR, and the chosen backtest mode.
          - Expected shape/type: `pd.DataFrame`.
        - `semantic_to_ref`
          - Semantic-name mapping. Use it to connect user-facing factor names to generated feature ids or column references.
          - Expected shape/type: `Dict[str, str]`.
        - `create_columns`
          - Column creation switch. Use it when a registered artifact should also materialize DataFrame columns for inspection or legacy consumers.
          - Expected shape/type: `bool`.
        - `strict`
          - Validation strictness switch. Enable strict mode when missing aliases or incompatible inputs should fail fast.
          - Expected shape/type: `bool`.
        - `param_map`
          - Mapping object used for lookup/resolution. Keys should match the semantic or column names used by downstream code.
          - Expected shape/type: `Optional[Dict[str, Any]]`.
        - `manifest_df`
          - Manifest table. Use it as the user-facing summary of what artifacts are registered and how to retrieve them.
          - Expected shape/type: `Optional[pd.DataFrame]`.

        #### Usage Example
        `resolved = AutoParam.resolve_feature_references(test_data, {"entry": "RSI_14"})`

        ---

        ### Parameters
        - `test_data`: **pd.DataFrame**.
        - `semantic_to_ref`: **Dict[str, str]**.

        #### Optional Parameters
        - `create_columns`: **bool** = *False*.
        - `strict`: **bool** = *True*.
        - `param_map`: **Optional[Dict[str, Any]]** = *None*.
        - `manifest_df`: **Optional[pd.DataFrame]** = *None*.

        ---

        ### Returns
        - `resolved`: **Dict[str, str]**.
        """

        del param_map

        if isinstance(manifest_df, pd.DataFrame):
            manifest = manifest_df

        else:
            rows: List[Dict[str, Any]] = []
            attrs = dict(getattr(test_data, "attrs", {}) or {})
            registry = attrs.get(AutoParam.REGISTRY_KEY, {})

            if isinstance(registry, dict):

                for column_name, meta_raw in dict(registry.get("columns", {}) or {}).items():
                    meta = dict(meta_raw) if isinstance(meta_raw, dict) else {}

                    rows.append(
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
                                 "params": AutoParam.clean_value(meta.get("params", {})),
                                }
                               )

            known = {str(row.get("column_name", "")) for row in rows}

            for column_name in test_data.columns:
                text = str(column_name)

                if text in known:

                    continue

                rows.append(
                            {
                             "column_name": text,
                             "feature_id": text,
                             "semantic_id": text,
                             "factor": "",
                             "target_freq": None,
                             "cal_column": None,
                             "output_idx": 0,
                             "timing_semantics": "",
                             "prev_idx": False,
                             "params": {},
                            }
                           )

            manifest = pd.DataFrame(rows)

        feature_to_col = {str(row.get("feature_id", "")): str(row.get("column_name", "")) for _, row in manifest.iterrows()}
        semantic_to_col = {str(row.get("semantic_id", "")): str(row.get("column_name", "")) for _, row in manifest.iterrows()}
        resolved: Dict[str, str] = {}

        for semantic_id, ref in dict(semantic_to_ref or {}).items():

            ref_text = str(ref).strip()
            actual_col: Optional[str] = None

            if ref_text in test_data.columns:
                actual_col = ref_text

            elif ref_text in feature_to_col:
                actual_col = feature_to_col[ref_text]

            elif ref_text in semantic_to_col:
                actual_col = semantic_to_col[ref_text]

            if actual_col is None:

                if strict:

                    raise KeyError(f"[FEATURE][RESOLVE] unresolved ref='{ref_text}' for semantic_id='{semantic_id}'")

                continue

            resolved[str(semantic_id)] = actual_col

            if create_columns and str(semantic_id) not in test_data.columns:
                test_data[str(semantic_id)] = test_data[actual_col]

        if resolved:
            AutoParam.set_feature_ref_map(test_data, resolved, merge = True)


        return resolved
