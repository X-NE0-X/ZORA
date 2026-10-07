from ENV_MGMT.imports import *
from .Manager import FactorManager



class GetProxy:
    """
    ### What It Does
    Provides the ergonomic `manager.get(...)`, `manager.get.single(...)`, and `manager.get.batch(...)` API.

    #### Responsibility
    Routes user-facing factor calls into the owning `FactorManager` without exposing the private `_get(...)` dispatcher.

    #### How To Use
    Access it through `FactorManager.get` or `FactorLibrary.get`; direct construction is reserved for manager initialization.

    #### Key Parameters In Practice
    - `manager`
      - Owning factor manager. Pass the exact manager instance whose registry/cache should be used.
      - Expected shape/type: `FactorManager`.


    #### Usage Example
    `artifact = manager.get.batch("RSI", {"period": [14, 28]})`

    ---

    ### Parameters
    - `manager`: **FactorManager**.
    """

    def __init__(self, manager: FactorManager) -> None:

        self._manager = manager


    def __call__(self, factor_name: str, *args: Any, **kwargs: Any) -> FactorManager.CubeArtifact:

        return self._manager._get(factor_name, *args, **kwargs)


    def batch(self, factor_name: str, factor_param_ranges: Mapping[str, Any], **kwargs: Any) -> FactorManager.CubeArtifact:
        """
        ### What It Does
        Computes or retrieves a factor across a parameter grid.

        #### Responsibility
        Passes the factor name and `factor_param_ranges` into the manager dispatcher for batch cube construction.

        #### How To Use
        Call it when a factor should be evaluated for multiple parameter combinations.

        #### Key Parameters In Practice
        - `factor_name`
          - Factor lookup key. Use a TA-Lib name, predefined FactorEngine name, or registered semantic/custom factor name owned by the active manager.
          - Expected shape/type: `str`.
        - `factor_param_ranges`
          - Canonical search domain. Traversal expands it as a Cartesian grid, GA treats it as gene domains, and artifact slicing uses it to map a selected combo back to tensor rows.
          - Expected shape/type: `Mapping[str, Any]`.
        - `**kwargs`
          - Factor-specific keyword parameters forwarded to the manager dispatcher, such as `period`, `cal_column`, `output_name`, `semantic_name`, or custom callable parameters.
          - Expected shape/type: `Any`.

        #### Usage Example
        `artifact = manager.get.batch("RSI", {"period": [14, 28]})`

        ---

        ### Parameters
        - `factor_name`: **str**.
        - `factor_param_ranges`: **Mapping[str, Any]**.

        #### Optional Parameters
        - `**kwargs`: **Any**.

        ---

        ### Returns
        - `artifact`: **FactorManager.CubeArtifact**.
        """

        return self._manager._get(factor_name, factor_param_ranges = factor_param_ranges, **kwargs)


    def single(self, factor_name: str, params: Optional[Mapping[str, Any]] = None, **kwargs: Any) -> FactorManager.CubeArtifact:
        """
        ### What It Does
        Computes or retrieves a factor for one parameter set.

        #### Responsibility
        Passes scalar parameters into the manager dispatcher and returns a single-combo cube artifact.

        #### How To Use
        Call it when a factor only needs one configuration or when preparing a simple registered feature.

        #### Key Parameters In Practice
        - `factor_name`
          - Factor lookup key. Use a TA-Lib name, predefined FactorEngine name, or registered semantic/custom factor name owned by the active manager.
          - Expected shape/type: `str`.
        - `params`
          - Selected factor parameter values for one computation. Use scalar values for `single` calls and keep grid domains in `factor_param_ranges` for batch/search paths.
          - Expected shape/type: `Optional[Mapping[str, Any]]`.
        - `**kwargs`
          - Factor-specific keyword parameters forwarded to the manager dispatcher, such as `period`, `cal_column`, `output_name`, `semantic_name`, or custom callable parameters.
          - Expected shape/type: `Any`.

        #### Usage Example
        `artifact = manager.get.single("RSI", {"period": 14})`

        ---

        ### Parameters
        - `factor_name`: **str**.

        #### Optional Parameters
        - `params`: **Optional[Mapping[str, Any]]** = *None*.
        - `**kwargs`: **Any**.

        ---

        ### Returns
        - `artifact`: **FactorManager.CubeArtifact**.
        """

        return self._manager._get(factor_name, params = params, **kwargs)
