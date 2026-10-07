#================================= ZORA HARNESS (vendored) ==============================
# INFRASTRUCTURE (Factor Library)
#========================================================================================

# No import-time bootstrap lives here on purpose. The upstream package opened with a
# monorepo-root search plus a dotenv credential loader that reached OUTSIDE the
# repository; neither is wanted in a self-contained, publishable tree.
# `harness._infra.ensure_infra()` is the one supported bootstrap and it already puts
# `_vendor` on sys.path, which is what makes the shared ENV_MGMT bundle resolvable.

# ---- public API re-exports (facade keeps `import FactorEngine` stable) ----
from .Specs import INDICATOR_SPECS
from .Params import AutoParam, clean_value
from .Manager import FactorManager, _factor_worker
from .Proxy import GetProxy
from .Library import FactorLibrary

__all__ = ["FactorManager", "FactorLibrary", "AutoParam", "GetProxy", "clean_value",
           "INDICATOR_SPECS"]



"""
### Structure:
```
FactorManager(test_data)                       # <- the engine you instantiate
|-- FactorManager.__init__()
|   |-- FactorManager._asset_panel(cal_column = "Close")
|   |-- FactorManager.FactorCache(memory_cache / ondisk_cache)
|   |-- GetProxy(FactorManager)
|
|   (FactorLibrary = passive predefined-kernel namespace, never instantiated;
|    consumed by FactorManager._compute_customized below)
|
|-- TALib path
|   |-- FactorManager.get("RSI" / "MACD" / ...)
|   |-- GetProxy.__call__() / GetProxy.single() / GetProxy.batch()
|       |-- FactorManager._get()
|           |-- FactorManager._asset_panel(cal_column)
|           |-- FactorManager._normalize_params()
|           |-- FactorManager.FactorCache.key()
|           |-- FactorManager.FactorCache.get()
|           |-- logical_processors > 1
|           |   |-- FactorManager._parallel_get()
|           |       |-- ProcessPoolExecutor / ThreadPoolExecutor
|           |       |-- _factor_worker()
|           |       |-- FactorManager.CubeArtifact(P x T x N)
|           |       |-- FactorManager.FactorCache.put()
|           |       |-- FactorManager.add()
|           |-- FactorManager._compute_talib()
|               |-- vectorbt.talib(...).run(...)
|               |-- FactorManager._select_output()
|               |-- FactorManager._slice_combo()
|           |-- FactorManager.CubeArtifact(P x T x N)
|           |-- FactorManager.FactorCache.put()
|           |-- FactorManager.add()
|
|-- Pre-defined
|   |-- FactorManager.get("HMA" / "SLOPE" / "CONVERGENCE" / ...)
|   |-- GetProxy.__call__() / GetProxy.single() / GetProxy.batch()
|       |-- FactorManager._get()
|           |-- FactorManager._asset_panel(cal_column)
|           |-- FactorManager._normalize_params()
|           |-- FactorManager.FactorCache.key()
|           |-- FactorManager.FactorCache.get()
|           |-- logical_processors > 1
|           |   |-- FactorManager._parallel_get()
|           |       |-- ProcessPoolExecutor / ThreadPoolExecutor
|           |       |-- _factor_worker()
|           |       |-- FactorManager.CubeArtifact(P x T x N)
|           |       |-- FactorManager.FactorCache.put()
|           |       |-- FactorManager.add()
|           |-- FactorManager._compute_customized()
|               |-- FactorLibrary._hma() -> TA-Lib WMA
|               |-- FactorLibrary._acc_return() -> pandas rolling product
|               |-- FactorLibrary._slope() -> njit kernel
|               |-- FactorLibrary._prev_diff() -> njit kernel
|               |-- FactorLibrary._prev_rollmax_safe() -> njit kernel
|               |-- FactorLibrary._roll_quantile() -> njit kernel
|               |-- FactorLibrary._convergence() -> njit kernel
|               |-- FactorLibrary._divergence() -> njit kernel
|           |-- FactorManager.CubeArtifact(P x T x N)
|           |-- FactorManager.FactorCache.put()
|           |-- FactorManager.add()
|
|-- Manually Defined
|   |
|   |-- already materialized panel/cube
|   |   |-- FactorManager.register(values = DataFrame / ndarray)
|   |       |-- FactorManager.CubeArtifact(P x T x N)
|   |       |-- FactorManager.add()
|   |
|   |-- VBT custom function
|       |-- FactorManager.register_vbt_factor(apply_func = callable)
|           |-- vectorbt.IndicatorFactory(...).from_apply_func(apply_func)
|           |-- indicator.run(...)
|           |-- FactorManager._slice_combo()
|           |-- FactorManager.CubeArtifact(P x T x N)
|           |-- FactorManager.add()
|
|-- retrieval path
|   |-- FactorManager.get_feature()
|   |-- FactorManager.get_cube()
|   |-- FactorManager.manifest()
```
"""
