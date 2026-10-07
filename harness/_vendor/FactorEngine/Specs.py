from ENV_MGMT.imports import *


# --- Indicator specs (FactorEngine.json co-located in this package) ---
_SPEC_PATH = pathlib.Path(__file__).resolve().with_name("FactorEngine.json")
if not _SPEC_PATH.exists():
    raise FileNotFoundError(f"[WARNING] FactorEngine.json not found at {_SPEC_PATH}")
with _SPEC_PATH.open("r", encoding="utf-8") as f:
    _SPECS_RAW = json.load(f)


INDICATOR_SPECS: Dict[str, Dict[str, Any]] = {k.upper(): v for k, v in _SPECS_RAW.items()}

# Startup invariant: the N-field input channel (`input_fields`) and the namespaced-kernel channel
# (`kernel_class`) are the SAME feature and must co-occur. A `kernel_class` spec with no `input_fields`
# would fail loud at compute time (the kernel needs materialized named panels); an `input_fields` spec
# with no `kernel_class` would route a legacy factor through the N-field cache key (extra_hashes) and
# silently break its byte-identity. Catch either misconfiguration at import, not deep in a backtest.
for _name, _spec in INDICATOR_SPECS.items():
    _has_fields = bool(_spec.get("input_fields"))
    _has_kernel = bool(_spec.get("kernel_class"))
    if _has_fields != _has_kernel:
        raise ValueError(f"[WARNING] FactorEngine spec {_name!r}: 'input_fields' and 'kernel_class' must co-occur (got input_fields={_has_fields}, kernel_class={_has_kernel}).")
