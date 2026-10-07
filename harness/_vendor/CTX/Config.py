from ENV_MGMT.imports import *


# --- project root for path constants (sys.path bootstrap handled by harness._infra) ---
# Config.py lives at <project>/harness/_vendor/CTX/Config.py, so parents[3] is the
# harness project root. Upstream walked up to a monorepo root instead; that ancestor
# does not exist here, which silently left every path constant rooted at this FILE.
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


# --- load defaults (CTX.json co-located in this package) ---
_JSON_PATH = pathlib.Path(__file__).resolve().with_name("CTX.json")
if not _JSON_PATH.exists():

    raise FileNotFoundError(f"[CTX WARNING] CTX.json not found at {_JSON_PATH}")

with _JSON_PATH.open("r", encoding = "utf-8") as f:
    _DEFAULTS = json.load(f)

def _deep_config_merge(base: Optional[Dict[str, Any]], overlay: Optional[Dict[str, Any]]) -> Dict[str, Any]:

    merged = copy.deepcopy(base or {})

    for key, value in dict(overlay or {}).items():

        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_config_merge(cast(Dict[str, Any], merged[key]), value)

        else:
            merged[key] = copy.deepcopy(value)

    return merged


DEFAULT_DATA_CONFIG: Dict[str, Any] = dict(_DEFAULTS.get("Data_Config", {}))
DEFAULT_FACTOR_CONFIG: Dict[str, Any] = dict(_DEFAULTS.get("Factor_Config", {}))
DEFAULT_EXECUTION_CONFIG: Dict[str, Any] = dict(_DEFAULTS.get("Execution_Config", {}))
DEFAULT_SIGNAL_CONFIG: Dict[str, Any] = dict(_DEFAULTS.get("Signal_Config", {}))
DEFAULT_TRAVERSAL_CONFIG: Dict[str, Any] = dict(_DEFAULTS.get("Traversal_Config", {}))

DEFAULT_PARQUET_CATALOG: Dict[str, Any] = dict(DEFAULT_DATA_CONFIG.get("parquet_catalog", {}))

if not isinstance(DEFAULT_PARQUET_CATALOG, dict):
    DEFAULT_PARQUET_CATALOG = {}

DEFAULT_TIME_PROFILES: Dict[str, Any] = dict(DEFAULT_DATA_CONFIG.get("time_profiles", {}))

if not isinstance(DEFAULT_TIME_PROFILES, dict):
    DEFAULT_TIME_PROFILES = {}

# Fallback market-data lake, scanned only when a caller supplies no explicit parquet
# path (see CTX._resolve_parquet_paths). The harness always passes explicit paths, so
# this staying empty is the normal case.
DATA_ARCHIVES_ROOT = _PROJECT_ROOT / "data"
DEFAULT_PARQUET_PATTERNS: Dict[str, Tuple[str, ...]] = {
                                                        "MIN": ("*_MIN.parquet", "*_M1.parquet"),
                                                        "H": ("*_H.parquet",),
                                                        "D": ("*_D.parquet",),
                                                        "M": ("*_M.parquet",),
                                                        "ALTER": ("*_ALTER.parquet",),
                                                        }

SESSION_SIGNATURE_FIELDS: Tuple[str, ...] = (
                                            "region",
                                            "timezone",
                                            "session_calendar",
                                            "session_boundary",
                                            "regular_sessions",
                                            "breaks",
                                            "session_label",
                                            )
