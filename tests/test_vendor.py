"""Re-vendor guards: the deliberate carve-outs and the one local patch must survive.

``harness/_vendor`` is a hand-carved MINIMAL subset of the upstream engine plus a
small local patch set --- not a faithful snapshot. A blind "copy the new version
over" re-vendor silently undoes both, and the two most dangerous cases fail
*quietly*: dropping the ``periods_per_year`` override re-annualises every metric
from 252 trading days to a 365-calendar-day inference (flipping OOS Pass/Fail
verdicts), and leaving a renamed-away module behind keeps a stale import
resolving against dead code. These tests make either one fail loudly.

The tree is also DE-BRANDED: the vendored packages must read as Zora Harness
code, with no upstream project tokens left in any docstring, comment or
identifier, and no import-time bootstrap that reaches outside the repository.
"""
import inspect
import json
import pathlib

from harness import _infra, infra_engine

VENDOR = pathlib.Path(_infra.__file__).resolve().parent / "_vendor"

# Deleted on purpose (dead on the propose->backtest->verdict path); a re-vendor
# that copies the upstream facade verbatim re-adds eager imports of these.
# GeneticAlgo was never vendored at all, yet Manager.py kept three
# `from .GeneticAlgo import ...` sites gating ~480 lines of unreachable code --
# a live ModuleNotFoundError. Attribution/Neutralize were vendored but reachable
# from nothing except the facade's own re-export line.
_DELETED_MODULES = ("GeneticAlgo.py", "Research.py", "Scenarios.py")
_DELETED_EXPORTS = ("GeneticAlgo", "ResearchEngine", "ImpulseReactor",
                    "WarpCore", "MycelialNetwork")
_DELETED_FACTOR_MODULES = ("Attribution.py", "Neutralize.py")
_DELETED_FACTOR_EXPORTS = ("Attributor", "attribution", "cs_corr", "cs_ic",
                           "cs_exposure", "forward_returns", "Neutralizer",
                           "neutralize", "cs_residual", "cs_zscore", "cs_winsor",
                           "cs_group_demean")
# Renamed upstream; the old files must not linger, or the old bridge imports
# keep resolving and the harness silently backtests against the previous engine.
_RENAMED_AWAY = (("BacktestEngine", "SignalRegularization.py", "Position.py"),
                 ("BacktestEngine", "PortfolioBacktest.py", "PortfolioEngine.py"))


def _infra_ready() -> bool:
    """Bootstrap the vendored tree; False if it (or a runtime dep) is unavailable."""
    if not infra_engine.available():
        return False
    try:
        _infra.ensure_infra()
        import BacktestEngine  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True


def test_minimal_vendor_deletions_survive():
    for name in _DELETED_MODULES:
        assert not (VENDOR / "BacktestEngine" / name).exists(), (
            f"BacktestEngine/{name} was deliberately not vendored; a re-vendor "
            "re-added it (and its eager facade import)"
        )


def test_renamed_modules_replaced_not_duplicated():
    for pkg, stale, replacement in _RENAMED_AWAY:
        assert not (VENDOR / pkg / stale).exists(), (
            f"{pkg}/{stale} was renamed upstream and must be deleted, not kept "
            "alongside its replacement"
        )
        assert (VENDOR / pkg / replacement).exists()


def test_sidecar_json_follows_the_package_rename():
    # Config.py / Specs.py locate these with a hardcoded with_name("<Pkg>.json")
    # literal, so the rename has to hit the FILES, not just the in-code tokens.
    assert (VENDOR / "CTX" / "CTX.json").is_file()
    assert (VENDOR / "FactorEngine" / "FactorEngine.json").is_file()


def test_no_upstream_package_tokens_leaked():
    # The vendored tree is the upstream engine with its package tokens renamed to
    # functional names; a stray original token means a file escaped the rename.
    # Matched case-INSENSITIVELY: the leaks that survived the first pass were
    # `PROJECT BABYPACA` banners and `_babykitty_registry` / `_aubergine_log`
    # identifiers, none of which a case-sensitive check would have caught.
    # `harness/_infra.py` is in scope too -- it is the bootstrap a reader meets first.
    upstream = ("babypaca", "babykitty", "babyaubergine", "arcfleet",
                "fleet_operations", "fleet_archives")
    paths = list(VENDOR.rglob("*.py")) + [pathlib.Path(_infra.__file__).resolve()]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for token in upstream:
            assert token not in text, f"{path.name} still carries `{token}`"


def test_no_dotenv_bootstrap_reaches_outside_the_repo():
    """The package facades must not load credentials at import time.

    Each vendored ``__init__.py`` used to open with the same ~38-line block: walk
    the parents for a monorepo root, push it onto ``sys.path``, then read a ``.env``
    from OUTSIDE this repository and seed ``os.environ`` from it. A publishable tree
    cannot carry a credential loader 25 lines into ``import BacktestEngine``, and the
    supported bootstrap (``harness._infra.ensure_infra``) already puts ``_vendor`` on
    the path, so the block was pure liability.
    """
    for pkg in ("CTX", "FactorEngine", "BacktestEngine"):
        text = (VENDOR / pkg / "__init__.py").read_text(encoding="utf-8")
        # comments may still *describe* the removed block; only code counts
        code = "\n".join(ln for ln in text.splitlines()
                         if not ln.lstrip().startswith("#"))
        assert "_DOTENV_PATH" not in code, f"{pkg}/__init__.py re-added the .env loader"
        assert "os.environ" not in code, f"{pkg}/__init__.py mutates os.environ at import"
        assert "sys.path" not in code, f"{pkg}/__init__.py mutates sys.path at import"


def test_shared_imports_bundle_resolves_without_a_namespace_shim():
    """Every vendored module opens with the shared bundle; it must import by bare name.

    Upstream wrote ``from FLEET_OPERATIONS.ENV_MGMT.imports import *`` and the
    bootstrap faked a ``FLEET_OPERATIONS`` namespace module in ``sys.modules``. An
    injected ``sys.modules`` entry is NOT inherited by a spawned worker process
    (``sys.path`` is), so that form silently broke the multiprocessing paths.
    """
    for path in VENDOR.rglob("*.py"):
        if path.name == "__init__.py":
            continue
        first = path.read_text(encoding="utf-8").lstrip("﻿").splitlines()[0]
        if "ENV_MGMT" in first:
            assert first == "from ENV_MGMT.imports import *", f"{path.name}: {first!r}"


def test_shared_imports_bundle_carries_no_unused_heavy_deps():
    """Imports the engine never uses are install-time weight, not features.

    seaborn / IPython / deap / statsmodels / pyDOE were imported eagerly by the
    bundle and referenced by nothing that is vendored (deap only by the GA backend,
    which is gone). ``pyDOE`` additionally forced a ``pyDOE -> pyDOE3`` alias shim
    into ``_infra`` because the installed ``pyDOE`` build ships no top-level module.
    """
    text = (VENDOR / "ENV_MGMT" / "imports.py").read_text(encoding="utf-8")
    for dep in ("seaborn", "pyDOE", "IPython", "deap", "statsmodels"):
        assert dep not in text, f"ENV_MGMT/imports.py re-added `{dep}`"
    # ...and the alias shim it existed for must go with it.
    assert "pyDOE" not in pathlib.Path(_infra.__file__).read_text(encoding="utf-8")
    # the local additions the harness DOES depend on must survive a re-vendor
    for kept in ("to_1d", "TYPE_CHECKING", "BrokenExecutor", "queue", "importlib"):
        assert kept in text, f"ENV_MGMT/imports.py lost the local `{kept}` addition"


def test_zero_coverage_factor_modules_are_gone():
    """Attribution/Neutralize were reachable from nothing but the facade re-export."""
    for name in _DELETED_FACTOR_MODULES:
        assert not (VENDOR / "FactorEngine" / name).exists(), (
            f"FactorEngine/{name} is dead on every harness path; a re-vendor re-added it"
        )
    facade = (VENDOR / "FactorEngine" / "__init__.py").read_text(encoding="utf-8")
    for mod in ("Neutralize", "Attribution"):
        assert f"from .{mod} import" not in facade, (
            f"FactorEngine/__init__.py re-imports .{mod} --- `import FactorEngine` will fail"
        )


def test_reachable_modules_were_not_pruned():
    """The prune must stop at the call graph, not at a coverage number.

    ForLoopEngine and Calendars look zero-coverage (only class bodies run at
    import) but ``Manager.performance_metrics`` calls
    ``BacktestEngine_ForLoop.max_drawdown`` and the weight stage calls
    ``default_holidays``; Signal looks unused from the harness but Manager.py
    imports it at module scope. Deleting any of the three breaks every run.
    """
    for name in ("ForLoopEngine.py", "Calendars.py", "Signal.py", "Position.py"):
        assert (VENDOR / "BacktestEngine" / name).is_file(), (
            f"BacktestEngine/{name} is reachable from Manager.py's call graph; "
            "deleting it breaks every backtest"
        )
    manager = (VENDOR / "BacktestEngine" / "Manager.py").read_text(encoding="utf-8")
    assert "from .Signal import Signal" in manager
    assert "from .ForLoopEngine import BacktestEngine_ForLoop" in manager


def test_ctx_asset_token_mirror_is_complete():
    """harness.data mirrors CTX's filename rule; the sidecar is the source of truth."""
    from harness import data

    sidecar = json.loads(
        (VENDOR / "CTX" / "CTX.json").read_text(encoding="utf-8")
    )
    otc = sidecar["Data_Config"]["time_profiles"]["otc"]
    expected = {"EQT", "ETF", "INDEX"}                       # region-listed (hardcoded)
    expected |= {k.split("_", 1)[1] for k in otc}            # region-less OTC
    assert set(data._CTX_ASSET_TOKENS) == expected, (
        "CTX gained/lost an asset token; discover_parquet would hide a loadable "
        "file (or offer an unloadable one) in the TUI"
    )


def test_periods_per_year_patch_survives():
    """The local vendor patch absent upstream --- losing it re-annualises silently."""
    if not _infra_ready():
        return
    from BacktestEngine.Manager import BacktestEngineManager

    sig = inspect.signature(BacktestEngineManager.performance_metrics)
    assert "periods_per_year" in sig.parameters, (
        "the periods_per_year override is a LOCAL patch with no upstream "
        "counterpart; a copy-only re-vendor drops it and every Sharpe/Sortino/"
        "Calmar/CAGR silently switches to a 365-calendar-day clock"
    )
    assert sig.parameters["periods_per_year"].default is None


def test_trimmed_facade_has_no_deleted_exports():
    if not _infra_ready():
        return
    import BacktestEngine

    for name in _DELETED_EXPORTS:
        assert name not in BacktestEngine.__all__, (
            f"BacktestEngine.__all__ re-exports `{name}`, whose module is not "
            "vendored --- `import BacktestEngine` will fail"
        )
        assert not hasattr(BacktestEngine, name)


def test_factor_facade_has_no_deleted_exports():
    if not _infra_ready():
        return
    import FactorEngine

    for name in _DELETED_FACTOR_EXPORTS:
        assert name not in FactorEngine.__all__, (
            f"FactorEngine.__all__ re-exports `{name}` from a module that is no "
            "longer vendored --- `import FactorEngine` will fail"
        )
        assert not hasattr(FactorEngine, name)
    # the surface the harness actually consumes must be intact
    for kept in ("FactorManager", "FactorLibrary", "AutoParam", "GetProxy",
                 "clean_value", "INDICATOR_SPECS"):
        assert hasattr(FactorEngine, kept), f"FactorEngine lost `{kept}`"


def test_genetic_backend_is_gone_and_asking_for_it_fails_loudly():
    """`optimizer='ga'` must raise, not silently run the exhaustive grid.

    Before the prune, ``optimizer='ga'`` hit ``from .GeneticAlgo import ...`` --- a
    module that was never vendored --- so it died with ModuleNotFoundError. Deleting
    the dispatch without a guard would have made the same config quietly fall through
    to traversal and expand the FULL Cartesian product, burning hours of compute on a
    search nobody asked for. Loud beats slow-and-wrong.
    """
    if not _infra_ready():
        return
    from BacktestEngine.Manager import BacktestEngineManager

    manager_src = (VENDOR / "BacktestEngine" / "Manager.py").read_text(encoding="utf-8")
    assert "GeneticAlgo" not in manager_src, (
        "Manager.py imports .GeneticAlgo, which is not vendored --- that path raises "
        "ModuleNotFoundError the moment it is reached"
    )
    assert "_ga_backend" not in manager_src

    sig = inspect.signature(BacktestEngineManager.window_processing)
    assert "GA_Config" not in sig.parameters, (
        "window_processing still takes GA_Config for a backend that no longer exists"
    )

    try:
        BacktestEngineManager.window_processing(
            Data_Config={"test_data": [1]}, Factor_Config={},
            factor_param_ranges={"p": [1]},
            Execution_Config={"BacktestEngine": {"initial_cash": 1.0},
                              "Protocol": {"optimizer": "ga"}},
        )
    except ValueError as exc:
        assert "unsupported optimizer 'ga'" in str(exc), exc
    else:
        raise AssertionError("optimizer='ga' was accepted and silently ran traversal")


def test_ctx_payload_matches_the_window_processing_signature():
    """CTX.build_payload feeds window_processing; a stale key there is a TypeError.

    ``build_payload`` returns exactly the kwargs a caller splats into
    ``window_processing``. Dropping ``GA_Config`` from one side only would blow up at
    call time, so the two surfaces are pinned to each other here.
    """
    if not _infra_ready():
        return
    from BacktestEngine.Manager import BacktestEngineManager
    from CTX.CTX import CTX

    payload_params = set(inspect.signature(CTX.build_payload).parameters) - {"self"}
    wp_params = set(inspect.signature(BacktestEngineManager.window_processing).parameters)
    assert payload_params <= wp_params, (
        "CTX.build_payload emits key(s) window_processing does not accept: "
        f"{sorted(payload_params - wp_params)}"
    )
    assert "GA_Config" not in payload_params
