#================================= ZORA HARNESS (vendored) ==============================
# INFRASTRUCTURE (Backtest Engine)
# Single/Serial Underlyings | Portfolio BackTesting
#========================================================================================

# No import-time bootstrap lives here on purpose. The upstream package opened with a
# monorepo-root search plus a dotenv credential loader that reached OUTSIDE the
# repository; neither is wanted in a self-contained, publishable tree.
# `harness._infra.ensure_infra()` is the one supported bootstrap and it already puts
# `_vendor` on sys.path, which is what makes the sibling packages (FactorEngine, CTX)
# and the shared ENV_MGMT bundle resolvable from here.

# ---- public API re-exports (facade keeps `import BacktestEngine` stable) ----
from .Utils import plot_display_set, output_display_set, print_override_set, multi_log_set_top, multi_log_set_main
from .Calendars import default_holidays, default_macro_release_dates
from .Signal import Signal
from .Position import Position
from .ForLoopEngine import BacktestEngine_ForLoop
from .VBTEngine import BacktestEngine_VBT
from .PortfolioEngine import PortfolioEngine
from .Manager import BacktestEngineManager, trav_init_worker, trav_worker



__all__ = [
            'plot_display_set',
            'output_display_set',
            'multi_log_set_top',
            'multi_log_set_main',
            'print_override_set',

            'Signal',
            'Position',
            'BacktestEngine_ForLoop',
            'BacktestEngine_VBT',
            'BacktestEngineManager',
            'PortfolioEngine',

            'default_holidays',
            'default_macro_release_dates',
          ]
