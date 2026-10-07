"""Interactive TUI --- expose every knob, run the harness, stream the log.

Layout is a **left rail + main pane**, not one tall column:

    +----------------------------------------------------------+
    |  Header                                                   |
    +--------------+-------------------------------------------+
    |  ZORA mark   |  [Data] [Windows] [Backtest] [Model] [Advanced] [?]    |
    |  MISSION     |  two field pairs per row (4-column grid),  |
    |  card        |  every field captioned underneath          |
    |  mode + run  |                                            |
    +--------------+-------------------------------------------+
    |  LOG (full width, live)                                   |
    +----------------------------------------------------------+

The MISSION card is a read-only digest of the form (provider, objective, Pass
Line, universe, search depth x breadth), so the current experiment is legible at
a glance without hunting across the pages. It is rebuilt from the widgets on any
change --- defensively, since a half-typed number must not crash the panel.

Every field carries a one-line caption *inside its own grid cell* (see
:func:`_field`), and the last tab renders ``README.md`` itself, so the manual is
one keystroke away (ctrl+g) instead of in another window.

The backtest runs on a worker thread so the UI never freezes; log lines are
marshalled back to the main thread.

Launch:  python -m harness.tui   (or:  python -m harness.cli tui)
"""
from __future__ import annotations

import os
import logging
import pathlib
import traceback

from rich.markup import escape
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Grid, Horizontal, Vertical, VerticalScroll
from textual.theme import Theme
from textual.widget import Widget
from textual.widgets import (
    Button, Footer, Header, Input, Label, Markdown, MarkdownViewer, RichLog,
    Select, Static, Switch, TabbedContent, TabPane,
)

from . import data as _data
from . import env
from . import memory
from . import objective as _obj
from . import palette as _pal
from . import preflight
from .config import ASSET_PRESETS, RunConfig
from .cancellation import (RunCancelled, RunControl, cancellation_scope,
                           check_cancelled)
from .parallel import pool_for
from .providers import PROVIDERS, close_provider, get_provider
from .providers.cli_base import redact_secrets
from .providers.codex_controls import model_defaults, reasoning_levels
from .runner import check_data_coverage, _journal_snapshot, run_once, run_walk_forward
from .store import Store

# --- decoration --------------------------------------------------------------
# Box-drawing wordmark (the terminal already renders box characters for every
# Textual border, so this needs no capability the app doesn't already require).
BANNER = "╔═╗ ╔═╗ ╦═╗ ╔═╗\n╔═╝ ║ ║ ╠╦╝ ╠═╣\n╚═╝ ╚═╝ ╩╚═ ╩ ╩"
SPARK = "▁▁▂▃▂▄▅▄▆▅▇▆█▇▆█▇"
TAGLINE = "words → factor → verdict"      # short enough for the narrow rail too

# One hue, three brightnesses (see harness/palette.py). Everything structural is
# gold-on-grey; only PASS/FAIL-shaped meaning gets its own colour.
ZORA_THEME = Theme(
    name="zora",
    background=_pal.INK,
    surface=_pal.SLATE,
    panel=_pal.STEEL,
    foreground=_pal.PARCHMENT,
    secondary=_pal.BRONZE,   # recessive frames, the sparkline
    primary=_pal.BRASS,      # active border, focus, the Run bar
    accent=_pal.GOLD,        # wordmark, border titles
    success=_pal.OK,
    warning=_pal.WARN,
    error=_pal.BAD,
    dark=True,
    variables={
        # Textual's $text defaults to "auto 87%" --- an auto-CONTRAST colour, i.e.
        # near-white on a dark app. Almost everything reads $foreground and never
        # notices, but the Guide's markdown does: fence bodies, bold runs and
        # links all came out pure #ffffff (and comments as a cold #a0a0a2 grey).
        # Pin it to the same warm parchment as the body text.
        "text": _pal.PARCHMENT,
        "text-muted": _pal.MUTED,
        "footer-key-foreground": _pal.GOLD,
        "block-cursor-background": _pal.BRASS,
        "block-cursor-foreground": _pal.INK,
        "input-cursor-background": _pal.BRASS,
        "input-cursor-foreground": _pal.INK,
        "input-selection-background": f"{_pal.BRASS} 35%",
        # captioned fields and a 374-line README both scroll; Textual would
        # derive the bar from $panel (a cool grey) --- put it on the gold ladder
        "scrollbar": _pal.BRONZE,
        "scrollbar-hover": _pal.BRASS,
        "scrollbar-active": _pal.GOLD,
        "scrollbar-background": _pal.INK,
        "scrollbar-corner-color": _pal.INK,
        "scrollbar-background-hover": _pal.INK,
        "scrollbar-background-active": _pal.INK,
    },
)


def _paths_str(paths) -> str:
    """Render config.parquet_daily (str | list | None) as an editable field."""
    if not paths:
        return ""
    if isinstance(paths, (list, tuple)):
        return ", ".join(str(p) for p in paths)
    return str(paths)


def _parse_paths(text: str):
    """Parse the parquet field back: "" -> None, one path -> str, many -> list."""
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else parts


def _symbols_field(symbols, asset_class: str) -> str:
    """Render the symbols box: show 'all' when the list is exactly the selected
    asset class's full basket (the clean default), else the explicit tickers."""
    preset = ASSET_PRESETS.get(asset_class)
    if preset is not None and list(symbols) == list(preset):
        return "all"
    return ", ".join(symbols)


def _parse_symbols(text: str):
    """Parse the symbols box: a lone 'all' (any case) -> the ['all'] sentinel
    (RunConfig expands it to the asset class's basket); else an explicit list."""
    parts = [x.strip() for x in text.split(",") if x.strip()]
    if len(parts) == 1 and parts[0].lower() == "all":
        return ["all"]
    return parts


def _clip(text, width: int) -> str:
    """One-line, width-bounded value for the narrow MISSION card."""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _field(label: str, widget: Widget, note: str, *, wide: bool = False
           ) -> ComposeResult:
    """One ``label | field + caption`` cell of a form grid.

    The caption sits *under the field, inside the same grid cell* --- not in the
    label column (which is ``auto``-width and would stretch to the longest
    sentence, shoving the inputs off the right edge) and not in a row of its own
    (which would divorce it from the pair it explains). Row gutter is 0 because
    the caption line now does the separating, so documenting every field costs
    ~1 row per row of form rather than doubling its height.

    ``wide`` spans the field across the second pair's columns --- for the boxes
    (symbols, parquet paths, API key) whose value is a long string. The span
    lives on the *cell*, not the widget, since the cell is what the grid places.
    """
    yield Label(label)
    with Vertical(classes="cell wide" if wide else "cell"):
        yield widget
        yield Static(note, classes="note")


def _guide_path() -> pathlib.Path:
    """The README rendered by the Guide tab (harness root, next to pyproject)."""
    return pathlib.Path(__file__).resolve().parents[1] / "README.md"


def _read_guide() -> str:
    """README text, or a note saying why it isn't there --- never an exception.

    The Guide is a convenience; a missing or unreadable file must not stop the
    TUI from starting (an installed harness may ship without its README).
    """
    path = _guide_path()
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 -- missing, unreadable, bad encoding
        return f"# Guide unavailable\n\nCould not read `{path}`:\n\n    {exc}\n"


class GuideViewer(MarkdownViewer):
    """README viewer that will not navigate out of the README.

    Stock :class:`MarkdownViewer` follows every link it is given: an ``https://``
    href opens a browser, and a relative one (this README has
    ``[harness/palette.py](harness/palette.py)``) is *loaded as markdown*,
    replacing the guide with a Python file and no way back. Anchors are the only
    links that make sense here, so those jump and everything else is reported.
    """

    async def _on_markdown_link_clicked(self, message: Markdown.LinkClicked
                                        ) -> None:
        message.stop()
        href = message.href
        if href.startswith("#"):
            self.document.goto_anchor(href[1:])
            return
        log = getattr(self.app, "_log", None)
        if log is not None:
            log(f"{_note('link not followed:')} {href}")


# Log tags. RichLog parses *Rich* markup, which knows terminal colour names but
# not Textual's `$variables`, so a bare colour-NAME tag would paint whatever green
# the terminal happens to have -- outside the palette entirely. Tag with the
# palette hexes instead, so the log obeys the same design language as the chrome.
def _ok(text: str) -> str:
    return f"[{_pal.OK}]{text}[/]"


def _warn(text: str) -> str:
    return f"[{_pal.WARN}]{text}[/]"


def _bad(text: str) -> str:
    return f"[{_pal.BAD}]{text}[/]"


def _note(text: str) -> str:
    return f"[{_pal.NOTE}]{text}[/]"


class _TUILogHandler(logging.Handler):
    """Send process-local logging records into Textual's RichLog."""

    def __init__(self, app):
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        try:
            text = escape(self.format(record))
            self._app.call_from_thread(self._app._log, text)
        except (AttributeError, RuntimeError):
            # App may be shutting down while a worker flushes its final record.
            pass


class HarnessTUI(App):
    TITLE = "ZORA Harness"
    SUB_TITLE = "natural language -> factor -> backtest -> Pass/Fail"

    # One design language: greys carry the structure, ONE gold carries emphasis,
    # and depth is luminance ($secondary frame -> $primary active -> $accent title).
    # No selector below introduces a hue that isn't in harness/palette.py.
    CSS = """
    Screen { layout: vertical; }
    Header { background: $surface; color: $accent; }
    Footer { background: $surface; }

    /* ---- top half: rail + main, side by side ---------------------------- */
    #body { height: 1fr; }

    #rail {
        width: 46;
        padding: 0 2;
        background: $surface;
        border-right: vkey $secondary;
        layout: vertical;
        overflow: hidden;
    }
    #rail-content { height: 1fr; layout: vertical; overflow-y: auto; }
    #mission-spacer, #decision-spacer { height: 1; min-height: 1; }
    #rail.compact #mission-spacer,
    #rail.compact #decision-spacer { display: none; }
    #brand    { color: $accent; text-style: bold; height: auto; }
    #spark    { color: $secondary; height: 1; }
    #tagline  { color: $text-muted; height: auto; padding: 0; }
    #status {
        height: auto; padding: 0 2; margin: 1 0 0 0;
        border: round $secondary;
        border-title-color: $accent;
        border-subtitle-color: $text-muted;
        text-wrap: nowrap; text-overflow: ellipsis;
    }
    #decision {
        layout: vertical; height: auto; padding: 0 2; margin: 0;
        border: round $secondary;
        border-title-color: $accent;
    }
    #decision .decision-row { height: 3; width: 1fr; align: left middle; margin: 0; }
    #decision .objective-row { margin-bottom: 1; }
    #decision .decision-row > Label { width: 11; color: $text-muted; }
    #decision Select, #decision Input { width: 1fr; height: 3; min-height: 3; }
    #decision Input {
        background: $panel; color: $foreground; border: tall $secondary;
    }
    #decision Select > SelectCurrent {
        background: $panel; color: $foreground; border: tall $secondary;
    }
    #decision Input:focus, #decision Select:focus > SelectCurrent {
        background: $panel; border: tall $primary;
    }
    #interpretability-control { height: 3; align: left middle; }
    #interpretability-control > Label {
        width: 1fr; min-width: 0; height: 3; content-align: left middle;
        padding: 0 0 0 1; color: $foreground;
    }
    #interpretability-control Switch {
        width: 10; min-width: 10; height: 3; min-height: 3;
        padding: 0 1; margin: 0; background: $panel; border: tall $secondary;
    }
    #interpretability-control Switch.-on { border: tall $primary; }
    #interpretability-control Switch > .switch--slider {
        color: $secondary; background: $panel-darken-2;
    }
    #interpretability-control Switch.-on > .switch--slider {
        color: $primary; background: $panel;
    }
    /* The card scrolls independently while this footer stays reachable on a
       short terminal. */
    #actions  { layout: grid; grid-size: 5; grid-gutter: 0 1;
                height: 3; padding: 0; margin-bottom: 0; background: $surface; }
    /* The wide rail gives every action one equal, readable cell. */
    #actions Button {
        width: 1fr; height: 3; min-width: 7; border: none; text-style: bold;
        content-align: center middle;
        background: $panel; color: $foreground;
    }
    #actions Button#run   { background: $primary; color: $background; }
    #actions Button#quit  { color: $error; }
    #actions Button:hover { background: $accent; color: $background; }
    #actions Button:focus { text-style: bold underline; }
    #rail.narrow #actions { grid-size: 4; height: 6; }
    #rail.narrow #run { column-span: 4; }
    #rail.narrow #status {
        padding: 0; text-wrap: nowrap; text-overflow: ellipsis;
    }
    #rail.narrow.spacious #status,
    #rail.narrow.spacious #decision { padding-top: 0; }
    #rail.spacious #status, #rail.spacious #decision {
        padding-top: 1; padding-left: 2; padding-right: 2; padding-bottom: 0;
    }
    #rail.spacious .objective-row { margin-bottom: 1; }

    #main { width: 1fr; }
    TabbedContent { height: 1fr; }
    .underline--bar { color: $primary; }
    /* the stock toggle is green; make it the same gold as every other "on" */
    Switch > .switch--slider { color: $panel; }
    Switch.-on > .switch--slider { color: $primary; }

    /* ---- forms: two label/field PAIRS per row --------------------------- */
    /* a captioned form is taller than the pane on a short terminal, so the
       pane scrolls rather than silently cropping the last row */
    TabPane { height: 1fr; overflow-y: auto; }
    .hint {
        color: $text-muted; width: 1fr; height: auto;
        padding: 0 1; margin: 0 2; border-left: thick $secondary;
    }
    .form {
        layout: grid; grid-size: 4; grid-columns: auto 1fr auto 1fr;
        grid-gutter: 0 1; padding: 0 2; height: auto;
    }
    .form Label { padding: 1 0 0 0; color: $text-muted; text-align: right; }
    .form Button { background: $panel; color: $foreground;
                   border: tall $panel; text-style: none; }
    .form Button:hover { background: $accent; color: $background;
                         border: tall $accent; }
    /* the field + its caption; the cell is what the grid places, so the span
       and the row height both belong here rather than on the input */
    .cell { height: auto; }
    /* no vertical padding: with the ROW gutter at 0 (grid-gutter: 0 1) the
       caption takes its place, so a fully documented form is the same height as the
       undocumented one was. The next field's top edge does the separating.

       A caption wraps rather than truncates: the text IS the point, and one
       extra line only makes its pane scroll. */
    .note { color: $text-muted; height: 1; padding: 0 1;
            text-wrap: nowrap; text-overflow: ellipsis; }
    /* ... except inside a spanning cell. Textual sizes an `auto` grid row by
       measuring each cell at ONE column's width even when that cell spans
       three, so a wrapping caption there is measured against the ~11-column
       label track, reports ~10 lines, and ballooned the Symbols row to 13 rows
       high. Kept to a single line it is 4 rows at any measured width. */
    .wide .note { text-wrap: nowrap; text-overflow: ellipsis; }
    /* narrow mode un-spans every cell (below), so wrapping is safe again */
    .form.narrow .wide .note { text-wrap: wrap; height: auto; }
    .wide { column-span: 3; }
    /* narrow terminal: fall back to one label/field pair per row (see _reflow) */
    .form.narrow { grid-size: 2; grid-columns: auto 1fr; }
    .form.narrow .wide { column-span: 1; }
    Select, Input { width: 1fr; }
    Switch { width: auto; }
    Input:disabled, Button:disabled, Select:disabled, Switch:disabled {
        opacity: 0.4;
    }

    /* ---- the Guide tab: README.md, rendered ----------------------------- */
    /* the viewer scrolls itself, so the pane gives it the whole rectangle */
    #tab-guide { padding: 0; overflow: hidden; }
    #guide { background: $surface; }
    /* the stock contents sidebar is $panel with no edge --- give it the same
       vertical keyline the left rail uses, so the app has one kind of divider */
    #guide MarkdownTableOfContents,
    #guide MarkdownTableOfContents > Tree {
        background: $surface; max-width: 38;
    }
    #guide MarkdownTableOfContents { border-right: vkey $secondary; }
    /* code fences hard-code rgb(210,210,210); their syntax spans already come
       from theme variables, so this is the last off-palette colour in the tab */
    #guide MarkdownFence, #guide MarkdownFence Label { color: $foreground; }

    /* ---- bottom: the log, full width ------------------------------------ */
    #log {
        height: 30%; min-height: 7; max-height: 14;
        border: round $secondary; background: $surface;
        border-title-color: $accent;
        border-subtitle-color: $text-muted;
        padding: 0 1; margin: 0 2 1 2;
    }
    """

    BINDINGS = [
        ("ctrl+r", "run", "Run"),
        ("ctrl+d", "doctor", "Check setup"),
        ("ctrl+s", "save", "Save config"),
        ("ctrl+l", "load", "Load config"),
        ("ctrl+g", "guide", "Guide"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, config: RunConfig | None = None,
                 config_path: str = "run_config.json",
                 artifacts_root: str = "artifacts",
                 memory_root: str = "memory"):
        super().__init__()
        self._cfg0 = config or RunConfig()
        self._config_path = config_path
        self._artifacts = artifacts_root
        self._memory = memory_root
        self._log_handler = None
        self._run_control: RunControl | None = None
        self._quit_requested = False
        self._ui_closed = False

    def _install_log_handler(self) -> None:
        if self._log_handler is not None:
            return
        handler = _TUILogHandler(self)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s: %(message)s"))
        logging.getLogger().addHandler(handler)
        logging.captureWarnings(True)
        self._log_handler = handler

    def _remove_log_handler(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        logging.captureWarnings(False)

    # --- layout ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            yield from self._compose_rail()
            with Vertical(id="main"):
                yield from self._compose_tabs()
        yield RichLog(id="log", highlight=True, markup=True)
        yield Footer()

    def _compose_rail(self) -> ComposeResult:
        c = self._cfg0
        with Vertical(id="rail"):
            with VerticalScroll(id="rail-content"):
                yield Static(BANNER, id="brand")
                yield Static(SPARK, id="spark")
                yield Static(TAGLINE, id="tagline")
                yield Static(id="status")
                yield Static("", id="mission-spacer")
                with Vertical(id="decision"):
                    with Horizontal(classes="decision-row objective-row"):
                        yield Label("Objective")
                        yield Select([(_obj.label(n), n) for n in _obj.names()],
                                     value=c.objective, allow_blank=False,
                                     id="objective")
                    with Horizontal(classes="decision-row"):
                        yield Label("Pass Line")
                        yield Input(str(c.pass_line), id="pass_line", type="number")
                    with Horizontal(id="interpretability-control"):
                        yield Switch(c.require_interpretability, id="interp",
                                     tooltip="Require a structured economic mechanism")
                        yield Label("Interpretability")
                yield Static("", id="decision-spacer")
            # Keep primary actions reachable while the card above scrolls on a
            # short terminal.
            with Grid(id="actions"):
                yield Button("Run", id="run", variant="success",
                             tooltip="Run the harness (ctrl+r)")
                yield Button("Check", id="doctor",
                             tooltip="Provider key / login readiness (ctrl+d)")
                yield Button("Save", id="save", variant="primary",
                             tooltip="Write the form to the config path (ctrl+s)")
                yield Button("Load", id="load",
                             tooltip="Read the config path into the form (ctrl+l)")
                yield Button("Quit", id="quit", variant="error",
                             tooltip="Leave the TUI (ctrl+q)")

    def _compose_tabs(self) -> ComposeResult:
        c = self._cfg0
        with TabbedContent(initial="tab-data"):
            with TabPane("Data", id="tab-data"):
                yield Label("The price panel every backtest runs on. Data dir / "
                            "Scan / Parquet apply only to source = ctx.",
                            classes="hint")
                with Grid(classes="form"):
                    yield from _field(
                        "Source",
                        Select([("synthetic", "synthetic"),
                                ("yfinance", "yfinance"),
                                ("ctx parquet", "ctx")],
                               value=c.data_source, allow_blank=False,
                               id="data_source"),
                        "the price source")
                    yield from _field(
                        "Asset class",
                        Select([(k, k) for k in ASSET_PRESETS],
                               value=c.asset_class, allow_blank=False,
                               id="asset_class"),
                        "a preset universe")

                    yield from _field(
                        "Symbols",
                        Input(_symbols_field(c.symbols, c.asset_class),
                              id="symbols",
                              placeholder="'all' for the whole asset class, "
                                          "or AAPL, MSFT, ..."),
                        "'all' = the whole asset class, or an explicit list",
                        wide=True)

                    yield from _field(
                        "Data dir",
                        Input(c.data_dir, id="data_dir",
                              placeholder="folder scanned for CTX parquet"),
                        "Scan looks here")
                    yield from _field(
                        " ", Button("Scan for parquet", id="scan"),
                        "fills Parquet below")

                    yield from _field(
                        "Parquet",
                        Input(_paths_str(c.parquet_daily), id="parquet_daily",
                              placeholder="auto-filled from Data dir when ctx"),
                        "Data_<TYPE>_<REGION>_D*.parquet, relative to harness root",
                        wide=True)

            with TabPane("Windows", id="tab-windows"):
                yield Label(
                    "Clock: walk or single. OOS: Advanced.",
                    classes="hint")
                with Grid(classes="form window-form"):
                    yield from _field(
                        "Walk-forward",
                         Switch(c.walk_forward, id="walk_forward",
                               tooltip="Repeat at every T_n from Walk T_0 to Walk T_p"),
                         "repeat each step")
                    yield from _field(
                        "Research T_n",
                        Input(c.research_date, id="research_date",
                              placeholder="YYYY-MM-DD"),
                        "single-date today")

                    yield from _field(
                        "Frequency",
                        Input(c.frequency, id="frequency",
                              placeholder="YS / QS / MS / W"),
                        "step: YS/QS/MS/W/D")

                    yield from _field(
                        "IS years",
                        Input(str(c.is_years), id="is_years", type="integer"),
                        "in-sample lookback")
                    yield from _field(
                        "IS start",
                        Input(c.is_start or "", id="is_start",
                              placeholder="blank = use clock"),
                        "override, both ends")
                    yield from _field(
                        "IS end",
                        Input(c.is_end or "", id="is_end",
                              placeholder="blank = use clock"),
                        "override, both ends")

                    yield from _field(
                        "OOS start",
                        Input(c.oos_start or "", id="oos_start",
                              placeholder="blank = use clock"),
                        "override, both ends")
                    yield from _field(
                        "OOS end",
                        Input(c.oos_end or "", id="oos_end",
                              placeholder="blank = use clock"),
                        "override, both ends")

                    yield from _field(
                        "Walk T_0",
                        Input(c.t_0, id="t_0", placeholder="YYYY-MM-DD"),
                        "walk starts here")
                    yield from _field(
                        "Walk T_p",
                        Input(c.t_p, id="t_p", placeholder="YYYY-MM-DD"),
                        "walk ends here")

            with TabPane("Backtest", id="tab-backtest"):
                yield Label(
                    "How each formula becomes a portfolio. Costs, exposure, "
                    "signal selection and rebalance policy are explicit here.",
                    classes="hint")
                with Grid(classes="form"):
                    yield from _field(
                        "Cost (bps)",
                        Input(str(c.cost_bps), id="cost_bps", type="number"),
                        "per unit turnover")
                    yield from _field(
                        "Gross",
                        Input(str(c.gross), id="gross", type="number"),
                        "gross exposure")
                    yield from _field(
                        "Initial cash",
                        Input(str(c.initial_cash), id="initial_cash",
                              type="number"),
                        "starting equity")
                    yield from _field(
                        "Sign check",
                        Switch(c.require_sign_consistency, id="sign"),
                        "IS/OOS signs agree")
                    yield from _field(
                        "Selection",
                        Select([("None", "none"),
                                ("TopQ", "top_q"),
                                ("TopK", "top_k")],
                               value=c.selection_mode, allow_blank=False,
                               id="selection_mode"),
                        "all, quantile or K")
                    yield from _field(
                        "Top Q",
                        Input(str(c.top_q), id="top_q", type="number"),
                        "fraction per side")
                    yield from _field(
                        "Top K",
                        Input(str(c.top_k), id="top_k", type="integer"),
                        "names per side")
                    yield from _field(
                        "Hold every",
                        Input(str(c.hold_every), id="hold_every", type="integer"),
                        "bars retained")
                    yield from _field(
                        "Rebalance every",
                        Input(str(c.rebalance_every), id="rebalance_every",
                              type="integer"),
                        "bars between entries")
                    yield from _field(
                        "Flatten delisted",
                        Switch(c.close_delisted_at_last, id="close_delisted"),
                        "flatten at last bar")

            with TabPane("Model", id="tab-model"):
                yield Label(f"Math contract: {c.math_contract}. Hard admission; economic mechanism remains a hypothesis.",
                            classes="hint", id="math_contract_label")
                yield Label("API providers need a key; CLI providers need login; scripted is offline.",
                            classes="hint", id="provider_controls")
                with Grid(classes="form model-form"):
                    yield from _field(
                        "Provider",
                        Select([(p, p) for p in PROVIDERS],
                               value=c.provider, allow_blank=False,
                               id="provider"),
                        "who writes formulas")
                    yield from _field(
                        "Model id", Input(c.model, id="model"),
                        "the model to call")

                    yield from _field(
                        "API key",
                        Input("", id="api_key", password=True,
                              placeholder="paste, then Save"),
                        "never echoed")
                    yield from _field(
                        " ", Button("Save key to .env", id="save_key"),
                        "-> .env, no restart")

                    yield from _field(
                        "Depth (iters)",
                        Input(str(c.max_iters), id="max_iters", type="integer"),
                        "refine rounds/date")
                    yield from _field(
                        "Breadth (K)",
                        Input(str(c.candidates_per_round),
                              id="candidates_per_round", type="integer"),
                        "proposals per round")

                    yield from _field(
                        "Workers",
                        Input(str(c.n_jobs), id="n_jobs", type="integer"),
                        "processes; needs K>1")
                    yield from _field(
                        "Reasoning",
                        Select([("Default?", "")] + [(v, v) for v in
                               reasoning_levels(c.model)],
                               value=c.reasoning_effort or "", allow_blank=False,
                               disabled=c.provider != "codex", id="reasoning_effort"),
                        "default unverified")

                    yield from _field(
                        "Research memory",
                        Switch(c.memory, id="memory"),
                        "read/write memory/")

                    yield from _field(
                        "Run name", Input(c.run_name, id="run_name"),
                        "artifacts/<name>/")

            with TabPane("Advanced", id="tab-advanced"):
                yield Label(
                    "Guardrails and automation. The runner computes the required "
                    "data window and fails before a truncated walk can start.",
                    classes="hint")
                with Grid(classes="form"):
                    yield from _field(
                        "Data start",
                        Input(c.data_start, id="data_start",
                              placeholder="YYYY-MM-DD"),
                        "earlier is allowed")
                    yield from _field(
                        "Data end",
                        Input(c.data_end, id="data_end",
                              placeholder="YYYY-MM-DD"),
                        "must cover full OOS")

                    yield from _field(
                        "OOS mode",
                        Select([("per research step", "per_step"),
                                ("fixed business days", "fixed_days")],
                               value=c.oos_mode, allow_blank=False,
                               id="oos_mode"),
                        "uses real panel bars")
                    yield from _field(
                        "Warmup bars",
                        Input(str(c.warmup), id="warmup", type="integer"),
                        "rolling pre-roll")

                    yield from _field(
                        "Fixed OOS days",
                        Input(str(c.oos_days), id="oos_days", type="integer"),
                        "fixed mode only")
                    yield from _field(
                        "Annualization",
                        Input(str(c.annualization), id="annualization",
                              type="integer"),
                        "periods per year")

                    yield from _field(
                        "Min IS days",
                        Input(str(c.min_is_days), id="min_is_days",
                              type="integer"),
                        "minimum live IS bars")
                    yield from _field(
                        "Min OOS days",
                        Input(str(c.min_oos_days), id="min_oos_days",
                              type="integer"),
                        "minimum valid OOS bars")
                    yield from _field(
                        "Allow missing",
                        Switch(c.allow_missing_symbols, id="allow_missing_symbols"),
                        "survivor universe only")

                    yield from _field(
                        "Tri alignment",
                        Switch(c.tri_align, id="tri_align"),
                        "annotations; math on")
                    yield from _field(
                        "Max tokens",
                        Input(str(c.max_tokens), id="max_tokens", type="integer"),
                        "provider output budget")
                    yield from _field(
                        "Seed", Input(str(c.seed), id="seed", type="integer"),
                        "local data/engine only")

                    yield from _field(
                        "Retry backoff",
                        Input(str(c.retry_backoff), id="retry_backoff", type="number"),
                        "seconds before retry")
                    yield from _field(
                        "Retry ceiling",
                        Input(str(c.retry_max_delay), id="retry_max_delay",
                              type="number"),
                        "maximum retry wait")

                    yield from _field(
                        "Temperature",
                        Input(str(c.temperature), id="temperature", type="number"),
                        "sampling control")
                    yield from _field(
                        "Config path", Input(self._config_path, id="config_path"),
                        "Save / Load target")

            # README.md itself, so the manual is a keystroke away (ctrl+g). The
            # text is loaded on first view, not at startup: parsing 20 kB of
            # markdown into widgets is not worth paying for a session that never
            # opens the tab.
            with TabPane("Guide", id="tab-guide"):
                yield GuideViewer(id="guide", show_table_of_contents=True,
                                  open_links=False)

    def on_mount(self) -> None:
        self._install_log_handler()
        self.register_theme(ZORA_THEME)
        self.theme = "zora"
        log = self.query_one("#log", RichLog)
        log.border_title = "LOG"
        log.border_subtitle = "idle"
        self.query_one("#status", Static).border_title = "MISSION"
        self.query_one("#decision", Vertical).border_title = "DECISION"
        self._gate_windows(self.query_one("#walk_forward", Switch).value)
        self._gate_oos_mode(str(self.query_one("#oos_mode", Select).value))
        self._gate_selection(str(self.query_one("#selection_mode", Select).value))
        self._controls_provider = self._cfg0.provider
        self._sync_provider_controls()
        self._reflow(self.size.width, self.size.height)
        self._sync_status()

    def on_unmount(self) -> None:
        self._ui_closed = True
        if self._run_control is not None:
            self._run_control.cancel()
        self._remove_log_handler()

    # --- mode gating ---------------------------------------------------------
    #: read only when the walk-forward switch is ON --- ``_evolution_dates``
    #: builds the schedule from these and nothing else touches them.
    _WALK_ONLY = ("t_0", "t_p", "frequency")
    #: read only when it is OFF --- ``runner._step_config`` overwrites
    #: ``research_date`` per stop and clears all four explicit overrides, so on a
    #: walk they are dead inputs that silently look live.
    _SINGLE_ONLY = ("research_date", "is_start", "is_end", "oos_start", "oos_end")

    @on(Switch.Changed, "#walk_forward")
    def _on_walk_forward(self, event: Switch.Changed) -> None:
        self._gate_windows(event.value)

    def _gate_windows(self, walking: bool) -> None:
        """Grey out whichever window fields the current mode does not read."""
        for wid in self._WALK_ONLY:
            self._set_disabled(wid, not walking)
        for wid in self._SINGLE_ONLY:
            self._set_disabled(wid, walking)

    @on(Select.Changed, "#oos_mode")
    def _on_oos_mode(self, event: Select.Changed) -> None:
        self._gate_oos_mode(str(event.value))

    def _gate_oos_mode(self, mode: str) -> None:
        # A per-step walk derives its holdout from the next research stop; showing
        # a live fixed-day input beside it invites a false sense of control.
        self._set_disabled("oos_days", mode == "per_step")

    @on(Select.Changed, "#selection_mode")
    def _on_selection_mode(self, event: Select.Changed) -> None:
        self._gate_selection(str(event.value))

    def _gate_selection(self, mode: str) -> None:
        self._set_disabled("top_q", mode != "top_q")
        self._set_disabled("top_k", mode != "top_k")

    def _set_disabled(self, wid: str, disabled: bool) -> None:
        try:
            self.query_one(f"#{wid}").disabled = disabled
        except Exception:  # noqa: BLE001 -- not mounted yet
            pass

    # --- the Guide tab -------------------------------------------------------
    _guide_loaded = False

    def action_guide(self) -> None:
        """Jump to the rendered README (ctrl+g)."""
        try:
            self.query_one(TabbedContent).active = "tab-guide"
        except Exception as exc:  # noqa: BLE001
            self._log(f"{_bad('cannot open the guide:')} {exc}")

    @on(TabbedContent.TabActivated)
    def _on_tab(self, event: TabbedContent.TabActivated) -> None:
        """Reading mode: the rail steps aside while the Guide is open.

        Prose plus a contents sidebar does not fit in the 78 columns left over
        after the rail, and nothing in the rail is needed to read the manual ---
        the Footer keys (Run, Check, Save, Load) all still work. Switching to any
        other tab brings it straight back.
        """
        reading = getattr(event.pane, "id", None) == "tab-guide"
        self.query_one("#rail").display = not reading
        self._reflow(self.size.width, self.size.height)

    @on(TabbedContent.TabActivated, pane="#tab-guide")
    async def _on_guide_shown(self) -> None:
        """Render README.md the first time the tab is looked at."""
        if self._guide_loaded:
            return
        self._guide_loaded = True          # set first: a failed parse must not
        viewer = self.query_one("#guide", GuideViewer)   # retry on every visit
        try:
            await viewer.document.update(_read_guide())
        except Exception as exc:  # noqa: BLE001
            self._log(f"{_bad('could not render the guide:')} {exc}")
            return
        self._reflow(self.size.width, self.size.height)  # TOC obeys the width

    # --- responsive layout ---------------------------------------------------
    #: below this width there is not enough room for two label/field pairs per
    #: row once the rail is subtracted, so the forms fall back to one pair.
    WIDE_AT = 112
    #: below this height the rail cannot show its decoration AND the whole
    #: MISSION card, and the card is the part that carries information.
    TALL_AT = 44
    #: the Guide's contents sidebar costs ~34 columns; below this the README
    #: would be left a column too thin to read, so the prose wins. Measured
    #: against the full width because reading mode has already hidden the rail.
    TOC_AT = 84

    def on_resize(self, event) -> None:
        self._reflow(event.size.width, event.size.height)

    def _reflow(self, width: int, height: int) -> None:
        """Pick rail width, form density and decoration for the terminal size."""
        narrow = width < self.WIDE_AT
        try:
            rail = self.query_one("#rail")
            rail.styles.width = 38 if narrow else 46
            rail.set_class(narrow, "narrow")
            rail.set_class(height < self.TALL_AT, "compact")
            rail.set_class(height >= self.TALL_AT, "spacious")
        except Exception:  # noqa: BLE001 -- not mounted yet
            return
        for form in self.query(".form"):
            form.set_class(narrow, "narrow")
        for viewer in self.query("#guide"):
            viewer.show_table_of_contents = width >= self.TOC_AT
        roomy = height >= self.TALL_AT
        for wid in ("#spark", "#tagline"):
            self.query_one(wid, Static).display = roomy

    # --- the MISSION card ----------------------------------------------------
    def _peek(self, wid: str, default: str = "—") -> str:
        """Current value of a widget as display text, tolerant of anything.

        The card refreshes on every keystroke, so it must survive a widget that
        is not mounted yet, a Select with no selection, and a half-typed number.
        """
        try:
            value = getattr(self.query_one(f"#{wid}"), "value", None)
        except Exception:  # noqa: BLE001 -- not mounted / wrong type
            return default
        if isinstance(value, bool):
            return "on" if value else "off"
        text = "" if value is None else str(value).strip()
        return text or default

    def _status_text(self) -> Text:
        name = self._peek("objective", "sortino")
        if _obj.is_valid(name):
            goal = f"{_obj.label(name)} {'≥' if _obj.higher_is_better(name) else '≤'} " \
                   f"{self._peek('pass_line')}"
        else:
            goal = name
        walking = self._peek("walk_forward") == "on"
        selection = self._peek("selection_mode")
        book_param = (f"q={self._peek('top_q')}" if selection == "top_q"
                      else f"k={self._peek('top_k')}" if selection == "top_k"
                      else "all")
        try:
            rail = self.query_one("#rail")
            narrow = rail.has_class("narrow")
            spacious = rail.has_class("spacious")
        except Exception:  # noqa: BLE001 -- status may refresh before layout
            narrow = spacious = False

        if narrow:
            narrow_book = (
                "all" if selection == "none"
                else f"{selection.replace('_', '')}="
                     f"{self._peek('top_q') if selection == 'top_q' else self._peek('top_k')}"
            )
            rows = [
                ("provider", f"{self._peek('provider')}/{self._peek('model')}"),
                ("goal", goal),
                ("memory", f"{self._peek('memory')}/{self._peek('asset_class')}"
                            f"/{self._peek('symbols')}/{narrow_book}"),
                ("search", f"{self._peek('max_iters')}d×"
                            f"{self._peek('candidates_per_round')}w/"
                            f"{'walk' if walking else 'single'}/"
                            f"{self._peek('oos_mode')}"),
            ]
        else:
            rows = [
                ("provider", f"{self._peek('provider')}/{self._peek('model')}"),
                ("goal", goal),
                ("memory", f"{self._peek('memory')}/{self._peek('asset_class')}/"
                            f"{self._peek('data_source')}"),
                ("universe", f"{self._peek('symbols')}/{selection} {book_param}"),
                ("search", f"{self._peek('max_iters')} deep × "
                            f"{self._peek('candidates_per_round')} wide"),
            ]
            if spacious:
                rows.append(("clock", f"{'walk' if walking else 'single'} / "
                             f"{self._peek('oos_mode')}"))
        card = Text(no_wrap=False, overflow="fold")
        label_prefix = " " if narrow else ""
        for i, (key, value) in enumerate(rows):
            card.append(f"{label_prefix}{key:<9}", style="dim")
            card.append(str(value))
            if i < len(rows) - 1:
                extra_line = (spacious and (i == 1 or (not narrow and i == 3)))
                card.append("\n\n" if extra_line else "\n")
        return card

    def _sync_status(self) -> None:
        try:
            card = self.query_one("#status", Static)
        except Exception:  # noqa: BLE001 -- not mounted yet
            return
        card.update(self._status_text())
        card.border_subtitle = _clip(self._peek("run_name"), 22)

    @on(Input.Changed)
    @on(Select.Changed)
    @on(Switch.Changed)
    def _refresh_status(self) -> None:
        self._sync_status()

    # --- helpers -----------------------------------------------------------
    def _log(self, msg: str) -> None:
        self.query_one("#log", RichLog).write(msg)

    def _s(self, wid: str) -> str:
        return self.query_one(f"#{wid}", Input).value.strip()

    def _opt(self, wid: str):
        v = self._s(wid)
        return v or None

    def _build_config(self) -> RunConfig:
        symbols = _parse_symbols(self._s("symbols"))
        return RunConfig(
            math_contract=self._cfg0.math_contract,
            symbols=symbols,
            asset_class=self.query_one("#asset_class", Select).value,
            data_source=self.query_one("#data_source", Select).value,
            data_start=self._s("data_start"),
            data_end=self._s("data_end"),
            parquet_daily=_parse_paths(self._s("parquet_daily")),
            data_dir=self._s("data_dir"),
            research_date=self._s("research_date"),
            is_years=int(self._s("is_years")),
            oos_days=int(self._s("oos_days")),
            oos_mode=self.query_one("#oos_mode", Select).value,
            warmup=int(self._s("warmup")),
            is_start=self._opt("is_start"),
            is_end=self._opt("is_end"),
            oos_start=self._opt("oos_start"),
            oos_end=self._opt("oos_end"),
            t_0=self._s("t_0"),
            t_p=self._s("t_p"),
            frequency=self._s("frequency"),
            cost_bps=float(self._s("cost_bps")),
            gross=float(self._s("gross")),
            initial_cash=float(self._s("initial_cash")),
            objective=self.query_one("#objective", Select).value,
            pass_line=float(self._s("pass_line")),
            require_sign_consistency=self.query_one("#sign", Switch).value,
            require_interpretability=self.query_one("#interp", Switch).value,
            memory=self.query_one("#memory", Switch).value,
            provider=self.query_one("#provider", Select).value,
            model=self._s("model"),
            reasoning_effort=(self.query_one("#reasoning_effort", Select).value or None)
            if self.query_one("#provider", Select).value == "codex" else None,
            max_iters=int(self._s("max_iters")),
            candidates_per_round=int(self._s("candidates_per_round")),
            n_jobs=int(self._s("n_jobs")),
            seed=int(self._s("seed")),
            selection_mode=self.query_one("#selection_mode", Select).value,
            top_q=float(self._s("top_q")),
            top_k=int(self._s("top_k")),
            hold_every=int(self._s("hold_every")),
            rebalance_every=int(self._s("rebalance_every")),
            close_delisted_at_last=self.query_one("#close_delisted", Switch).value,
            walk_forward=self.query_one("#walk_forward", Switch).value,
            run_name=self._s("run_name"),
            allow_missing_symbols=self.query_one("#allow_missing_symbols", Switch).value,
            annualization=int(self._s("annualization")),
            min_is_days=int(self._s("min_is_days")),
            min_oos_days=int(self._s("min_oos_days")),
            tri_align=self.query_one("#tri_align", Switch).value,
            retry_backoff=float(self._s("retry_backoff")),
            retry_max_delay=float(self._s("retry_max_delay")),
            max_tokens=int(self._s("max_tokens")),
            temperature=float(self._s("temperature")),
            logging=bool(getattr(self._cfg0, "logging", True)),
        )

    def _apply_config(self, cfg: RunConfig) -> None:
        self._controls_provider = cfg.provider
        # asset_class first (its Changed handler may touch symbols), symbols last.
        self.query_one("#asset_class", Select).value = cfg.asset_class
        self.query_one("#data_source", Select).value = cfg.data_source
        self.query_one("#symbols", Input).value = _symbols_field(cfg.symbols,
                                                                  cfg.asset_class)
        self.query_one("#parquet_daily", Input).value = _paths_str(cfg.parquet_daily)
        for wid in ("data_start", "data_end", "data_dir", "research_date", "t_0",
                    "t_p", "frequency", "model", "run_name"):
            self.query_one(f"#{wid}", Input).value = str(getattr(cfg, wid))
        for wid in ("is_years", "oos_days", "warmup", "cost_bps", "gross",
                    "top_q", "top_k", "initial_cash", "hold_every", "rebalance_every",
                    "annualization", "min_is_days", "min_oos_days", "pass_line", "max_iters",
                    "candidates_per_round", "n_jobs", "seed", "max_tokens",
                    "retry_backoff", "retry_max_delay", "temperature"):
            self.query_one(f"#{wid}", Input).value = str(getattr(cfg, wid))
        self.query_one("#oos_mode", Select).value = cfg.oos_mode
        self.query_one("#walk_forward", Switch).value = cfg.walk_forward
        for wid in ("is_start", "is_end", "oos_start", "oos_end"):
            self.query_one(f"#{wid}", Input).value = getattr(cfg, wid) or ""
        self.query_one("#objective", Select).value = cfg.objective
        self.query_one("#selection_mode", Select).value = cfg.selection_mode
        self.query_one("#close_delisted", Switch).value = cfg.close_delisted_at_last
        self.query_one("#provider", Select).value = cfg.provider
        effort = self.query_one("#reasoning_effort", Select)
        effort.set_options([("Default?", "")] + [(v, v) for v in reasoning_levels(cfg.model)])
        effort.value = cfg.reasoning_effort or ""
        self.query_one("#sign", Switch).value = cfg.require_sign_consistency
        self.query_one("#interp", Switch).value = cfg.require_interpretability
        self.query_one("#memory", Switch).value = cfg.memory
        self.query_one("#allow_missing_symbols", Switch).value = cfg.allow_missing_symbols
        self.query_one("#tri_align", Switch).value = cfg.tri_align
        self.query_one("#math_contract_label", Label).update(
            f"Math contract: {cfg.math_contract}. Hard admission; economic mechanism remains a hypothesis.")
        # The hidden fields are inherited from _cfg0 when a run is built, so a
        # config loaded here (ctrl+l) must replace it --- otherwise the run would
        # silently use the session-start config's allow_missing_symbols etc.
        self._cfg0 = cfg
        self._sync_provider_controls()
        self._sync_status()

    def _sync_provider_controls(self) -> None:
        """Display only advertised efforts and label controls the CLI ignores.

        Preserve an invalid current selection visibly instead of quietly lowering
        effort while a model id is being edited. Run/Save will reject that choice.
        """
        try:
            provider = self.query_one("#provider", Select).value
            effort = self.query_one("#reasoning_effort", Select)
            model = self._s("model")
            levels = reasoning_levels(model)
            value = effort.value if isinstance(effort.value, str) else ""
            options = [("Default?", "")] + [(v, v) for v in levels]
            if value and value not in levels:
                options.append((f"{value} (unsupported; Run blocked)", value))
            effort.set_options(options)
            effort.value = value
            effort.disabled = provider != "codex"
            for wid in ("max_tokens", "temperature"):
                self.query_one(f"#{wid}", Input).disabled = provider == "codex"
            hint = self.query_one("#provider_controls", Label)
        except Exception:  # noqa: BLE001 -- form may still be composing
            return
        if provider == "codex":
            hint.update("Codex: explicit model/effort. LLM seed/temp/token cap unsupported; seed is local.")
        else:
            hint.update("API providers need a key; CLI providers need login; scripted is offline.")

    @on(Input.Changed, "#model")
    def _on_model_controls(self) -> None:
        self._sync_provider_controls()

    # --- events ------------------------------------------------------------
    @on(Select.Changed, "#asset_class")
    def _on_asset(self, event: Select.Changed) -> None:
        """Keep the universe as 'all' (= every symbol of the newly selected class)
        only when the box is empty or already 'all'; a typed/loaded explicit list
        is left untouched. We must NOT treat "the list equals some preset" as
        swappable: matching against ANY class's preset would silently collapse a
        saved cross-class basket to 'all' (and thus the new class's universe) --
        this handler also fires once on mount and after Load (Select.Changed is
        posted asynchronously, so `_apply_config`'s "symbols last" ordering gives
        no protection). Under the 'all' model the default box is the sentinel, so
        the old preset-matching heuristic is vestigial as well as wrong."""
        if event.value not in ASSET_PRESETS:
            return
        box = self.query_one("#symbols", Input)
        if not box.value.strip() or box.value.strip().lower() == "all":
            box.value = "all"

    @on(Select.Changed, "#data_source")
    def _on_data_source(self, event: Select.Changed) -> None:
        """Parquet path / data dir / scan matter only for data_source='ctx':
        disable them otherwise (so they aren't confusing clutter for synthetic /
        yfinance), and auto-discover into an empty parquet box when ctx is picked
        so the user need not hand-type paths. Fires on mount and after Load too."""
        is_ctx = event.value == "ctx"
        try:
            pq = self.query_one("#parquet_daily", Input)
            dd = self.query_one("#data_dir", Input)
            scan = self.query_one("#scan", Button)
        except Exception:  # noqa: BLE001 -- widgets not mounted yet
            return
        for w in (pq, dd, scan):
            w.disabled = not is_ctx
        if is_ctx and not pq.value.strip():
            self._discover_into_field(announce=False)

    @on(Select.Changed, "#provider")
    def _on_provider(self, event: Select.Changed) -> None:
        """Toggle the API-key box (only key providers store a key here) and show the
        provider's readiness immediately, so a first-time user sees what's missing
        without hunting for 'Check setup'. Fires on mount (scripted -> stays quiet)."""
        provider = event.value
        is_key = provider in preflight.KEY_PROVIDERS
        try:
            box = self.query_one("#api_key", Input)
            btn = self.query_one("#save_key", Button)
        except Exception:  # noqa: BLE001 -- widgets not mounted yet
            return
        box.value = ""                      # drop any key typed for the OLD provider
        box.disabled = not is_key           # so it can't be misfiled under the new
        btn.disabled = not is_key           # provider's env var on Save
        previous = getattr(self, "_controls_provider", self._cfg0.provider)
        if previous != provider:
            old_defaults, new_defaults = model_defaults(previous), model_defaults(provider)
            field = self.query_one("#model", Input)
            if field.value == old_defaults:
                field.value = new_defaults
                self._log(f"provider default: model={new_defaults or 'client default'}")
        self._controls_provider = provider
        self._sync_provider_controls()
        if is_key:
            box.placeholder = (
                f"paste {preflight.KEY_PROVIDERS[provider]['env_key']}, then Save "
                "key (stored in .env, never shown)")
        if provider != "scripted":          # scripted is always ready -> no noise
            r = preflight.check_provider(provider, self._s("model"))
            self._log(preflight.format_readiness(r, markup=True))

    @on(Button.Pressed, "#scan")
    def _on_scan(self) -> None:
        self._discover_into_field()

    def _discover_into_field(self, *, announce: bool = True) -> None:
        """Fill the parquet field from a scan of the Data dir (CTX-named parquet)."""
        data_dir = self._s("data_dir")
        found = _data.discover_parquet(data_dir)
        where = data_dir or "(unset)"
        if found:
            self.query_one("#parquet_daily", Input).value = ", ".join(found)
            if announce:
                self._log(f"{_ok(f'found {len(found)} parquet')} under '{where}' "
                          "-- edit the list if needed")
        elif announce:
            self._log(f"{_warn(f'no CTX-named parquet under {where!r}')} -- drop files "
                      "named Data_<EQT|ETF|INDEX|FX|SPOT>_<REGION>_D*.parquet there, "
                      "or type paths manually")

    @on(Button.Pressed, "#quit")
    def _on_quit(self) -> None:
        self.action_quit()

    def action_quit(self) -> None:
        if self._run_control is None:
            self.exit()
            return
        self._quit_requested = True
        self._run_control.cancel()
        self.query_one("#quit", Button).disabled = True
        self.query_one("#log", RichLog).border_subtitle = "stopping…"
        self._log(_warn("Stopping run and owned child processes before closing…"))

    def _finish_run(self) -> None:
        self._run_control = None
        if self._quit_requested:
            self.exit()
        else:
            self._set_run_enabled(True)

    @on(Button.Pressed, "#save")
    def _on_save(self) -> None:
        self.action_save()

    @on(Button.Pressed, "#load")
    def _on_load(self) -> None:
        self.action_load()

    def action_load(self) -> None:
        path = self.query_one("#config_path", Input).value.strip()
        try:
            cfg = RunConfig.from_json(path)
            self._apply_config(cfg)
            self._log(f"{_ok('loaded config')} from {path}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"{_bad('load failed:')} {exc}")

    @on(Button.Pressed, "#run")
    def _on_run(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#doctor")
    def _on_doctor(self) -> None:
        self.action_doctor()

    def action_doctor(self) -> None:
        """Print full run readiness + setup guidance into the log pane.

        ``check_all`` rather than ``check_provider``: a doctor that only checks
        credentials answers "is my key set?" when the question the user is asking
        is "could a run start right now?". A missing engine runtime dependency or
        a broken prompt asset used to pass this check and then kill the run a
        minute in. (``_on_provider`` deliberately stays on ``check_provider`` ---
        it fires on every keystroke and must not pay the engine import.)
        """
        try:
            provider = self.query_one("#provider", Select).value
        except Exception as exc:  # noqa: BLE001
            self._log(f"{_bad('cannot read provider:')} {exc}")
            return
        r = preflight.check_all(provider, self._s("model"))
        self._log(preflight.format_readiness(r, markup=True))

    @on(Button.Pressed, "#save_key")
    def _on_save_key(self) -> None:
        self.action_save_key()

    def action_save_key(self) -> None:
        """Store the entered API key into the git-ignored .env (and the live process
        env, so it's effective immediately) for the selected key-provider. The value
        is never echoed or logged -- only that the key was saved."""
        provider = self.query_one("#provider", Select).value
        spec = preflight.KEY_PROVIDERS.get(provider)
        if spec is None:
            if provider in preflight.CLI_PROVIDERS:
                cmd = preflight.CLI_PROVIDERS[provider]["login_cmd"]
                self._log(f"{_warn(f'{provider!r} uses a CLI login, not a key')} -- "
                          f"run once in a terminal:  {cmd}")
            else:
                self._log(_warn(f"{provider!r} is offline and needs no key"))
            return
        box = self.query_one("#api_key", Input)
        try:
            # save_key returns the file it actually wrote; print THAT rather than
            # a constant, or a wheel install (which writes to the per-user config
            # dir, not the vendored .env) would be told the wrong path.
            saved_to = env.save_key(spec["env_key"], box.value)
        except ValueError as exc:
            self._log(_warn(str(exc)))
            return
        except Exception as exc:  # noqa: BLE001
            self._log(f"{_bad('could not save key:')} {exc}")
            return
        box.value = ""                       # never keep the secret in the widget
        self._log(f"{_ok('saved ' + spec['env_key'])} to {saved_to} "
                  "(value hidden); active now -- no restart needed")
        r = preflight.check_provider(provider, self._s("model"))
        self._log(preflight.format_readiness(r, markup=True))

    def action_save(self) -> None:
        try:
            cfg = self._build_config()
            path = self.query_one("#config_path", Input).value.strip()
            cfg.to_json(path)
            self._log(f"{_ok('saved config')} to {path}")
        except Exception as exc:  # noqa: BLE001
            self._log(f"{_bad('save failed:')} {exc}")

    def _set_run_enabled(self, enabled: bool) -> None:
        self.query_one("#run", Button).disabled = not enabled
        # the log frame doubles as the run indicator, so the state is visible even
        # when the rail is scrolled away
        self.query_one("#log", RichLog).border_subtitle = \
            "idle" if enabled else "running…"

    def action_run(self) -> None:
        # A thread worker cannot be force-killed, so a second Run (button or
        # ctrl+r) would race the first on the SAME run_dir --- both reset()ing and
        # appending to one ledger. Refuse to start while one is already in flight.
        if self.query_one("#run", Button).disabled:
            self._log(_warn("a run is already in progress"))
            return
        try:
            cfg = self._build_config()
        except Exception as exc:  # noqa: BLE001
            self._log(f"{_bad('invalid config:')} {exc}")
            return
        # First-run gate: block a run that cannot succeed rather than letting the
        # worker die deep inside it. Show the setup guide instead. Checked
        # unconditionally --- check_config now covers the vendored engine, its
        # runtime deps and the prompt assets as well as the credential, and
        # provider='scripted' needs every one of those just as much as a real
        # provider does. The old 'skip for scripted' guard meant the offline demo
        # was the ONE path that reached the worker with a broken engine.
        r = preflight.check_config(cfg)
        if not r.ok:
            self._log(preflight.format_readiness(r, markup=True))
            self._log(_warn("fix the above (or press 'Check setup'), "
                            "then Run again."))
            return
        walk = self.query_one("#walk_forward", Switch).value
        self._log(f"[b {_pal.GOLD}]starting[/] provider={cfg.provider} "
                  f"objective={cfg.objective} walk_forward={walk}")
        self._set_run_enabled(False)
        self._run_control = RunControl()
        self._run_worker(cfg, walk)

    # --- worker ------------------------------------------------------------
    def _remember(self, cfg: RunConfig, store: Store, emit) -> None:
        """Copy the finished run into the persistent journal under ``memory/``."""
        if not cfg.memory:
            return
        entry = memory.remember(store, self._memory)
        if entry is not None:
            emit(f"{_note('journal:')} {entry['run_id']} -> "
                 f"{memory.notes_path(self._memory)}")

    @work(thread=True, exclusive=True, group="run")
    def _run_worker(self, cfg: RunConfig, walk: bool) -> None:
        control = self._run_control or RunControl()
        try:
            with cancellation_scope(control):
                self._execute_run(cfg, walk)
        except RunCancelled:
            pass  # cancellation before the worker entered its run scope
        finally:
            if not self._ui_closed:
                try:
                    self.call_from_thread(self._finish_run)
                except RuntimeError:
                    pass  # the app was externally closed during cleanup

    def _execute_run(self, cfg: RunConfig, walk: bool) -> None:
        # Every log line also lands in the run's own text log
        # (artifacts/runs/<run_name>/run.log), so a session whose window closes
        # still leaves a tailable record of what ran --- the RichLog widget
        # itself dies with the process. The file is re-opened (truncating) for
        # each Run, matching the ledger/trace reset.
        log_fh = None
        provider = None

        def emit(msg: str) -> None:
            clean = redact_secrets(str(msg))
            if log_fh is not None:
                try:
                    log_fh.write(clean + "\n")
                    log_fh.flush()
                except OSError:
                    pass
            if not self._ui_closed:
                try:
                    self.call_from_thread(self._log, clean)
                except RuntimeError:
                    pass

        store = Store(self._artifacts, cfg.run_name)
        try:
            # Provider construction can fail (missing binary/SDK, malformed
            # local auth). It must happen before reset() or run.log truncation so
            # a failed launch cannot destroy the previous completed run.
            provider = get_provider(cfg.provider)
            check_cancelled()
            with store.ownership():
                from .runner import prepare_panel
                panel = prepare_panel(cfg, walk_forward=walk, log=emit)
                check_cancelled()
                try:
                    log_fh = open(os.path.join(store.run_dir, "run.log"), "w",
                                  encoding="utf-8")
                except OSError as exc:
                    self.call_from_thread(
                        self._log, f"{_bad('cannot write run log:')} "
                        f"{redact_secrets(str(exc))}"
                    )

                emit(f"run started: provider={cfg.provider} model={cfg.model} "
                     f"objective={cfg.objective} walk_forward={walk} "
                     f"run_name={cfg.run_name}")
                if walk:
                    out = run_walk_forward(
                        cfg, artifacts_root=self._artifacts, log=emit,
                        memory_root=self._memory, provider=provider, panel=panel,
                    )
                    recs = out["records"]
                    npass = sum(1 for r in recs if r["verdict"]["passed"])
                    emit(f"[b {_pal.GOLD}]{len(recs)} factor(s), {npass} PASS[/] "
                         f"-> {out['run_dir']}")
                    self._remember(cfg, store, emit)
                else:
                    check_cancelled()
                    store.reset()
                    journal = _journal_snapshot(cfg, self._memory)
                    store.write_manifest({
                        "run_name": cfg.run_name, "config": cfg.to_dict(),
                        "execution_mode": "single-date",
                        "provider": provider.name, "model": cfg.model,
                        "objective": cfg.objective, "pass_line": cfg.pass_line,
                        "seed": cfg.seed, "journal": journal,
                    })
                    pool = pool_for(cfg, log=emit)
                    try:
                        rec = run_once(cfg, provider=provider, store=store,
                                       log=emit, journal=journal, pool=pool, panel=panel)
                    finally:
                        if pool is not None:
                            pool.close()
                    manifest = store.read_manifest()
                    manifest.update({"n_factors": 1,
                                     "n_pass": int(rec["verdict"]["passed"]),
                                     "data_version": rec["data_version"]})
                    store.write_manifest(manifest)
                    v = rec["verdict"]
                    color = _pal.OK if v["passed"] else _pal.BAD
                    emit(f"[b {color}]{v['label']}[/]: {rec['formula']}")
                    self._remember(cfg, store, emit)
        except RunCancelled:
            emit("run cancelled; no further model calls will be made")
        except Exception as exc:  # noqa: BLE001
            emit(f"{_bad('run failed:')} {exc}")
            # the full traceback belongs in the file (the UI line stays terse):
            # the window that displayed the error may already be gone by the
            # time anyone goes looking for it.
            if log_fh is not None:
                try:
                    log_fh.write("\n--- traceback ---\n")
                    log_fh.write(redact_secrets(traceback.format_exc()))
                    log_fh.flush()
                except OSError:
                    pass
            # an auth-shaped failure (e.g. a lapsed CLI login) becomes a guide
            guide = preflight.explain_failure(cfg.provider, exc, cfg.model)
            if guide is not None:
                emit(preflight.format_readiness(guide, markup=True))
        finally:
            if provider is not None:
                try:
                    close_provider(provider)
                except Exception as exc:  # noqa: BLE001 - cleanup cannot hide result
                    emit(f"{_warn('provider cleanup failed:')} {exc}")
            # re-arm the Run button on the UI thread, whatever the outcome
            if log_fh is not None:
                try:
                    log_fh.close()
                except OSError:
                    pass


def main() -> int:
    HarnessTUI().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
