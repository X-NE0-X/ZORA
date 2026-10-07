"""Headless TUI tests via Textual's run_test()/Pilot (no real terminal)."""
import os
import pathlib
import re

from textual.widgets import (
    Button, Input, Markdown, RichLog, Select, Static, Switch, TabbedContent,
    TabPane,
)

from harness import objective as o
from harness import palette
from harness import preflight
from harness import tui
from harness.config import ASSET_PRESETS, RunConfig
from harness.preflight import Readiness
from harness.tui import HarnessTUI


async def test_tui_mounts_and_builds_config():
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        cfg = app._build_config()
        assert isinstance(cfg, RunConfig)
        assert cfg.objective in o.names()
        assert cfg.provider == "scripted"


async def test_tui_edits_reflect_in_config():
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#objective", Select).value = "sharpe"
        app.query_one("#pass_line", Input).value = "0.7"
        app.query_one("#selection_mode", Select).value = "top_q"
        app.query_one("#top_q", Input).value = "0.2"
        app.query_one("#hold_every", Input).value = "2"
        app.query_one("#rebalance_every", Input).value = "2"
        app.query_one("#oos_mode", Select).value = "per_step"
        app.query_one("#warmup", Input).value = "252"
        app.query_one("#interp", Switch).value = True
        await pilot.pause()
        cfg = app._build_config()
        assert cfg.objective == "sharpe"
        assert cfg.pass_line == 0.7
        assert cfg.selection_mode == "top_q"
        assert cfg.top_q == 0.2
        assert cfg.hold_every == 2
        assert cfg.rebalance_every == 2
        assert cfg.oos_mode == "per_step"
        assert cfg.warmup == 252
        assert cfg.require_interpretability is True


async def test_tui_selection_controls_follow_selection_mode():
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#top_q", Input).disabled is True
        assert app.query_one("#top_k", Input).disabled is True

        app.query_one("#selection_mode", Select).value = "top_q"
        await pilot.pause()
        assert app.query_one("#top_q", Input).disabled is False
        assert app.query_one("#top_k", Input).disabled is True

        app.query_one("#selection_mode", Select).value = "top_k"
        app.query_one("#top_k", Input).value = "5"
        await pilot.pause()
        assert app.query_one("#top_q", Input).disabled is True
        assert app.query_one("#top_k", Input).disabled is False
        assert app._build_config().top_k == 5


async def test_tui_advanced_guardrails_roundtrip_and_gate_fixed_oos():
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#tab-advanced", TabPane)
        app.query_one("#oos_mode", Select).value = "per_step"
        await pilot.pause()
        assert app.query_one("#oos_days", Input).disabled is True
        app.query_one("#warmup", Input).value = "252"
        app.query_one("#data_start", Input).value = "2017-01-01"
        app.query_one("#allow_missing_symbols", Switch).value = True
        await pilot.pause()
        cfg = app._build_config()
        assert cfg.oos_mode == "per_step"
        assert cfg.warmup == 252
        assert cfg.data_start == "2017-01-01"
        assert cfg.allow_missing_symbols is True
        app.query_one("#oos_mode", Select).value = "fixed_days"
        await pilot.pause()
        assert app.query_one("#oos_days", Input).disabled is False


async def test_tui_decision_and_memory_are_on_the_intended_pages():
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.query("#tab-objective")) == 0
        assert app.query_one("#decision")
        assert any(a.id == "tab-model" for a in app.query_one("#memory", Switch).ancestors)
        assert any(a.id == "tab-windows" for a in app.query_one("#walk_forward", Switch).ancestors)
        assert any(a.id == "tab-backtest" for a in app.query_one("#cost_bps", Input).ancestors)


async def test_tui_restores_saved_walk_forward_state_on_launch():
    app = HarnessTUI(config=RunConfig(walk_forward=True, run_name="saved"))
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#walk_forward", Switch).value is True
        assert app._build_config().walk_forward is True


async def test_tui_apply_config_roundtrip():
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        target = RunConfig(objective="maxdd", pass_line=-0.1,
                           require_interpretability=True, provider="scripted",
                           max_iters=3, candidates_per_round=4,
                           selection_mode="top_q", top_q=0.2,
                           hold_every=2, rebalance_every=2,
                           oos_mode="per_step", warmup=252,
                           allow_missing_symbols=True)
        app._apply_config(target)
        await pilot.pause()
        cfg = app._build_config()
        assert cfg.objective == "maxdd"
        assert cfg.pass_line == -0.1
        assert cfg.require_interpretability is True
        assert cfg.max_iters == 3
        assert cfg.candidates_per_round == 4          # breadth knob round-trips
        assert cfg.selection_mode == "top_q"
        assert cfg.top_q == 0.2
        assert cfg.hold_every == 2
        assert cfg.rebalance_every == 2
        assert cfg.oos_mode == "per_step"
        assert cfg.warmup == 252
        assert cfg.allow_missing_symbols is True


async def test_tui_refuses_concurrent_run():
    # A run already in flight disables the Run button; a second Run (button or
    # ctrl+r -> action_run) must refuse rather than race the first on one run_dir.
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#run", Button).disabled = True     # simulate run in flight
        app.action_run()                                   # must return early
        await pilot.pause()
        assert app.query_one("#run", Button).disabled is True


async def test_tui_symbols_default_is_all():
    # On open the symbols box shows the clean 'all' keyword, not a long ticker
    # list; building the config expands it to the whole default-class basket.
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#symbols", Input).value == "all"
        cfg = app._build_config()
        assert cfg.symbols == list(ASSET_PRESETS["equity"])


async def test_tui_all_follows_selected_asset_class():
    # switching class keeps 'all' and now means all of the NEW class
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#asset_class", Select).value = "crypto"
        await pilot.pause()
        assert app.query_one("#symbols", Input).value == "all"
        cfg = app._build_config()
        assert cfg.symbols == list(ASSET_PRESETS["crypto"])


async def test_tui_custom_symbols_preserved_across_class_switch():
    # a hand-typed basket must NOT be clobbered when the asset class changes
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#symbols", Input).value = "AAPL, TSLA"
        app.query_one("#asset_class", Select).value = "fx"
        await pilot.pause()
        assert app.query_one("#symbols", Input).value == "AAPL, TSLA"
        assert app._build_config().symbols == ["AAPL", "TSLA"]


async def test_tui_launch_preserves_cross_class_basket():
    # A saved config whose symbols exactly equal a DIFFERENT class's full preset
    # must survive launch: Select.Changed fires once on mount, so _on_asset must
    # NOT collapse the just-rendered list to 'all' (which would expand to the new
    # class's universe -- the confirmed silent wrong-universe bug).
    cfg = RunConfig(asset_class="crypto", symbols=list(ASSET_PRESETS["etf"]),
                    run_name="t", logging=False)
    app = HarnessTUI(config=cfg)
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#symbols", Input).value != "all"
        assert app._build_config().symbols == list(ASSET_PRESETS["etf"])


async def test_tui_load_preserves_cross_class_basket():
    # Same guarantee via the async _apply_config (Load) path: asset_class change
    # posts Changed AFTER symbols is written, so the deferred _on_asset must leave
    # a preset-equal cross-class list alone rather than reset it to 'all'.
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._apply_config(RunConfig(asset_class="crypto",
                                    symbols=list(ASSET_PRESETS["etf"]),
                                    run_name="t", logging=False))
        await pilot.pause()
        assert app._build_config().symbols == list(ASSET_PRESETS["etf"])


async def test_tui_simplified_removed_still_builds():
    # the dead 'simplified' knob is gone; the form still builds a valid config
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.query("#simplified")) == 0
        assert isinstance(app._build_config(), RunConfig)


async def test_tui_data_source_gates_parquet():
    # parquet path / data dir / scan matter only for ctx: disabled by default
    # (synthetic), enabled once ctx is picked.
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#parquet_daily", Input).disabled is True
        assert app.query_one("#data_dir", Input).disabled is True
        assert app.query_one("#scan", Button).disabled is True
        app.query_one("#data_source", Select).value = "ctx"
        await pilot.pause()
        assert app.query_one("#parquet_daily", Input).disabled is False
        assert app.query_one("#data_dir", Input).disabled is False
        assert app.query_one("#scan", Button).disabled is False


async def test_tui_provider_change_toggles_key_and_reports():
    # switching to a key-provider enables the API-key box and prints readiness
    # immediately; scripted (default) keeps the key box disabled.
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#api_key", Input).disabled is True    # scripted default
        logged: list[str] = []
        app._log = lambda msg: logged.append(msg)
        app.query_one("#provider", Select).value = "openai"
        await pilot.pause()
        assert app.query_one("#api_key", Input).disabled is False
        assert app.query_one("#save_key", Button).disabled is False
        assert any("openai" in m for m in logged)                   # readiness shown


async def test_tui_provider_switch_clears_api_key():
    # a key typed for one provider must be dropped on switch, so it can't be saved
    # under the NEXT provider's env var (misfiled credential).
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#provider", Select).value = "claude"
        await pilot.pause()
        app.query_one("#api_key", Input).value = "sk-ant-SECRET"
        app.query_one("#provider", Select).value = "openai"
        await pilot.pause()
        assert app.query_one("#api_key", Input).value == ""


async def test_tui_save_key_forwards_to_env_and_clears():
    # Save key forwards (env_var, value) to env.save_key and clears the box; the
    # secret is never kept in the widget. env.save_key is stubbed so no real file /
    # process env is touched.
    from harness import env
    calls: list[tuple[str, str]] = []
    old = env.save_key
    env.save_key = lambda name, value: calls.append((name, value))
    try:
        app = HarnessTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#provider", Select).value = "openai"
            await pilot.pause()
            app.query_one("#api_key", Input).value = "sk-test-xyz"
            app.action_save_key()
            await pilot.pause()
            assert calls == [("OPENAI_API_KEY", "sk-test-xyz")]
            assert app.query_one("#api_key", Input).value == ""     # cleared
    finally:
        env.save_key = old


async def test_tui_save_key_noop_for_cli_provider():
    # a CLI provider (codex) has no key to store: Save key must not call env.save_key
    from harness import env
    calls: list = []
    old = env.save_key
    env.save_key = lambda name, value: calls.append((name, value))
    try:
        app = HarnessTUI()
        async with app.run_test() as pilot:
            await pilot.pause()
            app.query_one("#provider", Select).value = "codex"
            await pilot.pause()
            app.action_save_key()
            await pilot.pause()
            assert calls == []
    finally:
        env.save_key = old


async def test_tui_run_blocked_when_provider_not_ready():
    # A real provider with no key/login must NOT start a run: action_run shows the
    # setup guide and leaves the Run button armed (never reaches _set_run_enabled).
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#provider", Select).value = "openai"
        await pilot.pause()
        logged: list[str] = []
        app._log = lambda msg: logged.append(msg)          # capture log output
        old = preflight.check_config
        preflight.check_config = lambda cfg, **k: Readiness(
            "openai", ok=False, reason="OPENAI_API_KEY is not set",
            steps=["get a key"], checks=[("OPENAI_API_KEY", False)])
        try:
            app.action_run()
            await pilot.pause()
        finally:
            preflight.check_config = old
        assert app.query_one("#run", Button).disabled is False   # not started
        assert any("needs setup" in m for m in logged)


async def test_tui_check_setup_button_writes_guide():
    # The 'Check setup' button prints a readiness block without starting a run.
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        logged: list[str] = []
        app._log = lambda msg: logged.append(msg)
        app.action_doctor()                                # default provider=scripted
        await pilot.pause()
        assert logged and "scripted" in logged[0]
        assert app.query_one("#run", Button).disabled is False


async def test_tui_mission_card_digests_the_form():
    # The rail's MISSION card is the at-a-glance digest of settings spread over
    # four tabs, so it must track edits made on any of them.
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#objective", Select).value = "sharpe"
        app.query_one("#pass_line", Input).value = "0.7"
        app.query_one("#max_iters", Input).value = "6"
        app.query_one("#memory", Switch).value = False
        await pilot.pause()
        card = str(app.query_one("#status", Static).content)
        assert "Sharpe" in card and "0.7" in card
        assert "6 deep" in card
        assert "memory   off" in card
        assert "universe" in card and "search" in card


async def test_tui_mission_card_survives_a_half_typed_number():
    # It refreshes on every keystroke, so an in-progress value ("1." / "") must
    # render rather than raise -- _build_config would throw on both.
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one("#pass_line", Input).value = ""
        app.query_one("#max_iters", Input).value = ""
        await pilot.pause()
        assert "provider" in str(app.query_one("#status", Static).content)


async def test_tui_reflows_between_wide_and_narrow():
    # Two label/field pairs per row need the width; below the threshold the forms
    # fall back to one pair and the rail narrows, so nothing is clipped.
    app = HarnessTUI()
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        forms = app.query(".form")
        assert forms, "the tabs must be built from .form grids"
        assert not any(f.has_class("narrow") for f in forms)
        assert app.query_one("#rail").styles.width.value == 46

        await pilot.resize_terminal(90, 40)
        await pilot.pause()
        assert all(f.has_class("narrow") for f in app.query(".form"))
        assert app.query_one("#rail").styles.width.value == 38


async def test_tui_wide_rail_needs_no_scroll_at_launch():
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#rail").max_scroll_y == 0
        assert app.query_one("#rail-content").max_scroll_y == 0
        assert "Interpretability" in str(app.query_one("#interpretability-control Label").content)
        assert app.query_one("#run", Button).styles.height.value == 3


async def test_tui_wide_render_keeps_controls_readable():
    """The compositor must paint values, not just widget borders."""
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for wid in ("#objective", "#pass_line", "#data_source"):
            assert app.query_one(wid).region.height >= 3, f"{wid} is vertically clipped"
        app.query_one(TabbedContent).active = "tab-model"
        await pilot.pause()
        assert app.query_one("#model").region.height >= 3, "#model is vertically clipped"

        rendered = "\n".join(strip.text for strip in app.screen._compositor.render_strips())
        for text in ("Sortino", "1.0", "Interpretability", "Run", "Check",
                     "Save", "Load", "Quit"):
            assert text in rendered, f"{text!r} is missing from the rendered UI"


async def test_tui_action_bar_matches_rail_width():
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        actions = app.query_one("#actions")
        buttons = [app.query_one(f"#{wid}") for wid in
                   ("run", "doctor", "save", "load", "quit")]
        assert actions.size.height == 3
        assert {button.region.height for button in buttons} == {3}
        widths = [button.region.width for button in buttons]
        assert max(widths) - min(widths) <= 1
        assert buttons[0].region.width * 5 <= actions.region.width


async def test_tui_short_rail_keeps_actions_outside_scroll_content():
    """A short terminal scrolls the card without putting buttons over it."""
    app = HarnessTUI()
    async with app.run_test(size=(90, 24)) as pilot:
        await pilot.pause()
        content = app.query_one("#rail-content")
        actions = app.query_one("#actions")
        rail = app.query_one("#rail")
        assert content.region.bottom <= actions.region.y
        assert actions.region.bottom <= rail.region.bottom
        assert content.max_scroll_y > 0


async def test_tui_drops_rail_decoration_on_a_short_terminal():
    # The wordmark stays, but the sparkline/tagline give way to the expanded
    # MISSION card when there are not enough rows for both.
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#spark", Static).display is False
        await pilot.resize_terminal(120, 50)
        await pilot.pause()
        assert app.query_one("#spark", Static).display is True
        await pilot.resize_terminal(120, 24)
        await pilot.pause()
        assert app.query_one("#spark", Static).display is False
        assert app.query_one("#tagline", Static).display is False
        assert app.query_one("#brand", Static).display is True


async def test_tui_walk_forward_gates_the_window_fields():
    """The switch changes WHICH window fields the run reads, so it greys the rest.

    ``runner._step_config`` overwrites ``research_date`` at every stop and clears
    all four explicit overrides, so on a walk those five inputs are dead --- and
    a live-looking dead input is how someone spends an afternoon wondering why
    their OOS window was ignored. Off, the reverse: T_0/T_p/Frequency are unread.
    """
    walk_only = ("t_0", "t_p", "frequency")
    single_only = ("research_date", "is_start", "is_end", "oos_start", "oos_end")
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert all(app.query_one(f"#{w}").disabled for w in walk_only)
        assert not any(app.query_one(f"#{w}").disabled for w in single_only)

        app.query_one("#walk_forward", Switch).value = True
        await pilot.pause()
        assert not any(app.query_one(f"#{w}").disabled for w in walk_only)
        assert all(app.query_one(f"#{w}").disabled for w in single_only)

        # the clock lengths feed BOTH modes and must stay live either way
        for wid in ("is_years", "oos_days"):
            assert app.query_one(f"#{wid}").disabled is False
        # ... and greying never changes what gets run
        assert app._build_config().t_0 == "2020-01-01"
        assert app._build_config().walk_forward is True

        saved = RunConfig(walk_forward=True, run_name="saved", logging=False)
        app._apply_config(saved)
        await pilot.pause()
        assert app.query_one("#walk_forward", Switch).value is True
        assert app._build_config().walk_forward is True


async def test_tui_palette_actually_reaches_the_screen():
    """The theme is only real if the compositor paints it.

    Guards the design language end to end: the wordmark is the palette's brightest
    gold, the sparkline its recessive bronze, and both sit on the rail's grey.
    """
    def rgb(color):
        c = color.get_truecolor()
        return (c.red, c.green, c.blue)

    def want(hex_str):
        h = hex_str.lstrip("#")
        return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))

    def close(a, b):
        # Textual blends text at the theme's text_alpha, so allow a hair of drift
        return all(abs(x - y) <= 4 for x, y in zip(a, b))

    # NO_COLOR/TERM=dumb are legitimate inherited preferences. This test asserts
    # the product palette, so isolate it from both variables.
    saved = {name: os.environ.get(name) for name in ("NO_COLOR", "TERM")}
    os.environ.pop("NO_COLOR", None)
    os.environ.pop("TERM", None)
    try:
        app = HarnessTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            for wid, expect in (("#brand", palette.GOLD), ("#spark", palette.BRONZE)):
                region = app.query_one(wid, Static).region
                style = app.screen._compositor.get_style_at(region.x, region.y)
                assert close(rgb(style.color), want(expect)), \
                    f"{wid} painted {rgb(style.color)}, wanted {expect}"
                assert close(rgb(style.bgcolor), want(palette.SLATE)), \
                    f"{wid} sits on {rgb(style.bgcolor)}, wanted {palette.SLATE}"

            hint = app.query(".hint").first()
            style = app.screen._compositor.get_style_at(hint.region.x + 3, hint.region.y)
            assert close(rgb(style.color), want(palette.MUTED)), \
                f"hint text painted {rgb(style.color)}, wanted {palette.MUTED}"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_no_terminal_colour_names_survive_in_the_ui():
    """Every colour must come from harness.palette --- one design language.

    A bare Rich tag like a green/red colour NAME renders in whatever the terminal
    calls green, which is outside the palette and reads as a different app. Both
    modules that write into the log are checked, because ``preflight`` renders
    straight into the TUI's RichLog.
    """
    bad = re.compile(r"\[/?(?:b |bold )?"
                     r"(?:green|red|yellow|blue|cyan|magenta|white|black)\b")
    root = pathlib.Path(tui.__file__).resolve().parent
    for name in ("tui.py", "preflight.py"):
        text = (root / name).read_text(encoding="utf-8")
        hits = [ln for ln in text.splitlines() if bad.search(ln)]
        assert not hits, f"{name} still uses terminal colour names: {hits}"


async def test_tui_log_frame_shows_run_state():
    # The log border doubles as the busy indicator, so it is visible even when
    # the rail is scrolled; it must re-arm together with the Run button.
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#log", RichLog).border_subtitle == "idle"
        app._set_run_enabled(False)
        await pilot.pause()
        assert app.query_one("#log", RichLog).border_subtitle == "running…"
        app._set_run_enabled(True)
        await pilot.pause()
        assert app.query_one("#log", RichLog).border_subtitle == "idle"


async def test_tui_every_field_carries_a_caption():
    """No knob is left unexplained, and no caption is a paragraph.

    Each form cell is exactly ``field + one-line note``: a caption that wraps is
    both unreadable in a 21-column track and (see the next test) able to wreck
    the grid's row heights.
    """
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        cells = list(app.query(".cell"))
        assert len(cells) >= 25, f"only {len(cells)} captioned fields"
        for cell in cells:
            field, note = cell.children
            assert note.has_class("note"), f"{field} has no caption"
            text = str(note.content).strip()
            assert text, f"{field} has an empty caption"
            assert "\n" not in text, f"caption is multi-line: {text!r}"
            # the wide cells span three tracks; the rest must stay terse enough
            # to fit the ~20 columns a single field track gives them
            limit = 60 if cell.has_class("wide") else 22
            assert len(text) <= limit, f"caption too long ({len(text)}): {text!r}"


async def test_tui_captions_cost_no_height():
    """Every tab must still fit a 120x40 terminal with nothing hidden.

    This is the regression guard for a Textual grid quirk: an `auto` row is sized
    by measuring each cell at ONE column's width even when the cell spans three,
    so a caption allowed to wrap inside a spanning cell is measured against the
    ~11-column label track and reports ~10 lines. That silently pushed the last
    two rows of the Data tab off the screen.
    """
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        for tab in ("tab-data", "tab-windows", "tab-model"):
            app.query_one(TabbedContent).active = tab
            await pilot.pause()
            pane = app.query_one(f"#{tab}", TabPane)
            assert pane.max_scroll_y == 0, (
                f"{tab} hides {pane.max_scroll_y} row(s) at 120x40 "
                f"(pane {pane.size.height}, content {pane.virtual_size.height})")


async def test_tui_forms_scroll_when_they_do_not_fit():
    # ... and when the terminal really is too small, the pane scrolls rather
    # than cropping the last field into invisibility.
    app = HarnessTUI()
    async with app.run_test(size=(120, 28)) as pilot:
        await pilot.pause()
        pane = app.query_one("#tab-data", TabPane)
        assert pane.max_scroll_y > 0
        assert pane.allow_vertical_scroll


async def test_tui_guide_renders_the_readme_lazily():
    """The Guide tab is README.md, parsed on first view rather than at startup."""
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app._guide_loaded is False, "the README must not load at startup"
        assert not app.query_one("#guide", tui.GuideViewer).document.children

        app.query_one(TabbedContent).active = "tab-guide"
        await pilot.pause()
        await pilot.pause()
        assert app._guide_loaded is True
        blocks = app.query_one("#guide", tui.GuideViewer).document.children
        assert len(blocks) > 20, f"only {len(blocks)} blocks rendered"
        rendered = " ".join(str(getattr(b, "content", "")) for b in blocks[:4])
        assert "ZORA Harness" in rendered


async def test_tui_guide_source_is_the_real_readme():
    # Not a copy that can rot: the tab reads the file the repo ships.
    path = tui._guide_path()
    assert path.name == "README.md" and path.is_file()
    text = tui._read_guide()
    assert text == path.read_text(encoding="utf-8")
    assert text.startswith("# ZORA Harness")


def test_read_guide_reports_a_missing_readme_instead_of_raising():
    # An installed harness may ship without its README; that must not stop the
    # TUI from starting, so the reader degrades to a rendered explanation.
    original = tui._guide_path
    tui._guide_path = lambda: pathlib.Path("no", "such", "README.md")
    try:
        text = tui._read_guide()
    finally:
        tui._guide_path = original
    assert text.startswith("# Guide unavailable")


async def test_tui_guide_refuses_to_navigate_out_of_the_document():
    """Stock MarkdownViewer follows links; this README has relative ones.

    ``[harness/palette.py](harness/palette.py)`` would be *loaded as markdown*,
    replacing the guide with a Python file and no way back, and an http href
    would open a browser. Both are reported instead.
    """
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(TabbedContent).active = "tab-guide"
        await pilot.pause()
        await pilot.pause()
        viewer = app.query_one("#guide", tui.GuideViewer)
        before = list(viewer.document.children)

        for href in ("https://example.com", "harness/palette.py"):
            await viewer._on_markdown_link_clicked(
                Markdown.LinkClicked(viewer.document, href))
            await pilot.pause()
            assert list(viewer.document.children) == before, \
                f"{href} navigated away from the README"

        log = app.query_one("#log", RichLog)
        written = " ".join(str(line) for line in log.lines)
        assert "example.com" in written and "palette.py" in written


async def test_tui_guide_takes_the_rail_and_gives_it_back():
    # Reading mode: prose + contents sidebar does not fit in what is left after
    # the rail, and nothing in the rail is needed to read the manual.
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#rail").display is True

        app.action_guide()
        await pilot.pause()
        assert app.query_one(TabbedContent).active == "tab-guide"
        assert app.query_one("#rail").display is False
        assert app.query_one("#guide", tui.GuideViewer).show_table_of_contents

        app.query_one(TabbedContent).active = "tab-model"
        await pilot.pause()
        assert app.query_one("#rail").display is True


async def test_tui_guide_drops_its_sidebar_on_a_narrow_terminal():
    # Below TOC_AT the contents tree would leave the prose a column too thin.
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        app.query_one(TabbedContent).active = "tab-guide"
        await pilot.pause()
        viewer = app.query_one("#guide", tui.GuideViewer)
        assert viewer.show_table_of_contents is True
        await pilot.resize_terminal(70, 40)
        await pilot.pause()
        assert viewer.show_table_of_contents is False


async def test_tui_exposes_workers_and_the_quarterly_step():
    """The two knobs that were configurable but invisible: n_jobs and QS.

    ``n_jobs`` had no widget at all, and the Frequency caption advertised only
    YS/MS/W, so a quarterly walk --- which the runner has always supported ---
    looked impossible from the UI.
    """
    app = HarnessTUI()
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app._build_config().n_jobs == 1               # inline by default
        app.query_one("#n_jobs", Input).value = "4"
        app.query_one("#frequency", Input).value = "QS"
        await pilot.pause()
        cfg = app._build_config()
        assert cfg.n_jobs == 4 and cfg.frequency == "QS"

        # both survive a Load, so a saved config round-trips through the form
        app._apply_config(RunConfig(n_jobs=6, frequency="2QS"))
        await pilot.pause()
        assert app.query_one("#n_jobs", Input).value == "6"
        assert app.query_one("#frequency", Input).value == "2QS"

        cells = {str(c.children[1].content).strip() for c in app.query(".cell")}
        assert any("QS" in note for note in cells), "no caption mentions QS"
