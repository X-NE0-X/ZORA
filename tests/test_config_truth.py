"""Configuration claims must match the Codex invocation, without live requests."""
import json
import contextlib
import io
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from textual.widgets import Input, Select, Switch, Label

from harness.config import RunConfig
from harness import cli, preflight
from harness.providers.codex import CodexProvider
from harness.providers.codex_controls import (
    ProviderConfigurationError, reasoning_levels, validate_controls,
)
from harness.providers.replay import RecordingProvider, ReplayProvider
from harness.tui import HarnessTUI
from harness.runner import _config_conflicts


def _reject(fn, fragment):
    try:
        fn()
    except ProviderConfigurationError as exc:
        assert fragment in str(exc), str(exc)
    else:
        raise AssertionError("invalid configuration was accepted")


def test_provider_defaults_are_explicit():
    cfg = RunConfig(provider="codex", reasoning_effort="max")
    assert cfg.model == "gpt-6-luna"
    assert RunConfig.from_dict(cfg.to_dict()) == cfg
    assert RunConfig().model == "claude-opus-4-8"


def test_invalid_model_and_effort_fail_before_request():
    _reject(lambda: RunConfig(provider="codex", model="claude-opus-4-8"), "incompatible")
    _reject(lambda: RunConfig(provider="codex", model="gpt-6-luna", reasoning_effort="ultra"), "ultra")
    for effort in ("bad", 'max" injected'):
        with pytest.raises(ValueError):
            RunConfig(provider="codex", reasoning_effort=effort)



def test_mutated_config_is_blocked_by_preflight_before_runtime_or_login():
    cfg = RunConfig(provider="codex")
    cfg.model = "claude-haiku"
    with patch("harness.preflight.check_all") as check:
        ready = preflight.check_config(cfg)
    assert not ready.ok and "incompatible" in ready.reason
    check.assert_not_called()


def test_codex_has_no_constructor_override_or_foreign_model_fallback():
    with patch("harness.providers.codex.ensure_binary", return_value="codex"), \
         patch("harness.providers.codex.run_cli") as call:
        provider = CodexProvider()
        _reject(lambda: provider.complete("sys", [], model="claude-haiku"), "incompatible")
        _reject(lambda: provider.complete("sys", [], model=None), "explicit")
        _reject(lambda: CodexProvider(model="gpt-6-sol").complete(
            "sys", [], model="gpt-6-luna"), "conflicts")
        _reject(lambda: provider.complete("sys", [], model="gpt-6-luna", reasoning_effort="ultra"), "ultra")
        call.assert_not_called()


def test_catalog_capabilities_and_unknown_model_are_honest():
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "models_cache.json"
        path.write_text(json.dumps({"models": [{"slug": "gpt-test", "supported_reasoning_levels":
                                                [{"effort": "low"}, {"effort": "high"}]}]}), encoding="utf-8")
        with patch.dict("os.environ", {"CODEX_HOME": root}):
            assert reasoning_levels("gpt-test") == ("low", "high")
            assert "ultra" not in reasoning_levels("gpt-6-luna")
            _reject(lambda: validate_controls("gpt-test", "max"), "max")
            _reject(lambda: validate_controls("gpt-unknown", "high"), "unknown")
            assert validate_controls("gpt-unknown", None) == "gpt-unknown"
        path.write_text("bad json", encoding="utf-8")
        with patch.dict("os.environ", {"CODEX_HOME": root}):
            assert "max" in reasoning_levels("gpt-6-luna")


def test_exact_cli_controls_survive_record_and_replay():
    commands = []
    events = json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}})
    with patch("harness.providers.codex.ensure_binary", return_value="codex"), \
         patch("harness.providers.codex.run_cli", side_effect=lambda cmd, **kw: commands.append(cmd) or events):
        recording = RecordingProvider(CodexProvider())
        response = recording.complete("sys", [], model="gpt-6-luna", reasoning_effort="max", seed=42,
                                      temperature=0.7, max_tokens=123)
    assert commands[0][commands[0].index("-m") + 1] == "gpt-6-luna"
    assert 'model_reasoning_effort="max"' in commands[0]
    assert 'forced_login_method="chatgpt"' in commands[0]
    old = RunConfig().to_dict()
    old.pop("reasoning_effort")
    assert not _config_conflicts(old, RunConfig().to_dict())
    cfg = RunConfig(provider="codex", reasoning_effort="max")
    assert _config_conflicts(cfg.to_dict(), {**cfg.to_dict(), "reasoning_effort": "high"}) == ["reasoning_effort"]
    controls = response.meta["configuration"]
    assert controls["requested_model"] == controls["selected_model"] == "gpt-6-luna"
    assert controls["unsupported_llm_controls"] == ["seed", "temperature", "max_tokens"]
    assert recording.trace[0]["configuration"] == controls
    assert recording.trace[0]["reasoning_effort"] == "max"
    replay = ReplayProvider(recording.trace).complete("sys", [], model="gpt-6-luna", reasoning_effort="max")
    assert replay.meta == response.meta
    from harness.providers.replay import ReplayDesyncError
    try:
        ReplayProvider(recording.trace).complete("sys", [], model="gpt-6-sol", reasoning_effort="max")
    except ReplayDesyncError:
        pass
    else:
        raise AssertionError("requested-model drift was ignored")
    with pytest.raises(ReplayDesyncError):
        ReplayProvider(recording.trace).complete("sys", [], model="gpt-6-luna", reasoning_effort="high")


def test_cli_provider_switch_is_visible_and_preserves_explicit_choices():
    with tempfile.TemporaryDirectory() as root:
        out = str(Path(root) / "cfg.json")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            assert cli.main(["configure", "--out", out, "--set", "provider=codex",
                             "--set", "reasoning_effort=max"]) == 0
        assert "provider default: model=gpt-6-luna" in stdout.getvalue()
        cfg = RunConfig.from_json(out)
        assert cfg.model == "gpt-6-luna"
        args = cli.build_parser().parse_args(["doctor", "--provider", "codex"])
        assert cli._load_config(args).model == "gpt-6-luna"
        bad = str(Path(root) / "bad.json")
        with contextlib.redirect_stdout(io.StringIO()):
            assert cli.main(["configure", "--out", bad, "--set", "provider=codex",
                             "--set", "model=claude-opus-4-8"]) == 2
        assert not Path(bad).exists()


async def test_tui_codex_options_and_unsupported_controls():
    app = HarnessTUI(config=RunConfig(provider="codex", reasoning_effort="max"))
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        assert app.query_one("#max_tokens", Input).disabled
        assert app.query_one("#temperature", Input).disabled
        assert not app.query_one("#seed", Input).disabled
        assert "unsupported" in str(app.query_one("#provider_controls", Label).render())
        effort = app.query_one("#reasoning_effort", Select)
        try:
            effort.value = "ultra"
        except Exception:
            pass
        else:
            raise AssertionError("Luna picker offered Ultra")
        assert app._build_config().reasoning_effort == "max"
        app.query_one("#model", Input).value = "gpt-5.5"
        await pilot.pause()
        assert effort.value == "max"  # never silently downgrade the configured effort
        _reject(app._build_config, "max")


async def test_provider_switch_defaults_and_loaded_custom_values():
    app = HarnessTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#provider", Select).value = "codex"
        await pilot.pause()
        assert app._build_config().model == "gpt-6-luna"
        assert not app.query("#judge_prescreen")
        assert not app.query("#judge_model")
        loaded = RunConfig(provider="codex", model="gpt-6-sol", reasoning_effort="max")
        app._apply_config(loaded)
        await pilot.pause()
        assert app._build_config().model == "gpt-6-sol"
        assert app._build_config().reasoning_effort == "max"
