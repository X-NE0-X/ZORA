"""The `configure` CLI subcommand: write a valid run_config.json from defaults
plus --set overrides, with JSON-aware value coercion and validation."""
import os
import contextlib
import io
import tempfile

from harness import cli
from harness.config import RunConfig


def _run(argv):
    return cli.main(argv)


def test_configure_writes_and_coerces_types():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "cfg.json")
        code = _run([
            "configure", "--out", out,
            "--set", "pass_line=1.5",                 # float
            "--set", "max_iters=3",                   # int
            "--set", "require_interpretability=true",  # bool
            "--set", 'symbols=["AAPL","MSFT","GOOGL"]',  # list
            "--set", "objective=sharpe",              # bare string
        ])
        assert code == 0
        cfg = RunConfig.from_json(out)
        assert cfg.pass_line == 1.5 and isinstance(cfg.pass_line, float)
        assert cfg.max_iters == 3 and isinstance(cfg.max_iters, int)
        assert cfg.require_interpretability is True
        assert cfg.symbols == ["AAPL", "MSFT", "GOOGL"]
        assert cfg.objective == "sharpe"


def test_configure_round_trips_from_base_config():
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base.json")
        RunConfig(run_name="base_run", max_iters=7).to_json(base)
        out = os.path.join(d, "out.json")
        code = _run(["configure", "--config", base, "--out", out,
                     "--set", "pass_line=0.8"])
        assert code == 0
        cfg = RunConfig.from_json(out)
        assert cfg.run_name == "base_run"     # inherited from base
        assert cfg.max_iters == 7             # inherited from base
        assert cfg.pass_line == 0.8           # overridden


def test_configure_defaults_out_to_config_path():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "inplace.json")
        RunConfig(run_name="r").to_json(path)
        code = _run(["configure", "--config", path, "--set", "seed=99"])
        assert code == 0
        assert RunConfig.from_json(path).seed == 99   # written back in place


def test_configure_rejects_unknown_key():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "cfg.json")
        code = _run(["configure", "--out", out, "--set", "not_a_field=1"])
        assert code == 2
        assert not os.path.exists(out)        # nothing written on error


def test_configure_rejects_invalid_value():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "cfg.json")
        # max_iters must be >= 1 -> RunConfig.__post_init__ rejects it
        code = _run(["configure", "--out", out, "--set", "max_iters=0"])
        assert code == 2
        assert not os.path.exists(out)


def test_configure_rejects_malformed_set():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "cfg.json")
        code = _run(["configure", "--out", out, "--set", "no_equals_sign"])
        assert code == 2


def test_all_cli_commands_report_invalid_config_without_traceback():
    with tempfile.TemporaryDirectory() as d:
        bad = os.path.join(d, "bad.json")
        with open(bad, "w", encoding="utf-8") as fh:
            fh.write('{"max_iters": 0}')
        for command in ("run", "show", "log", "replay", "doctor"):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = _run([command, "--config", bad])
            text = out.getvalue()
            assert code == 2, (command, code, text)
            assert "invalid configuration" in text
            assert "Traceback" not in text
