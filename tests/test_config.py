"""RunConfig validation + window semantics."""
import pathlib
import tomllib
import pytest

import pandas as pd

from harness.config import ASSET_PRESETS, RunConfig

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ok(**kw) -> RunConfig:
    base = dict(run_name="test_config", logging=False)
    base.update(kw)
    return RunConfig(**base)


def test_symbols_all_expands_to_asset_class_universe():
    # the default universe is "all" of the default class (equity)
    assert RunConfig(run_name="t", logging=False).symbols == list(ASSET_PRESETS["equity"])
    # "all" follows whichever asset class is selected
    for cls in ("equity", "etf", "crypto", "fx"):
        assert _ok(asset_class=cls, symbols=["all"]).symbols == list(ASSET_PRESETS[cls])
    # case-insensitive, and the bare-string form works too
    assert _ok(asset_class="crypto", symbols=["ALL"]).symbols == list(ASSET_PRESETS["crypto"])
    assert _ok(asset_class="fx", symbols="all").symbols == list(ASSET_PRESETS["fx"])
    # an explicit list is left untouched
    assert _ok(symbols=["AAPL", "MSFT"]).symbols == ["AAPL", "MSFT"]
    # a lone non-'all' ticker is NOT the sentinel (no accidental expansion)
    assert _ok(symbols=["ALLY"]).symbols == ["ALLY"]
    # expansion happens before the empty-universe guard, so 'all' never trips it
    assert len(_ok(symbols=["all"]).symbols) >= 1


def test_bare_string_symbols_normalised_to_list():
    # A bare non-'all' string (e.g. `configure --set symbols=AAPL`) must become a
    # list, not slip past validation as a truthy str and later crash in the data
    # layer (pandas would treat the str as a per-character column collection).
    assert _ok(symbols="AAPL").symbols == ["AAPL"]
    assert _ok(symbols="AAPL,MSFT").symbols == ["AAPL", "MSFT"]
    assert _ok(symbols=" AAPL , MSFT ").symbols == ["AAPL", "MSFT"]
    # an empty / whitespace-only string is an empty universe -> rejected
    for bad in ("", "   ", ",", " , ", []):
        try:
            _ok(symbols=bad)
        except ValueError as exc:
            assert "symbol" in str(exc).lower()
            continue
        raise AssertionError(f"empty symbols {bad!r} must be rejected")


def test_removed_and_unknown_config_keys_rejected():
    for key in ("simplified", "judge_prescreen", "judge_model", "totally_bogus"):
        data = RunConfig(run_name="t", logging=False).to_dict()
        data[key] = True
        with pytest.raises(ValueError, match="unknown config keys"):
            RunConfig.from_dict(data)


def test_data_dir_defaults_and_roundtrips():
    assert RunConfig(run_name="t", logging=False).data_dir == "data"
    d = RunConfig(run_name="t", logging=False, data_dir="mydata").to_dict()
    assert RunConfig.from_dict(d).data_dir == "mydata"


def test_half_open_windows_rejected():
    # only one bound of a window set -> would silently fall back to the clock
    for kw in [dict(is_start="2020-01-01"), dict(is_end="2020-12-31"),
               dict(oos_start="2021-01-01"), dict(oos_end="2021-06-30")]:
        try:
            _ok(**kw)
        except ValueError:
            continue
        raise AssertionError(f"half-open window {kw} must be rejected")

    # both bounds together is fine
    _ok(is_start="2018-01-01", is_end="2020-12-31",
        oos_start="2021-01-01", oos_end="2021-12-31")


def test_numeric_validation():
    bad = [
        dict(is_years=0), dict(oos_days=0), dict(oos_days=-5),
        dict(max_iters=0), dict(min_is_days=0), dict(min_oos_days=0), dict(cost_bps=-1),
        dict(gross=0), dict(gross=-1), dict(initial_cash=0),
        dict(annualization=0), dict(pass_line=float("nan")),
        dict(pass_line=float("inf")),
        dict(selection_mode="percentile"), dict(top_q=0), dict(top_q=1.1),
        dict(top_k=0),
        dict(hold_every=0), dict(rebalance_every=0),
        dict(oos_mode="calendar"), dict(warmup=-1),
    ]
    for kw in bad:
        try:
            _ok(**kw)
        except ValueError:
            continue
        raise AssertionError(f"invalid numeric config {kw} must be rejected")

    # a NEGATIVE pass_line is legitimate (e.g. a maxdd Pass Line of -0.2)
    _ok(objective="maxdd", pass_line=-0.2)


def test_backtest_weight_config_roundtrips():
    cfg = _ok(selection_mode="top_q", top_q=0.2, top_k=4,
              hold_every=2, rebalance_every=2)
    assert cfg.backtest_weight_config() == {
        "selection_mode": "top_q",
        "top_q": 0.2,
        "top_k": 4,
        "hold_every": 2,
        "rebalance_every": 2,
    }


def test_oos_window_counts_trading_days():
    # research_date on a Monday; 10 trading days -> lands 10 business days later
    cfg = _ok(research_date="2022-01-03", oos_days=10)
    start, end = cfg.oos_window()
    assert start == pd.Timestamp("2022-01-03")
    # BDay(10) skips weekends -> more than 10 calendar days
    assert (end - start).days > 10
    # exactly 10 business days after the start
    assert end == pd.Timestamp("2022-01-03") + pd.tseries.offsets.BDay(10)


def test_oos_mode_defaults_to_fixed_days_and_roundtrips():
    cfg = _ok()
    assert cfg.oos_mode == "fixed_days" and cfg.warmup == 0
    assert RunConfig.from_dict(cfg.to_dict()).oos_mode == "fixed_days"


def test_shipped_run_config_matches_offline_defaults():
    shipped = RunConfig.from_json(str(ROOT / "run_config.json"))
    assert shipped.to_dict() == RunConfig().to_dict()
    assert shipped.provider == "scripted"
    assert shipped.data_source == "synthetic"
    assert shipped.n_jobs == 1 and shipped.candidates_per_round == 1
    assert shipped.walk_forward is False


def test_field_config_loads_operator_parquet_from_a_temporary_checkout(tmp_path, monkeypatch):
    from harness import data

    cfg = RunConfig.from_json(str(ROOT / "field_config.json"))
    assert cfg.data_source == "ctx"
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    monkeypatch.setattr(data, "_harness_root", lambda: checkout)
    monkeypatch.setattr(data, "_state_dir", lambda: checkout / "artifacts")
    paths = cfg.parquet_daily if isinstance(cfg.parquet_daily, list) \
        else [cfg.parquet_daily]
    assert paths
    dates = pd.bdate_range("2021-01-04", periods=20)
    frames = []
    for number, symbol in enumerate(cfg.symbols):
        prices = [100.0 + number + day / 10 for day in range(len(dates))]
        frames.append(pd.DataFrame({
            "Datetime": dates, "Symbol": symbol, "Open": prices,
            "High": [price + 1 for price in prices],
            "Low": [price - 1 for price in prices],
            "Close": [price + 0.5 for price in prices],
            "Volume": 1000.0, "Vwap": [price + 0.25 for price in prices],
        }))
    fixture = pd.concat(frames, ignore_index=True)
    for path in paths:
        relative = pathlib.Path(path)
        assert not relative.is_absolute() and ".." not in relative.parts
        parquet = checkout / relative
        parquet.parent.mkdir(parents=True, exist_ok=True)
        fixture.to_parquet(parquet, index=False)

    # Private operator data is deliberately absent from Git and CI. Exercise the
    # shipped relative paths and real CTX loader without reading the live data.
    monkeypatch.chdir(tmp_path)
    assert data.resolve_parquet_paths(paths) == [str(checkout / path) for path in paths]
    panel = data.load_panel(cfg)
    assert panel.symbols == cfg.symbols
    assert panel.fields["close"].notna().any().all()
    assert panel.vwap_source == "source"


def test_data_extra_contains_the_real_ctx_execution_stack():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = project["project"]["optional-dependencies"]["data"]
    joined = "\n".join(deps).lower()
    for package in ("yfinance", "pyarrow", "vectorbt", "ta-lib", "numba",
                    "duckdb", "cloudpickle", "matplotlib", "rich"):
        assert package in joined, f".[data] is missing {package}"
