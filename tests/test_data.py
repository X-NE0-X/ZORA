"""Data ingest: the CTX/yfinance tail must not silently shrink the universe.

``_panel_from_wide`` is the shared tail for both online (yfinance) and local
(ctx) parquet. When a caller asks for a specific universe, a symbol the source
dropped (no Close, or all-NaN Close) must raise rather than quietly evaluate the
factor on a smaller cross-section -- that is survivorship bias, so the refusal is
the default and the opt-out (``allow_missing_symbols``) has to be explicit.

Also covered here: the price fields are named for what they hold
(``typical_price`` is (H+L+C)/3, ``vwap`` is a real volume-weighted price only
when the source supplies one), the provenance string records what changed the
numbers, and nothing the harness generates is written into the process CWD or
into site-packages.
"""
import json
import os
import pathlib
import sys
import tempfile
import types
import warnings

import numpy as np
import pandas as pd

from harness import data
from harness.data import _panel_from_wide

IDX = pd.bdate_range("2020-01-01", periods=12)


def _wide(with_symbols, *, nan_close=(), vwap=()):
    cols = {}
    for s in with_symbols:
        for f in ("Open", "High", "Low", "Close", "Volume"):
            vals = np.arange(1, len(IDX) + 1, dtype=float)
            if f == "Close" and s in nan_close:
                vals = np.full(len(IDX), np.nan)
            cols[f"{s}_{f}"] = vals
        if s in vwap:
            # deliberately NOT equal to (High+Low+Close)/3, so a test can tell a
            # real source VWAP apart from the typical-price proxy
            cols[f"{s}_Vwap"] = np.arange(1, len(IDX) + 1, dtype=float) * 7.0
    return pd.DataFrame(cols, index=IDX)


def test_missing_symbol_raises():
    # C is present for Open only (no Close column) -> unusable
    wide = _wide(["A", "B"])
    wide["C_Open"] = np.arange(1, len(IDX) + 1, dtype=float)
    try:
        _panel_from_wide(wide, "ctx", expected=["A", "B", "C"])
    except ValueError as exc:
        assert "C" in str(exc)
        return
    raise AssertionError("a requested symbol with no Close must raise")


def test_all_nan_close_counts_as_missing():
    wide = _wide(["A", "B", "D"], nan_close=["D"])
    try:
        _panel_from_wide(wide, "ctx", expected=["A", "B", "D"])
    except ValueError as exc:
        assert "D" in str(exc)
        return
    raise AssertionError("an all-NaN Close must count as a missing symbol")


def test_missing_symbol_error_names_survivorship_and_the_opt_out():
    """The fatal default must say WHY it is fatal, not just that it is.

    Refusing an incomplete universe is a bias decision (dropping the names with
    no coverage leaves the survivors), so the message has to name the bias and
    the knob that opts into it -- otherwise a user just widens the date range
    until the error goes away and never learns what was traded off.
    """
    wide = _wide(["A", "B"])
    try:
        _panel_from_wide(wide, "ctx", expected=["A", "B", "GONE"])
    except ValueError as exc:
        msg = str(exc).lower()
        assert "survivorship" in msg, "the consequence must be named"
        assert "allow_missing_symbols" in msg, "the opt-out must be named"
        return
    raise AssertionError("incomplete coverage must be fatal by default")


def test_allow_missing_opts_into_a_survivors_only_panel_loudly():
    wide = _wide(["A", "B"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        panel = _panel_from_wide(wide, "ctx", expected=["A", "B", "GONE"],
                                 allow_missing=True)
    assert panel.symbols == ["A", "B"], "the survivors still build a panel"
    assert any("survivorship" in str(w.message).lower() for w in caught), \
        "opting in must still warn that the metrics are survivorship-biased"


def test_expected_subset_ok_and_ordered():
    wide = _wide(["B", "A"])            # column order deliberately reversed
    panel = _panel_from_wide(wide, "ctx", expected=["A", "B"])
    assert panel.symbols == ["A", "B"], "must follow the requested order"
    assert panel.source.startswith("ctx")


def test_no_expected_uses_all_usable():
    wide = _wide(["A", "B"])
    panel = _panel_from_wide(wide, "yfinance")
    assert panel.symbols == ["A", "B"]


# =====================================================================
# the price fields are named for what they actually hold
# =====================================================================
def test_typical_price_is_hlc3_and_is_not_called_vwap():
    """(high+low+close)/3 has no volume term, so it is not a VWAP.

    It used to be published under the name ``vwap``, which made every
    "volume-weighted execution price" rationale written about it false.
    """
    panel = data.make_synthetic(["A", "B"], "2020-01-01", "2020-03-01")
    f = panel.fields
    expected = (f["high"] + f["low"] + f["close"]) / 3.0
    pd.testing.assert_frame_equal(f["typical_price"], expected)
    assert panel.vwap_source == "hlc3"
    # no source can volume-weight a one-bar-a-day GBM, so vwap is the proxy here
    pd.testing.assert_frame_equal(f["vwap"], f["typical_price"])
    assert "vwap=hlc3" in panel.source, "the substitution must be in the provenance"


def test_a_real_source_vwap_is_kept_not_overwritten_by_the_proxy():
    """A user who brings genuine VWAP data must not have it silently discarded."""
    wide = _wide(["A", "B"], vwap=["A", "B"])
    panel = _panel_from_wide(wide, "ctx", expected=["A", "B"])
    assert panel.vwap_source == "source"
    assert "vwap=source" in panel.source
    pd.testing.assert_frame_equal(
        panel.fields["vwap"],
        pd.DataFrame({s: np.arange(1, len(IDX) + 1, dtype=float) * 7.0
                      for s in ["A", "B"]}, index=panel.dates),
    )
    assert not panel.fields["vwap"].equals(panel.fields["typical_price"]), \
        "the real VWAP must survive the tail, not be replaced by HLC3"


def test_partial_source_vwap_falls_back_to_the_proxy_with_a_warning():
    # mixing a genuine VWAP for one name with NaN for another would make the
    # cross-section incomparable, so the whole panel falls back -- loudly.
    wide = _wide(["A", "B"], vwap=["A"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        panel = _panel_from_wide(wide, "ctx", expected=["A", "B"])
    assert panel.vwap_source == "hlc3"
    pd.testing.assert_frame_equal(panel.fields["vwap"], panel.fields["typical_price"])
    assert any("only 1 of 2" in str(w.message) for w in caught)


def test_panel_version_and_slice_carry_the_vwap_provenance():
    real = _panel_from_wide(_wide(["A", "B"], vwap=["A", "B"]), "ctx",
                            expected=["A", "B"])
    proxy = _panel_from_wide(_wide(["A", "B"]), "ctx", expected=["A", "B"])
    assert real.version != proxy.version, \
        "two panels whose vwap means different things are not the same data"
    assert real.slice().vwap_source == "source", "slice must not lose provenance"


def test_test_data_frame_stays_ohlcv_only():
    # the vendored backtest speaks {SYMBOL}_{Open,High,Low,Close,Volume}; the
    # derived price fields must not leak extra columns into it.
    panel = _panel_from_wide(_wide(["A"], vwap=["A"]), "ctx", expected=["A"])
    assert set(panel.test_data.columns) == {
        "A_Open", "A_High", "A_Low", "A_Close", "A_Volume", "Datetime"
    }


# =====================================================================
# provenance: what silently changed the numbers has to be on the record
# =====================================================================
class _FakeYF(types.ModuleType):
    """Minimal stand-in for yfinance that records how it was called."""

    def __init__(self):
        super().__init__("yfinance")
        self.kwargs = {}

    def download(self, symbols, **kwargs):
        self.kwargs = kwargs
        cols = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], list(symbols)])
        vals = np.tile(np.arange(1, len(IDX) + 1, dtype=float)[:, None],
                       (1, len(cols)))
        return pd.DataFrame(vals, index=IDX, columns=cols)


def test_yfinance_records_auto_adjust_in_the_panel_provenance():
    """auto_adjust=True back-fills today's split/dividend factors over history.

    The price at date t is then not the price a trader saw at t, so the run
    record has to say so: it lands in ``source`` (and therefore in the version
    hash and in ``data_source`` on every record).
    """
    fake = _FakeYF()
    real_yf = sys.modules.get("yfinance")
    real_ctx_wide = data._ctx_wide
    sys.modules["yfinance"] = fake
    data._ctx_wide = lambda paths, symbols, start, end: _wide(list(symbols))
    try:
        panel = data.load_yfinance(["A", "B"], "2020-01-01", "2020-02-01")
    finally:
        data._ctx_wide = real_ctx_wide
        if real_yf is None:
            del sys.modules["yfinance"]
        else:
            sys.modules["yfinance"] = real_yf
    assert fake.kwargs.get("auto_adjust") is True, "guards the premise of the test"
    assert panel.source == "yfinance(auto_adjust=True,vwap=hlc3)"


# =====================================================================
# nothing the harness generates lands in the CWD or in site-packages
# =====================================================================
def test_generated_state_lives_outside_the_package_when_not_a_checkout():
    """A non-editable install must not write next to the package.

    ``site-packages`` is wiped by ``pip install -U`` and is read-only under a
    system Python, so the parquet cache and the CTX logs move to the per-user
    cache dir there. A source checkout keeps the in-repo path.
    """
    assert data._is_source_checkout(), "this repo is a checkout (pyproject.toml)"
    assert data._state_dir() == data._harness_root() / "artifacts"

    real = data._is_source_checkout
    data._is_source_checkout = lambda: False
    try:
        installed = data._state_dir()
    finally:
        data._is_source_checkout = real
    assert data._harness_root() not in installed.parents, \
        "an installed harness must not write into its own package tree"
    assert installed.name == "zora-harness"
    assert installed.is_absolute()


def test_yf_cache_and_ctx_logs_live_under_the_state_dir():
    assert data._yf_cache_dir() == data._state_dir() / "cache"
    work = data._ctx_work_dir()
    assert work == data._state_dir() / "ctx"
    assert (work / "ArchiveDeck" / "ComputerLog").is_dir()


def test_ctx_log_pruning_keeps_only_the_newest():
    with tempfile.TemporaryDirectory() as d:
        log_dir = pathlib.Path(d)
        for i in range(6):
            p = log_dir / f"run_{i}.log"
            p.write_text("x", encoding="utf-8")
            os.utime(p, (1_600_000_000 + i, 1_600_000_000 + i))
        (log_dir / "keepme.txt").write_text("x", encoding="utf-8")
        data._prune_ctx_logs(log_dir, keep=2)
        assert {p.name for p in log_dir.glob("*.log")} == {"run_4.log", "run_5.log"}
        assert (log_dir / "keepme.txt").is_file(), "only *.log is pruned"


def test_a_ctx_panel_build_writes_nothing_into_the_process_cwd():
    """gap-03: CTX defaults notebook_dir to the CWD and drops ArchiveDeck/ there.

    Runs a real (tiny) CTX load from a scratch directory and asserts the only
    thing in it afterwards is the parquet we put there. Skipped when the vendored
    infra's runtime deps are absent.
    """
    from harness import infra_engine
    try:
        infra_engine._ensure()
    except infra_engine.InfraUnavailable:
        return                                   # infra extras not installed

    syms = ["AAA", "BBB"]
    idx = pd.bdate_range("2021-01-04", periods=20)
    rng = np.random.default_rng(0)
    frames = []
    for s in syms:
        px = 100.0 + np.cumsum(rng.normal(0, 1, len(idx)))
        frames.append(pd.DataFrame({
            "Datetime": idx, "Symbol": s,
            "Open": px, "High": px + 1.0, "Low": px - 1.0, "Close": px + 0.5,
            "Volume": rng.lognormal(12, 0.2, len(idx)),
            "Vwap": px + 0.25,                   # a REAL volume-weighted price
        }))
    long_df = pd.concat(frames, ignore_index=True)

    with tempfile.TemporaryDirectory() as scratch:
        pq = pathlib.Path(scratch) / "Data_EQT_US_D_probe.parquet"
        long_df.to_parquet(pq, index=False)
        cwd = os.getcwd()
        os.chdir(scratch)
        try:
            wide = data._ctx_wide([str(pq)], syms, "2021-01-01", "2021-03-01")
        finally:
            os.chdir(cwd)
        leftovers = sorted(p.name for p in pathlib.Path(scratch).iterdir())
        assert leftovers == ["Data_EQT_US_D_probe.parquet"], \
            f"a panel build littered the CWD: {leftovers}"

    # and the Vwap column really does survive CTX into the panel
    panel = data._panel_from_wide(wide, source="ctx", expected=syms)
    assert panel.vwap_source == "source"
    assert not panel.fields["vwap"].equals(panel.fields["typical_price"])


def test_discover_parquet_surfaces_only_ctx_named():
    # Only files whose name carries a CTX session profile
    # (Data_<TYPE>_<REGION>_D*.parquet, TYPE in EQT/ETF/INDEX/FX/SPOT/DIGITAL) are
    # offered;
    # scans the dir + one level of sub-dirs. Content is irrelevant (name-only check).
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "Data_EQT_US_D_abc.parquet").write_bytes(b"x")     # keep
        (root / "Data_FX_G10_D_2024.parquet").write_bytes(b"x")    # keep
        sub = root / "sub"
        sub.mkdir()
        (sub / "Data_ETF_US_D_x.parquet").write_bytes(b"x")        # keep (one level)
        (root / "random.parquet").write_bytes(b"x")                # no Data_ prefix
        (root / "Data_BOGUS_US_D_x.parquet").write_bytes(b"x")     # bad asset token
        (root / "EQT_DATA_D.csv").write_bytes(b"x")                # not parquet
        found = data.discover_parquet(str(root))
        assert {pathlib.Path(p).name for p in found} == {
            "Data_EQT_US_D_abc.parquet",
            "Data_FX_G10_D_2024.parquet",
            "Data_ETF_US_D_x.parquet",
        }
        assert found == sorted(found)          # stable, sorted output


def test_listed_types_require_a_region_ctx_actually_knows():
    # CTX resolves a LISTED type's region against a closed set
    # (DEFAULT_TIME_PROFILES['regions'] == US/HK/CN/UK/EU); an unknown token
    # falls through to an empty profile and ctx_portfolio then REFUSES the file.
    # The scan used to accept any region, so the TUI offered names the loader
    # rejected with "requires region/session-profiled parquet names" -- which
    # reads as a corrupt file rather than a mis-named one.
    for good in ("Data_EQT_US_D_k.parquet", "Data_ETF_HK_D_k.parquet",
                 "Data_INDEX_EU_D_k.parquet", "Data_EQT_us_D_k.parquet"):
        assert data._is_ctx_parquet_name(good), good
    for bad in ("Data_EQT_XX_D_k.parquet",      # region CTX does not know
                "Data_ETF_G10_D_k.parquet",     # G10 is an OTC-style token
                "Data_INDEX_D.parquet"):        # listed type with no region at all
        assert not data._is_ctx_parquet_name(bad), bad
    # OTC types stay region-less by design and must not regress.
    for otc in ("Data_FX_D.parquet", "Data_SPOT_D_2024.parquet",
                "Data_DIGITAL_D_k.parquet", "Data_FX_G10_D_2024.parquet"):
        assert data._is_ctx_parquet_name(otc), otc


def test_discover_parquet_daily_and_depth_boundaries():
    # region-less OTC daily (FX/SPOT carry NO region) must be accepted; non-daily
    # names and files two levels deep must be excluded.
    with tempfile.TemporaryDirectory() as d:
        root = pathlib.Path(d)
        (root / "Data_FX_D.parquet").write_bytes(b"x")           # keep (OTC, no region)
        (root / "Data_SPOT_D.parquet").write_bytes(b"x")         # keep (OTC, no region)
        (root / "Data_DIGITAL_D.parquet").write_bytes(b"x")      # keep (OTC, no region)
        (root / "Data_EQT_US_D.parquet").write_bytes(b"x")       # keep (minimal 4-token)
        (root / "Data_EQT_US_W_x.parquet").write_bytes(b"x")     # drop (weekly, no D)
        (root / "Data_EQT_US.parquet").write_bytes(b"x")         # drop (no daily marker)
        deep = root / "a" / "b"
        deep.mkdir(parents=True)
        (deep / "Data_EQT_US_D_deep.parquet").write_bytes(b"x")  # drop (two levels deep)
        found = {pathlib.Path(p).name for p in data.discover_parquet(str(root))}
        assert found == {"Data_FX_D.parquet", "Data_SPOT_D.parquet",
                         "Data_DIGITAL_D.parquet", "Data_EQT_US_D.parquet"}


def test_discover_parquet_empty_or_missing_dir():
    assert data.discover_parquet("") == []
    assert data.discover_parquet(None) == []
    with tempfile.TemporaryDirectory() as d:
        assert data.discover_parquet(str(pathlib.Path(d) / "does_not_exist")) == []


def test_relative_parquet_path_anchors_to_the_harness_root_not_the_cwd():
    """A shipped config must not depend on where the process was launched.

    ``field_config.json`` carries a repo-relative parquet path so it holds no
    machine-specific absolute path. That only works if a relative entry is
    anchored to the harness root --- the same rule ``data_dir`` follows --- so
    running from the repo root (or anywhere else) resolves to the same file.
    """
    root = data._harness_root()
    rel = "artifacts/cache/Data_EQT_US_D_abc.parquet"
    expected = str(root / "artifacts" / "cache" / "Data_EQT_US_D_abc.parquet")

    cwd = os.getcwd()
    try:
        os.chdir(tempfile.gettempdir())          # deliberately NOT the harness root
        assert data.resolve_parquet_paths(rel) == [expected]
        assert data.resolve_parquet_paths([rel]) == [expected]
    finally:
        os.chdir(cwd)

    # absolute entries are passed through untouched
    absolute = str(pathlib.Path(tempfile.gettempdir()) / "Data_FX_D.parquet")
    assert data.resolve_parquet_paths([absolute]) == [absolute]


def test_discover_returns_portable_paths_for_files_under_the_harness_root():
    """What the TUI auto-fills gets SAVED, so it must not re-pin an absolute path.

    A hit inside the harness root comes back relative to it (forward slashes, so
    the saved string is the same on Windows and POSIX) and still resolves back to
    the very same file. A hit outside can only be named absolutely.
    """
    probe = data._harness_root() / "artifacts" / "_discover_probe"
    probe.mkdir(parents=True, exist_ok=True)
    target = probe / "Data_EQT_US_D_probe.parquet"
    target.write_bytes(b"x")
    try:
        found = data.discover_parquet("artifacts/_discover_probe")
        assert found == ["artifacts/_discover_probe/Data_EQT_US_D_probe.parquet"]
        assert data.resolve_parquet_paths(found) == [str(target)], \
            "the portable form must round-trip back to the same file"
    finally:
        target.unlink()
        probe.rmdir()

    with tempfile.TemporaryDirectory() as d:          # outside the harness root
        outside = pathlib.Path(d) / "Data_FX_D.parquet"
        outside.write_bytes(b"x")
        assert data.discover_parquet(str(d)) == [str(outside)]


def test_field_config_carries_no_absolute_path():
    """The shipped configs must stay machine-independent (open-source hygiene)."""
    root = data._harness_root()
    for name in ("run_config.json", "field_config.json"):
        path = root / name
        if not path.is_file():
            continue
        cfg = json.loads(path.read_text(encoding="utf-8"))
        paths = cfg.get("parquet_daily") or []
        if isinstance(paths, str):
            paths = [paths]
        for p in paths:
            assert not pathlib.Path(p).is_absolute(), \
                f"{name} pins an absolute parquet path: {p}"
            assert ":" not in p and not p.startswith("\\\\"), \
                f"{name} pins a machine-specific path: {p}"
