"""Market-data panel + pluggable sources.

A ``Panel`` holds two aligned views of the same daily OHLCV basket:

  * ``fields`` --- one (date x symbol) DataFrame per field, consumed by the
    safe-AST factor evaluator (``factor/evaluate.py``);
  * ``test_data`` --- the *wide* ``{SYMBOL}_{Field}`` frame (plus a ``Datetime``
    column) that the vendored infra speaks natively: FactorEngine's
    ``FactorManager`` derives its asset panel from the ``_Close`` columns, and
    BacktestEngine's portfolio backtest slices ``_Open``/``_Close`` per underlying.

Three sources ship, all producing the same schema:

  * ``make_synthetic`` --- deterministic GBM basket (offline, reproducible);
  * ``load_yfinance``  --- real daily OHLCV pulled online via yfinance, then
    transcoded to a local parquet and cleaned through the SAME
    CTX ``CTX.ctx_portfolio`` pipeline as local parquet;
  * ``load_ctx``       --- the user's own parquet, loaded + cleaned through
    CTX's ``CTX.ctx_portfolio`` pipeline.

``load_yfinance`` and ``load_ctx`` share one ingest tail (parquet -> CTX -> wide
``{SYMBOL}_{Field}`` -> Panel); they differ only in where the parquet comes from.

Two price fields are easy to confuse, so they are named apart:

  * ``typical_price`` --- ALWAYS ``(high + low + close) / 3``. That is the classic
    *typical price*; it has NO volume term in it.
  * ``vwap``          --- the source's genuine volume-weighted average price when
    the source supplies one, otherwise the ``typical_price`` proxy. Which of the
    two a panel actually holds is recorded in :attr:`Panel.vwap_source` and in the
    provenance string, and the D2 alignment check reads it (see
    :mod:`harness.alignment`) so a "volume-weighted execution" story cannot be
    stamped consistent against a panel that has no volume in the field.

``source`` is a provenance string, not just a name: it carries the tags that
change the numbers (``auto_adjust`` for yfinance, ``vwap=source|hlc3``) so a run
record says how its data was built. ``version`` is a content hash used for
reproducibility (it covers ``source``, so a provenance change is visible there
too).
"""
from __future__ import annotations

import hashlib
import logging
import os
import pathlib
import warnings
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field as _field

import numpy as np
import pandas as pd

FIELD_NAMES = ("open", "high", "low", "close", "volume",
               "typical_price", "vwap", "returns")
_OHLCV = (("open", "Open"), ("high", "High"), ("low", "Low"),
          ("close", "Close"), ("volume", "Volume"))
# Wide-frame columns the panel reads but the *infra* frame does not carry. CTX
# passes a ``Vwap`` column through to ``{SYMBOL}_Vwap`` and re-weights it by
# volume/turnover on resample (a REAL volume-weighted price), so it must be picked
# up here instead of being thrown away and overwritten by the HLC3 proxy. Kept out
# of ``_OHLCV`` because ``_wide_test_data`` feeds the vendored backtest, which
# speaks Open/High/Low/Close/Volume only.
_OPTIONAL_WIDE = (("vwap", "Vwap"),)


@dataclass
class Panel:
    """One daily OHLCV basket in both views, plus its data provenance.

    ``vwap_source`` says what the ``vwap`` field actually contains --- ``"source"``
    (a genuine volume-weighted price the data source supplied) or ``"hlc3"`` (the
    ``(high + low + close) / 3`` typical-price proxy, which has no volume term).
    It is part of the panel's identity: it is hashed into :attr:`version` and
    mirrored into :attr:`source`, and :mod:`harness.alignment` reads it to decide
    whether a volume-weighted narrative is truthful for this panel.
    """

    fields: dict[str, pd.DataFrame]
    symbols: list[str]
    source: str = "unknown"
    vwap_source: str = "hlc3"
    test_data: pd.DataFrame | None = _field(default=None, repr=False)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.fields["close"].index

    @property
    def version(self) -> str:
        """blake2b content hash over all field values + axes (reproducibility)."""
        h = hashlib.blake2b(digest_size=16)
        h.update(self.source.encode())
        h.update(self.vwap_source.encode())
        h.update(",".join(self.symbols).encode())
        idx = self.fields["close"].index
        h.update(np.asarray(idx.view("int64")).tobytes())
        for name in FIELD_NAMES:
            df = self.fields[name]
            h.update(name.encode())
            h.update(np.ascontiguousarray(df.to_numpy(dtype="float64")).tobytes())
        return h.hexdigest()

    def slice(self, start=None, end=None) -> "Panel":
        sub = {k: v.loc[start:end] for k, v in self.fields.items()}
        td = None
        if self.test_data is not None:
            td = self.test_data.loc[start:end]
        return Panel(fields=sub, symbols=list(self.symbols),
                     source=self.source, vwap_source=self.vwap_source,
                     test_data=td)


def _wide_test_data(fields: dict[str, pd.DataFrame], symbols: list[str]) -> pd.DataFrame:
    """Assemble the wide ``{SYMBOL}_{Field}`` frame the infra expects."""
    cols: dict[str, pd.Series] = {}
    for sym in symbols:
        for lo, hi in _OHLCV:
            cols[f"{sym}_{hi}"] = fields[lo][sym]
    wide = pd.DataFrame(cols, index=fields["close"].index)
    wide.index.name = "Datetime"
    wide["Datetime"] = wide.index
    return wide


def _panel_from_ohlcv(open_, high, low, close, volume, symbols, source,
                      *, vwap=None, tags=()) -> Panel:
    """Derive the remaining fields from OHLCV and wrap everything in a Panel.

    ``typical_price`` is ALWAYS ``(high + low + close) / 3``. That average has no
    volume term in it, so it is named for what it is; the field used to be called
    ``vwap``, which made every "volume-weighted execution price" rationale written
    about it false.

    ``vwap`` holds the source's genuine volume-weighted price when ``vwap`` is
    passed (the CTX path supplies one whenever the parquet carries a ``Vwap``
    column), and otherwise falls back to the same typical-price proxy so that the
    DSL field set does not change shape from one data source to the next. The
    difference is not hidden: it is recorded on the panel (``vwap_source``), in the
    provenance string, and in the version hash, and D2 uses it to reject a
    volume-weighted story told about a proxy.

    ``tags`` are extra provenance tokens the caller wants recorded in ``source``
    (e.g. ``auto_adjust=True`` for a retroactively adjusted yfinance pull).
    """
    typical = (high + low + close) / 3.0
    vwap_source = "hlc3" if vwap is None else "source"
    returns = close.pct_change(fill_method=None)
    fields = {
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "typical_price": typical,
        "vwap": typical if vwap is None else vwap, "returns": returns,
    }
    provenance = f"{source}({','.join((*tags, f'vwap={vwap_source}'))})"
    return Panel(fields=fields, symbols=list(symbols), source=provenance,
                 vwap_source=vwap_source,
                 test_data=_wide_test_data(fields, list(symbols)))


def _panel_from_wide(wide: pd.DataFrame, source: str,
                     expected: list[str] | None = None, *,
                     allow_missing: bool = False, tags=()) -> Panel:
    """Build a Panel from a wide ``{SYMBOL}_{Field}`` frame (e.g. from CTX).

    When ``expected`` (the requested universe) is given, every requested symbol
    must come back with at least some usable Close data. A symbol the source
    dropped ENTIRELY (never listed, delisted before the window, bad ticker, no
    coverage) is fatal by default, because silently continuing on the remainder
    would run the study on the SURVIVORS only --- the classic survivorship bias,
    which flatters every metric the harness then reports. Note this is about
    *total absence*: a name that trades for part of the window and then delists
    keeps its history (its Close is only partly NaN) and enters the panel normally.

    ``allow_missing=True`` (config ``allow_missing_symbols``) is the deliberate
    opt-out: the panel is built from whatever came back and a RuntimeWarning names
    the dropped symbols, so the bias is chosen and recorded rather than accidental.

    A genuine ``{SYMBOL}_Vwap`` column is picked up here and becomes the ``vwap``
    field instead of being discarded in favour of the HLC3 proxy --- but only when
    EVERY selected symbol carries one, since a cross-section that mixes a real
    volume-weighted price with NaN is worse than a consistent proxy.
    """
    w = wide.copy()
    if not isinstance(w.index, pd.DatetimeIndex):
        if "Datetime" in w.columns:
            w.index = pd.DatetimeIndex(pd.to_datetime(w["Datetime"], errors="coerce"))
        else:
            w.index = pd.DatetimeIndex(pd.to_datetime(w.index, errors="coerce"))
    w = w.sort_index()
    per: dict[str, dict[str, pd.Series]] = {
        hi: {} for _, hi in (*_OHLCV, *_OPTIONAL_WIDE)
    }
    for col in w.columns:
        if not isinstance(col, str) or col == "Datetime" or "_" not in col:
            continue
        asset, _, fld = col.rpartition("_")
        if asset and fld in per:
            per[fld][asset] = pd.to_numeric(w[col], errors="coerce")
    # a symbol counts as present only if its Close has at least one real value
    usable = {s for s, ser in per["Close"].items() if ser.notna().any()}
    if not usable:
        raise ValueError(
            "data source returned no usable '{SYMBOL}_Close' columns; "
            "cannot build a panel"
        )
    if expected is not None:
        missing = [s for s in expected if s not in usable]
        if missing and not allow_missing:
            raise ValueError(
                f"data source is missing usable Close data for {len(missing)} of "
                f"{len(expected)} requested symbol(s): {missing} "
                f"(available: {sorted(usable)}). Continuing on the remaining names "
                "would silently restrict the study to the SURVIVORS and bias every "
                "reported metric upward (survivorship bias), so this is fatal by "
                "default; set allow_missing_symbols=True to accept the smaller "
                "cross-section deliberately, or fix the source/date range."
            )
        if missing:
            warnings.warn(
                f"allow_missing_symbols: dropping {len(missing)} of "
                f"{len(expected)} requested symbol(s) with no usable Close data "
                f"({missing}); the panel now covers survivors only, so treat the "
                "reported metrics as survivorship-biased.",
                RuntimeWarning, stacklevel=2,
            )
        symbols = [s for s in expected if s in usable]
    else:
        symbols = sorted(usable)

    def _frame(hi: str) -> pd.DataFrame:
        parts = per[hi]
        return pd.DataFrame(
            {s: parts.get(s, pd.Series(np.nan, index=w.index)) for s in symbols},
            index=w.index,
        )

    have_vwap = [s for s in symbols
                 if s in per["Vwap"] and per["Vwap"][s].notna().any()]
    vwap = None
    if len(have_vwap) == len(symbols) and symbols:
        vwap = _frame("Vwap")
    elif have_vwap:
        warnings.warn(
            f"source supplies a real Vwap for only {len(have_vwap)} of "
            f"{len(symbols)} symbol(s); falling back to the (high+low+close)/3 "
            "proxy for the whole cross-section rather than mixing a genuine "
            "volume-weighted price with NaN.",
            RuntimeWarning, stacklevel=2,
        )

    return _panel_from_ohlcv(
        _frame("Open"), _frame("High"), _frame("Low"),
        _frame("Close"), _frame("Volume"), symbols, source,
        vwap=vwap, tags=tags,
    )


def make_synthetic(symbols: list[str], start: str, end: str, seed: int = 17) -> Panel:
    """Deterministic geometric-brownian-motion basket (offline, reproducible).

    The generator draws one bar per day, so there is no intraday tape to weight by
    volume: ``vwap`` here is always the HLC3 proxy (``vwap_source='hlc3'``).
    """
    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        raise ValueError("empty date range for synthetic panel")
    t, n = len(dates), len(symbols)
    rng = np.random.default_rng(seed)

    drift = rng.normal(0.0003, 0.0002, n)
    logret = rng.normal(drift, 0.012, (t, n))
    close = 100.0 * np.exp(np.cumsum(logret, axis=0))
    close = pd.DataFrame(close, index=dates, columns=symbols)

    prev = close.shift(1)
    prev.iloc[0] = close.iloc[0]
    open_ = prev * (1.0 + rng.normal(0.0, 0.002, (t, n)))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.004, (t, n))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.004, (t, n))))
    volume = pd.DataFrame(
        rng.lognormal(14.0, 0.4, (t, n)), index=dates, columns=symbols
    )
    return _panel_from_ohlcv(open_, high, low, close, volume, symbols, "synthetic")


_YF_LONG_COLS = ["Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume"]

# CTX infers a session profile from the parquet *filename*
# (CTX._infer_parquet_session_profile): region assets must be
# ``Data_<EQT|ETF|INDEX>_<REGION>_...`` (REGION in US/HK/CN/UK/EU) and OTC assets
# ``Data_<FX|SPOT|DIGITAL>_...``. Map the harness asset class -> (asset token, region);
# yfinance baskets are treated as US listings / OTC spot by default.
_CTX_ASSET_PROFILE: dict[str, tuple[str, str]] = {
    "equity": ("EQT", "US"),
    "etf": ("ETF", "US"),
    "crypto": ("SPOT", "CRYPTO"),
    "fx": ("FX", "G10"),
}


def _harness_root() -> pathlib.Path:
    """Repo-relative harness root (the dir that holds artifacts/, data/, ...).

    This is a READ anchor only --- it resolves relative parquet paths in configs.
    Anything the harness *writes* goes under :func:`_state_dir`, which is not the
    package directory unless we are running from a source checkout.
    """
    return pathlib.Path(__file__).resolve().parents[1]


def _is_source_checkout() -> bool:
    """True when the harness runs from a source/editable checkout, not site-packages.

    ``pyproject.toml`` (or a ``.git``) next to the package is the marker: a wheel
    installs the ``harness`` package alone, so neither survives a non-editable
    ``pip install``.
    """
    root = _harness_root()
    return (root / "pyproject.toml").is_file() or (root / ".git").exists()


def _state_dir() -> pathlib.Path:
    """Writable base for everything the harness generates (cache, CTX logs).

    A source checkout keeps the historic in-repo location (``<harness>/artifacts``)
    so existing workflows, shipped relative paths and ``.gitignore`` rules keep
    working. A NON-editable install must not write next to the package: that path
    lands in ``site-packages``, which ``pip install -U`` wipes and which raises
    PermissionError under a system Python. There it falls back to the per-user
    cache location (``%LOCALAPPDATA%`` on Windows, ``$XDG_CACHE_HOME`` / ``~/.cache``
    elsewhere).
    """
    if _is_source_checkout():
        return _harness_root() / "artifacts"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or (pathlib.Path.home() / "AppData"
                                                  / "Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or (pathlib.Path.home() / ".cache")
    return pathlib.Path(base) / "zora-harness"


def _yf_cache_dir() -> pathlib.Path:
    """Where transcoded yfinance parquet is cached (see :func:`_state_dir`)."""
    d = _state_dir() / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


# One CTX panel build writes one timestamped log file, so an unbounded log dir
# grows once per run forever. Cap it at the most recent few dozen.
_CTX_LOG_KEEP = 50


def _ctx_work_dir() -> pathlib.Path:
    """Working dir handed to the vendored CTX so it never writes into the CWD.

    CTX defaults ``notebook_dir`` to ``pathlib.Path.cwd()`` and then creates
    ``ArchiveDeck/ComputerLog`` (a per-run log file) and ``ArchiveDeck/
    TransmissionLog`` (its payload) underneath it. Left alone, one harness run
    litters whatever directory the user happened to launch from, outside the
    documented artifacts tree. Pinning it here keeps CTX's output inside
    :func:`_state_dir` and lets us bound it.
    """
    d = _state_dir() / "ctx"
    logs = d / "ArchiveDeck" / "ComputerLog"
    logs.mkdir(parents=True, exist_ok=True)
    _prune_ctx_logs(logs)
    return d


def _prune_ctx_logs(log_dir: pathlib.Path, keep: int = _CTX_LOG_KEEP) -> None:
    """Keep only the ``keep`` newest CTX log files; never fail a load over it.

    A log the current process (or a concurrent run) still holds open cannot be
    unlinked on Windows --- that is fine, it is simply skipped and collected on a
    later build.
    """
    try:
        logs = sorted(log_dir.glob("*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:                       # pragma: no cover - unreadable dir
        return
    for stale in logs[keep:]:
        try:
            stale.unlink()
        except OSError:                   # pragma: no cover - locked by a live run
            pass


# CTX asset tokens that CTX._infer_parquet_session_profile accepts (region-listed
# EQT/ETF/INDEX + OTC FX/SPOT/DIGITAL). Used only to *surface* likely-loadable files
# in the TUI; ctx_portfolio still does the authoritative profile check at load time.
_CTX_ASSET_TOKENS: frozenset[str] = frozenset(
    {"EQT", "ETF", "INDEX", "FX", "SPOT", "DIGITAL"}
)

# The listed types are region-profiled, and CTX resolves the region against a
# CLOSED set (DEFAULT_TIME_PROFILES['regions']) -- an unknown region falls
# through to an empty profile and ctx_portfolio then refuses the file. Mirrored
# here so the TUI's Scan does not offer names the loader will reject.
_CTX_LISTED_TOKENS: frozenset[str] = frozenset({"EQT", "ETF", "INDEX"})
_CTX_REGIONS: frozenset[str] = frozenset({"US", "HK", "CN", "UK", "EU"})


def _is_ctx_parquet_name(name: str) -> bool:
    """Cheap, dependency-free proxy for CTX's filename rule, for *daily* parquet.

    A CTX session-profiled name is ``Data_<TYPE>_..._D*.parquet`` where the asset
    token (the part after ``Data_``) is in :data:`_CTX_ASSET_TOKENS`. Region-listed
    assets (EQT/ETF/INDEX) carry a region -- ``Data_EQT_US_D_key`` -- while OTC
    assets (FX/SPOT/DIGITAL) are region-less -- ``Data_FX_D`` -- so for those we do
    NOT require a region; we only require the daily marker (a lone ``D`` token
    after the asset). Mirrors CTX._infer_parquet_session_profile without importing
    the heavy vendored CTX, so scans stay fast.

    For a LISTED type the region is checked against the same closed set CTX uses
    (``DEFAULT_TIME_PROFILES['regions']``). Accepting any region token here made
    the TUI's Scan offer files the loader then refused --- ``Data_EQT_XX_D_k``
    passed this proxy and died at load with "[CTX WARNING] ctx_portfolio price
    parquet requires region/session-profiled parquet names", which reads as a
    corrupt file rather than a mis-named one.
    """
    if not name.lower().endswith(".parquet"):
        return False
    parts = name[: -len(".parquet")].split("_")
    if not (len(parts) >= 3 and parts[0] == "Data"
            and parts[1] in _CTX_ASSET_TOKENS and "D" in parts[2:]):
        return False
    if parts[1] in _CTX_LISTED_TOKENS:
        return len(parts) >= 4 and parts[2].upper() in _CTX_REGIONS
    return True


def _portable(p: pathlib.Path) -> str:
    """Harness-root-relative (forward slashes) if under it, else absolute.

    What the TUI discovers lands in ``parquet_daily`` and is saved to a config
    file, so emitting the relative form keeps a saved config portable instead of
    silently re-pinning one machine's absolute path into the repo.
    """
    try:
        return p.relative_to(_harness_root()).as_posix()
    except ValueError:
        return str(p)


def discover_parquet(data_dir) -> list[str]:
    """Find CTX-loadable daily parquet under ``data_dir`` for data_source='ctx'.

    Returns a sorted list of path strings whose filename carries a CTX session
    profile (so ``ctx_portfolio`` will accept them). A relative ``data_dir`` is
    resolved against the harness root; an unset or missing directory yields ``[]``.
    Scans the directory itself and one level of sub-directories. The TUI uses this
    to auto-fill the parquet field so paths need not be hand-typed.

    Hits *inside* the harness root come back relative to it (see :func:`_portable`)
    so the config the user then saves stays machine-independent; anything outside
    is returned absolute, because only an absolute path can name it.
    """
    if not data_dir:
        return []
    root = pathlib.Path(data_dir).expanduser()
    if not root.is_absolute():
        root = _harness_root() / root
    if not root.is_dir():
        return []
    found: set[str] = set()
    for pattern in ("*.parquet", "*/*.parquet"):
        for p in root.glob(pattern):
            if p.is_file() and _is_ctx_parquet_name(p.name):
                found.add(_portable(p))
    return sorted(found)


def resolve_parquet_paths(paths) -> list[str]:
    """Normalise ``config.parquet_daily`` to a list of absolute path strings.

    A **relative** entry is resolved against the harness root -- the same rule
    ``data_dir`` already follows in :func:`discover_parquet` -- not against the
    process CWD. That is what lets a config ship with the repo: a portable
    ``artifacts/cache/Data_EQT_US_D_....parquet`` loads identically whether the
    harness is launched from the harness root, the repo root, or anywhere else.
    Absolute paths and ``~`` are left alone.
    """
    items = [paths] if isinstance(paths, str) else list(paths)
    out = []
    for p in items:
        q = pathlib.Path(p).expanduser()
        if not q.is_absolute():
            q = _harness_root() / q
        out.append(str(q))
    return out


def _ctx_parquet_name(asset_class: str, key: str) -> str:
    """Build a CTX-session-profiled parquet filename for a transcoded pull."""
    atype, region = _CTX_ASSET_PROFILE.get(asset_class, ("EQT", "US"))
    return f"Data_{atype}_{region}_D_{key}.parquet"


def _yf_to_long(raw: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Reshape a yfinance download into the CTX long schema (one row per bar/sym)."""
    def _field(name: str) -> pd.DataFrame:
        col = raw[name]
        if isinstance(col, pd.Series):  # single symbol -> flat frame
            col = col.to_frame(symbols[0])
        return col.reindex(columns=symbols)

    per = {hi: _field(hi) for _, hi in _OHLCV}
    idx = pd.to_datetime(raw.index)
    frames = []
    for sym in symbols:
        frames.append(pd.DataFrame({
            "Datetime": idx,
            "Symbol": sym,
            "Open": per["Open"][sym].to_numpy(),
            "High": per["High"][sym].to_numpy(),
            "Low": per["Low"][sym].to_numpy(),
            "Close": per["Close"][sym].to_numpy(),
            "Volume": per["Volume"][sym].to_numpy(),
        }, columns=_YF_LONG_COLS))
    long_df = pd.concat(frames, ignore_index=True)
    return long_df.dropna(subset=["Open", "High", "Low", "Close"])


def _ctx_wide(paths, symbols: list[str], start: str, end: str) -> pd.DataFrame:
    """Clean daily parquet through the CTX pipeline; return the wide frame.

    ``notebook_dir``/``log_dir`` are passed EXPLICITLY: CTX otherwise defaults
    ``notebook_dir`` to the process CWD and drops an ``ArchiveDeck/`` tree there
    (see :func:`_ctx_work_dir`).
    """
    from . import infra_engine

    infra_engine._ensure()
    from CTX import CTX

    work = _ctx_work_dir()
    ctx = CTX.ctx_portfolio(
        data_d=paths,
        UNDERLYING=list(symbols),
        frequency="D",
        start_date=start,
        end_date=end,
        notebook_dir=work,
        log_dir=work / "ArchiveDeck" / "ComputerLog",
    )
    return pd.DataFrame(ctx.get_data("TARGET"))


_YF_DOWNLOAD_LOCK = threading.Lock()
_YF_CACHE_CONFIGURED = False


@contextmanager
def _yfinance_ownership():
    """Serialize the library's global state and shared SQLite/parquet writes.

    yfinance's threaded first access can race SQLite WAL/schema initialization.
    A process lock handles concurrent TUI/library threads; the OS lock handles
    independent ZORA runs. This never touches another application's yf cache.
    """
    from .cancellation import check_cancelled, cancellable_sleep
    from .store import Store, RunLockedError

    deadline = time.monotonic() + 120
    while not _YF_DOWNLOAD_LOCK.acquire(timeout=0.1):
        check_cancelled()
        if time.monotonic() >= deadline:
            raise RuntimeError("yfinance download busy in this process; try again later")
    store = Store(str(_state_dir() / "yfinance"), "cache")
    try:
        while True:
            check_cancelled()
            try:
                store.acquire()
                break
            except RunLockedError:
                if time.monotonic() >= deadline:
                    raise RuntimeError("yfinance cache busy in another ZORA run; try again later")
                cancellable_sleep(0.1)
        yield
    finally:
        store.release()
        _YF_DOWNLOAD_LOCK.release()


def load_yfinance(symbols: list[str], start: str, end: str,
                  *, asset_class: str = "equity",
                  allow_missing: bool = False) -> Panel:
    with _yfinance_ownership():
        return _load_yfinance_serial(symbols, start, end, asset_class=asset_class,
                                     allow_missing=allow_missing)


def _load_yfinance_serial(symbols: list[str], start: str, end: str,
                  *, asset_class: str = "equity",
                  allow_missing: bool = False) -> Panel:
    """Pull daily OHLCV online via yfinance, cache to parquet, load through CTX.

    The online frame is transcoded to a local parquet in the CTX long format
    (Datetime, Symbol, Open, High, Low, Close, Volume) with a CTX
    session-profiled filename (see ``_ctx_parquet_name``), then routed through the
    SAME ``CTX.ctx_portfolio`` pipeline as ``load_ctx`` -- so both online and
    local parquet share one clean/resample tail. Lazy import.

    ``auto_adjust=True``: yfinance returns prices RETROACTIVELY adjusted for
    splits and dividends, i.e. today's adjustment factors are applied to the whole
    history, so the price at date t is not the price a trader saw at t. That is a
    real (if mild) look-back in the input data, so it is stamped into the panel's
    provenance string --- ``yfinance(auto_adjust=True,vwap=hlc3)`` --- and hence
    into ``data_source``/``data_version`` on every run record.

    yfinance ships no VWAP, so ``vwap`` here is always the HLC3 proxy.
    ``allow_missing`` opts into a survivors-only universe; see
    :func:`_panel_from_wide`.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "yfinance is required for data_source='yfinance' "
            "(pip install yfinance), or use data_source='synthetic'"
        ) from exc

    global _YF_CACHE_CONFIGURED
    if not _YF_CACHE_CONFIGURED and hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(str(_state_dir() / "yfinance" / "sqlite"))
        _YF_CACHE_CONFIGURED = True

    errors = []
    owner = threading.get_ident()

    class DownloadErrors(logging.Handler):
        def emit(self, record):
            if record.thread == owner and record.levelno >= logging.ERROR:
                from .providers.cli_base import redact_secrets
                errors.append(redact_secrets(record.getMessage())[:500])

    handler = DownloadErrors()
    logger = logging.getLogger("yfinance")
    logger.addHandler(handler)
    try:
        raw = yf.download(
            symbols, start=start, end=end, auto_adjust=True,
            progress=False, group_by="column", threads=False, timeout=15,
        )
    finally:
        logger.removeHandler(handler)
    if raw is None or len(raw) == 0:
        detail = "; ".join(errors[-3:])
        raise RuntimeError(f"yfinance returned no data for {symbols}"
                           + (f": {detail}" if detail else ""))

    long_df = _yf_to_long(raw, list(symbols))
    if long_df.empty:
        raise RuntimeError(f"yfinance returned no usable rows for {symbols}")

    # transcode to a local parquet so the online pull flows through CTX exactly
    # like a user-supplied parquet (requires pyarrow). The filename must carry a
    # CTX session profile or ctx_portfolio rejects it.
    key = hashlib.blake2b(
        (f"{asset_class}|" + "|".join(sorted(symbols)) + f"|{start}|{end}").encode(),
        digest_size=8,
    ).hexdigest()
    path = _yf_cache_dir() / _ctx_parquet_name(asset_class, key)
    long_df.to_parquet(path, index=False)

    wide = _ctx_wide([str(path)], list(symbols), start, end)
    return _panel_from_wide(wide, source="yfinance", expected=list(symbols),
                            allow_missing=allow_missing,
                            tags=("auto_adjust=True",))


def _ctx_all_symbols(paths: list[str]) -> list[str]:
    """Every distinct ``Symbol`` across the daily parquet files, sorted.

    This is what ``symbols = ["all"]`` means for ``data_source = "ctx"``: the
    ENTIRE universe in the parquet, not the asset-class basket
    (``config.ASSET_PRESETS``). Reads only the ``Symbol`` column in batches, so
    a 10M-row file costs a columnar scan, not a full load; the sorted order
    keeps the panel deterministic.
    """
    import pyarrow.parquet as pq

    syms: set[str] = set()
    for p in paths:
        pf = pq.ParquetFile(p)
        for batch in pf.iter_batches(columns=["Symbol"], batch_size=262144):
            col = batch.column("Symbol")
            syms.update(str(v) for v in col.to_pylist())
    return sorted(syms)


def load_ctx(config) -> Panel:
    """Load the user's own parquet daily OHLCV through the CTX pipeline.

    Requires ``config.parquet_daily`` (path or list of paths to daily parquet
    datasets in the CTX long format: Datetime, Symbol, Open, High, Low, Close
    [, Volume]). Each file must carry a CTX session-profiled name --
    ``Data_<EQT|ETF|INDEX>_<REGION>_...`` (REGION in US/HK/CN/UK/EU) or
    ``Data_<FX|SPOT|DIGITAL>_...`` -- or ``ctx_portfolio`` rejects it. CTX resamples/cleans
    to the daily working clock and returns the wide ``{SYMBOL}_{Field}`` frame the
    rest of the harness consumes.

    ``symbols = ["all"]`` (kept as the sentinel by :class:`RunConfig` for ctx)
    expands here to EVERY underlying in the parquet --- see
    :func:`_ctx_all_symbols`.

    Two CTX behaviours decide which rows reach the panel, and both matter for
    reproducibility across engine versions:

    * a *string* ``Datetime`` column carrying an ISO offset (``+08:00``, ``Z``, ...)
      has that suffix stripped and is read as naive wall-clock, so the panel index
      -- and therefore :attr:`Panel.version` -- is the local timestamp, not UTC.
      Mixed offsets in one file are handled instead of aborting the load.
    * the date-only ``data_end`` bound is *half-open* (``< end + 1 day``), so a
      daily file stamped at session close (e.g. 16:00) keeps its final day rather
      than losing it.

    Write ``Datetime`` as a real timestamp (as the yfinance transcode path does)
    to stay clear of both.

    This is the ONLY source that can deliver a genuine ``vwap``: add a ``Vwap``
    column to the parquet and CTX volume-weights it on resample, in which case the
    panel's ``vwap`` field is that real series (``vwap_source='source'``) instead
    of the HLC3 proxy.

    Paths may be relative; see :func:`resolve_parquet_paths` for the anchor.
    """
    paths = config.parquet_daily
    if not paths:
        raise ValueError(
            "data_source='ctx' requires config.parquet_daily "
            "(path or list of paths to daily parquet datasets)"
        )
    paths = resolve_parquet_paths(paths)
    symbols = list(config.symbols)
    if symbols == ["all"]:
        symbols = _ctx_all_symbols(paths)
    wide = _ctx_wide(paths, symbols, config.data_start, config.data_end)
    return _panel_from_wide(wide, source="ctx", expected=symbols,
                            allow_missing=_allow_missing(config))


def _allow_missing(config) -> bool:
    """Read the survivorship opt-out off a config that may predate the knob.

    ``getattr`` (not ``config.allow_missing_symbols``) so an older RunConfig, an
    older saved config file, and a resumed run manifest all keep loading; the
    default is the safe one --- fatal on incomplete coverage.
    """
    return bool(getattr(config, "allow_missing_symbols", False))


def load_panel(config) -> Panel:
    """Build a panel from a RunConfig (dispatches on config.data_source)."""
    if config.data_source == "synthetic":
        return make_synthetic(
            config.symbols, config.data_start, config.data_end, config.seed
        )
    if config.data_source == "yfinance":
        return load_yfinance(config.symbols, config.data_start, config.data_end,
                             asset_class=config.asset_class,
                             allow_missing=_allow_missing(config))
    if config.data_source == "ctx":
        return load_ctx(config)
    raise ValueError(f"unknown data_source: {config.data_source!r}")
