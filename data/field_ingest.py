"""Field-test ingest: transcode the user's daily CSVs into CTX-profiled parquet.

The raw files under ``ZORA/Data`` are long CSVs with columns
``Date, Open, High, Low, Close, Volume, Dividends, Stock Splits, Symbol`` and a
non-CTX name. CTX's ``ctx_portfolio`` needs a *parquet* named with a session
profile (``Data_<EQT|INDEX>_<REGION>_D_...`` for region assets, ``Data_FX_...``
for OTC) and the long schema ``Datetime, Symbol, Open, High, Low, Close, Volume``.

This carves single-session universes out of the mixed CSVs (US equities; G10 FX),
normalises the mixed tz-suffixed ``Date`` column to naive midnight, and writes
CTX-profiled parquet under the (git-ignored) harness cache. Re-runnable.

Usage:  python data/field_ingest.py            # CSVs at <repo>/ZORA/Data
        ZORA_CSV_DIR=/path/to/csvs python data/field_ingest.py

Every path is derived from this file's own location (or ``ZORA_CSV_DIR``), so
the script is machine-independent.
"""
from __future__ import annotations

import hashlib
import os
import pathlib

import pandas as pd

# this file lives in <repo>/harness/data/, so the harness root is one level up
# and the repo root two. Nothing here is hard-coded to a particular machine.
HARNESS_ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = HARNESS_ROOT.parent

SRC = pathlib.Path(os.environ.get("ZORA_CSV_DIR") or REPO_ROOT / "ZORA" / "Data")
OUT = HARNESS_ROOT / "artifacts" / "cache"

_LONG_COLS = ["Datetime", "Symbol", "Open", "High", "Low", "Close", "Volume"]


def transcode(csv_name: str, symbols: list[str], asset_token: str, region: str) -> pathlib.Path:
    src = SRC / csv_name
    if not src.is_file():
        raise SystemExit(f"source CSV not found: {src}\n"
                         f"point ZORA_CSV_DIR at the folder holding your daily CSVs")
    df = pd.read_csv(src)
    df = df[df["Symbol"].isin(symbols)].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"no rows for {symbols} in {csv_name}")

    # The Date column carries genuinely MIXED formats: most rows are naive
    # (``2010-01-01 00:00:00``) but many carry a seasonal tz offset
    # (``...+01:00`` in EU summer, ``...-05:00`` for US listings). A single
    # inferred format silently NaTs the off-format rows (a 58% FX data loss), so
    # parse per-element with format='mixed', normalise every bar to UTC, drop the
    # tz, and floor to the trading date -- daily bars, so the intraday offset is
    # immaterial once normalised to midnight.
    dt = (pd.to_datetime(df["Date"], format="mixed", utc=True, errors="coerce")
            .dt.tz_localize(None).dt.normalize())

    out = pd.DataFrame({
        "Datetime": dt,
        "Symbol": df["Symbol"],
        "Open": pd.to_numeric(df["Open"], errors="coerce"),
        "High": pd.to_numeric(df["High"], errors="coerce"),
        "Low": pd.to_numeric(df["Low"], errors="coerce"),
        "Close": pd.to_numeric(df["Close"], errors="coerce"),
        "Volume": pd.to_numeric(df["Volume"], errors="coerce"),
    })[_LONG_COLS]
    out = (out.dropna(subset=["Datetime", "Open", "High", "Low", "Close"])
              .drop_duplicates(subset=["Symbol", "Datetime"])
              .sort_values(["Symbol", "Datetime"])
              .reset_index(drop=True))

    OUT.mkdir(parents=True, exist_ok=True)
    key = hashlib.blake2b("|".join(sorted(symbols)).encode(), digest_size=8).hexdigest()
    path = OUT / f"Data_{asset_token}_{region}_D_{key}.parquet"
    out.to_parquet(path, index=False)

    cov = out.groupby("Symbol")["Close"].count()
    print(f"[{asset_token}_{region}] -> {path.name}")
    print(f"   rows={len(out)}  symbols={out['Symbol'].nunique()}  "
          f"dates={out['Datetime'].min().date()}..{out['Datetime'].max().date()}")
    print(f"   rows/sym: min={cov.min()} median={int(cov.median())} max={cov.max()}")
    return path


if __name__ == "__main__":
    us_eqt = ["AAPL", "AMZN", "GOOG", "META", "MSFT", "NVDA", "TSLA"]
    fx = ["AUDUSD", "EURUSD", "GBPUSD", "NZDUSD", "USDCAD",
          "USDCNY", "USDHKD", "USDJPY", "USDSGD"]
    p_eqt = transcode("EQT_DATA_D.csv", us_eqt, "EQT", "US")
    p_fx = transcode("FX_DATA_D.csv", fx, "FX", "G10")
    # print the harness-relative form: that is what belongs in a config's
    # `parquet_daily`, and it keeps absolute machine paths out of the repo.
    print("\nPARQUET_EQT_US=", p_eqt.relative_to(HARNESS_ROOT).as_posix())
    print("PARQUET_FX_G10=", p_fx.relative_to(HARNESS_ROOT).as_posix())
