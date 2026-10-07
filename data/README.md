# `data/` — drop-in folder for `data_source = "ctx"`

The TUI scans this folder (the `data_dir` field, default `data`) and auto-fills the
**Parquet path(s)** box when you pick **data_source = ctx**, so you don't have to
type paths by hand. Press **Scan data dir for parquet** to re-scan after adding files.
The scan looks in the folder itself and one level of sub-directories.

## The filename is not cosmetic

CTX infers a **session profile** — market, timezone, trading hours — from the
*filename*, and `ctx_portfolio` refuses a price parquet it cannot profile:

```
[CTX WARNING] ctx_portfolio price parquet requires region/session-profiled
parquet names; unprofiled datasets=[...]
```

The rule (`CTX._infer_parquet_session_profile`, mirrored cheaply by
`harness.data._is_ctx_parquet_name` for the scan) reads the underscore-separated
tokens of the stem:

```
Data_<TYPE>_<REGION>_D[_<anything>].parquet     listed assets — REGION required
Data_<TYPE>_[<anything>_]D[_<anything>].parquet OTC assets    — no REGION needed
        │        │        │
        │        │        └─ "D" = the daily-bar marker
        │        └────────── token 3. For a LISTED type it must be a known region:
        │                    US | HK | CN | UK | EU. Anything else (G10, XX, ...)
        │                    leaves the file unprofiled and it is rejected.
        │                    For an OTC type this token is free — it is recorded
        │                    but never used to look the profile up.
        └───────────────── TYPE. Listed: EQT | ETF | INDEX
                                 OTC:    FX  | SPOT | DIGITAL
```

Verified against the vendored CTX by loading each name:

| filename | result |
|---|---|
| `Data_EQT_US_D_myset.parquet` | loads |
| `Data_EQT_US_D.parquet` | loads |
| `Data_FX_G10_D_2024.parquet` | loads |
| `Data_SPOT_CRYPTO_D_x.parquet` | loads |
| `Data_DIGITAL_D_x.parquet` | loads |
| `Data_EQT_XX_D_test.parquet` | **rejected** — `XX` is not a known region, so no profile |
| `prices.parquet` | **rejected** — no profile at all |
| `Data_FX_D.parquet` | **rejected** — see the reserved names below |

### Five reserved filenames

`Data_EQT_D.parquet`, `Data_ETF_D.parquet`, `Data_FX_D.parquet`,
`Data_INDEX_D.parquet` and `Data_SPOT_D.parquet` are entries in the vendored
engine's built-in parquet catalog (`DEFAULT_PARQUET_CATALOG["D"]`), which lists
the symbols each of *those* datasets is known to contain. Name your own file
exactly one of them and CTX will check your universe against that catalog instead
of against your file, and fail with:

```
[CTX WARNING] requested UNDERLYING not present in D parquet catalog: ['AAA', 'BBB']
```

So a region-less OTC name is legal in general (`Data_DIGITAL_D_x.parquet` is
fine) but the *bare* three-token form collides with the catalog. Add a region or
a key token — `Data_FX_G10_D_2024.parquet` — and it loads. The harness's scanner
does not know about the catalog, so it will still offer you a reserved name; the
failure only shows up at load time.

## The file contents

Each parquet must be **long format**, one row per bar per symbol, with columns

```
Datetime, Symbol, Open, High, Low, Close [, Volume] [, Vwap]
```

CTX resamples/cleans to the daily clock on load.

- **`Vwap` is the one way to get a real VWAP.** This is the only data source that
  can: CTX volume-weights that column on resample, so the panel's `vwap` field
  becomes a genuine volume-weighted price (`Panel.vwap_source == "source"`, and
  the provenance string reads `ctx(vwap=source)`). Without it — and on yfinance
  or synthetic data, which ship no VWAP at all — `vwap` falls back to the
  `(high + low + close) / 3` typical-price proxy, which has **no volume term in
  it**. The fallback is all-or-nothing: if only some symbols carry a usable
  `Vwap`, the whole cross-section uses the proxy and a `RuntimeWarning` says so,
  because mixing a real volume-weighted price with NaN is worse than a consistent
  proxy.
- **Write `Datetime` as a real timestamp.** A *string* column carrying an ISO
  offset (`+08:00`, `Z`) has the suffix stripped and is read as naive wall-clock,
  so the panel index — and therefore the content-hashed data version — is the
  local timestamp, not UTC. Mixed offsets in one file are handled rather than
  fatal, but the two spellings do not produce the same panel.
- The `data_end` bound is **half-open** (`< end + 1 day`), so a daily file stamped
  at session close (16:00) keeps its final day instead of losing it.

## `field_ingest.py` — CSV → CTX parquet

Raw `ZORA/Data/*.csv` files are **not** loaded as-is: they're CSV (not parquet),
their date column is `Date` (CTX wants `Datetime`), and the filenames lack the
`Data_<TYPE>_<REGION>_` prefix. `field_ingest.py` transcodes them — it carves a
single-session universe out of a mixed CSV, normalises the mixed tz-suffixed
`Date` column to naive midnight (a single inferred format silently `NaT`s the
off-format rows — that alone cost 58% of the FX data), and writes a properly
profiled parquet into `artifacts/cache/`:

```bash
python data/field_ingest.py
```

It looks for the CSVs in `<repo>/ZORA/Data` by default; point `ZORA_CSV_DIR` at
another folder to override. Every other path is derived from the script's own
location, so there is nothing machine-specific to edit. Change the symbol lists
and source filenames in the `__main__` block at the bottom.

> Paths in a config may be **relative** — they resolve against the harness root,
> not the shell's CWD — so `"parquet_daily": ["artifacts/cache/Data_EQT_US_D_….parquet"]`
> works from anywhere and keeps absolute machine paths out of the repo. A
> relative path is also what keeps one committed config a *single* experiment
> across machines: the run's identity normalises it to a harness-root-relative
> POSIX form, so Windows and Linux do not fork the journal or block a `--resume`.
