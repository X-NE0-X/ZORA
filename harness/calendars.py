"""Coverage boundaries on the data source's daily session calendar."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import resolve_parquet_paths

_REGIONS = {"US": "XNYS", "HK": "XHKG", "CN": "XSHG", "UK": "XLON", "EU": "XETR"}
_YAHOO_SUFFIXES = {"HK": "XHKG", "SS": "XSHG", "SZ": "XSHG", "L": "XLON",
                   "DE": "XETR", "T": "XTKS", "PA": "XPAR", "TO": "XTSE", "AX": "XASX"}


def source_calendar(config) -> str:
    if config.data_source == "synthetic":
        return "business-day"  # make_synthetic explicitly uses pandas BDay.
    if config.data_source == "yfinance":
        if config.asset_class in {"crypto", "fx"}:
            return {"crypto": "24/7", "fx": "24/5"}[config.asset_class]
        calendars = set()
        for symbol in config.symbols:
            suffix = symbol.upper().rsplit(".", 1)[1] if "." in symbol else ""
            if suffix in {"", "A", "B"}:
                calendars.add("XNYS")
            elif suffix in _YAHOO_SUFFIXES:
                calendars.add(_YAHOO_SUFFIXES[suffix])
            else:
                raise ValueError(f"cannot establish Yahoo session calendar for {symbol}")
        if len(calendars) != 1:
            raise ValueError("coverage requires one data-source session calendar")
        return calendars.pop()
    calendars = set()
    for path in resolve_parquet_paths(config.parquet_daily):
        parts = Path(path).stem.upper().split("_")
        asset = parts[1] if len(parts) > 1 else ""
        if asset in {"EQT", "ETF", "INDEX"}:
            region = parts[2] if len(parts) > 2 else ""
            if region not in _REGIONS:
                raise ValueError(f"cannot establish session calendar for {Path(path).name}")
            calendars.add(_REGIONS[region])
        elif asset == "FX":
            calendars.add("24/5")
        elif asset in {"SPOT", "DIGITAL"}:
            calendars.add("24/7")
        else:
            raise ValueError(f"cannot establish session calendar for {Path(path).name}")
    if len(calendars) != 1:
        raise ValueError("coverage requires one data-source session calendar")
    return calendars.pop()


def session_bounds(config, start, end) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First session on/after start and last session on/before end.

    Unknown calendars or unsupported ranges fail explicitly; an observed last
    bar is never substituted for the last expected session.
    """
    start, end = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    name = source_calendar(config)
    if name == "business-day":
        sessions = pd.bdate_range(start, end)
    elif name in {"24/5", "24/7"}:
        sessions = pd.date_range(start, end, freq="B" if name == "24/5" else "D")
    else:
        import exchange_calendars as xcals
        calendar = xcals.get_calendar(name, start=start - pd.Timedelta(days=14),
                                      end=end + pd.Timedelta(days=14))
        sessions = calendar.sessions_in_range(start, end).tz_localize(None)
    if len(sessions) == 0:
        raise ValueError(f"required interval has no {name} sessions")
    return pd.Timestamp(sessions[0]), pd.Timestamp(sessions[-1])
