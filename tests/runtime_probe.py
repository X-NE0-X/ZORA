"""Isolated child programs for runtime regression tests (no external model calls)."""
import json
import os
import pathlib
import subprocess
import sys
import time


def child_tree(path):
    child = subprocess.Popen([sys.executable, __file__, "sleep"])
    pathlib.Path(path).write_text(json.dumps([os.getpid(), child.pid]), encoding="utf-8")
    time.sleep(60)


def exited_parent(path):
    child = subprocess.Popen([sys.executable, __file__, "sleep"],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    pathlib.Path(path).write_text(json.dumps([os.getpid(), child.pid]), encoding="utf-8")


def blocking_metrics(payload):
    _, ready_path, _ = payload
    pathlib.Path(ready_path).write_text(str(os.getpid()), encoding="utf-8")
    time.sleep(60)
    return True, {}


def yf_cache(root):
    import pandas as pd
    import yfinance as yf
    from concurrent.futures import ThreadPoolExecutor
    from harness import data

    data._state_dir = lambda: pathlib.Path(root)
    active = []

    def history(self, **kwargs):
        # Real SQLite WAL/schema initialization and writes, without Yahoo access.
        tz = yf.cache.get_tz_cache()
        tz.lookup(self.ticker)
        tz.store(self.ticker, "America/New_York")
        cookie = yf.cache.get_cookie_cache()
        cookie.lookup("basic")
        cookie.store("basic", {"test": "not-an-auth-cookie"})
        active.append(self.ticker)
        idx = pd.date_range("2024-01-02", periods=8, freq="B")
        return pd.DataFrame({f: range(10, 18) for f in
                             ("Open", "High", "Low", "Close", "Volume")}, index=idx)

    yf.Ticker.history = history

    def wide(paths, symbols, *_args):
        long = pd.read_parquet(paths[0])
        return pd.DataFrame({f"{s}_{f}": long[long.Symbol == s].set_index("Datetime")[f]
                             for s in symbols for f in
                             ("Open", "High", "Low", "Close", "Volume")})

    data._ctx_wide = wide
    baskets = [["AAPL", "MSFT"], ["GOOGL", "AMZN", "NVDA", "META"]]
    with ThreadPoolExecutor(2) as executor:
        panels = list(executor.map(lambda syms: data.load_yfinance(
            syms, "2024-01-02", "2024-01-15"), baskets))
    assert [p.symbols for p in panels] == baskets
    assert len(active) == 6
    assert (pathlib.Path(root) / "yfinance/sqlite/tkr-tz.db").is_file()
    print(json.dumps({"baskets": baskets, "cache_writes": len(active)}))


if __name__ == "__main__":
    if sys.argv[1] == "tree":
        child_tree(sys.argv[2])
    elif sys.argv[1] == "exited-parent":
        exited_parent(sys.argv[2])
    elif sys.argv[1] == "sleep":
        time.sleep(60)
    elif sys.argv[1] == "yf-cache":
        yf_cache(sys.argv[2])
