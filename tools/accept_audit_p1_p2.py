"""Offline F10 acceptance using the actual, unmodified field_config settings.

Creates an isolated artifact root; never modifies the research journal or an
existing run. Run from the checkout with its reference Python environment.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from harness.config import RunConfig
from harness.providers.replay import RecordingProvider
from harness.providers.scripted import ScriptedProvider
from harness.runner import prepare_panel, run_once
from harness.store import Store
from tests.regression_support import source_fingerprint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.iterdir()):
        raise ValueError("Acceptance output must be empty; prior evidence is preserved")
    config = RunConfig.from_json(str(ROOT / "field_config.json"))
    store = Store(str(destination), config.run_name)
    with (destination / "execution.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log), store.ownership():
            panel = prepare_panel(config, walk_forward=False)
            provider = RecordingProvider(ScriptedProvider(), on_record=store.append_trace)
            record = run_once(config, store=store, provider=provider, panel=panel, register=False)
            store.write_manifest({"run_name": config.run_name, "config": config.to_dict(),
                                  "data_version": panel.version, "n_factors": 1,
                                  "n_pass": int(record["verdict"]["passed"]),
                                  "provider": "scripted", "model": config.model})
    payload = {"complete": True, "source": source_fingerprint(),
               "config": config.to_dict(), "resolved_symbols": panel.symbols,
               "data_version": panel.version, "panel_bars": len(panel.dates),
               "scored_oos": {key: record[key] for key in ("oos_start", "oos_end")},
               "oos_metrics": record["oos_metrics"], "verdict": record["verdict"]}
    assert panel.symbols == config.symbols
    assert "GOOGL" in panel.symbols and "GOOG" not in panel.symbols
    assert record["oos_metrics"]["n_active"] >= config.min_oos_days
    (destination / "result.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("complete", "panel_bars", "resolved_symbols", "scored_oos", "verdict")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
