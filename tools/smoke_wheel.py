"""Offline smoke from an installed wheel, with no checkout imports or journal."""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import math
from pathlib import Path
import sys
import warnings


def verify_location(module_file: str, prefix: str, checkout: Path) -> None:
    module = Path(module_file).resolve()
    if not module.is_relative_to(Path(prefix).resolve()) or module.is_relative_to((checkout / "harness").resolve()):
        raise ValueError("Smoke must import the installed wheel, not the checkout")


def verify_metrics(record: dict, config) -> None:
    for label, floor in (("is_metrics", config.min_is_days), ("oos_metrics", config.min_oos_days)):
        if record[label]["n_active"] < floor:
            raise ValueError("Installed-wheel smoke has insufficient active bars")
        if any(not math.isfinite(record[label][key]) for key in
               ("cagr", "sortino", "avg_gross", "avg_turnover")):
            raise ValueError("Installed-wheel smoke produced nonfinite core metrics")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    destination = args.output.resolve()
    if destination.exists():
        parser.error("Smoke output must not already exist")
    destination.mkdir(parents=True)
    import harness
    from harness import cli, promptlib
    from harness.config import RunConfig
    from harness.factor.contract import POLICY_HASH
    from harness.providers.replay import RecordingProvider
    from harness.providers.scripted import ScriptedProvider
    from harness.runner import _deep_eq, _replay_core, prepare_panel, replay, run_once
    from harness.store import Store

    verify_location(harness.__file__, sys.prefix, Path(__file__).resolve().parents[1])
    cfg = RunConfig(symbols=["A", "B", "C", "D", "E", "F"],
                    data_source="synthetic", provider="scripted", memory=False,
                    data_start="2015-01-01", data_end="2023-12-31",
                    research_date="2021-01-01", max_iters=4, logging=False,
                    n_jobs=1, run_name="wheel-smoke")
    config_path = destination / "config.json"
    config_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    store = Store(str(destination), cfg.run_name)
    with warnings.catch_warnings(record=True) as caught, \
            (destination / "execution.log").open("w", encoding="utf-8") as log, \
            redirect_stdout(log), redirect_stderr(log), store.ownership():
        warnings.simplefilter("always")
        doctor = cli.main(["doctor", "--config", str(config_path)])
        if doctor != 0:
            raise ValueError("Installed-wheel doctor failed")
        for name in promptlib.REQUIRED_PROMPTS:
            promptlib.load(name)
        panel = prepare_panel(cfg, walk_forward=False)
        provider = RecordingProvider(ScriptedProvider(), on_record=store.append_trace)
        original = run_once(cfg, provider=provider, store=store, panel=panel, register=False)
        manifest = {"config": cfg.to_dict(), "data_version": panel.version,
                    "math_policy_hash": POLICY_HASH, "execution_mode": "single-date"}
        store.write_manifest(manifest)
        restored = replay(store.read_manifest(), store.read_trace())
        if len(restored) != 1 or not _deep_eq(_replay_core(original), _replay_core(restored[0])):
            raise ValueError("Installed-wheel replay core differs")
        verify_metrics(original, cfg)
    if caught:
        raise ValueError("Installed-wheel smoke emitted warnings; inspect execution log")
    receipt = {"complete": True, "doctor_exit": doctor, "installed_wheel": True,
               "prompt_count": len(promptlib.REQUIRED_PROMPTS), "core_identical_replay": True,
               "data_version": panel.version, "is_n_active": original["is_metrics"]["n_active"],
               "oos_n_active": original["oos_metrics"]["n_active"],
               "strategy_passed": original["verdict"]["passed"], "warning_count": len(caught)}
    (destination / "result.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
