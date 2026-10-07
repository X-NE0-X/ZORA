"""Command-line entry point: configure / run / show / log / replay / memory / tui / doctor.

Installed as the ``zora-harness`` console script (see pyproject ``[project.scripts]``);
``python -m harness.cli`` is the equivalent from a source checkout. Both accept the
same subcommands:

  zora-harness configure --set pass_line=1.2 --set max_iters=6 --out run_config.json
  zora-harness run     [--config run_config.json] [--walk-forward] [--resume]
  zora-harness show    [--config run_config.json]
  zora-harness log     [--config run_config.json]
  zora-harness replay  [--config run_config.json]
  zora-harness memory  [--list] [--archive] [--synthesize]
  zora-harness tui     [--config run_config.json]
  zora-harness doctor  [--config run_config.json] [--provider NAME] [--all]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import memory, preflight, promptlib
from .config import FREQUENCIES, RunConfig
from .parallel import pool_for
from .providers import get_provider, provider_lifecycle
from .providers.replay import ReplayDesyncError
from .runner import (_deep_eq, _journal_snapshot, _replay_core, check_data_coverage,
                     replay, run_once, run_walk_forward)
from .store import RunLockedError, Store

_ARTIFACTS = "artifacts"
# The durable research journal, next to (not inside) the disposable artifacts
# tree --- see harness/memory.py for why the ledger alone is not a record.
_MEMORY = "memory"


class ConfigError(ValueError):
    """A user-supplied configuration could not be read or validated."""


def _config_from_path(path: str) -> RunConfig:
    try:
        return RunConfig.from_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _load_config(args) -> RunConfig:
    try:
        cfg = _config_from_path(args.config) if args.config else RunConfig()
        previous = cfg.provider
        for key in ("provider", "data_source", "research_date", "run_name",
                    "max_iters", "candidates_per_round", "frequency", "n_jobs"):
            val = getattr(args, key, None)
            if val is not None:
                setattr(cfg, key, val)
        from .providers.codex_controls import switch_defaults
        values = cfg.to_dict()
        for notice in switch_defaults(values, previous):
            print(notice)
        # setattr bypasses __post_init__; re-validate the merged config so a bad
        # CLI override fails here instead of deep inside a run.
        return RunConfig.from_dict(values)
    except ConfigError:
        raise
    except (ValueError, TypeError) as exc:
        raise ConfigError(str(exc)) from exc


def _stdout_log(msg: str) -> None:
    print(msg, flush=True)


def _coerce(raw: str):
    """Parse a --set value as JSON (number/bool/list/null) or fall back to str.

    So ``pass_line=1.2`` -> float, ``require_interpretability=true`` -> bool,
    ``symbols=["AAPL","MSFT"]`` -> list, and a bare ``objective=sortino`` -> str.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def cmd_configure(args) -> int:
    base = _config_from_path(args.config) if args.config else RunConfig()
    d = base.to_dict()
    known = set(d)
    explicit = set()
    for pair in (args.set or []):
        if "=" not in pair:
            print(f"--set expects KEY=VALUE, got {pair!r}")
            return 2
        key, _, raw = pair.partition("=")
        key = key.strip()
        if key not in known:
            print(f"unknown config key {key!r}; choose from {sorted(known)}")
            return 2
        d[key] = _coerce(raw)
        explicit.add(key)
    from .providers.codex_controls import switch_defaults
    notices = switch_defaults(d, base.provider, explicit)
    try:
        cfg = RunConfig.from_dict(d)          # re-validate the merged config
    except (ValueError, TypeError) as exc:
        print(f"invalid configuration: {exc}")
        return 2
    out = args.out or args.config or "run_config.json"
    try:
        cfg.to_json(out)
    except OSError as exc:
        print(f"could not write configuration to {out}: {exc}")
        return 2
    n_set = len(args.set or [])
    for notice in notices:
        print(notice)
    print(f"wrote config ({n_set} override(s)) -> {out}")
    return 0


def _remember(args, cfg: RunConfig, store: Store) -> None:
    """Copy a finished run into the persistent journal (unless memory is off).

    Deliberately at the CLI layer, not inside the runner: ``replay`` must stay a
    pure read-only reproduction, and library callers of ``run_walk_forward``
    should not acquire a filesystem side effect they never asked for.
    """
    root = _journal_root(args)
    if root is None or not cfg.memory:
        return
    entry = memory.remember(store, root)
    if entry is not None:
        print(f"journal: {entry['run_id']} -> {memory.notes_path(root)}")


def _journal_root(args) -> str | None:
    """Where the cross-run memory lives for this invocation (None = disabled)."""
    return None if getattr(args, "no_memory", False) else _MEMORY


def _do_run(args, cfg: RunConfig) -> int:
    if args.walk_forward or cfg.walk_forward:
        with provider_lifecycle(get_provider(cfg.provider)) as provider:
            store = Store(_ARTIFACTS, cfg.run_name)
            with store.ownership():
                out = run_walk_forward(
                    cfg, artifacts_root=_ARTIFACTS, log=_stdout_log,
                    resume=args.resume, memory_root=_journal_root(args),
                    provider=provider,
                )
                recs = out["records"]
                n_pass = sum(1 for r in recs if r["verdict"]["passed"])
                print(f"\n{len(recs)} factor(s), {n_pass} PASS -> {out['run_dir']}")
                _remember(args, cfg, store)
        return 0

    # single research date (M1 path)
    check_data_coverage(cfg, walk_forward=False, log=_stdout_log)
    with provider_lifecycle(get_provider(cfg.provider)) as provider:
        store = Store(_ARTIFACTS, cfg.run_name)
        with store.ownership():
            store.reset()
            journal = _journal_snapshot(cfg, _journal_root(args))
            store.write_manifest({
                "run_name": cfg.run_name, "config": cfg.to_dict(),
                "execution_mode": "single-date",
                "provider": provider.name, "model": cfg.model, "seed": cfg.seed,
                "journal": journal,
            })
            pool = pool_for(cfg, log=_stdout_log)
            try:
                rec = run_once(cfg, provider=provider, store=store, log=_stdout_log,
                               journal=journal, pool=pool)
            finally:
                if pool is not None:
                    pool.close()
            manifest = store.read_manifest()
            manifest["n_factors"] = 1
            manifest["n_pass"] = int(rec["verdict"]["passed"])
            manifest["data_version"] = rec["data_version"]
            store.write_manifest(manifest)
            print(f"\n{rec['verdict']['label']}: {rec['formula']} -> {store.run_dir}")
            _remember(args, cfg, store)
    return 0


def cmd_run(args) -> int:
    cfg = _load_config(args)
    if args.resume and not (args.walk_forward or cfg.walk_forward):
        print("--resume is only supported for walk-forward runs; the single-date "
              "path has no partial-date checkpoint to resume.")
        return 2
    # First-run gate: a real provider needs a key/login the user may not have set
    # yet, AND every run (scripted included) needs an importable backtest engine
    # plus a complete prompt library. Check those cheap, reliable signals BEFORE
    # any work and print a setup guide instead of dying deep in the provider or
    # ~60s later on an InfraUnavailable traceback (see harness/preflight.py).
    r = preflight.check_config(cfg)
    if not r.ok:
        print(preflight.format_readiness(r))
        print(f"\nfix the above, then re-run "
              f"(or run: python -m harness.cli doctor --provider {cfg.provider}).")
        return 3
    try:
        return _do_run(args, cfg)
    except RunLockedError as exc:
        print(f"\nrun refused: {exc}")
        return 4
    except Exception as exc:  # noqa: BLE001 - translate auth-shaped failures to a guide
        guide = preflight.explain_failure(cfg.provider, exc, cfg.model)
        if guide is None:
            raise
        print(f"\nrun failed: {exc}\n")
        print(preflight.format_readiness(guide))
        return 3


def cmd_doctor(args) -> int:
    """Check run readiness (provider credential + local runtime) with a fix guide.

    Exit 0 only if a run could actually start: ``--all`` still lists every
    provider, but a broken backtest engine or prompt library fails the command,
    because no provider is runnable on that machine.
    """
    if getattr(args, "all", False):
        print(preflight.format_all())
        return 0 if preflight.check_runtime().ok else 1
    if args.provider:
        r = preflight.check_all(args.provider)
    else:
        r = preflight.check_config(_load_config(args))
    print(preflight.format_readiness(r))
    return 0 if r.ok else 1


def cmd_show(args) -> int:
    cfg = _load_config(args)
    store = Store(_ARTIFACTS, cfg.run_name)
    recs = store.read_factors()
    if not recs:
        print(f"no factors recorded for run '{cfg.run_name}'")
        return 1
    obj = recs[0]["verdict"].get("objective", "sortino")
    print(f"objective = {obj}")
    print(f"{'date':<12} {'verdict':<6} {'IS':>9} {'OOS':>9}  formula")
    print("-" * 78)
    for r in recs:
        v = r["verdict"]
        iss = v.get("is_value")
        oos = v.get("oos_value")
        iss_s = f"{iss:>9.4f}" if iss is not None else f"{'nan':>9}"
        oos_s = f"{oos:>9.4f}" if oos is not None else f"{'nan':>9}"
        print(f"{r['research_date']:<12} {v['label']:<6} "
              f"{iss_s} {oos_s}  {r['formula']}")
        if "parameters" in r:
            print("  parameters = " + json.dumps(r["parameters"], sort_keys=True))
    return 0


def cmd_tui(args) -> int:
    from .tui import HarnessTUI
    cfg = _load_config(args)
    path = args.config or "run_config.json"
    if args.config is None and os.path.exists(path):
        # Session state persistence: the TUI is interactive, so a config saved
        # to the default path (Ctrl+S / Save) must be picked up automatically on
        # the next launch --- otherwise every session starts from the built-in
        # defaults and the user has to Load by hand. An explicit --config keeps
        # its documented meaning (and a missing explicit path still errors).
        try:
            cfg = RunConfig.from_json(path)
        except (OSError, ValueError) as exc:
            print(f"warning: could not load saved config at {path} ({exc}); "
                  "starting from the built-in defaults")
    HarnessTUI(config=cfg, config_path=path, artifacts_root=_ARTIFACTS,
               memory_root=_MEMORY).run()
    return 0


def _diff_core(recorded: dict, replayed: dict) -> list[str]:
    """Keys whose deterministic slice differs between a recorded + replayed rec."""
    a, b = _replay_core(recorded), _replay_core(replayed)
    return [k for k in a if not _deep_eq(a[k], b[k])]


def cmd_replay(args) -> int:
    cfg = _load_config(args)
    store = Store(_ARTIFACTS, cfg.run_name)
    manifest = store.read_manifest()
    if not manifest or "config" not in manifest:
        print(f"no replayable manifest for run '{cfg.run_name}' "
              "(run a --walk-forward first)")
        return 1
    trace = store.read_trace()
    if not trace:
        print(f"no LLM trace recorded for run '{cfg.run_name}'; nothing to replay")
        return 1
    recorded = store.read_factors()

    try:
        replayed = replay(manifest, trace, log=_stdout_log)
    except (ReplayDesyncError, ValueError) as exc:
        print(f"replay failed: {exc}")
        return 1
    print(f"\nreplay: {len(replayed)} record(s) re-driven from "
          f"{len(trace)} cached completion(s)")

    if len(recorded) != len(replayed):
        print(f"  MISMATCH: recorded {len(recorded)} record(s) but replay "
              f"produced {len(replayed)}")
        return 1

    mism = 0
    for rec, rep in zip(recorded, replayed):
        diff = _diff_core(rec, rep)
        if diff:
            mism += 1
        print(f"  [{'MISMATCH' if diff else 'ok'}] {rep['research_date']}  "
              f"{rep['formula']}"
              + (f"   diverged: {', '.join(diff)}" if diff else ""))
    if mism:
        print(f"\n{mism} date(s) diverged --- NOT bit-reproducible from trace")
        return 1
    print(f"\nall {len(replayed)} date(s) reproduced bit-identically")
    return 0


def cmd_memory(args) -> int:
    """Inspect / refresh the persistent research journal under ``memory/``."""
    if args.list:
        entries = memory.load_entries(_MEMORY)
        if not entries:
            print(f"no runs archived yet under {memory.raw_dir(_MEMORY)}")
            return 1
        print(f"{'run_id':<28} {'archived':<21} {'n':>3} {'pass':>4}  objective")
        print("-" * 78)
        for e in entries:
            facts = e.get("factors") or []
            npass = sum(1 for f in facts
                        if (f.get("verdict") or {}).get("passed"))
            print(f"{str(e.get('run_id', '')):<28} "
                  f"{str(e.get('first_archived', '')):<21} "
                  f"{len(facts):>3} {npass:>4}  {memory.entry_objective(e)}")
        return 0

    if args.archive:
        cfg = _load_config(args)
        store = Store(_ARTIFACTS, cfg.run_name)
        entry = memory.archive_store(store, _MEMORY)
        if entry is None:
            print(f"nothing to archive for run '{cfg.run_name}' "
                  "(no manifest or empty ledger)")
            return 1
        print(f"archived {entry['run_id']} "
              f"({len(entry['factors'])} factor(s))")

    if args.synthesize:
        cfg = _load_config(args)
        if cfg.provider == "scripted":
            # The offline double replays canned FACTOR json and holds on its last
            # entry; it would happily write that blob into the notes as if it were
            # a narrative. Refuse rather than poison the journal.
            print("the 'scripted' provider cannot write a synthesis (it replays "
                  "canned factor JSON); choose a real provider with "
                  "--provider NAME or in the config.")
            return 2
        # The narrative costs one real completion, so gate it on the provider's
        # credential rather than dying inside the provider. Deliberately NOT the
        # full run gate: a synthesis reads the journal and calls the model, it
        # never touches the backtest engine, so an unrunnable engine must not
        # block it.
        r = preflight.check_provider(cfg.provider, cfg.model)
        if not r.ok:
            print(preflight.format_readiness(r))
            return 3
        with provider_lifecycle(get_provider(cfg.provider)) as provider:
            payload = memory.synthesize(provider, cfg, _MEMORY)
        if payload is None:
            print("nothing to synthesise --- the journal is empty")
            return 1
        print(f"synthesis written ({payload['model']}, "
              f"{payload['n_runs']} run(s))")

    path = memory.rebuild(_MEMORY)
    n = len(memory.load_entries(_MEMORY))
    print(f"{n} run(s) -> {path}")
    return 0


def cmd_log(args) -> int:
    cfg = _load_config(args)
    store = Store(_ARTIFACTS, cfg.run_name)
    manifest = store.read_manifest()
    if not manifest:
        print(f"no manifest for run '{cfg.run_name}'")
        return 1
    print(f"run_name    : {manifest.get('run_name')}")
    print(f"provider    : {manifest.get('provider')}")
    print(f"model       : {manifest.get('model')}")
    print(f"seed        : {manifest.get('seed')}")
    print(f"data_version: {manifest.get('data_version')}")
    print(f"n_factors   : {manifest.get('n_factors')}")
    print(f"n_pass      : {manifest.get('n_pass')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="zora-harness",
        description="ZORA factor-mining harness: propose a factor with an LLM, "
                    "backtest it, and let one out-of-sample window decide "
                    "Pass/Fail.",
        epilog=(
            "invoke as 'zora-harness <cmd>' (installed console script) or as\n"
            "'python -m harness.cli <cmd>' (from a source checkout).\n"
            "\n"
            "a first session, end to end:\n"
            "  zora-harness doctor --all        # can this machine run at all?\n"
            "  zora-harness configure --set provider=claude --set pass_line=1.0\n"
            "  zora-harness run --walk-forward  # mine a factor per research date\n"
            "  zora-harness show                # the recorded verdict table\n"
            "  zora-harness replay              # re-derive it from the cached\n"
            "                                   # LLM trace, no model call\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True,
                           metavar="{configure,run,show,log,replay,memory,tui,doctor}")

    def _common(sp):
        """Flags every subcommand shares: which config, and the four overrides."""
        sp.add_argument("--config", default=None,
                        help="path to run_config.json (default: built-in defaults)")
        sp.add_argument("--run-name", dest="run_name", default=None,
                        help="artifacts/<run-name>/ subdirectory to read or write")
        sp.add_argument("--provider", default=None,
                        help="LLM provider: scripted (offline), claude, openai, "
                             "deepseek, codex, opencode; overrides the config")
        sp.add_argument("--data-source", dest="data_source", default=None,
                        help="price data: synthetic (offline), yfinance, ctx; "
                             "overrides the config")
        sp.add_argument("--research-date", dest="research_date", default=None,
                        help="T_n, the as-of date splitting in-sample from "
                             "out-of-sample (YYYY-MM-DD); overrides the config")

    sp_cfg = sub.add_parser("configure",
                            help="write a run_config.json from defaults + --set overrides",
                            description="Write a validated run_config.json. Start "
                                        "from the built-in defaults (or --config), "
                                        "apply each --set KEY=VALUE, re-validate, "
                                        "and write the result to --out.")
    sp_cfg.add_argument("--config", default=None,
                        help="base config to start from (defaults to built-in defaults)")
    sp_cfg.add_argument("--out", default=None,
                        help="output path (defaults to --config or run_config.json)")
    sp_cfg.add_argument("--set", action="append", metavar="KEY=VALUE", default=None,
                        help="override a config field (repeatable); "
                             "VALUE parsed as JSON, else treated as a string")
    sp_cfg.set_defaults(func=cmd_configure)

    sp_run = sub.add_parser(
        "run", help="mine a factor and decide Pass/Fail",
        description="Propose -> backtest -> refine on the in-sample window, then "
                    "score the out-of-sample window ONCE against the Pass Line. "
                    "Blocked up front if the provider has no credential or the "
                    "backtest engine / prompt library is unusable (exit 3).")
    _common(sp_run)
    sp_run.add_argument("--walk-forward", action="store_true",
                        help="step T_n from T_0 to T_p (time-forward evolution)")
    sp_run.add_argument("--resume", action="store_true",
                        help="continue a killed walk-forward from its ledger "
                             "(skip done dates; keep learned factors)")
    sp_run.add_argument("--max-iters", dest="max_iters", type=int, default=None,
                        help="refine rounds per date (search DEPTH); "
                             "overrides the config")
    sp_run.add_argument("--candidates-per-round", dest="candidates_per_round",
                        type=int, default=None,
                        help="candidate factors returned per round (search "
                             "BREADTH, one provider call); overrides the config")
    sp_run.add_argument("--frequency", default=None,
                        help="walk-forward step, a pandas offset alias: "
                             + " / ".join(FREQUENCIES) +
                             " (year/quarter/month start, weekly, business, "
                             "daily), or a multiple like 2QS; overrides the config")
    sp_run.add_argument("--n-jobs", dest="n_jobs", type=int, default=None,
                        help="worker PROCESSES scoring one round's candidates "
                             "in parallel (1 = inline; needs "
                             "--candidates-per-round > 1 to do anything); "
                             "overrides the config")
    sp_run.add_argument("--no-memory", dest="no_memory", action="store_true",
                        help="do not copy this run into the persistent research "
                             "journal under memory/ (throwaway run)")
    sp_run.set_defaults(func=cmd_run)

    sp_show = sub.add_parser(
        "show", help="print recorded factors + verdicts",
        description="Print one row per recorded research date: verdict, "
                    "in-sample and out-of-sample objective value, formula.")
    _common(sp_show)
    sp_show.set_defaults(func=cmd_show)

    sp_log = sub.add_parser(
        "log", help="print run manifest",
        description="Print the run manifest: provider, model, seed, data "
                    "version and the factor/pass counts.")
    _common(sp_log)
    sp_log.set_defaults(func=cmd_log)

    sp_replay = sub.add_parser(
        "replay",
        help="reproduce a recorded walk-forward from its cached LLM trace "
             "(no model call)",
        description="Re-drive a recorded run from "
                    "artifacts/runs/<run_name>/llm_trace.jsonl "
                    "with no provider call, then diff the deterministic slice "
                    "of every record against what was recorded. Exit 1 if any "
                    "date fails to reproduce bit-identically.")
    _common(sp_replay)
    sp_replay.set_defaults(func=cmd_replay)

    sp_mem = sub.add_parser(
        "memory",
        help="rebuild / inspect the persistent research journal "
             "(memory/raw + ResearchNotes.md)",
        description="The cross-run journal: with no flag it rebuilds "
                    "ResearchNotes.md from the archived runs.")
    _common(sp_mem)
    sp_mem.add_argument("--list", action="store_true",
                        help="list the archived runs instead of rebuilding")
    sp_mem.add_argument("--archive", action="store_true",
                        help="archive the current run's ledger before rebuilding")
    sp_mem.add_argument("--synthesize", action="store_true",
                        help="spend ONE model call to write the narrative "
                             "'Synthesis' section over the whole journal")
    sp_mem.set_defaults(func=cmd_memory)

    sp_tui = sub.add_parser(
        "tui", help="launch the interactive TUI",
        description="Textual UI over the same config and runner "
                    "(needs the 'tui' extra: pip install '.[tui]').")
    _common(sp_tui)
    sp_tui.set_defaults(func=cmd_tui)

    sp_doc = sub.add_parser(
        "doctor",
        help="check run readiness (provider credential + backtest engine + "
             "prompts) and print first-run setup guidance",
        description="Answer 'could a run start right now?' without starting "
                    "one: provider SDK/key/CLI login, the vendored backtest "
                    "engine and its runtime dependencies, and the prompt "
                    "library. Exit 0 only if everything needed is in place.")
    _common(sp_doc)
    sp_doc.add_argument("--all", action="store_true",
                        help="show a readiness line for every provider "
                             "(plus the one shared runtime block)")
    sp_doc.set_defaults(func=cmd_doctor)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"\ninvalid configuration: {exc}")
        return 2
    except promptlib.PromptAssetError as exc:
        # A missing / hand-broken prompt file is a CONFIGURATION fault, not a
        # crash: the README invites editing these assets, so say what is wrong
        # in one line instead of a traceback the user will read as a harness bug
        # (or, worse, as a model failure).
        print(f"\nprompt configuration error: {exc}")
        print("prompts are hand-editable assets under harness/prompts/ --- check "
              "the filename spelling, then re-check with: "
              "python -m harness.cli doctor")
        return 3


if __name__ == "__main__":
    sys.exit(main())
