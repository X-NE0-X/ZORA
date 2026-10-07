"""Persistent cross-session research journal --- ``memory/raw`` + ``ResearchNotes.md``.

A run's ledger lives under ``artifacts/runs/<run_name>/``, but that tree is
git-ignored, is wiped by ``Store.reset()`` whenever the same ``run_name`` is
re-used, and is documented as safe to delete. So it is a *working* directory,
not a record. This module is the record: every finished run is copied out to

    memory/
      raw/<run_id>.json     one self-contained snapshot per run (manifest + ledger)
      ResearchNotes.md      the consolidated, human-readable digest of all of them

``raw/`` is append-only in spirit and self-sufficient: an entry carries the full
manifest *and* the full factor rows, so the journal still reads correctly after
``artifacts/`` is deleted.

**run_id is content-addressed, not a timestamp**: ``<run_name>-<hash>`` over the
config (minus the fields a resume may legitimately change) plus the data version.
So re-running or ``--resume``-ing the same experiment UPDATES one entry instead
of littering the journal with near-duplicates, while changing any real knob
starts a new entry --- which is the honest thing, because it is a new experiment.

``ResearchNotes.md`` is fully REGENERATED from ``raw/`` on every rebuild, so it
can never drift from the underlying records. Hand-written commentary belongs in
an entry's ``note`` field (preserved across re-archives), not in the markdown.

The journal also **feeds back**: :func:`lessons` is the compact slice a run shows
its proposer (see ``config.memory``, on by default). That feedback is **point-in-time
gated**, on the same rule as an earlier date's OOS score inside one walk
(``runner._prior_oos_closed``). A lesson carries an OOS grade, and a grade
computable only from bars AFTER T_n is look-ahead whichever run produced it ---
otherwise the second run of an experiment would be handed, at simulated
T_n = 2023-01-01, the score a formula earned on the holdout window that *starts*
there. So every lesson carries ``oos_end``, the date its own run's OOS window
closed (from THAT run's ``oos_days``, never this one's), and
``proposer.journal_as_of`` withholds it until T_n reaches that date. A lesson
whose close date cannot be recovered is withheld: fail safe.

What the gate cannot remove is the cost in independence: consulting the same OOS
window across many runs biases the reported Pass rate optimistically.

Three rules keep the feedback from breaking bit-exact replay: the snapshot a run
was shown is recorded in its manifest UNFILTERED (so replay regenerates the same
prompts), it is taken ONCE before the first date (so a walk is internally
consistent and a resume can reuse it), and the gate is applied at RENDER time
from the date's own T_n --- which is deterministic, so the same snapshot always
renders the same prompt.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone

from . import objective as _obj
from .runner import _RESUME_IGNORE, norm_config
from .config import RunConfig
from .factor.contract import POLICY_HASH, SPECS, ContractError, compile_factor
from .store import _safe_run_name

RAW_DIRNAME = "raw"
NOTES_NAME = "ResearchNotes.md"

# Config fields that must NOT change a run's identity. Deliberately the same set
# the resume guard tolerates (runner._RESUME_IGNORE): `logging` is cosmetic, `t_p`
# only EXTENDS a walk-forward (a continued run is the same experiment, and must
# land back in the same entry rather than forking a prefix-duplicate), `data_dir`
# is a TUI scan hint no execution path reads. Anything else differing means a
# genuinely different run.
_IDENTITY_IGNORE = _RESUME_IGNORE

_SYNTHESIS_NAME = "_synthesis.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(payload: object, size: int = 5) -> str:
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.blake2b(blob.encode("utf-8"), digest_size=size).hexdigest()


def raw_dir(root: str) -> str:
    return os.path.join(root, RAW_DIRNAME)


def notes_path(root: str) -> str:
    return os.path.join(root, NOTES_NAME)


def run_id(manifest: dict) -> str:
    """Content-addressed id for a run: ``<run_name>-<hash of what defines it>``.

    The config is put through :func:`runner.norm_config` first, so the identity is
    over what the config MEANS, not how it is spelt --- re-pointing at the same
    parquet by a relative instead of an absolute path lands in the same entry.
    """
    config = norm_config(dict(manifest.get("config") or {}))
    identity = {k: v for k, v in config.items() if k not in _IDENTITY_IGNORE}
    identity["__data_version__"] = manifest.get("data_version")
    name = _safe_run_name(str(manifest.get("run_name") or config.get("run_name") or "run"))
    return f"{name}-{_digest(identity)}"


# --- archive -----------------------------------------------------------------
def _merge_factors(old: list[dict], new: list[dict]) -> list[dict]:
    """Union two ledgers by research date, newest wins, chronological order.

    A resumed run re-archives the same early dates it already wrote; re-running a
    date must update it, not duplicate it. Dates with no parseable
    ``research_date`` are kept as-is (never silently dropped).
    """
    by_date: dict[str, dict] = {}
    loose: list[dict] = []
    for rec in list(old) + list(new):
        key = rec.get("research_date")
        if isinstance(key, str) and key:
            by_date[key] = rec
        else:
            loose.append(rec)
    return [by_date[k] for k in sorted(by_date)] + loose


def archive(manifest: dict, factors: list[dict], root: str, *,
            now: str | None = None) -> dict:
    """Write/refresh one ``raw/<run_id>.json`` entry; return the stored entry."""
    if not manifest:
        raise ValueError("cannot archive a run with no manifest")
    stamp = now or _utc_now()
    rid = run_id(manifest)
    os.makedirs(raw_dir(root), exist_ok=True)
    path = os.path.join(raw_dir(root), rid + ".json")

    prev = _read_json(path) or {}
    entry = {
        "run_id": rid,
        "run_name": manifest.get("run_name"),
        # A hand-written line about what this run was for / what it showed. The
        # only field a human is meant to edit; carried forward on every refresh.
        "note": prev.get("note", ""),
        "first_archived": prev.get("first_archived", stamp),
        "last_archived": stamp,
        "manifest": manifest,
        "factors": _merge_factors(prev.get("factors") or [], factors),
    }
    _write_json(path, entry)
    return entry


def archive_store(store, root: str, *, now: str | None = None) -> dict | None:
    """Archive whatever a :class:`~harness.store.Store` currently holds.

    Returns ``None`` (instead of raising) when there is nothing worth recording
    --- no manifest or an empty ledger --- so a caller can archive
    unconditionally after a run without special-casing failures.
    """
    manifest = store.read_manifest()
    factors = store.read_factors()
    if not manifest or not factors:
        return None
    return archive(manifest, factors, root, now=now)


def _read_json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _write_atomic(path: str, text: str) -> None:
    """Write via a temp file + ``os.replace`` so a crash cannot truncate a record.

    An entry carries the only copy of a hand-written ``note`` and of
    ``first_archived``; a half-written file reads back as unparseable, and the
    next archive would silently start the entry over and lose both.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def _write_json(path: str, obj: dict) -> None:
    _write_atomic(path, json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_entries(root: str) -> list[dict]:
    """Every archived run, oldest first. Unreadable files are skipped, not fatal."""
    d = raw_dir(root)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        if not name.endswith(".json") or name.startswith("_"):
            continue          # `_synthesis.json` and friends are not run entries
        entry = _read_json(os.path.join(d, name))
        if entry and entry.get("factors") is not None:
            out.append(entry)
    out.sort(key=lambda e: (str(e.get("first_archived") or ""), str(e.get("run_id") or "")))
    return out


# --- digest helpers ----------------------------------------------------------
def _passed(rec: dict) -> bool:
    return bool((rec.get("verdict") or {}).get("passed"))


def _oos(rec: dict):
    return (rec.get("verdict") or {}).get("oos_value")


def _is_val(rec: dict):
    return (rec.get("verdict") or {}).get("is_value")


def _fmt(v) -> str:
    if not isinstance(v, (int, float)) or v != v:      # None / NaN
        return "n/a"
    return f"{float(v):.4f}"


def _reason_list(rec: dict) -> list[str]:
    rs = (rec.get("verdict") or {}).get("reasons") or []
    return [str(r) for r in rs] or ["below the Pass Line"]


def _reasons(rec: dict) -> str:
    return "; ".join(_reason_list(rec))


def _reason_class(reason: str) -> str:
    """Collapse a reason to its CLASS by blanking the numbers it embeds.

    Verdict reasons interpolate the realised values ("OOS Sortino -1.2969 fails
    Pass Line (>= 1.0000)"), so every failure is a unique string and a raw tally
    would only ever count 1s. Blanking the numerals groups the recurring failure
    MODES, which is the thing worth looking at --- and it stays correct if the
    wording in validate.py ever changes.
    """
    return re.sub(r"-?\d+(?:\.\d+)?", "#", reason)


def entry_objective(entry: dict) -> str:
    m = entry.get("manifest") or {}
    cfg = m.get("config") or {}
    return str(m.get("objective") or cfg.get("objective") or "sortino")


def _label(name: str) -> str:
    """Pretty objective label, tolerant of one the registry no longer knows.

    The journal outlives the code: an entry archived before a metric was renamed
    or retired must still render, not take the whole rebuild down with a KeyError.
    """
    return _obj.label(name) if _obj.is_valid(name) else name


def _all_factors(entries: list[dict]) -> list[tuple[dict, dict]]:
    """Flatten to ``(entry, factor_record)`` pairs so a row can name its run."""
    return [(e, f) for e in entries for f in (e.get("factors") or [])]


def _operator_usage(pairs: list[tuple[dict, dict]]) -> tuple[Counter, Counter]:
    """Count registered operators in current checked IR, without an old parser."""

    won: Counter = Counter()
    lost: Counter = Counter()
    for _entry, rec in pairs:
        try:
            checked = compile_factor(str(rec.get("formula") or ""), rec.get("parameters", {}))
        except ContractError:
            continue
        pending = [checked.node]
        ops = set()
        while pending:
            node = pending.pop()
            if node.op in SPECS:
                ops.add(node.op)
            pending.extend(node.children)
        (won if _passed(rec) else lost).update(ops)
    return won, lost


def corpus_fingerprint(entries: list[dict]) -> str:
    """Content hash of the journal --- what a synthesis was written against.

    Deliberately excludes the archive timestamps, so merely re-archiving an
    unchanged run does not mark an existing synthesis stale.
    """
    payload = [
        {
            "run_id": e.get("run_id"),
            "note": e.get("note", ""),
            "objective": entry_objective(e),
            "factors": [
                {"date": f.get("research_date"), "formula": f.get("formula"),
                 "parameters": f.get("parameters", {}),
                 "context": evidence_context(e, f),
                 "passed": _passed(f), "oos": _oos(f), "reasons": _reasons(f)}
                for f in (e.get("factors") or [])
            ],
        }
        for e in entries
    ]
    return _digest(payload, size=8)


# --- lessons: the compact slice the proposer is shown -------------------------
# Bounds on what a run carries into the prompt. The journal grows without limit;
# the prompt cannot. Losses are ranked by how OFTEN a formula has died (a
# mechanism that failed five times is the strongest lesson in the book), wins by
# recency (the most recent evidence about what still works).
_MAX_JOURNAL_WINS = 10
_MAX_JOURNAL_LOSSES = 15


def _factor_oos_end(rec: dict) -> str | None:
    """Use the explicit close date recorded by the current execution engine."""
    end = rec.get("oos_end")
    return str(end) if end else None


def _current_entries(root: str) -> list[dict]:
    """Historical files remain readable, but cannot enter current model prompts."""
    selected = []
    for entry in load_entries(root):
        manifest = entry.get("manifest") or {}
        if manifest.get("math_policy_hash") != POLICY_HASH:
            continue
        try:
            RunConfig.from_recorded_dict(manifest.get("config"))
        except (TypeError, ValueError):
            continue
        selected.append(entry)
    return selected


def _merge_oos_end(a: str | None, b: str | None) -> str | None:
    """Fold two occurrences' close dates into the one that gates the lesson.

    A lesson aggregates every occurrence of a formula (``n_seen``), so it may only
    be shown once the LAST of them has closed --- otherwise the repeat count
    itself would reveal a verdict that is not yet knowable. An occurrence whose
    close date could not be recovered makes the whole lesson unknowable.
    """
    if a is None or b is None:
        return None
    return max(a, b)


def evidence_context(entry: dict, rec: dict) -> dict:
    """Scoring context, excluding cosmetic controls and search/model choices."""
    manifest = entry.get("manifest") or {}
    raw = manifest.get("config") or {}
    try:
        cfg = RunConfig.from_dict(raw)
    except (ValueError, TypeError):
        # Archived notes may contain an unsupported historical config. Preserve
        # its evidence verbatim for display; never silently infer a new contract.
        return {"recorded_config": raw,
                "data_version": rec.get("data_version", manifest.get("data_version")),
                "objective": rec.get("objective") or entry_objective(entry),
                **(rec.get("experiment_context") or {})}
    context = {
        "symbols": cfg.symbols, "asset_class": cfg.asset_class,
        "data_source": rec.get("data_source", cfg.data_source),
        "data_version": rec.get("data_version", manifest.get("data_version")),
        "cost_bps": cfg.cost_bps, "gross": cfg.gross,
        "initial_cash": cfg.initial_cash, "annualization": cfg.annualization,
        "weight_config": cfg.backtest_weight_config(),
        "objective": rec.get("objective") or entry_objective(entry),
        "pass_line": cfg.pass_line, "require_sign_consistency": cfg.require_sign_consistency,
        "min_is_days": cfg.min_is_days, "min_oos_days": cfg.min_oos_days,
        "close_delisted_at_last": cfg.close_delisted_at_last,
        "allow_missing_symbols": cfg.allow_missing_symbols,
        "clock": {key: getattr(cfg, key) for key in (
            "frequency", "is_years", "oos_days", "oos_mode", "warmup",
            "is_start", "is_end", "oos_start", "oos_end")},
        "math_contract": cfg.math_contract,
        "math_policy_hash": manifest.get("math_policy_hash"),
    }
    context.update(rec.get("experiment_context") or {})
    context["objective"] = rec.get("objective") or entry_objective(entry)
    return context


def lessons(root: str, *, max_wins: int = _MAX_JOURNAL_WINS,
            max_losses: int = _MAX_JOURNAL_LOSSES) -> dict | None:
    """Cross-run experience for the proposer: what has worked, what keeps dying.

    Deduplicated by checked expression (including bindings) AND scoring context.
    Repeated evidence at different dates of the same experiment accumulates.
    Different objectives, costs, universes or clocks never share a score. Returns
    ``None`` when the journal has nothing to say, so a caller can drop the block
    entirely rather than render an empty one.

    Every lesson carries ``oos_end`` --- the date the LAST of its occurrences
    became knowable, on that occurrence's own run's clock. Nothing is filtered
    here: this snapshot is taken once per run and recorded verbatim in the
    manifest, and the point-in-time gate is applied per research date at render
    time (``proposer.journal_as_of``). Keeping the two apart is what lets a
    journal-informed run replay bit-for-bit.
    """
    wins: dict[str, dict] = {}
    losses: dict[str, dict] = {}
    current_entries = _current_entries(root)
    for entry, rec in _all_factors(current_entries):
        # A frozen OOS numerical failure is terminal evidence, never a
        # generation/repair lesson. Keep its archived record, omit feedback.
        if (rec.get("verdict") or {}).get("label") == "NUMERICAL_REJECT":
            continue
        formula = str(rec.get("formula") or "").strip()
        if not formula:
            continue
        try:
            checked = compile_factor(formula, rec.get("parameters", {}))
        except ContractError:
            continue
        context = evidence_context(entry, rec)
        identity = _digest([checked.record()["expression_hash"], context], size=32)
        oos_end = _factor_oos_end(rec)
        bucket = wins if _passed(rec) else losses
        seen = bucket.setdefault(identity, {
            "formula": formula, "parameters": checked.bindings(),
            "expression_hash": checked.record()["expression_hash"],
            "experiment_context": context, "objective": context["objective"],
            "oos_value": _oos(rec), "research_date": rec.get("research_date"),
            "oos_end": oos_end,
            "run_id": entry.get("run_id"), "n_seen": 0, "modes": [],
        })
        seen["n_seen"] += 1
        seen["oos_end"] = _merge_oos_end(seen["oos_end"], oos_end)
        # keep the most recent occurrence as the representative row
        if str(rec.get("research_date") or "") >= str(seen["research_date"] or ""):
            seen["formula"] = formula
            seen["parameters"] = checked.bindings()
            seen["oos_value"] = _oos(rec)
            seen["research_date"] = rec.get("research_date")
            seen["run_id"] = entry.get("run_id")
        if bucket is losses:
            for mode in (_reason_class(r) for r in _reason_list(rec)):
                if mode not in seen["modes"]:
                    seen["modes"].append(mode)

    top_wins = sorted(wins.values(),
                      key=lambda w: (str(w["research_date"] or ""), w["formula"]),
                      reverse=True)[:max_wins]
    top_losses = sorted(losses.values(),
                        key=lambda l: (-l["n_seen"], l["formula"]))[:max_losses]
    if not top_wins and not top_losses:
        return None
    return {"wins": top_wins, "losses": top_losses,
            "n_runs": len(current_entries)}


# --- render ------------------------------------------------------------------
_HEADER = "# ZORA Research Notes"
_BANNER = (
    "<!-- GENERATED FILE. Rebuilt from memory/raw/ by `python -m harness.cli memory`.\n"
    "     Hand edits are overwritten --- put durable commentary in an entry's\n"
    "     \"note\" field in memory/raw/<run_id>.json, which survives every rebuild. -->"
)


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _evidence_text(entry: dict, rec: dict) -> str:
    bindings = json.dumps(rec.get("parameters", {}), sort_keys=True, ensure_ascii=False)
    context = json.dumps(evidence_context(entry, rec), sort_keys=True, ensure_ascii=False)
    return _clip_cell(f"`{rec.get('formula', '')}`; parameters={bindings}; context={context}")


def _clip_cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def render_notes(entries: list[dict], synthesis: dict | None = None) -> str:
    """Render the whole journal. Pure function of its inputs (no clock, no I/O)."""
    lines = [_HEADER, "", _BANNER, ""]
    if not entries:
        lines += [
            "_No runs archived yet._", "",
            "Every finished run is copied here automatically "
            "(`python -m harness.cli run ...`).",
            "Rebuild this file by hand with `python -m harness.cli memory`.",
            "",
        ]
        return "\n".join(lines)

    pairs = _all_factors(entries)
    n_pass = sum(1 for _e, f in pairs if _passed(f))
    dates = sorted(str(f.get("research_date")) for _e, f in pairs
                   if f.get("research_date"))
    updated = max((str(e.get("last_archived") or "") for e in entries), default="")
    rate = f"{100.0 * n_pass / len(pairs):.1f}%" if pairs else "n/a"
    span = f"{dates[0]} -> {dates[-1]}" if dates else "n/a"
    lines += [
        f"**{len(entries)} run(s)** | **{len(pairs)} factor(s)** | "
        f"**{n_pass} PASS** ({rate}) | research dates {span} | last archived {updated}",
        "",
    ]

    # --- synthesis (only when one has been generated) ------------------------
    if synthesis and synthesis.get("text"):
        stale = synthesis.get("fingerprint") != corpus_fingerprint(entries)
        lines += ["## Synthesis", ""]
        if stale:
            lines += [
                "> **STALE** --- written against an earlier state of the journal "
                f"({synthesis.get('n_runs', '?')} run(s), "
                f"{synthesis.get('generated_at', 'unknown date')}). "
                "Re-run `python -m harness.cli memory --synthesize` to refresh.",
                "",
            ]
        lines += [str(synthesis["text"]).strip(), "",
                  f"_{synthesis.get('model', 'llm')} | "
                  f"{synthesis.get('generated_at', '')}_", ""]

    # --- what worked ---------------------------------------------------------
    wins = [(e, f) for e, f in pairs if _passed(f)]
    lines += ["## What worked", ""]
    if wins:
        lines += _table(
            ["research date", "OOS", "objective", "formula", "run"],
            [[str(f.get("research_date", "?")), _fmt(_oos(f)),
              _label(f.get("objective") or entry_objective(e)), _evidence_text(e, f),
              str(e.get("run_id", ""))]
             for e, f in sorted(wins, key=lambda p: str(p[1].get("research_date")))],
        ) + [""]
    else:
        lines += ["_Nothing has cleared its Pass Line yet._", ""]

    # --- what did not --------------------------------------------------------
    losses = [(e, f) for e, f in pairs if not _passed(f)]
    lines += ["## What did not work", ""]
    if losses:
        why: Counter = Counter()
        for _e, f in losses:
            # count each reason on its own: one factor can fail for two reasons
            # and both belong in the tally.
            why.update(_reason_class(r) for r in _reason_list(f))
        lines += ["### Dominant failure modes", "",
                  "_Numbers blanked to `#` so recurring modes group together._", ""]
        lines += _table(["n", "failure mode"],
                        [[str(n), r] for r, n in why.most_common()]) + [""]
        lines += ["### Rejected factors", ""]
        lines += _table(
            ["research date", "OOS", "objective", "formula", "why", "run"],
            [[str(f.get("research_date", "?")), _fmt(_oos(f)),
              _label(f.get("objective") or entry_objective(e)),
              _evidence_text(e, f), _reasons(f), str(e.get("run_id", ""))]
             for e, f in sorted(losses, key=lambda p: str(p[1].get("research_date")))],
        ) + [""]
    else:
        lines += ["_Nothing has failed yet._", ""]

    # --- which primitives keep showing up ------------------------------------
    won, lost = _operator_usage(pairs)
    if won or lost:
        lines += [
            "## Operator usage", "",
            "How often each DSL primitive appears in a winning vs a rejected "
            "formula --- a cheap read on which mechanisms keep earning their keep.",
            "",
        ]
        lines += _table(
            ["operator", "in PASS", "in FAIL"],
            [[op, str(won.get(op, 0)), str(lost.get(op, 0))]
             for op in sorted(set(won) | set(lost),
                              key=lambda o: (-(won.get(o, 0) + lost.get(o, 0)), o))],
        ) + [""]

    # --- per-run log ---------------------------------------------------------
    lines += ["## Runs", ""]
    for e in entries:
        lines += _render_run(e)
    return "\n".join(lines)


def _render_run(entry: dict) -> list[str]:
    m = entry.get("manifest") or {}
    cfg = m.get("config") or {}
    facts = entry.get("factors") or []
    obj = entry_objective(entry)
    npass = sum(1 for f in facts if _passed(f))
    symbols = cfg.get("symbols") or []
    universe = (f"{len(symbols)} {cfg.get('asset_class', '?')} "
                f"({', '.join(map(str, symbols[:6]))}"
                f"{', ...' if len(symbols) > 6 else ''})") if symbols else "?"

    lines = [f"### `{entry.get('run_id', '?')}`", ""]
    if entry.get("note"):
        # one blockquote line: a raw newline would drop out of the quote
        lines += ["> " + " ".join(str(entry["note"]).split()), ""]
    lines += [
        f"- **archived** {entry.get('first_archived', '?')}"
        + (f" (refreshed {entry['last_archived']})"
           if entry.get("last_archived") != entry.get("first_archived") else ""),
        f"- **model** {m.get('provider', '?')} / {m.get('model', '?')} | seed {m.get('seed', '?')}",
        f"- **universe** {universe} | data {cfg.get('data_source', '?')}"
        f"@{str(m.get('data_version') or '')[:8]} "
        f"[{cfg.get('data_start', '?')} .. {cfg.get('data_end', '?')}]",
        f"- **clock** T_0={cfg.get('t_0', '?')} T_p={cfg.get('t_p', '?')} "
        f"step={cfg.get('frequency', '?')} | IS={cfg.get('is_years', '?')}y "
        f"OOS={cfg.get('oos_days', '?')}bd",
        f"- **objective** {_label(obj)} | Pass Line {cfg.get('pass_line', '?')} | "
        f"depth {cfg.get('max_iters', '?')} x breadth {cfg.get('candidates_per_round', '?')}",
        f"- **result** {len(facts)} factor(s), {npass} PASS",
        "",
    ]
    lines += _table(
        ["research date", "verdict", "IS", "OOS", "formula", "mechanism"],
        [[str(f.get("research_date", "?")),
          str((f.get("verdict") or {}).get("label", "?")),
          _fmt(_is_val(f)), _fmt(_oos(f)),
          _evidence_text(entry, f),
          _clip(str(f.get("mechanism") or "").replace("|", "\\|"))]
         for f in facts],
    ) + [""]
    return lines


def _clip(text: str, width: int = 80) -> str:
    """Make model-authored prose safe for a markdown table cell, and bound it.

    Collapses whitespace (a newline inside a cell would end the row early) and
    marks a truncation, so a cut sentence cannot read as a complete one.
    """
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3].rstrip() + "..."


# --- top-level operations ----------------------------------------------------
def read_synthesis(root: str) -> dict | None:
    return _read_json(os.path.join(raw_dir(root), _SYNTHESIS_NAME))


def rebuild(root: str) -> str:
    """Regenerate ``ResearchNotes.md`` from ``raw/``; return the file path."""
    entries = load_entries(root)
    path = notes_path(root)
    _write_atomic(path, render_notes(entries, read_synthesis(root)))
    return path


def remember(store, root: str, *, now: str | None = None) -> dict | None:
    """Archive a finished run and refresh the notes. Never raises on I/O trouble.

    Called at the end of a run, where a journal problem must not turn a
    successful, already-persisted run into a failure --- the ledger under
    ``artifacts/`` is still intact either way.
    """
    try:
        entry = archive_store(store, root, now=now)
        if entry is not None:
            rebuild(root)
        return entry
    except (OSError, ValueError):
        return None


def synthesize(provider, config, root: str, *, now: str | None = None) -> dict | None:
    """Ask the configured model to write the narrative over the whole journal.

    Deterministic aggregation (see :func:`render_notes`) answers *what* happened;
    this answers *so what*. Cached against :func:`corpus_fingerprint` so the notes
    can flag it STALE rather than quietly presenting conclusions drawn from a
    journal that has since moved on.
    """
    from . import promptlib

    entries = _current_entries(root)
    if not entries:
        return None
    facts = render_notes(entries)          # the deterministic body, minus synthesis
    # Name the objective the verdicts were actually scored on. Runs may disagree,
    # and claiming one when the journal holds several would have the model reason
    # against a goal that was never in force (see the refine-prompt precedent:
    # give the data AND state the goal, but never state the wrong one).
    objectives = sorted({_label(entry_objective(e)) for e in entries})
    messages = [{"role": "user", "content": promptlib.render(
        "user_synthesis", notes=facts,
        objective=objectives[0] if len(objectives) == 1
        else " / ".join(objectives) + " (it varies by run --- see each run's block)",
    )}]
    from .providers.base import complete_provider

    resp = complete_provider(
        provider,
        promptlib.render("system_synthesis"), messages,
        model=config.model, seed=config.seed, max_tokens=config.max_tokens,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
    )
    payload = {
        "text": resp.text.strip(),
        "model": f"{resp.provider}/{resp.model}",
        "generated_at": now or _utc_now(),
        "fingerprint": corpus_fingerprint(entries),
        "n_runs": len(entries),
    }
    _write_json(os.path.join(raw_dir(root), _SYNTHESIS_NAME), payload)
    rebuild(root)
    return payload
