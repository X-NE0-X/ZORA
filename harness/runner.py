"""Orchestration --- the factor-mining loop and the time-forward evolution.

  run_once(cfg)         : one research date T_n. Iteratively propose+refine a
                          factor on the IS window (raising the chosen objective),
                          then score the best formula ONCE on the OOS window and
                          emit a Pass/Fail verdict against the Pass Line.
  run_walk_forward(cfg) : step T_n from T_0 to T_p by ``frequency`` and run_once
                          at each stop, so later dates can learn from earlier ones.

The LLM lives only inside proposer.propose; everything below (backtest, split,
verdict) is deterministic and vendor-free.
"""
from __future__ import annotations

import os
import pathlib
import time
import warnings
from typing import cast

import pandas as pd

from . import alignment
from . import infra_engine
from . import objective as _obj
from . import preflight
from . import promptlib
from .backtest import metrics, run_backtest
from .config import RunConfig
from .cancellation import check_cancelled, cancellable_sleep
from .data import _harness_root, load_panel, resolve_parquet_paths
from .calendars import session_bounds
from .factor.evaluate import FactorEvalError, evaluate
from .factor.contract import compile_factor, ContractError, POLICY_HASH
from .factor.checked_evaluate import ResearchScope
from .factor.protocol import configured_contract, require_contract
from .parallel import pool_for
from .proposer import ProposalError, journal_as_of, propose_many
from .providers import close_provider, get_provider
from .providers.base import LLMProvider, complete_provider
from .providers.codex_controls import ProviderConfigurationError
from .providers.replay import (RecordingProvider, ReplayDesyncError,
                               ReplayProvider, TracePersistenceError)
from .store import Store
from .validate import evaluate_windows, verdict


@configured_contract
def _register_factor(panel, formula, config: RunConfig) -> dict | None:
    """Register the chosen factor's signal into FactorEngine (lineage/manifest)."""
    if not infra_engine.available():
        return None
    manager = infra_engine.get_manager(panel.test_data)
    feature_id = f"{config.run_name}__{config.research_date}"
    signal = evaluate(formula, panel, scope=ResearchScope(end=config.oos_window()[1], phase="registration"))
    infra_engine.register_signal(manager, feature_id, signal)
    return {"feature_id": feature_id, "engine": "FactorEngine",
            "n_registered": int(len(manager._artifacts))}


@configured_contract
def _is_metrics(formula, panel, config: RunConfig) -> dict:
    """Metrics on the IS window only (no OOS is ever looked at here)."""
    is_start, is_end = config.is_window()
    panel = panel.slice(end=is_end)
    signal = evaluate(formula, panel, scope=ResearchScope(is_start, is_end, "IS"))
    bt = run_backtest(
        signal, panel, config.cost_bps, config.gross, config.initial_cash,
        weight_config=config.backtest_weight_config(),
        close_delisted_at_last=config.close_delisted_at_last,
    )
    is_start, is_end = config.is_window()
    idx = bt["strategy_returns"].index
    mask = (idx >= is_start) & (idx <= is_end)
    sub = {"strategy_returns": bt["strategy_returns"][mask],
           "turnover": bt["turnover"][mask]}
    if "gross" in bt:
        sub["gross"] = bt["gross"][mask]
    return metrics(sub, config.annualization)


# --- transient-failure backoff ----------------------------------------------
def _backoff_delay(n_consecutive: int, config: RunConfig) -> float:
    """Seconds to wait before the next round, after ``n_consecutive`` failures.

    Exponential --- ``retry_backoff * 2**(n-1)`` --- capped at ``retry_max_delay``
    so a long search cannot stall for hours on a dead endpoint. Returns 0.0 when
    ``retry_backoff`` is 0 (the pause is disabled) or the config predates the
    knobs, in which case the caller does not sleep at all.
    """
    base = float(getattr(config, "retry_backoff", 0.0) or 0.0)
    if base <= 0.0:
        return 0.0
    cap = float(getattr(config, "retry_max_delay", 0.0) or 0.0)
    if cap <= 0.0:
        return 0.0
    delay = base * (2.0 ** max(0, n_consecutive - 1))
    return min(delay, cap)


def _no_sleep(_seconds: float) -> None:
    """A sleep that does not: used where there is nothing to wait for (replay)."""


# --- token accounting --------------------------------------------------------
# Canonical count -> the spellings a provider/SDK may report it under. Anthropic
# says input_tokens/output_tokens; the OpenAI-compatible APIs say
# prompt_tokens/completion_tokens/total_tokens.
_USAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "input_tokens": ("input_tokens", "prompt_tokens"),
    "output_tokens": ("output_tokens", "completion_tokens"),
    "total_tokens": ("total_tokens",),
}
_USAGE_KEYS: tuple[str, ...] = tuple(_USAGE_ALIASES)


def _usage_field(payload, key: str):
    """Read ``key`` off a usage payload, which may be a dict or an SDK object."""
    if isinstance(payload, dict):
        return payload.get(key)
    return getattr(payload, key, None)


def extract_usage(resp) -> dict | None:
    """Token counts carried by ONE provider response, or ``None`` if it reported none.

    The provider protocol only pins ``.text``; usage is optional and every SDK
    spells it differently, so this looks for it in the two places a provider
    adapter can put it (a ``usage`` attribute, or ``meta['usage']`` / ``meta``
    itself) and normalises the aliases. Absent or unreadable counts return
    ``None`` --- deliberately distinct from zero, so a run whose provider reports
    nothing is recorded as "not costed" rather than as "cost nothing".
    """
    payload = getattr(resp, "usage", None)
    if payload is None:
        meta = getattr(resp, "meta", None)
        if isinstance(meta, dict):
            payload = meta.get("usage")
            if payload is None and any(
                a in meta for aliases in _USAGE_ALIASES.values() for a in aliases
            ):
                payload = meta
    if payload is None:
        return None
    out: dict[str, int] = {}
    for canon, aliases in _USAGE_ALIASES.items():
        for alias in aliases:
            raw = _usage_field(payload, alias)
            if raw is None:
                continue
            try:
                out[canon] = int(raw)
            except (TypeError, ValueError):
                continue
            break
    if not out:
        return None
    out.setdefault("total_tokens",
                   out.get("input_tokens", 0) + out.get("output_tokens", 0))
    return out


class UsageMeter(LLMProvider):
    """Transparent proxy that SUMS the token usage of every completion.

    A run cannot be costed --- before or after --- unless somebody adds the counts
    up: the provider layer reports them per call and then drops them on the floor.
    This wraps a provider (in practice the :class:`RecordingProvider`, so the
    all proposal completions are counted), accumulates whatever
    usage each response carries, and returns the response untouched.

    ``n_reported`` is the honest denominator: a provider that reports no usage
    leaves it at 0 and the runner then records ``None`` instead of claiming the
    run cost zero tokens. Everything except ``complete`` delegates to ``inner``
    (``trace``, ``prompt_mismatches``, ...), so the meter can be dropped in front
    of any provider without its callers noticing.
    """

    def __init__(self, inner: LLMProvider):
        self.inner = inner
        self.name = inner.name
        # LLMProvider defines these as False class attributes, so __getattr__
        # cannot delegate them from the proxy. Copy capabilities explicitly or
        # complete_provider will drop structured controls before forwarding.
        self.supports_response_format = getattr(
            inner, "supports_response_format", False
        )
        self.supports_structured_output = getattr(
            inner, "supports_structured_output", False
        )
        self.totals: dict[str, int] = {k: 0 for k in _USAGE_KEYS}
        self.totals["n_calls"] = 0
        self.totals["n_reported"] = 0

    def __getattr__(self, item):
        # Only reached for attributes the proxy itself does not define. Guard
        # ``inner`` explicitly: without it, an attribute lookup made before
        # __init__ has run would recurse forever.
        if item == "inner":
            raise AttributeError(item)
        return getattr(self.inner, item)

    @property
    def usage_totals(self) -> dict:
        """A snapshot copy, so a caller can diff two points in the run."""
        return dict(self.totals)

    def complete(self, system, messages, *, model, seed=17, max_tokens=10000,
                 temperature: float | None = None,
                 reasoning_effort: str | None = None,
                 response_format: dict | None = None,
                 structured_schema: dict | None = None,
                 structured_retry_count: int = 2):
        resp = complete_provider(
            self.inner, system, messages, model=model, seed=seed,
            max_tokens=max_tokens, temperature=temperature,
            reasoning_effort=reasoning_effort,
            response_format=response_format, structured_schema=structured_schema,
            structured_retry_count=structured_retry_count,
        )
        self.totals["n_calls"] += 1
        used = extract_usage(resp)
        if used is not None:
            self.totals["n_reported"] += 1
            for key in _USAGE_KEYS:
                self.totals[key] += int(used.get(key, 0))
        return resp


def _usage_delta(before: dict | None, after: dict | None) -> dict | None:
    """What a :class:`UsageMeter` accumulated between two snapshots.

    ``None`` when the provider is not metered at all, or when no completion in
    the interval reported usage --- see :func:`extract_usage` on why that is not
    the same as zero.
    """
    if before is None or after is None:
        return None
    delta = {k: int(after.get(k, 0)) - int(before.get(k, 0))
             for k in set(before) | set(after)}
    return delta if delta.get("n_reported", 0) > 0 else None


def _usage_total(records: list[dict]) -> dict | None:
    """Whole-run token cost: the per-date usage rows added up.

    Summed off the LEDGER rather than off the meter so a resumed run costs the
    dates it inherited too --- the meter only ever saw this session's calls.
    """
    out: dict[str, int] = {}
    seen = False
    for rec in records:
        used = rec.get("usage")
        if not isinstance(used, dict):
            continue
        seen = True
        for key, val in used.items():
            try:
                out[key] = out.get(key, 0) + int(val)
            except (TypeError, ValueError):
                continue
    return out if seen else None


@configured_contract
def optimize(provider, config: RunConfig, panel, log=None,
             prior_factors: list[dict] | None = None,
             prior_failures: list[dict] | None = None,
             journal: dict | None = None, sleep=None, pool=None) -> dict:
    """Iteratively propose+refine on IS; return best proposal + history.

    ``prior_factors`` (PASS factors from earlier walk-forward dates) and
    ``prior_failures`` (their OOS FAILURES) are passed to the proposer as
    learning-by-doing context; ``journal`` adds the same from PREVIOUS RUNS (see
    :mod:`harness.memory`), point-in-time gated at render time against this
    date's T_n. None of the three touch the deterministic scoring, only what the
    model sees.

    Within a date, candidates that are rejected or uncomputable
    are collected into ``negatives`` and fed back to the proposer as dead ends, so
    the model stops re-inventing them. Mandatory static and numerical admission
    is entirely local; no model participates in certification.

    ``sleep`` is the pacing hook used by the transient-failure backoff (defaults
    to :func:`time.sleep`); replay and the test suite pass a no-op so a recorded
    failure costs no wall clock.

    ``pool`` (a :class:`harness.parallel.CandidatePool`, or ``None``) moves the
    round's candidate BACKTESTS into worker processes. It is pure wall clock: the
    proposer still runs here, and every result
    is folded back in candidate order, so the history, the dead ends, the trace
    and the winner come out identical either way. ``None`` --- the default and
    what ``n_jobs=1`` produces --- runs the historic inline path.
    """
    obj = config.objective
    n_cand = max(1, int(getattr(config, "candidates_per_round", 1) or 1))
    config._available_fields = set(panel.fields) - ({"vwap"} if panel.vwap_source != "source" else set())
    wait = sleep if sleep is not None else cancellable_sleep
    history: list[dict] = []
    negatives: list[dict] = []
    # consecutive transient provider failures --- drives the exponential backoff
    # below, and resets the moment the provider answers at all.
    n_transient = 0
    best = None
    for it in range(config.max_iters):
        check_cancelled()
        # ONE provider call per round returns up to n_cand candidates (default 1),
        # so the recorded trace holds exactly one propose completion per round
        # regardless of breadth --- the record/replay contract is unchanged.
        # The provider call can take minutes (a CLI spawn + model generation), so
        # it must be bracketed by log lines: without them a long call looks like a
        # hung run (the TUI/CLI prints nothing in between).
        if log:
            log(f"  [iter {it}] asking provider={provider.name} model={config.model} ...")
        try:
            proposals, parse_errors, resp = propose_many(
                provider, config, history or None, prior_factors,
                negatives or None, prior_failures, journal,
            )
        except ProposalError as exc:
            # The provider ANSWERED (its output merely failed to validate), so the
            # link is healthy: clear the transient streak before feeding the dead
            # end forward, or an unlucky parse would keep inflating the backoff.
            n_transient = 0
            if log:
                log(f"  [iter {it}] rejected proposal: {exc}")
            negatives.append({"formula": None, "reason": str(exc)})
            continue
        except (ReplayDesyncError, ProviderConfigurationError):
            # Running past the recorded trace is a hard reproducibility failure,
            # not a transient provider hiccup: never smother it in the catch-all
            # below (which would silently truncate the replayed search).
            raise
        except TracePersistenceError:
            # A provider may have answered successfully while the durable trace
            # append failed. Retrying would spend another call and mislabel an
            # artifact-integrity failure as a transient endpoint problem.
            raise
        except promptlib.PromptAssetError:
            # A missing/malformed prompt file is a CONFIGURATION error, not a
            # provider failure: it fails identically every round, so the
            # catch-all below would relabel it "provider error", burn every
            # remaining iteration on it and end with the misleading "no valid
            # active factor proposal was produced". The readiness gate normally
            # stops this before the first round, but a direct optimize() call
            # (or a prompt deleted mid-run) still reaches here. Surface it as
            # itself --- and before the backoff, since sleeping between eight
            # guaranteed-identical failures only makes the wrong answer slower.
            raise
        except Exception as exc:  # noqa: BLE001 - provider (API/CLI) failure
            # An auth/credential failure (missing/invalid key, lapsed CLI login)
            # is NOT transient: it fails identically every round, so retrying to
            # max_iters just burns calls and ends in a misleading "no valid
            # factor" error. Surface it so the onboarding layer can translate it
            # into a setup guide (cli.cmd_run / the TUI worker route it through
            # preflight.explain_failure). It must be re-raised BEFORE the backoff,
            # too: sleeping between eight guaranteed-identical failures only makes
            # the wrong answer slower.
            if preflight.looks_like_auth_error(str(exc)):
                raise
            # Transient (network, rate limit, CLI crash): skip the round so one
            # bad call can't abort the search. The NEXT round re-asks the very
            # same question --- nothing is appended to ``negatives`` here --- so
            # the retry is the next round, which is why a retried search still
            # consumes exactly one completion per round and the record/replay
            # contract is untouched. What was missing is the pause: without it a
            # 429 on round 1 fires every remaining round back-to-back and burns
            # the whole budget in about a second.
            n_transient += 1
            delay = _backoff_delay(n_transient, config)
            more_rounds = it + 1 < config.max_iters
            if log:
                log(f"  [iter {it}] provider error: {type(exc).__name__}: {exc}"
                    + (f"  [retrying in {delay:.1f}s]" if delay and more_rounds
                       else ""))
            if delay and more_rounds:
                wait(delay)
            continue
        n_transient = 0
        if log:
            log(f"  [iter {it}] got {len(proposals)} candidate(s)"
                + (f", {len(parse_errors)} rejected" if parse_errors else "")
                + " ...")

        # candidates that individually failed to validate become dead ends too, so
        # the next round's proposer sees them (they cost no extra completion).
        for reason in parse_errors:
            if log:
                log(f"  [iter {it}] rejected candidate: {reason}")
            negatives.append({"formula": None, "reason": reason})

        # --- fold ONE candidate's outcome into the run state ------------------
        # Every mutation of ``negatives`` / ``history`` / ``best`` happens here,
        # and BOTH drivers below call it in ascending candidate order. That is
        # what lets the parallel path reproduce the serial one exactly: the two
        # lists are rendered into the NEXT round's prompt in list order, and the
        # incumbent keeps an exact tie (which is reachable --- ``f`` and ``2.0*f``
        # score identically once the book is demeaned and rescaled), so any
        # reordering would change the prompt, the recorded trace, and sometimes
        # the winner.
        def fold(ci: int, proposal, kind: str, payload) -> None:
            nonlocal best
            tag = f"iter {it}" if n_cand == 1 else f"iter {it}.{ci}"

            if kind == "structure":
                if log:
                    log(f"  [{tag}] structure reject: {payload}  {proposal.formula}")
                negatives.append(
                    {"formula": proposal.formula, "parameters": proposal.parameters,
                     "reason": f"structure: {payload}"})
                return
            if kind == "uncomputable":
                if log:
                    log(f"  [{tag}] uncomputable formula: {payload}")
                negatives.append(
                    {"formula": proposal.formula,
                     "parameters": proposal.parameters,
                     "reason": f"uncomputable: {payload}"})
                return

            m = payload
            # D2/D3 (tri-alignment, warn): a computable candidate whose story
            # ignores a field it uses (D2) or whose realised in-sample IC sign
            # contradicts the declared expected_sign (D3). Non-fatal --- surfaced
            # here and fed back to the proposer via history, never a rejection.
            # Deterministic (no provider call), so record/replay is unaffected.
            # Stays in the PARENT even with workers: it is one extra signal
            # evaluation --- milliseconds against a multi-second backtest --- and
            # keeping it here means a worker ships back nothing but a metrics dict.
            align_notes: list[str] = []
            if config.tri_align:
                align_notes = alignment.soft_warnings(proposal, panel, config)
                if align_notes and log:
                    for note in align_notes:
                        log(f"  [{tag}] alignment warn: {note}")

            entry = {
                "iteration": it,
                "candidate": ci,
                "formula": proposal.formula,
                "rationale": proposal.rationale,
                "mechanism": proposal.mechanism,
                "expected_sign": proposal.expected_sign,
                "is_metrics": m,
                "alignment": align_notes,
                "prompt_hash": resp.prompt_hash,
            }
            entry["parameters"] = proposal.parameters
            entry["checked_factor"] = proposal.checked(panel)
            history.append(entry)
            active = _obj.is_active(m)
            n_active = int(m.get("n_active", 0))
            enough_is = n_active >= config.min_is_days
            if log:
                log(f"  [{tag}] IS {_obj.label(obj)}="
                    f"{_obj.value(m, obj):.4f}  {proposal.formula}"
                    f"{'' if active else '  [inactive book -- skipped]'}"
                    f"{'' if enough_is else f'  [IS live bars {n_active} < {config.min_is_days} -- skipped]'}")

            # Never let a degenerate empty book (~zero exposure) win the search:
            # its metrics are vacuous and some objectives would rank it best.
            if not active or not enough_is:
                return
            if best is None or _obj.better(
                _obj.value(m, obj), _obj.value(best["is_metrics"], obj), obj
            ):
                best = entry

        def screen(ci: int, proposal):
            """``(kind, payload)`` if the candidate is already dead, else ``(None, None)``.

            Always runs in the parent. Mandatory admission is deterministic and
            provider-free; the optional annotations never control this gate.
            """
            # Compile against this panel's field provenance before scoring.
            try:
                proposal.checked(panel)
            except ContractError as exc:
                return "structure", str(exc)

            return None, None

        if pool is None:
            # The historic inline path, sequence for sequence: screen a
            # candidate, back-test it, fold it, move to the next.
            for ci, proposal in enumerate(proposals):
                check_cancelled()
                kind, payload = screen(ci, proposal)
                if kind is not None:
                    fold(ci, proposal, kind, payload)
                    continue
                try:
                    factor = proposal.checked(panel)
                    m = _is_metrics(factor, panel, config)
                except FactorEvalError as exc:
                    fold(ci, proposal, "uncomputable", str(exc))
                    continue
                fold(ci, proposal, "ok", m)
                check_cancelled()
        else:
            # Same work, same order of provider calls, same fold order --- only
            # the backtests overlap. Screening still walks the candidates in
            # order, the survivors go out as ONE
            # ordered batch, and nothing touches the run state until every
            # result is back. A dead worker raises straight through.
            screened = [(ci, p) + screen(ci, p) for ci, p in enumerate(proposals)]
            live = [(ci, p) for ci, p, kind, _ in screened if kind is None]
            scored = dict(zip([ci for ci, _ in live],
                              pool.metrics([p.checked(panel) for _, p in live],
                                           panel, config)))
            for ci, proposal, kind, payload in screened:
                if kind is None:
                    ok, payload = scored[ci]
                    kind = "ok" if ok else "uncomputable"
                fold(ci, proposal, kind, payload)

    if best is None:
        raise RuntimeError(
            "no eligible in-sample factor proposal was produced "
            "(all candidates were rejected, uncomputable, inactive, or had fewer "
            f"than min_is_days={config.min_is_days} live-exposure bars)"
        )
    return {"best": best, "history": history}


@configured_contract
def run_once(config: RunConfig, provider=None, store: Store | None = None,
             log=None, prior_factors: list[dict] | None = None,
             register: bool = True,
             prior_failures: list[dict] | None = None,
             journal: dict | None = None, sleep=None, pool=None, panel=None) -> dict:
    """Run one research date and close an internally-created provider."""
    owns_provider = provider is None
    inner = provider or get_provider(config.provider)
    recorded = inner
    trace_owner = inner
    while isinstance(trace_owner, UsageMeter):
        trace_owner = trace_owner.inner
    if store is not None and not isinstance(trace_owner, RecordingProvider):
        recorded = RecordingProvider(inner, on_record=store.append_trace)
    try:
        return _run_once_owned(
            config, provider=recorded, store=store, log=log,
            prior_factors=prior_factors, register=register,
            prior_failures=prior_failures, journal=journal, sleep=sleep,
            pool=pool, panel=panel,
        )
    finally:
        if owns_provider:
            close_provider(inner)


@configured_contract
def _run_once_owned(config: RunConfig, provider, store: Store | None = None,
                    log=None, prior_factors: list[dict] | None = None,
                    register: bool = True,
                    prior_failures: list[dict] | None = None,
                    journal: dict | None = None, sleep=None, pool=None, panel=None) -> dict:
    """Full single-date cycle: optimise on IS, validate once on OOS, record.

    ``prior_factors`` (earlier PASS factors) and ``prior_failures`` (earlier OOS
    failures) feed learning-by-doing context to the proposer; ``journal`` feeds
    the cross-run research memory (:mod:`harness.memory`), point-in-time gated
    against this date's T_n before it is rendered.
    ``register`` gates FactorEngine lineage registration (off during replay, so
    a replay stays a pure read-only reproduction). ``sleep`` and ``pool`` are
    forwarded to :func:`optimize` --- the transient-failure backoff and the
    optional candidate worker pool.
    """
    from .providers.codex_controls import validate_config
    validate_config(config)
    required_start, required_end = required_data_window(config, walk_forward=False)
    _check_config_data_window(config, required_start, required_end)
    if panel is None:
        panel = load_panel(config)
    _check_panel_data_window(panel, required_start, required_end, config)
    obj = config.objective
    oos_start, oos_end = config.oos_window()

    if log:
        log(f"T_n={config.research_date}  universe={len(panel.symbols)}  "
            f"objective={_obj.label(obj)}  pass_line={config.pass_line}  "
            f"OOS={oos_start.date()}..{oos_end.date()}  "
            f"data={panel.source}@{panel.version[:8]}")

    # count the completions this date consumes off a RecordingProvider's trace
    # (its live-appended entries) so resume can realign the trace to the ledger.
    _trace = getattr(provider, "trace", None)
    n_trace0 = len(_trace) if isinstance(_trace, list) else None
    # token accounting is opt-in on the provider (see UsageMeter): an unmetered
    # provider simply has no totals and this date records no usage row.
    usage0 = getattr(provider, "usage_totals", None)

    opt = optimize(provider, config, panel, log=log, prior_factors=prior_factors,
                   prior_failures=prior_failures, journal=journal, sleep=sleep,
                   pool=pool)
    best = opt["best"]

    n_completions = (len(provider.trace) - n_trace0) if n_trace0 is not None else None
    usage = _usage_delta(usage0, getattr(provider, "usage_totals", None))

    check_cancelled()
    factor = best.get("checked_factor", best["formula"])
    numerical_rejection = None
    try:
        windows = evaluate_windows(factor, panel, config)
    except FactorEvalError as exc:
        numerical_rejection = str(exc)
        is_start, is_end = config.is_window()
        windows = {"is_metrics": best["is_metrics"], "oos_metrics": {},
                   "is_index": panel.dates[(panel.dates >= is_start) & (panel.dates <= is_end)],
                   "oos_index": pd.DatetimeIndex([]),
                   "math_admission": {**factor.record(), "numerical_status": "rejected",
                                      "scope": {"phase": "frozen-OOS"}, "reason": numerical_rejection}}
    check_cancelled()
    v = (verdict(windows["is_metrics"], windows["oos_metrics"], config)
         if numerical_rejection is None else
         {"passed": False, "label": "NUMERICAL_REJECT", "objective": obj,
          "pass_line": config.pass_line, "is_value": _obj.value(best["is_metrics"], obj),
          "oos_value": None, "reasons": [numerical_rejection]})
    factor_ref = _register_factor(panel, factor, config) if register and numerical_rejection is None else None
    # the same gate build_messages applied, re-evaluated for the ledger row below
    shown_journal = journal_as_of(journal, config.research_date)

    configured_is_start, configured_is_end = config.is_window()

    def _actual_bound(index, which: str) -> str | None:
        if len(index) == 0:
            return None
        value = index[0] if which == "start" else index[-1]
        return pd.Timestamp(value).date().isoformat()

    record = {
        "research_date": config.research_date,
        "formula": best["formula"],
        "rationale": best["rationale"],
        "mechanism": best["mechanism"],
        "expected_sign": best["expected_sign"],
        "alignment": best.get("alignment", []),
        "objective": obj,
        "pass_line": config.pass_line,
        # Unqualified bounds are the bars actually scored. Configured bounds are
        # retained explicitly so a reader can distinguish calendar intent from
        # the effective market-data slice (including the forced post-IS OOS cut).
        "is_start": _actual_bound(windows["is_index"], "start"),
        "is_end": _actual_bound(windows["is_index"], "end"),
        "oos_start": _actual_bound(windows["oos_index"], "start"),
        "oos_end": _actual_bound(windows["oos_index"], "end"),
        "configured_is_start": configured_is_start.date().isoformat(),
        "configured_is_end": configured_is_end.date().isoformat(),
        "configured_oos_start": oos_start.date().isoformat(),
        "configured_oos_end": oos_end.date().isoformat(),
        "is_n_bars": len(windows["is_index"]),
        "oos_n_bars": len(windows["oos_index"]),
        "is_metrics": windows["is_metrics"],
        "oos_metrics": windows["oos_metrics"],
        "verdict": v,
        "n_iterations": len(opt["history"]),
        "iterations": [
            {"iteration": h["iteration"], "formula": h["formula"],
             "is_value": _obj.value(h["is_metrics"], obj)}
            for h in opt["history"]
        ],
        "data_version": panel.version,
        "data_source": panel.source,
        "factor_ref": factor_ref,
        "provider": provider.name,
        "model": config.model,
        "require_interpretability": config.require_interpretability,
        "n_prior_factors": len(prior_factors or []),
        "n_prior_failures": len(prior_failures or []),
        # what the proposer was ACTUALLY shown at this T_n --- the journal
        # snapshot minus the lessons whose OOS window had not closed yet. The
        # unfiltered snapshot stays in the manifest; these two are the provenance
        # of this date's prompt, so they must count the gated view.
        "n_journal_wins": len((shown_journal or {}).get("wins") or []),
        "n_journal_losses": len((shown_journal or {}).get("losses") or []),
        "n_completions": n_completions,
        "usage": usage,
        "seed": config.seed,
    }
    record["parameters"] = best["parameters"]
    record["experiment_context"] = {
        "symbols": list(panel.symbols), "asset_class": config.asset_class,
        "data_source": panel.source, "data_version": panel.version,
        "cost_bps": config.cost_bps, "gross": config.gross,
        "initial_cash": config.initial_cash, "annualization": config.annualization,
        "weight_config": config.backtest_weight_config(),
        "objective": obj, "pass_line": config.pass_line,
        "require_sign_consistency": config.require_sign_consistency,
        "min_is_days": config.min_is_days, "min_oos_days": config.min_oos_days,
        "close_delisted_at_last": config.close_delisted_at_last,
        "allow_missing_symbols": config.allow_missing_symbols,
        "clock": {key: getattr(config, key) for key in (
            "frequency", "is_years", "oos_days", "oos_mode", "warmup",
            "is_start", "is_end", "oos_start", "oos_end")},
        "math_contract": config.math_contract, "math_policy_hash": POLICY_HASH,
    }
    record["math_admission"] = windows["math_admission"]
    record["is_math_admission"] = evaluate(
        factor, panel, scope=ResearchScope(*config.is_window(), phase="IS")
    ).attrs["zora_admission"]
    for output, attempt in zip(record["iterations"], opt["history"]):
        output["parameters"] = attempt["parameters"]
        output["math_contract"] = attempt["checked_factor"].record()

    if log:
        log(f"  -> best {best['formula']}")
        log(f"  -> IS {_obj.label(obj)}={_obj.value(windows['is_metrics'], obj):.4f}  "
            f"OOS {_obj.label(obj)}={_obj.value(windows['oos_metrics'], obj):.4f}  "
            f"({len(windows['oos_index'])} bars)  "
            f"=> {v['label']}")

    if store is not None:
        check_cancelled()
        store.append_factor(record)
    return record


def _evolution_dates(config: RunConfig) -> list[pd.Timestamp]:
    dates = pd.date_range(config.t_0, config.t_p, freq=config.frequency)
    if len(dates) == 0:
        raise ValueError(
            f"empty walk-forward schedule: no {config.frequency!r} dates between "
            f"t_0={config.t_0} and t_p={config.t_p} (need t_0 <= t_p and a range "
            "spanning at least one step)"
        )
    return list(dates)


def _ts(value) -> pd.Timestamp:
    """Normalize pandas' scalar/NaT union to a validated timestamp."""
    result = cast(pd.Timestamp, pd.Timestamp(str(value)))
    if pd.isna(result):
        raise ValueError(f"invalid date value {value!r}")
    return result


def _date_text(value) -> str:
    return str(_ts(value)).split(" ", 1)[0]


def _next_evolution_date(config: RunConfig, tn) -> pd.Timestamp:
    """Return next research stop for a valid frequency alias."""
    dates = pd.date_range(_ts(tn), periods=2, freq=config.frequency)
    if len(dates) < 2:
        raise ValueError(
            f"cannot derive next research date from frequency {config.frequency!r}"
        )
    return _ts(dates[1])


def _step_config(config: RunConfig, tn, next_tn=None) -> RunConfig:
    """Per-T_n config: set per-step OOS bounds, then drop stale overrides."""
    step_date = _ts(tn)
    payload = {
        **config.to_dict(),
        "research_date": _date_text(step_date),
        "is_start": None, "is_end": None,
        "oos_start": None, "oos_end": None,
    }
    if config.oos_mode == "per_step" and next_tn is not None:
        next_date = _ts(next_tn)
        payload["oos_start"] = _date_text(step_date)
        payload["oos_end"] = _date_text(next_date - pd.Timedelta(days=1))
    return RunConfig.from_dict(payload)


def required_data_window(config: RunConfig, *, walk_forward: bool) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Minimum loaded date range needed for configured IS/OOS evaluation."""
    if not walk_forward and config.oos_mode == "per_step" \
            and not (config.oos_start and config.oos_end):
        raise ValueError(
            "oos_mode='per_step' requires walk-forward mode or an explicit "
            "oos_start/oos_end window"
        )
    dates = (_evolution_dates(config) if walk_forward
    else [_ts(config.research_date)])
    first, last = dates[0], dates[-1]
    if walk_forward:
        is_start = first - pd.DateOffset(years=config.is_years)
    else:
        is_start, _ = config.is_window()
        is_start = _ts(is_start)
    required_start = is_start - pd.tseries.offsets.BDay(config.warmup)

    if walk_forward and config.oos_mode == "per_step":
        required_end = _next_evolution_date(config, last) - pd.Timedelta(days=1)
    elif not walk_forward and config.oos_start and config.oos_end:
        required_end = pd.Timestamp(config.oos_end)
    else:
        required_end = last + pd.tseries.offsets.BDay(config.oos_days)
    return _ts(required_start), _ts(required_end)


def _check_config_data_window(config: RunConfig, required_start, required_end) -> None:
    required_start, required_end = session_bounds(config, required_start, required_end)
    data_start = pd.Timestamp(config.data_start)
    data_end = pd.Timestamp(config.data_end)
    if data_start > required_start:
        raise ValueError(
            f"data_start={data_start.date()} is too late for required research "
            f"history; need data_start <= {pd.Timestamp(required_start).date()} "
            f"(is_years={config.is_years}, warmup={config.warmup})"
        )
    if data_end < required_end:
        raise ValueError(
            f"data_end={data_end.date()} is too early for required OOS coverage; "
            f"need data_end >= {pd.Timestamp(required_end).date()} "
            f"(oos_mode={config.oos_mode}, oos_days={config.oos_days})"
        )


def _check_panel_data_window(panel, required_start, required_end, config=None) -> None:
    if len(panel.dates) == 0:
        raise ValueError("loaded data panel has no dates")
    if config is not None:
        required_start, required_end = session_bounds(config, required_start, required_end)
    actual_start = _ts(panel.dates.min()).normalize()
    actual_end = _ts(panel.dates.max()).normalize()
    if actual_start > _ts(required_start):
        raise ValueError(
            f"loaded data begins at {actual_start.date()}, after required "
            f"research warmup start {_ts(required_start).date()}"
        )
    if actual_end < _ts(required_end):
        raise ValueError(
            f"loaded data ends at {actual_end.date()}, before required OOS end "
            f"{_ts(required_end).date()}"
        )


def check_data_coverage(config: RunConfig, *, walk_forward: bool,
                        log=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate configured and loaded data coverage before a run can reset state."""
    prepare_panel(config, walk_forward=walk_forward, log=log)
    return required_data_window(config, walk_forward=walk_forward)


def prepare_panel(config: RunConfig, *, walk_forward: bool, log=None, panel=None):
    """Validate and freeze one loaded panel before any artifact mutation."""
    required_start, required_end = required_data_window(
        config, walk_forward=walk_forward)
    _check_config_data_window(config, required_start, required_end)
    if panel is None:
        panel = load_panel(config)
    _check_panel_data_window(panel, required_start, required_end, config)
    if log:
        log(f"required data window: {required_start.date()} .. {required_end.date()} "
            f"(oos_mode={config.oos_mode}, warmup={config.warmup})")
    return panel


# Cap how many cross-date FAILURES are fed forward as negative memory, so a long
# walk with many misses cannot bloat the proposer prompt without bound. The most
# recent failures are kept (closest to the current regime).
_MAX_PRIOR_FAILURES = 8


def _win_summary(rec: dict) -> dict:
    """The learning-by-doing context a PASS factor contributes to later dates."""
    return {
        "research_date": rec["research_date"],
        "formula": rec["formula"],
        "parameters": rec.get("parameters", {}),
        "experiment_context": rec.get("experiment_context", {}),
        "objective": rec.get("objective") or rec["verdict"].get("objective"),
        "oos_value": rec["verdict"].get("oos_value"),
        "oos_end": rec.get("oos_end"),
        "rationale": rec.get("rationale", ""),
    }


def _fail_summary(rec: dict) -> dict:
    """The negative-memory context a FAILED factor contributes to later dates."""
    return {
        "research_date": rec["research_date"],
        "formula": rec["formula"],
        "parameters": rec.get("parameters", {}),
        "experiment_context": rec.get("experiment_context", {}),
        "objective": rec.get("objective") or rec["verdict"].get("objective"),
        "oos_value": rec["verdict"].get("oos_value"),
        "oos_end": rec.get("oos_end"),
        "reasons": rec["verdict"].get("reasons") or [],
    }


def oos_close_date(research_date, oos_days: int) -> pd.Timestamp:
    """The date an OOS verdict for ``research_date`` first becomes knowable.

    T_n + ``oos_days`` BUSINESS days --- the same clock ``RunConfig.oos_window``
    ends on. Factored out because two gates must agree on it: the within-walk one
    (:func:`_prior_oos_closed`) and the cross-run journal one
    (``memory.lessons`` -> ``proposer.journal_as_of``). If they ever disagreed,
    the same class of information would be point-in-time in one direction and
    look-ahead in the other.
    """
    return _ts(research_date) + pd.tseries.offsets.BDay(int(oos_days))


def _prior_oos_closed(prior: dict, config: RunConfig, tn: pd.Timestamp) -> bool:
    """True iff ``prior``'s OOS window has fully closed on or before T_n ``tn``.

    Learning-by-doing must be point-in-time: at research date T_n the researcher
    can only know an earlier factor's OOS result once that factor's OOS window has
    actually ended. With the default clock (``oos_days`` short relative to the
    walk-forward step) every earlier window is long closed, so this is a no-op.
    But if ``oos_days`` is large enough that a prior date's OOS window still
    extends past T_n, feeding that factor --- and its OOS score --- would leak an
    evaluation the researcher could not yet have. We withhold it until it closes.

    ``prior`` came from THIS walk and normally carries its resolved ``oos_end``;
    old records fall back to this run's fixed-day clock. A cross-run journal
    lesson carries its own resolved close date (see ``memory._factor_oos_end``).
    """
    recorded_end = prior.get("oos_end")
    if recorded_end:
        return pd.Timestamp(recorded_end) <= tn
    return oos_close_date(prior["research_date"], config.oos_days) <= tn


def _evolve(config: RunConfig, provider, *, store: Store | None = None, log=None,
            prior_seed: list[dict] | None = None,
            fail_seed: list[dict] | None = None,
            skip_dates: frozenset = frozenset(), register: bool = True,
            journal: dict | None = None, sleep=None, pool=None, panel=None) -> list[dict]:
    """Drive T_n over the schedule, carrying PASS factors (and FAILURES) forward.

    Later dates learn from earlier PASS factors (``prior``) AND from earlier OOS
    failures (``prior_fail``, negative memory). ``journal`` is the cross-run
    research memory and is CONSTANT for the whole walk --- it is snapshotted once
    before the first date, so every date sees the same accumulated experience and
    the walk stays self-consistent (and replayable). Constant does not mean
    ungated: each date renders only the lessons whose OOS window has closed by its
    own T_n, so the snapshot is the same while the visible slice grows with the
    clock. Shared by :func:`run_walk_forward` (records + persists) and
    :func:`replay` (reproduces), so the two follow byte-for-byte the same
    evolution logic.
    """
    records: list[dict] = []
    prior = list(prior_seed or [])
    prior_fail = list(fail_seed or [])
    dates = _evolution_dates(config)
    for i, tn in enumerate(dates):
        check_cancelled()
        ds = tn.date().isoformat()
        if ds in skip_dates:
            continue
        # point-in-time learning: only carry forward PASS/FAIL factors whose OOS
        # window has already closed by T_n (see _prior_oos_closed) --- a still-open
        # verdict is not yet knowable to the researcher and must not leak.
        available = [p for p in prior if _prior_oos_closed(p, config, tn)]
        available_fail = [p for p in prior_fail if _prior_oos_closed(p, config, tn)]
        if len(available_fail) > _MAX_PRIOR_FAILURES:
            available_fail = available_fail[-_MAX_PRIOR_FAILURES:]
        next_tn = dates[i + 1] if i + 1 < len(dates) else _next_evolution_date(config, tn)
        rec = run_once(_step_config(config, tn, next_tn), provider=provider, store=store,
                       log=log, prior_factors=available or None,
                       prior_failures=available_fail or None, register=register,
                       journal=journal, sleep=sleep, pool=pool, panel=panel)
        records.append(rec)
        if rec["verdict"]["passed"]:
            prior.append(_win_summary(rec))
        elif rec["verdict"].get("label") != "NUMERICAL_REJECT":
            prior_fail.append(_fail_summary(rec))
    return records


# Config fields that may legitimately differ on resume without desyncing the
# recorded ledger/trace: ``logging`` is cosmetic, ``t_p`` only extends the
# walk-forward horizon (already-recorded dates stay a clean prefix), ``data_dir``
# is a TUI-only scan hint (the discovered paths land in ``parquet_daily``, which
# IS compared) that no execution path reads, ``retry_backoff`` /
# ``retry_max_delay`` only pace a failing search, and ``n_jobs`` only decides how
# many processes run the SAME backtests (results are folded back in candidate
# order either way) --- all four change wall-clock, never what is computed or how
# many completions a clean round consumes. Everything else --- per-date
# computation (max_iters, windows, objective, ...) and the schedule shape (t_0,
# frequency) --- must match or replay would break.
_RESUME_IGNORE = frozenset({
    "logging", "t_p", "data_dir", "retry_backoff", "retry_max_delay", "n_jobs",
    "walk_forward",
})


def _portable_parquet(paths) -> list[str]:
    """``parquet_daily`` reduced to a PLATFORM-INDEPENDENT identity for a run.

    Two steps. First :func:`~harness.data.resolve_parquet_paths`, which anchors a
    relative entry at the harness root, so ``data/x.parquet`` and its absolute
    spelling name one file. Then --- and this is the part that matters --- the
    absolute path is re-expressed RELATIVE to the harness root with forward
    slashes, because an absolute native path is itself machine-dependent: hashing
    it into ``run_id`` and comparing it in the resume guard forked the same
    committed experiment between a Windows and a Linux colleague (two journal
    entries for one experiment, wrong dedup counts, and ``--resume`` refusing with
    "config differs on ['parquet_daily']").

    A file OUTSIDE the harness root cannot be named portably at all, so it keeps
    its absolute path with the separators normalised --- still stable across
    absolute-vs-relative spellings and across ``..`` detours on one machine, which
    is the most that path alone can promise. (Hashing the file's CONTENT would be
    portable but is not usable here: this runs on manifests whose parquet may no
    longer exist, and on files far too large to re-read for an identity check.)
    """
    root = str(_harness_root())
    out: list[str] = []
    for p in resolve_parquet_paths(paths):
        native = os.path.normpath(p)
        try:
            rel = os.path.relpath(native, root)
        except ValueError:          # a different drive on Windows: no relative form
            rel = None
        outside = rel is None or rel.split(os.sep)[0] == os.pardir
        out.append(pathlib.PurePath(native if outside else rel).as_posix())
    return out


def norm_config(d: dict) -> dict:
    """Normalize a valid current configuration and portable parquet spelling.

    Unsupported fields or contracts fail explicitly. Recorded execution uses
    the complete-schema gate before this comparison; no old schema is inferred.
    """
    out = RunConfig.from_dict(dict(d)).to_dict()
    if out.get("parquet_daily"):
        try:
            out["parquet_daily"] = _portable_parquet(out["parquet_daily"])
        except (TypeError, ValueError, OSError):
            pass
    return out


def _config_conflicts(prev: dict, cur: dict) -> list[str]:
    """Keys on which a resume config diverges from the recorded run's config."""
    prev, cur = norm_config(prev), norm_config(cur)
    return sorted(
        k for k in (set(prev) | set(cur))
        if k not in _RESUME_IGNORE and prev.get(k) != cur.get(k)
    )


def _journal_snapshot(config: RunConfig, memory_root: str | None) -> dict | None:
    """The cross-run lessons this run may be shown, read once up front.

    Deliberately UNFILTERED: this is the snapshot recorded in the manifest, and
    replay has to regenerate prompts from it. The point-in-time gate is applied
    per date, at render time, from that date's T_n
    (``proposer.journal_as_of``) --- so the recorded snapshot stays a constant and
    the prompts stay reproducible.

    ``None`` when the feature is off or the journal has nothing to say. Journal
    trouble is never fatal to a run: the record under ``artifacts/`` is what makes
    a run real, and a missing lesson block only makes the search less informed.
    """
    if not memory_root or not getattr(config, "memory", False):
        return None
    from . import memory as _memory

    try:
        return _memory.lessons(memory_root)
    except (OSError, ValueError):
        return None


def run_walk_forward(config: RunConfig, artifacts_root: str = "artifacts",
                     log=None, resume: bool = False,
                     memory_root: str | None = None, provider=None, panel=None) -> dict:
    """Run a walk while exclusively owning ``artifacts/runs/<run_name>``."""
    owns_provider = provider is None
    inner = provider or get_provider(config.provider)
    store = Store(artifacts_root, config.run_name)
    try:
        with store.ownership():
            return _run_walk_forward_owned(
                config, artifacts_root=artifacts_root, log=log, resume=resume,
                memory_root=memory_root, inner=inner, store=store, panel=panel,
            )
    finally:
        if owns_provider:
            close_provider(inner)


def _run_walk_forward_owned(config: RunConfig, artifacts_root: str,
                            log, resume: bool, memory_root: str | None,
                            *, inner, store: Store, panel=None) -> dict:
    """Step T_n from T_0 to T_p; run_once at each; return the full ledger.

    Later dates see earlier dates' PASS factors (learning-by-doing). Every LLM
    completion is recorded to the run's ``llm_trace.jsonl`` (for :func:`replay`).
    With ``resume=True`` the ledger is a checkpoint: already-recorded dates are
    skipped and their PASS factors seed the learning context, so a killed run can
    continue instead of starting over.

    ``memory_root`` points at the persistent research journal; when given (and
    ``config.memory`` is on) its accumulated lessons are read ONCE, before the
    first date, and offered to the proposer at every date --- each date seeing
    only the lessons whose own OOS window has closed by its T_n. It defaults to
    ``None`` --- no cross-run memory --- so a library/test caller is hermetic and
    only the CLI/TUI opt in to the real journal. The snapshot is stored in the
    manifest (unfiltered) so :func:`replay` can regenerate byte-identical prompts.

    Token usage is accumulated per date into each ledger row's ``usage`` and
    summed into ``manifest['usage']``, so a finished run can be costed; it is
    ``None`` for a provider that reports no usage (see :func:`extract_usage`).
    """
    from .providers.codex_controls import validate_config
    validate_config(config)
    store.ensure()

    # Build the schedule BEFORE anything destructive. ``store.reset()`` below
    # deletes the previous run's ledger and --- irreplaceably --- its LLM trace,
    # and this is the last thing that can still reject the config: an empty
    # schedule (t_0 > t_p) raises here. RunConfig now rejects a malformed
    # ``frequency`` at construction, which is earlier still; this is the belt to
    # that pair of braces, and it costs one date_range call.
    _evolution_dates(config)
    # Verify source coverage before resume/reset can touch the artifact directory.
    # Pin this validated panel for every stop. run_once also checks coverage as
    # a public single-date entry point, without redownloading during a walk.
    panel = prepare_panel(config, walk_forward=True, log=log, panel=panel)
    data_version = panel.version

    journal = _journal_snapshot(config, memory_root)

    existing: list[dict] = []
    prior_seed: list[dict] = []
    fail_seed: list[dict] = []
    skip_dates: set[str] = set()
    if resume:
        # The recorded manifest's config must match: resuming with a changed
        # config (max_iters, windows, objective, pass_line, ...) would desync the
        # ledger/trace from what replay expects and silently corrupt it.
        prev_manifest = store.read_manifest()
        if prev_manifest:
            RunConfig.from_recorded_dict(prev_manifest.get("config"))
            if prev_manifest.get("math_policy_hash") != POLICY_HASH:
                raise ValueError("MATH_POLICY_MISMATCH: cannot resume under a different policy")
        stale_artifacts = [path for path in (store.ledger_path, store.trace_path)
                           if os.path.exists(path)]
        if not prev_manifest and stale_artifacts:
            raise ValueError(
                f"cannot resume run '{config.run_name}': manifest.json is missing "
                "while ledger/trace artifacts still exist. Their config and data "
                "identity cannot be proven; recover the manifest or start a fresh "
                "run_name."
            )
        # Continue with the journal the ORIGINAL run was given, not today's. The
        # journal grows between sessions, and a resumed date must see exactly what
        # it would have seen in an unbroken run --- otherwise the second half of a
        # walk is prompted differently from the first, and the trace stops
        # regenerating the prompts it was recorded against.
        if "journal" in prev_manifest:
            journal = prev_manifest["journal"]
        prev_cfg = prev_manifest.get("config")
        if prev_cfg is not None:
            conflicts = _config_conflicts(prev_cfg, config.to_dict())
            if conflicts:
                raise ValueError(
                    f"cannot resume run '{config.run_name}': config differs from "
                    f"the recorded run on {conflicts}. Resuming with a changed "
                    "config would desync the ledger/trace and corrupt replay --- "
                    "match the original config or use a new run_name."
                )
        existing = store.read_factors()
        # Identity admission precedes even orphan-trace cleanup or a manifest
        # rewrite. Reject drift with the original three artifact bytes intact.
        versions = {r.get("data_version") for r in existing}
        if existing and (None in versions or len(versions) != 1):
            raise ValueError("cannot resume: ledger data identity is missing or mixed")
        recorded_version = prev_manifest.get("data_version")
        if recorded_version is not None:
            versions.add(recorded_version)
        if versions and versions != {data_version}:
            raise ValueError("cannot resume: loaded data version differs from the recorded run; use a new run_name")
        skip_dates = {r["research_date"] for r in existing}
        # t_p is exempt from the conflict check so a run can EXTEND its horizon,
        # but it must not SHRINK: a smaller t_p would drop already-recorded dates
        # off the schedule while their ledger/trace rows stay on disk, desyncing
        # the manifest from them and breaking replay. Require every recorded date
        # to still be on the schedule.
        schedule = {d.date().isoformat() for d in _evolution_dates(config)}
        dropped = sorted(skip_dates - schedule)
        if dropped:
            raise ValueError(
                f"cannot resume run '{config.run_name}': the new schedule drops "
                f"already-recorded date(s) {dropped} (t_p={config.t_p} shrinks the "
                "horizon). t_p may only extend a run --- grow it back or use a new "
                "run_name."
            )
        prior_seed = [_win_summary(r) for r in existing if r["verdict"]["passed"]]
        # negative memory must survive a resume too, so continued dates still see
        # the earlier failures they would have seen in an unbroken run.
        fail_seed = [_fail_summary(r) for r in existing
                     if not r["verdict"]["passed"] and r["verdict"].get("label") != "NUMERICAL_REJECT"]
        # The trace is appended live per completion; a date killed mid-run (or one
        # that errored out before its ledger row was written) left orphan trace
        # entries after the last recorded date. Cut the trace back to exactly the
        # recorded dates' completions so it stays aligned with the ledger and the
        # run remains replayable. n_completions is authoritative; fall back to
        # max_iters for ledgers written before it was recorded.
        n_keep = sum(r.get("n_completions") or config.max_iters for r in existing)
        store.truncate_trace(n_keep)
        if log and skip_dates:
            log(f"resume: {len(skip_dates)} date(s) already done, continuing")
    else:
        store.reset()

    # Record every completion live so even a killed run leaves a replayable trace,
    # and meter the token usage on the way past. The meter is the OUTER wrapper so
    # that the single object handed to _evolve exposes both ``trace`` (delegated to
    # the recorder) and ``usage_totals``, which is how run_once attributes each
    # date's cost. A failed call raises through the meter uncounted --- it consumed
    # no tokens worth billing, and the recorder has already logged the error.
    provider = UsageMeter(RecordingProvider(inner, on_record=store.append_trace))

    manifest = {
        **(prev_manifest if resume else {}),
        "run_name": config.run_name,
        "config": config.to_dict(),
        "provider": inner.name,
        "model": config.model,
        "objective": config.objective,
        "pass_line": config.pass_line,
        "seed": config.seed,
        # the exact cross-run lessons this run was shown --- part of the prompt,
        # so replay needs it to regenerate byte-identical prompts.
        "journal": journal,
        "data_version": data_version,
        "n_factors": len(existing),
        "n_pass": sum(1 for r in existing if r["verdict"]["passed"]),
        "usage": _usage_total(existing),
    }
    # Persist the manifest up front so a run killed mid-loop still has its config
    # on disk (enough for replay + the resume guard above); the counts and
    # data_version below are filled in once the walk completes.
    store.write_manifest(manifest)

    # One pool for the WHOLE walk (a worker pays seconds of engine import and
    # numba compilation before its first backtest, so a pool per date would
    # spend more than it saves). ``None`` unless the config actually asks for
    # workers AND has the breadth to use them --- see parallel.pool_for.
    pool = pool_for(config, log=log)
    try:
        new_records = _evolve(config, provider, store=store, log=log,
                              prior_seed=prior_seed, fail_seed=fail_seed,
                              skip_dates=skip_dates, journal=journal, pool=pool, panel=panel)
    finally:
        if pool is not None:
            pool.close()
    records = existing + new_records

    manifest["n_factors"] = len(records)
    manifest["n_pass"] = sum(1 for r in records if r["verdict"]["passed"])
    # whole-run token cost, so a finished run can be priced without re-reading
    # the trace (``None`` when the provider reported no usage at all)
    manifest["usage"] = _usage_total(records)
    # Defense in depth: every stop receives the pinned panel, so no completed
    # record may acquire a different identity, even from a future code change.
    versions = {r["data_version"] for r in records}
    if len(versions) > 1:
        raise ValueError(
            f"run '{config.run_name}' mixes {len(versions)} data versions "
            f"{sorted(versions)}: a resume reloaded market data that differs from "
            "the recorded run, so replay could not reproduce it. Start a fresh "
            "run_name against the current data."
        )
    manifest["data_version"] = records[0]["data_version"] if records else None
    store.write_manifest(manifest)
    return {"records": records, "manifest": manifest, "run_dir": store.run_dir}


# --- replay: reproduce a recorded run from cached LLM outputs (no model call) --
REPLAY_CORE_KEYS = (
    "research_date", "formula", "expected_sign",
    "is_metrics", "oos_metrics", "verdict", "data_version",
)


def replay(manifest: dict, trace: list[dict], log=None) -> list[dict]:
    """Re-drive the recorded walk-forward from its LLM ``trace`` --- no model call.

    Deterministic *given the same data*: with the recorded seed and the same data
    version, replay reproduces the execution outputs (formula / metrics / verdict)
    bit-for-bit. It reloads the market data (offline + byte-stable for synthetic;
    a re-read that may hit the network/disk for yfinance/ctx), so if the upstream
    data has drifted since the run, results can differ --- a ``RuntimeWarning`` is
    emitted for any date whose reloaded data version no longer matches the one in
    the manifest. FactorEngine registration is skipped, so replay stays read-only.
    """
    replay_config = dict(manifest["config"])
    require_contract(replay_config.get("math_contract"))
    cfg = RunConfig.from_recorded_dict(replay_config)
    if manifest.get("math_policy_hash") != POLICY_HASH:
        raise ValueError("MATH_POLICY_MISMATCH: replay requires the current recorded policy hash; start a new run")
    # A clean run consumes exactly max_iters completions per date, so the trace
    # length is pinned. Check it up front: a truncated/padded trace (or one from a
    # different config) would otherwise misalign silently and replay the wrong
    # completions from some date onward.
    single_date = manifest.get("execution_mode") == "single-date"
    n_dates = 1 if single_date else len(_evolution_dates(cfg))
    base = n_dates * cfg.max_iters
    if len(trace) != base:
        raise ValueError(
            f"replay trace has {len(trace)} completion(s) but this config consumes "
            f"{base} ({n_dates} date(s) x {cfg.max_iters} iters): the trace is "
            "truncated, padded, or from a different config and cannot reproduce "
            "the run."
        )
    provider = ReplayProvider(trace)
    # Replay the journal the run was actually shown (recorded in the manifest),
    # NOT whatever the journal holds today --- it is part of the prompt, so
    # anything else regenerates a different one and trips the drift warning below.
    # sleep=_no_sleep: a recorded transient failure is re-raised instantly from
    # the trace, so there is nothing to back off from --- pausing here would only
    # make a replay of a flaky run slower than the run itself.
    pool = pool_for(cfg, log=log)
    try:
        if single_date:
            records = [run_once(cfg, provider=provider, store=None, log=log, register=False,
                                journal=manifest.get("journal"), sleep=_no_sleep, pool=pool)]
        else:
            records = _evolve(cfg, provider, store=None, log=log, register=False,
                              journal=manifest.get("journal"), sleep=_no_sleep, pool=pool)
    finally:
        if pool is not None:
            pool.close()

    # A faithful replay consumes EVERY recorded completion, in order: running past
    # the trace already raises ReplayDesyncError, and any completion left unread at
    # the end means the trace has trailing entries this config never asks for ---
    # padded or from a different config, so it cannot reproduce the run.
    if provider._i != len(trace):
        raise ValueError(
            f"replay consumed {provider._i} of {len(trace)} recorded completion(s): "
            f"{len(trace) - provider._i} trailing entry/entries were never "
            "reproduced, so the trace does not align with this config."
        )

    # A faithful replay regenerates byte-identical prompts, so a hash mismatch
    # means the trace was recorded under a different config/prompt version --- the
    # cached completions no longer correspond to what would be asked now. Surface
    # it (a warning, like data drift), without failing: the returned records are
    # still the recorded ones, only their provenance is suspect.
    if provider.prompt_mismatches:
        warnings.warn(
            f"replay prompt drift: {len(provider.prompt_mismatches)} call(s) "
            "regenerated a prompt whose hash != the recorded one (first at call "
            f"{provider.prompt_mismatches[0]}); the trace looks recorded under a "
            "different config/prompt version, so replayed results may not match "
            "the original run.",
            RuntimeWarning, stacklevel=2,
        )

    recorded_version = manifest.get("data_version")
    if recorded_version is not None:
        drifted = sorted({r["research_date"] for r in records
                          if r["data_version"] != recorded_version})
        if drifted:
            warnings.warn(
                f"replay data drift: {len(drifted)} date(s) reloaded a data "
                f"version != the recorded {recorded_version!r} "
                f"(first: {drifted[0]}); replayed results are NOT guaranteed to "
                "match the original run.",
                RuntimeWarning, stacklevel=2,
            )
    return records


def _replay_core(rec: dict) -> dict:
    """The provider-independent, deterministic slice of a record for comparison."""
    keys = REPLAY_CORE_KEYS + (("parameters", "math_admission", "is_math_admission") if "math_admission" in rec else ())
    return {k: rec.get(k) for k in keys}


def _deep_eq(a, b) -> bool:
    """Structural equality treating NaN == NaN (float metrics can be NaN, and
    ``float('nan') != float('nan')`` would flag every NaN metric as a mismatch)."""
    if isinstance(a, float) and isinstance(b, float):
        if a != a and b != b:            # both NaN
            return True
        return a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_deep_eq(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
    return a == b
