"""Ask the LLM to invent / refine a factor formula.

The model writes the maths itself; we only hand it the operator vocabulary, the
objective + Pass Line, and the feedback from the last backtest. All prompt text
lives in harness/prompts/*.json (see promptlib) so it can be reviewed by hand.

Output is strict JSON so parsing is deterministic. Every returned formula is
validated against the safe AST parser before it leaves this module; when
interpretability is required, the rationale/mechanism are enforced too.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pandas as pd

from . import objective as _obj
from . import promptlib
from .factor import operators
from .factor.errors import FactorSyntaxError
from .providers.base import complete_provider
from .factor.contract import CheckedFactor, compile_factor, operator_reference


_MIN_RATIONALE = 40      # chars, when interpretability is required
_MIN_MECHANISM = 40


class ProposalError(ValueError):
    """Raised when the model's output cannot be turned into a valid factor."""


def _strict_json(text: str) -> dict:
    def pairs(items):
        obj = {}
        for k, v in items:
            if k in obj:
                raise ProposalError(f"duplicate JSON key: {k}")
            obj[k] = v
        return obj

    def nonfinite(value):
        raise ProposalError(f"nonfinite JSON number: {value}")

    try:
        obj = json.loads(text, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (ValueError, TypeError) as exc:
        raise ProposalError(f"strict JSON required: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProposalError("strict JSON must be one object")
    return obj


@dataclass
class FactorProposal:
    formula: str
    rationale: str
    mechanism: str
    expected_sign: int
    raw_text: str
    parameters: dict = field(default_factory=dict)

    def checked(self, panel=None) -> CheckedFactor:
        return compile_factor(self.formula, self.parameters, panel=panel)


# --- prompt assembly -------------------------------------------------------
def _cmp_and_dir(config) -> tuple[str, str]:
    hib = _obj.higher_is_better(config.objective)
    return (">=" if hib else "<="), ("higher is better" if hib else "lower is better")


def _n_candidates(config, requested: int | None = None) -> int:
    if requested is not None:
        return max(1, int(requested or 1))
    return max(1, int(getattr(config, "candidates_per_round", 1) or 1))


def _proposal_schema(n_candidates: int) -> dict:
    item = {
        "type": "object",
        "properties": {
            "formula": {"type": "string"},
            "rationale": {"type": "string"},
            "mechanism": {"type": "string"},
            "expected_sign": {"type": "integer", "enum": [1, -1]},
        },
        "required": ["formula", "rationale", "mechanism", "expected_sign"],
        "additionalProperties": False,
    }
    item["properties"]["parameters"] = {
        "type": "object", "maxProperties": 2,
        "additionalProperties": {
            "type": "object", "properties": {
                "type": {"type": "string", "enum": ["Window", "Coefficient", "Exponent"]},
                "value": {"type": "number"}},
            "required": ["type", "value"], "additionalProperties": False}}
    item["required"].append("parameters")
    if n_candidates <= 1:
        return item
    return {
        "type": "object",
        "properties": {
            "candidates": {
                "type": "array",
                "items": item,
                "minItems": n_candidates,
                "maxItems": n_candidates,
            },
        },
        "required": ["candidates"],
        "additionalProperties": False,
    }


def system_prompt(config, n_candidates: int | None = None) -> str:
    fields = getattr(config, "_available_fields", operators.FIELDS - {"vwap"})
    cmp, direction = _cmp_and_dir(config)
    return promptlib.render(
        "system_propose", fields=", ".join(sorted(fields)), operators=operator_reference(),
        objective_label=_obj.label(config.objective), cmp=cmp, pass_line=f"{config.pass_line:g}",
        goodness_dir=direction,
        interpretability=promptlib.render("interpretability_block") if config.require_interpretability else "",
        output_format=_strict_output_format(_n_candidates(config, n_candidates)),
    )


def _strict_output_format(n):
    item = ('{"formula":"<expression with named IDs>","parameters":'
            '{"w":{"type":"Window","value":20}},"rationale":"<hypothesis>",'
            '"mechanism":"<hypothesized driver>","expected_sign":1}')
    return ("STRICT JSON only; keys and types exactly as shown; parameters may be empty.\n"
            + (item if n == 1 else f'{{"candidates":[{item}]}}\nExactly {n} objects required.'))


# Order in which the FULL in-sample metric set is surfaced to the model on each
# refine turn. Every metric is listed evenly in the block --- none is pre-labelled
# "the score" here --- so the model always sees the whole risk/return/activity
# picture. The optimisation objective IS named once in the surrounding refine
# prompt (user_refine.json) so the model knows what it is improving, but no
# per-metric value is pre-judged, so it still weighs the trade-offs (and does not
# game the objective with a degenerate book) rather than tunnelling blind.
_HISTORY_METRICS = ("sortino", "sharpe", "calmar", "cagr", "ann_return",
                    "ann_vol", "maxdd", "hit_rate", "avg_turnover", "avg_gross",
                    "n", "n_active")


def _fmt_metric(key: str, value) -> str:
    if value is None:
        return "n/a"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if v != v:                                  # NaN
        return "n/a"
    # `n` (bars in the window) and `n_active` (bars the book actually held
    # exposure) are counts: rendering them as "1250.0000" reads as a ratio.
    return str(int(v)) if key in ("n", "n_active") else f"{v:.4f}"


def _history_block(history: list[dict]) -> str:
    lines = []
    for h in history:
        m = h["is_metrics"]
        # every key in the metric dict, stable order first then any extras, so the
        # model sees the complete result --- no metric marked as the score here, no
        # prior verdict, just the numbers (the objective is named once in the
        # surrounding refine prompt, not in this block).
        keys = [k for k in _HISTORY_METRICS if k in m]
        keys += [k for k in m if k not in _HISTORY_METRICS]
        stats = "  ".join(f"{k}={_fmt_metric(k, m[k])}" for k in keys)
        block = f"  formula: {h['formula']}\n    IS: {stats}"
        # Tri-alignment warnings (D2 field-mention / D3 sign-vs-IC) so the next
        # refine turn sees, and can fix, where the story diverged from the maths.
        notes = h.get("alignment")
        if notes:
            block += "\n    alignment: " + "; ".join(notes)
        lines.append(block)
    return "\n".join(lines)


def _prior_block(prior_factors: list[dict], config) -> str:
    obj = config.objective
    lines = []
    for p in prior_factors:
        ov = p.get("oos_value")
        ov_s = f"{ov:.4f}" if isinstance(ov, (int, float)) and ov == ov else "n/a"
        lines.append(
            f"  [{p.get('research_date', '?')}] {_evidence_text(p)}  "
            f"(OOS {_objective_label(p.get('objective') or obj)}={ov_s})"
        )
    return "\n".join(lines)


def _prior_fail_block(prior_failures: list[dict], config) -> str:
    """Cross-date negative memory: factors that FAILED out-of-sample earlier."""
    obj = config.objective
    lines = []
    for p in prior_failures:
        ov = p.get("oos_value")
        ov_s = f"{ov:.4f}" if isinstance(ov, (int, float)) and ov == ov else "n/a"
        why = "; ".join(p.get("reasons") or []) or "failed the Pass Line"
        lines.append(
            f"  [{p.get('research_date', '?')}] {_evidence_text(p)}  "
            f"(OOS {_objective_label(p.get('objective') or obj)}={ov_s}; {why})"
        )
    return "\n".join(lines)


def journal_as_of(journal: dict | None, as_of) -> dict | None:
    """The cross-run journal reduced to what is KNOWABLE at research date ``as_of``.

    A journal lesson is an OOS grade, so it is exactly as much look-ahead as an
    earlier date's OOS score inside this walk (``runner._prior_oos_closed``) --- the
    only difference is which run computed it. Rendering an ungated lesson would
    hand the model, at simulated T_n, a number that can only be computed from bars
    after T_n. So a lesson is shown only once its own OOS window has closed:
    ``oos_end <= as_of``, where ``oos_end`` was derived in ``memory.lessons`` from
    the ORIGINATING run's ``oos_days`` (a different run may have held out a
    different horizon). A lesson with no usable ``oos_end`` is withheld --- fail
    safe, because "unknown" must never read as "already knowable".

    Filtering happens HERE, at render time, and never at snapshot time: the
    unfiltered snapshot is what the manifest records, and ``as_of`` is
    deterministic, so replay regenerates byte-identical prompts.
    """
    if not journal:
        return journal
    try:
        cut = pd.Timestamp(as_of)
    except (TypeError, ValueError):
        # an unparseable T_n cannot establish that anything has closed
        return {**journal, "wins": [], "losses": []}

    def _closed(lesson: dict) -> bool:
        end = lesson.get("oos_end")
        if not end:
            return False
        try:
            return pd.Timestamp(end) <= cut
        except (TypeError, ValueError):
            return False

    return {
        **journal,
        "wins": [w for w in (journal.get("wins") or []) if _closed(w)],
        "losses": [l for l in (journal.get("losses") or []) if _closed(l)],
    }


def _objective_label(name) -> str:
    name = str(name or "sortino")
    return _obj.label(name) if _obj.is_valid(name) else name


def _evidence_text(rec: dict, *, repeat: str = "") -> str:
    return (f"{rec.get('formula', '')}{repeat} "
            f"parameters={json.dumps(rec.get('parameters', {}), sort_keys=True)} "
            f"context={json.dumps(rec.get('experiment_context', {}), sort_keys=True)}")


def _journal_block(journal: dict) -> str:
    """Cross-RUN memory: the accumulated lessons of every previous run.

    Renders whatever it is given; the point-in-time gate has already been applied
    by :func:`journal_as_of`.
    """
    def _val(v) -> str:
        return f"{v:.4f}" if isinstance(v, (int, float)) and v == v else "n/a"

    def _times(n) -> str:
        return f" x{n}" if isinstance(n, int) and n > 1 else ""

    def _lbl(name) -> str:
        # the journal outlives the code: an entry recorded against an objective
        # the registry no longer knows must still render, not abort the prompt
        name = str(name or "sortino")
        return _obj.label(name) if _obj.is_valid(name) else name

    out = ""
    if journal.get("wins"):
        out += "\n\nThese CLEARED their Pass Line in an earlier run:\n" + "\n".join(
            f"  [{w.get('research_date', '?')}] {_evidence_text(w)}  "
            f"(OOS {_lbl(w.get('objective'))}="
            f"{_val(w.get('oos_value'))}{_times(w.get('n_seen'))})"
            for w in journal["wins"]
        )
    if journal.get("losses"):
        out += "\n\nThese FAILED in an earlier run --- treat the mechanism as suspect:\n" \
            + "\n".join(
                f"  {_evidence_text(l, repeat=_times(l.get('n_seen')))}  "
                f"(OOS {_lbl(l.get('objective'))}="
                f"{_val(l.get('oos_value'))}; {'; '.join(l.get('modes') or []) or 'failed the Pass Line'})"
                for l in journal["losses"]
            )
    return out


def _dead_ends_block(negatives: list[dict]) -> str:
    """Within-date negative memory: candidates rejected before they were scored."""
    lines = []
    for n in negatives:
        formula = n.get("formula")
        reason = n.get("reason", "rejected")
        if formula:
            bindings = (f"\n    parameters: {json.dumps(n['parameters'], sort_keys=True)}"
                        if "parameters" in n else "")
            lines.append(f"  formula: {formula}{bindings}\n    reason: {reason}")
        else:
            lines.append(f"  (unparseable candidate) reason: {reason}")
    return "\n".join(lines)


def build_messages(config, history: list[dict] | None,
                   prior_factors: list[dict] | None = None,
                   negatives: list[dict] | None = None,
                   prior_failures: list[dict] | None = None,
                   n_candidates: int = 1,
                   journal: dict | None = None) -> list[dict]:
    cmp, _ = _cmp_and_dir(config)
    n = _n_candidates(config, n_candidates)
    if history:
        content = promptlib.render(
            "user_refine",
            history_block=_history_block(history),
            objective_label=_obj.label(config.objective),
            cmp=cmp,
            pass_line=f"{config.pass_line:g}",
            output_format=_strict_output_format(n),
        )
    else:
        is_start, tn = config.is_window()
        content = promptlib.render(
            "user_initial",
            n_symbols=len(config.symbols),
            asset_class=config.asset_class,
            tn=str(tn.date()),
            is_start=str(is_start.date()),
            objective_label=_obj.label(config.objective),
            cmp=cmp,
            pass_line=f"{config.pass_line:g}",
            output_format=_strict_output_format(n),
        )
    # Cross-run feedback is included only once its explicit OOS close is knowable.
    known = journal_as_of(journal, config.research_date)
    if known and (known.get("wins") or known.get("losses")):
        content = content + "\n" + promptlib.render(
            "journal_block", n_runs=str(known.get("n_runs", "?")),
            journal_block=_journal_block(known),
        )
    # learning-by-doing: earlier research dates' PASS factors become context for
    # later ones during walk-forward evolution.
    if prior_factors:
        content = content + "\n" + promptlib.render(
            "prior_factors_block", prior_block=_prior_block(prior_factors, config)
        )
    # negative memory across dates: earlier dates' OOS FAILURES, so later dates do
    # not re-invent known-dead mechanisms (point-in-time gated by the caller).
    if prior_failures:
        content = content + "\n" + promptlib.render(
            "negative_factors_block",
            negative_block=_prior_fail_block(prior_failures, config),
        )
    # Rejected or uncomputable IS candidates become within-date dead ends.
    if negatives:
        content = content + "\n" + promptlib.render(
            "dead_ends_block", dead_ends_block=_dead_ends_block(negatives)
        )
    # Breadth uses an exact-count array inside the sole response object.
    if n > 1:
        content = content + "\n" + promptlib.render(
            "multi_candidate_block", n_candidates=str(n)
        )
    bound_history = [h for h in (history or []) if "parameters" in h]
    if bound_history:
        content += "\nExplicit parameter bindings for IS feedback:\n" + "\n".join(
            f"{h['formula']}: {json.dumps(h['parameters'], sort_keys=True)}" for h in bound_history)
    return [{"role": "user", "content": content}]


# --- parsing / validation --------------------------------------------------


def _extract_candidate_dicts(text: str, max_candidates: int = 1) -> list[dict]:
    """Accept exactly the requested number of candidates in one JSON object."""
    obj = _strict_json(text)
    if set(obj) != {"candidates"} or not isinstance(obj["candidates"], list) or len(obj["candidates"]) != max_candidates:
        raise ProposalError(f"exactly {max_candidates} candidates required")
    return obj["candidates"]


def _proposal_from_dict(obj: dict, raw_text: str,
                        require_interpretability: bool = False) -> FactorProposal:
    """Validate ONE decoded proposal object into a FactorProposal.

    Split out of :func:`parse_proposal` so a multi-candidate response (a JSON
    array) can validate each element with identical rules; ``raw_text`` is kept on
    the proposal for provenance (all candidates from one call share it)."""
    if not isinstance(obj, dict) or set(obj) != {"formula", "parameters", "rationale", "mechanism", "expected_sign"}:
        raise ProposalError("proposal keys must exactly match strict v2 schema")
    if any(not isinstance(obj[k], str) for k in ("formula", "rationale", "mechanism")):
        raise ProposalError("formula/rationale/mechanism must be strings")
    if type(obj["expected_sign"]) is not int:
        raise ProposalError("expected_sign must be integer +1 or -1")
    formula = str(obj["formula"]).strip()
    parameters = obj["parameters"]
    if isinstance(parameters, list):
        bindings = {}
        for entry in parameters:
            if not isinstance(entry, dict) or set(entry) != {"name", "type", "value"}:
                raise ProposalError("wire parameter entries require name/type/value")
            name = entry["name"]
            if not isinstance(name, str) or name in bindings:
                raise ProposalError("wire parameter names must be unique strings")
            bindings[name] = {"type": entry["type"], "value": entry["value"]}
        parameters = bindings
    try:
        compile_factor(formula, parameters)
    except FactorSyntaxError as exc:
        raise ProposalError(f"invalid factor formula: {exc}") from exc

    try:
        sign = int(obj["expected_sign"])
    except (TypeError, ValueError) as exc:
        raise ProposalError("expected_sign must be +1 or -1") from exc
    if sign not in (1, -1):
        raise ProposalError(f"expected_sign must be +1 or -1, got {sign}")

    rationale = str(obj["rationale"]).strip()
    mechanism = str(obj.get("mechanism", "")).strip()

    if require_interpretability:
        if len(rationale) < _MIN_RATIONALE:
            raise ProposalError(
                "interpretability required: 'rationale' is too thin "
                f"({len(rationale)} < {_MIN_RATIONALE} chars)"
            )
        if len(mechanism) < _MIN_MECHANISM:
            raise ProposalError(
                "interpretability required: 'mechanism' is missing or too thin "
                f"({len(mechanism)} < {_MIN_MECHANISM} chars)"
            )
        # The rationale (the edge) and the mechanism (why it exists) must be
        # two distinct thoughts, and neither may just parrot the formula --- a
        # model dodging the requirement tends to duplicate one field or paste
        # the expression back.
        norm = lambda s: " ".join(s.lower().split())
        if norm(rationale) == norm(mechanism):
            raise ProposalError(
                "interpretability required: 'rationale' and 'mechanism' are "
                "identical; give a distinct economic edge and its mechanism"
            )
        if norm(rationale) == norm(formula) or norm(mechanism) == norm(formula):
            raise ProposalError(
                "interpretability required: rationale/mechanism must explain the "
                "factor in words, not restate the formula"
            )

    return FactorProposal(
        formula=formula,
        rationale=rationale,
        mechanism=mechanism,
        expected_sign=sign,
        raw_text=raw_text,
        parameters=parameters,
    )


def parse_proposal(raw_text: str, require_interpretability: bool = False) -> FactorProposal:
    """Parse a SINGLE-object model response into a validated FactorProposal."""
    return _proposal_from_dict(
        _strict_json(raw_text), raw_text, require_interpretability
    )


def propose(provider, config, history: list[dict] | None = None,
            prior_factors: list[dict] | None = None,
            negatives: list[dict] | None = None,
            prior_failures: list[dict] | None = None,
            journal: dict | None = None) -> tuple[FactorProposal, object]:
    """Ask ``provider`` for a factor; return (proposal, raw ProviderResponse)."""
    messages = build_messages(config, history, prior_factors, negatives,
                              prior_failures, journal=journal)
    resp = complete_provider(
        provider, system_prompt(config, n_candidates=1), messages,
        model=config.model, seed=config.seed, max_tokens=config.max_tokens,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
        response_format={"type": "json_object"},
        structured_schema=_proposal_schema(1), structured_retry_count=2,
    )
    proposal = parse_proposal(resp.text, config.require_interpretability)
    return proposal, resp


def propose_many(provider, config, history: list[dict] | None = None,
                 prior_factors: list[dict] | None = None,
                 negatives: list[dict] | None = None,
                 prior_failures: list[dict] | None = None,
                 journal: dict | None = None
                 ) -> tuple[list[FactorProposal], list[str], object]:
    """One completion proposes exactly the requested breadth.

    K=1 uses the single-object schema. K>1 requires the exact-count container;
    individually invalid candidates yield parse errors without discarding valid
    siblings. An invalid container fails the entire round before scoring.
    """
    n = _n_candidates(config)
    messages = build_messages(config, history, prior_factors, negatives,
                              prior_failures, n_candidates=n, journal=journal)
    resp = complete_provider(
        provider, system_prompt(config, n_candidates=n), messages,
        model=config.model, seed=config.seed, max_tokens=config.max_tokens,
        temperature=config.temperature,
        reasoning_effort=config.reasoning_effort,
        response_format={"type": "json_object"},
        structured_schema=_proposal_schema(n), structured_retry_count=2,
    )
    if n == 1:
        # Single-candidate rounds use the same strict parser as propose().
        proposal = parse_proposal(resp.text, config.require_interpretability)
        return [proposal], [], resp
    dicts = _extract_candidate_dicts(resp.text, n)
    proposals: list[FactorProposal] = []
    errors: list[str] = []
    for obj in dicts:
        try:
            proposals.append(_proposal_from_dict(
                obj, resp.text, config.require_interpretability))
        except ProposalError as exc:
            errors.append(str(exc))
    return proposals, errors, resp
