"""Versioned single-construction compiler. No model judgement or executable code."""
from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Callable

from . import operators
from .errors import FactorSyntaxError
from .protocol import STRICT

VERSION = STRICT
PRICE = (1.0, 0.0)
VOLUME = (0.0, 1.0)
UNITLESS = (0.0, 0.0)
FIELD_UNITS = {f: PRICE for f in operators.FIELDS}
FIELD_UNITS.update(volume=VOLUME, returns=UNITLESS)


class ContractError(FactorSyntaxError):
    """Deterministic rejection, including the rule and failing expression."""


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    slots: tuple[str, ...]
    semantics: str
    minimum_args: int | None = None
    kernel_version: str = "1"

    @property
    def kernel(self) -> Callable:
        return operators.FUNCS[self.name]

    @property
    def validity(self) -> str:
        if self.name in ("pow", "signed_pow"):
            return "source mask + declared real-power domain"
        if self.name in ("delay", "delta"):
            return "positive lag warmup + participating input masks"
        if self.name in ("ts_std", "ts_cov", "ts_corr"):
            return "full rolling window; ddof=1; corr requires both dispersions>0"
        if "Window" in self.slots:
            return "full rolling window; min_periods=Window"
        if self.name in ("zscore", "scale"):
            return "source mask + nonzero finite cross-sectional denominator"
        return "participating input masks"


_SPECS = [
    OperatorSpec("abs", ("Panel",), "elementwise absolute value; preserves unit"),
    OperatorSpec("sign", ("Panel",), "elementwise sign; dimensionless"),
    OperatorSpec("signed_log1p", ("Panel",), "sign(x)*ln(1+abs(x)); source-unit numerical values; dimensionless score"),
    OperatorSpec("pow", ("Panel", "Exponent"), "ordinary real power; negative base requires integer exponent; zero base requires p>0"),
    OperatorSpec("signed_pow", ("Panel", "Exponent"), "sign(x)*abs(x)**p; zero base requires p>0; NEVER hybrid power"),
    OperatorSpec("min", ("Panel", "Coefficient"), "elementwise scalar ceiling; threshold expressed in subject units"),
    OperatorSpec("max", ("Panel", "Coefficient"), "elementwise scalar floor; threshold expressed in subject units"),
    OperatorSpec("rank", ("Panel",), "cross-sectional percentile rank; average ties; axis=1; valid inputs only"),
    OperatorSpec("zscore", ("Panel",), "cross-sectional (x-mean)/std; ddof=1; zero std invalid"),
    OperatorSpec("demean", ("Panel",), "subtract cross-sectional mean; axis=1"),
    OperatorSpec("scale", ("Panel", "Coefficient"), "cross-sectional x/sum(abs(x))*k; default k=1 fixed; zero sum invalid", 1),
    *[OperatorSpec(n, ("Panel", "Window"), s) for n, s in {
        "delay": "positive past shift",
        "delta": "x-delay(x,w)",
        "ts_mean": "past/current rolling mean; min_periods=w",
        "ts_std": "past/current rolling std; min_periods=w; ddof=1",
        "ts_sum": "past/current rolling sum; min_periods=w",
        "ts_min": "past/current rolling minimum; min_periods=w",
        "ts_max": "past/current rolling maximum; min_periods=w",
        "ts_rank": "past/current rank of last value; ties count <=; min_periods=w",
        "decay_linear": "past/current weights 1..w normalized; min_periods=w",
        "product": "past/current rolling product; min_periods=w",
    }.items()],
    OperatorSpec("ts_corr", ("Observable", "Observable", "Window"), "rolling Pearson correlation; min_periods=w; zero variance invalid"),
    OperatorSpec("ts_cov", ("Observable", "Observable", "Window"), "rolling covariance; ddof=1; min_periods=w"),
]
SPECS = {s.name: s for s in _SPECS}
POLICY = {
    "version": VERSION, "parameters": 2, "formula_chars": 1000,
    "grammar": "observable/primitive-relation/unary/own-location-difference/own-location-or-scale-ratio",
    "units": FIELD_UNITS, "window": "JSON integer >=1, bound ID only",
    "coverage": "at least one valid cell in evaluation scope; reject globally constant signal; existing exposure floors remain separate",
    "numeric": "float64; checked axes; per-node nonfinite rejection; no epsilon/clip/fill; declared domain masks",
    "specs": [(s.name, s.slots, s.semantics, s.minimum_args, s.validity,
               s.kernel.__module__ + '.' + s.kernel.__name__, s.kernel_version) for s in _SPECS],
}
POLICY_HASH = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Parameter:
    name: str
    type: str
    value: int | float


@dataclass(frozen=True)
class Node:
    op: str
    children: tuple[Node, ...] = ()
    name: str = ""
    type: str = "Panel"
    unit: tuple[float, float] = UNITLESS
    subject: Node | None = None


@dataclass(frozen=True)
class CheckedFactor:
    formula: str
    parameters: tuple[Parameter, ...]
    node: Node
    policy_hash: str = POLICY_HASH

    def bindings(self) -> dict:
        return {p.name: {"type": p.type, "value": p.value} for p in self.parameters}

    def record(self) -> dict:
        # Alpha-renaming affects identity only, never execution order or the
        # original proposal. Different IDs remain distinct even at equal values.
        tree = ast.parse(self.formula, mode="eval")
        names = self.bindings()
        renames = {}
        for item in ast.walk(tree):
            if isinstance(item, ast.Name) and item.id in names and item.id not in renames:
                renames[item.id] = f"parameter{len(renames)+1}"
        for item in ast.walk(tree):
            if isinstance(item, ast.Name) and item.id in renames:
                item.id = renames[item.id]
        canonical = ast.unparse(tree)
        canonical_parameters = {renames[k]: v for k, v in names.items()}
        fingerprint = hashlib.sha256(json.dumps(
            [self.policy_hash, canonical, canonical_parameters], sort_keys=True
        ).encode("utf-8")).hexdigest()
        return {"version": VERSION, "policy_hash": self.policy_hash,
                "formula": self.formula, "parameters": self.bindings(),
                "normalized_formula": canonical, "normalized_parameters": canonical_parameters,
                "expression_hash": fingerprint,
                "parameter_count": len(self.parameters), "causality": "current/past operators only",
                "structure": "single-construction", "economic_mechanism": "hypothesis, not proved"}


def operator_reference() -> str:
    return "\n".join(f"{s.name}({', '.join(s.slots)}): {s.semantics}; mask={s.validity}; kernel v{s.kernel_version}" for s in _SPECS)


def _observable(n: Node) -> bool:
    return n.op == "field" or (n.op == "sub" and all(c.op == "field" for c in n.children))


def _locations(n: Node) -> list[Node]:
    result = [n]
    if n.op in ("ts_mean", "delay"):
        result.append(n.children[0])
    if n.subject is not None:
        result.append(n.subject)
    if n.op == "neg":
        result.extend(_locations(n.children[0]))
    return result


def compile_factor(formula: str | CheckedFactor, parameters: dict | None = None,
                   *, panel=None) -> CheckedFactor:
    if isinstance(formula, CheckedFactor):
        if parameters is not None or formula.policy_hash != POLICY_HASH:
            raise ContractError("POLICY_MISMATCH: checked factor or parameter override")
        checked = compile_factor(formula.formula, formula.bindings(), panel=panel)
        if checked != formula:
            raise ContractError("IR_MISMATCH: checked expression changed")
        return checked
    if not isinstance(formula, str) or not formula.strip() or len(formula) > 1000:
        raise ContractError("FORMULA: nonempty string of at most 1000 characters required")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict) or len(parameters) > 2:
        raise ContractError("PARAMETER_BUDGET: at most TWO parameter bindings required")
    bindings = {}
    for name, spec in parameters.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in operators.FIELDS or name in SPECS or name in operators.FUNCS:
            raise ContractError("PARAMETER_NAME: invalid or reserved binding")
        if not isinstance(spec, dict) or set(spec) != {"type", "value"}:
            raise ContractError(f"PARAMETER_SCHEMA: {name}")
        kind, value = spec["type"], spec["value"]
        if kind not in ("Window", "Coefficient", "Exponent") or type(value) not in (int, float):
            raise ContractError(f"PARAMETER_TYPE: {name}")
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ContractError(f"PARAMETER_FINITE: {name}")
        if kind == "Window" and (type(value) is not int or value < 1 or value > 2**63-1):
            raise ContractError(f"WINDOW: {name} requires a positive integer")
        bindings[name] = Parameter(name, kind, value)
    try:
        tree = ast.parse(formula, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError) as exc:
        raise ContractError("SYNTAX: invalid or excessive expression") from exc
    used = set()
    scalar_units = {}

    def fail(rule, a):
        raise ContractError(f"{rule}: {ast.unparse(a)}")

    def scalar_role(n, role, a, unit=UNITLESS):
        if n.op != "parameter" or n.type != role:
            fail(f"ARGUMENT_TYPE_{role}", a)
        if role == "Coefficient":
            previous = scalar_units.setdefault(n.name, unit)
            if previous != unit:
                fail("PARAMETER_UNIT_CONFLICT", a)

    def walk(a) -> Node:
        if isinstance(a, ast.Name):
            if a.id in bindings:
                used.add(a.id)
                return Node("parameter", name=a.id, type=bindings[a.id].type)
            if a.id in FIELD_UNITS:
                if panel is not None and (a.id not in panel.fields or (a.id == "vwap" and getattr(panel, "vwap_source", None) != "source")):
                    fail("FIELD_UNAVAILABLE", a)
                return Node("field", name=a.id, unit=FIELD_UNITS[a.id])
            fail("UNKNOWN_NAME", a)
        if isinstance(a, ast.UnaryOp) and isinstance(a.op, (ast.UAdd, ast.USub)):
            x = walk(a.operand)
            if x.type != "Panel":
                fail("UNARY_REQUIRES_PANEL", a)
            return x if isinstance(a.op, ast.UAdd) else Node("neg", (x,), unit=x.unit)
        if isinstance(a, ast.Call):
            if not isinstance(a.func, ast.Name) or a.func.id not in SPECS or a.keywords:
                fail("OPERATOR", a)
            spec = SPECS[a.func.id]
            if not (spec.minimum_args or len(spec.slots)) <= len(a.args) <= len(spec.slots):
                fail("ARITY", a)
            cs = tuple(walk(x) for x in a.args)
            for c, role in zip(cs, spec.slots):
                if role in ("Panel", "Observable"):
                    if c.type != "Panel" or (role == "Observable" and not _observable(c)):
                        fail(f"ARGUMENT_TYPE_{role}", a)
                else:
                    scalar_role(c, role, a, cs[0].unit if spec.name in ("min", "max") else UNITLESS)
            x = cs[0]
            unit = x.unit
            if spec.name in ("rank", "zscore", "scale", "sign", "signed_log1p", "ts_rank", "ts_corr"):
                unit = UNITLESS
            if spec.name == "ts_cov":
                unit = tuple(x+y for x, y in zip(cs[0].unit, cs[1].unit))
            if spec.name in ("pow", "signed_pow", "product"):
                p = bindings[cs[1].name].value
                try:
                    unit = tuple(u*p for u in x.unit)
                except OverflowError as exc:
                    raise ContractError("UNIT_RANGE: dimensional exponent overflow") from exc
                if not all(math.isfinite(u) for u in unit):
                    fail("UNIT_RANGE", a)
            return Node(spec.name, cs, unit=unit, subject=x if spec.name == "delta" else None)
        if isinstance(a, ast.BinOp):
            if isinstance(a.op, ast.Pow):
                return walk(ast.Call(ast.Name("pow", ast.Load()), [a.left, a.right], []))
            l, r = walk(a.left), walk(a.right)
            if isinstance(a.op, (ast.Mult, ast.Div)) and (l.type != "Panel" or r.type != "Panel"):
                if l.type == r.type or (l.type != "Panel" and r.type != "Panel"):
                    fail("SCALAR_OUTPUT", a)
                p = l if l.type != "Panel" else r
                scalar_role(p, "Coefficient", a)
                x = r if l.type != "Panel" else l
                op = "mul" if isinstance(a.op, ast.Mult) else "div"
                unit = x.unit if l.type == "Panel" or op == "mul" else tuple(-u for u in x.unit)
                return Node(op, (l, r), unit=unit)
            if l.type != "Panel" or r.type != "Panel":
                fail("SIGNAL_OFFSET_OR_SCALAR", a)
            if isinstance(a.op, ast.Sub):
                if l.unit != r.unit:
                    fail("UNIT_MISMATCH", a)
                if l.op == r.op == "field":
                    return Node("sub", (l, r), unit=l.unit)
                subject = None
                if r.op in ("ts_mean", "delay") and l == r.children[0]:
                    subject = l
                elif l.op == r.op == "ts_mean" and l.children[0] == r.children[0]:
                    subject = l.children[0]
                if subject is None:
                    fail("OWN_BASELINE_REQUIRED", a)
                return Node("sub", (l, r), unit=l.unit, subject=subject)
            if isinstance(a.op, ast.Div):
                if l.unit == r.unit:
                    if _observable(l) and _observable(r) and l.unit == PRICE:
                        return Node("div", (l, r), unit=UNITLESS)
                    owners = _locations(l)
                    if r in owners or (r.op in ("ts_mean", "delay", "ts_std") and r.children[0] in owners):
                        return Node("div", (l, r), unit=UNITLESS)
                fail("OWN_NORMALIZER_REQUIRED", a)
            fail("INDEPENDENT_SIGNAL_COMBINATION", a)
        fail("ANONYMOUS_LITERAL_OR_UNSAFE_SYNTAX", a)

    try:
        node = walk(tree.body)
    except RecursionError as exc:
        raise ContractError("DEPTH: excessive expression") from exc
    if node.type != "Panel":
        raise ContractError("SCALAR_OUTPUT: factor must be a Panel")
    if used != set(bindings):
        raise ContractError("UNUSED_PARAMETER: every binding must be referenced")
    return CheckedFactor(ast.unparse(tree), tuple(bindings[k] for k in sorted(bindings)), node)
