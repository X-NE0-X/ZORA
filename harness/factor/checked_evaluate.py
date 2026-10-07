"""Numerical evaluation with per-node, causal validity masks."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd

from . import operators as ops
from .contract import CheckedFactor, compile_factor, SPECS
from .errors import FactorEvalError


@dataclass(frozen=True)
class ResearchScope:
    start: object = None
    end: object = None
    phase: str = "direct"


@dataclass
class ValidatedSignal:
    values: pd.DataFrame
    admission: dict


def signal_hash(signal):
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(signal, index=True).to_numpy().tobytes())
    h.update(repr(tuple(signal.columns)).encode("utf-8"))
    return h.hexdigest()


def require_validated_signal(signal):
    from .contract import POLICY_HASH
    record = signal.attrs.get("zora_admission", {})
    if record.get("policy_hash") != POLICY_HASH or record.get("numerical_status") != "validated" or record.get("signal_hash") != signal_hash(signal):
        raise FactorEvalError("ADMISSION_REQUIRED: signal is unvalidated or changed")


def evaluate_checked(factor: str | CheckedFactor, panel, parameters=None,
                     scope: ResearchScope | None = None) -> ValidatedSignal:
    scope = scope or ResearchScope()
    fields = {k: v.loc[:scope.end] for k, v in panel.fields.items()}
    checked = compile_factor(factor, parameters, panel=panel)
    reference = fields["close"]
    if not reference.index.is_monotonic_increasing or reference.index.has_duplicates or reference.columns.has_duplicates:
        raise FactorEvalError("PANEL_AXES: dates must be ordered and axes unique")
    values = {p.name: p.value for p in checked.parameters}
    invalid = {}
    nodes = []

    def panel_mask(x):
        return x.notna()

    def rolling_ready(x, w):
        return x.notna().rolling(w, min_periods=w).sum().eq(w)

    def finish(out, valid, node, reason="propagated_input_mask"):
        if not isinstance(out, pd.DataFrame) or not out.index.equals(reference.index) or not out.columns.equals(reference.columns):
            raise FactorEvalError(f"PANEL_AXES: {node.op}")
        array = out.to_numpy(dtype=float)
        if np.isinf(array).any() or (valid.to_numpy() & ~np.isfinite(array)).any():
            raise FactorEvalError(f"NUMERICAL_NONFINITE: {node.op}; unexpected overflow/NaN")
        out = out.where(valid)
        invalid[node.op] = invalid.get(node.op, 0) + int((~valid).to_numpy().sum())
        nodes.append({"node": len(nodes), "operator": node.op,
                      "invalid_cells": int((~valid).to_numpy().sum()), "mask_rule": reason})
        return out

    def walk(n):
        if n.op == "parameter":
            return values[n.name]
        if n.op == "field":
            x = fields[n.name].astype(float)
            return finish(x, panel_mask(x), n, "source_missing")
        cs = [walk(c) for c in n.children]
        op = n.op
        x = cs[0]
        if op in ("mul", "div", "sub"):
            y = cs[1]
            panels = [c for c in cs if isinstance(c, pd.DataFrame)]
            valid = panels[0].notna()
            if len(panels) == 2:
                valid &= panels[1].notna()
            if op == "div":
                valid &= y.ne(0) if isinstance(y, pd.DataFrame) else y != 0
                out = ops.div(x, y)
            else:
                out = x*y if op == "mul" else x-y
            return finish(out, valid, n, "input_mask + zero_denominator" if op == "div" else "input_masks")
        valid = x.notna()
        if op == "neg":
            out = -x
        elif op in ("pow", "signed_pow"):
            p = cs[1]
            if p <= 0:
                valid &= x.ne(0)
            if op == "pow" and not float(p).is_integer():
                valid &= x.ge(0)
            base = x.where(valid)
            out = SPECS[op].kernel(base, p)
        elif op in ("ts_corr", "ts_cov"):
            y, w = cs[1:]
            valid = rolling_ready(x, w) & rolling_ready(y, w)
            if w <= 1:
                valid &= False
            if op == "ts_corr":
                sx, sy = x.rolling(w, min_periods=w).std(), y.rolling(w, min_periods=w).std()
                for sd in (sx, sy):
                    if np.isinf(sd.to_numpy()).any() or (valid & sd.isna()).to_numpy().any():
                        raise FactorEvalError("NUMERICAL_NONFINITE: correlation dispersion")
                valid &= sx.gt(0) & sy.gt(0)
                out = SPECS[op].kernel(x, y, w)
            else:
                out = SPECS[op].kernel(x, y, w)
            # Invalid-domain cells are specified before this kernel, never
            # inferred from an arbitrary nonfinite output after execution.
            out = out.where(valid)
        elif op in ("delay", "delta"):
            w = cs[1]
            past = x.shift(w)
            valid = past.notna() if op == "delay" else valid & past.notna()
            out = SPECS[op].kernel(x, w)
        elif op.startswith("ts_") or op in ("product", "decay_linear"):
            w = cs[1]
            valid = rolling_ready(x, w)
            if op == "ts_std" and w <= 1:
                valid &= False
            out = SPECS[op].kernel(x, w)
        elif op == "zscore":
            sd = x.std(axis=1, ddof=1)
            if np.isinf(sd.to_numpy()).any() or (x.notna().sum(axis=1).ge(2) & sd.isna()).any():
                raise FactorEvalError("NUMERICAL_NONFINITE: cross-sectional dispersion")
            valid = valid.mul(sd.gt(0), axis=0)
            out = SPECS[op].kernel(x)
        elif op == "scale":
            total = x.abs().sum(axis=1)
            if np.isinf(total.to_numpy()).any():
                raise FactorEvalError("NUMERICAL_NONFINITE: scale denominator")
            valid = valid.mul(total.gt(0), axis=0)
            out = SPECS[op].kernel(x, cs[1] if len(cs) == 2 else 1.0)
        elif op == "signed_log1p":
            out = SPECS[op].kernel(x)
        else:
            out = SPECS[op].kernel(*cs)
        return finish(out, valid, n, SPECS[op].validity if op in SPECS else "input_mask")

    try:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            signal = walk(checked.node)
    except FactorEvalError:
        raise
    except Exception as exc:
        raise FactorEvalError(f"KERNEL: {checked.formula}: {exc}") from exc
    in_scope = signal.loc[scope.start:scope.end]
    count = int(in_scope.notna().to_numpy().sum())
    if count == 0:
        raise FactorEvalError("VALID_COVERAGE: no valid signal cells in research scope")
    finite = in_scope.to_numpy()[in_scope.notna().to_numpy()]
    if np.all(finite == finite[0]):
        raise FactorEvalError("DEGENERATE_SIGNAL: all valid cells have the same value")
    admission = {**checked.record(), "data_source": getattr(panel, "source", "unknown"),
                 "vwap_source": getattr(panel, "vwap_source", None),
                 "scope": {"start": str(scope.start), "end": str(scope.end), "phase": scope.phase},
                 "valid_cells": count, "invalid_node_cells": invalid,
                 "node_masks": nodes, "data_version": getattr(panel, "version", None),
                 "numerical_status": "validated"}
    admission["signal_hash"] = signal_hash(signal)
    signal.attrs["zora_admission"] = admission
    return ValidatedSignal(signal, admission)
