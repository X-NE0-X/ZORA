"""Evaluate only compiler-admitted factors with causal numerical validation."""
import pandas as pd

from .errors import FactorEvalError


def evaluate(formula, panel, parameters=None, *, scope=None) -> pd.DataFrame:
    from .checked_evaluate import evaluate_checked
    return evaluate_checked(formula, panel, parameters, scope).values
