"""Factor admission failures, separated from parsing and execution."""


class FactorSyntaxError(ValueError):
    """A formula violates the static mathematical contract."""


class FactorEvalError(ValueError):
    """An admitted formula cannot produce a valid signal on the given data."""
