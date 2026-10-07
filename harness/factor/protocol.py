"""The sole mathematical contract and entry-point validation."""
from functools import wraps
from inspect import signature

STRICT = "strict-math-v2"


def require_contract(version):
    if version != STRICT:
        raise ValueError(f"unsupported mathematical contract: {version!r}; required {STRICT}")


def configured_contract(function):
    sig = signature(function)

    @wraps(function)
    def wrapped(*args, **kwargs):
        config = sig.bind(*args, **kwargs).arguments["config"]
        require_contract(config.math_contract)
        return function(*args, **kwargs)
    return wrapped
