"""Native v2 proposal fixtures, with explicit named window bindings."""
import re


def window_bindings(formula):
    return {name: {"type": "Window", "value": int(name[1:])}
            for name in sorted(set(re.findall(r"\bw[0-9]+\b", formula)))}


def factor_object(formula, sign=1, rationale="r" * 50, mechanism="m" * 50):
    return {"formula": formula, "parameters": window_bindings(formula),
            "rationale": rationale, "mechanism": mechanism, "expected_sign": sign}
