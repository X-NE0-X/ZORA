"""Translate the local proposal contract to API-supported closed JSON schemas.

Parameter names are data on the wire, not dynamically generated object keys.
The proposer converts entries back to bindings and still enforces every local
count/type/AST constraint. Schema relaxation never relaxes mathematical admission.
"""
from __future__ import annotations

from copy import deepcopy


def api_schema(schema: dict, provider: str) -> tuple[dict, str]:
    """Return an independent schema in the provider's supported subset."""
    converted_parameters = False

    def visit(node: dict) -> dict:
        nonlocal converted_parameters
        out = deepcopy(node)
        if out.get("type") == "object":
            props = out.get("properties", {})
            if isinstance(out.get("additionalProperties"), dict):
                raise ValueError("API structured schemas require closed objects")
            required = set(out.get("required", []))
            result = {}
            for key, child in props.items():
                if key == "parameters" and isinstance(child.get("additionalProperties"), dict):
                    binding = child["additionalProperties"]
                    result[key] = {
                        "type": "array",
                        "description": "At most two unique named parameter bindings; [] when unused.",
                        "items": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, **deepcopy(binding["properties"])},
                            "required": ["name", *binding["required"]],
                            "additionalProperties": False,
                        },
                    }
                    converted_parameters = True
                else:
                    transformed = visit(child)
                    if provider == "openai" and key not in required:
                        transformed = {"anyOf": [transformed, {"type": "null"}]}
                    result[key] = transformed
            out["properties"] = result
            out["additionalProperties"] = False
            if provider == "openai":
                out["required"] = list(props)
            out.pop("maxProperties", None)
            out.pop("minProperties", None)
        if "items" in out:
            out["items"] = visit(out["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in out:
                out[key] = [visit(child) for child in out[key]]
        for key in ("$defs", "definitions"):
            if key in out:
                out[key] = {name: visit(child) for name, child in out[key].items()}
        if provider == "claude":
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                        "multipleOf", "minLength", "maxLength", "maxItems"):
                out.pop(key, None)
            if out.get("minItems", 0) not in (0, 1):
                out.pop("minItems")
        return out

    wire = visit(schema)
    hint = ("\nAPI wire format override: each proposal's parameters MUST be an array "
            'of {"name":"w","type":"Window","value":20} entries, or [] when unused. '
            "All other proposal keys and the exact requested candidate count stay the same."
            if converted_parameters else "")
    return wire, hint
