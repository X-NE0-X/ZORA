"""Prompt library --- every LLM prompt lives in harness/prompts/*.json.

Prompts are kept OUT of the code so they can be reviewed and edited by hand.
Each file is one prompt:

    {
      "name": "...",
      "description": "what this prompt is for",
      "placeholders": ["fields", "operators", ...],
      "template": ["line 1 with $placeholder", "line 2", ...]
    }

Templates use ``$name`` placeholders (``string.Template``), NOT ``{}``, so the
literal ``{ }`` of the JSON schema inside a prompt needs no escaping. Unknown
placeholders are left untouched (safe_substitute), never raising.

Because the prompts are *invited* hand-edits, a missing or malformed asset is a
CONFIGURATION error, not a model failure --- so it gets its own exception type
(:class:`PromptAssetError`) and is checked up front by the readiness gate
(:func:`verify`, wired into :mod:`harness.preflight`). Without that, a typo'd
filename surfaces one iteration later as a nameless "provider error", is retried
every round, and the run ends with a misleading "no valid active factor
proposal" --- sending the user to debug their LLM instead of their typo.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from string import Template

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
_CACHE: dict[str, dict] = {}

# The prompt assets the code can ask for by name (proposer.py + memory.py). This
# is the manifest the readiness gate validates, so a file that is renamed, hand-
# broken or left out of the wheel is caught BEFORE a run instead of on the turn
# that happens to need it. tests/test_prompts.py asserts it matches the directory
# exactly, so it cannot drift from the files on disk.
REQUIRED_PROMPTS: tuple[str, ...] = (
    "system_propose",
    "system_synthesis",
    "user_initial",
    "user_refine",
    "user_synthesis",
    "interpretability_block",
    "journal_block",
    "prior_factors_block",
    "negative_factors_block",
    "dead_ends_block",
    "multi_candidate_block",
)


class PromptAssetError(RuntimeError):
    """A prompt asset is missing, unreadable or malformed.

    A distinct, clearly-named type (rather than the raw ``FileNotFoundError`` /
    ``JSONDecodeError``) so the run layers can tell "this installation's prompt
    directory is broken" apart from "the model call failed" --- the two need
    opposite responses: the first must stop the run immediately, the second is
    retried.
    """


def _available() -> list[str]:
    """Prompt names present on disk; ``[]`` if the directory is gone.

    Deliberately silent (unlike :func:`list_prompts`) because it is used to
    enrich an error message that is already being raised.
    """
    try:
        return sorted(f[:-5] for f in os.listdir(PROMPTS_DIR) if f.endswith(".json"))
    except OSError:
        return []


def load(name: str) -> dict:
    """Load and cache prompt ``name``; raise :class:`PromptAssetError` if broken.

    The error names the exact path searched and the prompts that WERE found, so
    a rename or a typo reads as itself ("you have user_intial.json, I wanted
    user_initial.json") rather than as a bare traceback.
    """
    if not isinstance(name, str) or not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*", name
    ):
        raise PromptAssetError(
            f"invalid prompt name {name!r}: expected one identifier, not a path"
        )
    if name not in _CACHE:
        path = os.path.join(PROMPTS_DIR, name + ".json")
        try:
            with open(path, encoding="utf-8") as fh:
                spec = json.load(fh)
        except FileNotFoundError as exc:
            found = ", ".join(_available()) or "(none)"
            raise PromptAssetError(
                f"prompt asset {name!r} not found at {path}; prompts present in "
                f"{PROMPTS_DIR}: {found}"
            ) from exc
        except OSError as exc:
            raise PromptAssetError(
                f"prompt asset {name!r} at {path} could not be read: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PromptAssetError(
                f"prompt asset {name!r} at {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(spec, dict) or not spec.get("template"):
            raise PromptAssetError(
                f"prompt asset {name!r} at {path} has no non-empty 'template' "
                "field (each file is one prompt: name/description/"
                "placeholders/template)"
            )
        _CACHE[name] = spec
    return _CACHE[name]


def _text(spec: dict) -> str:
    tmpl = spec["template"]
    if isinstance(tmpl, list):
        return "\n".join(tmpl)
    return str(tmpl)


def render(name: str, **kwargs) -> str:
    """Render prompt ``name`` with the given placeholder values."""
    return Template(_text(load(name))).safe_substitute(kwargs)


def list_prompts() -> list[str]:
    """Names of every prompt asset on disk.

    Raises :class:`PromptAssetError` when the directory itself is absent rather
    than returning ``[]``: an empty list is precisely how a packaging bug
    (prompts left out of the wheel) presents, and answering "there are no
    prompts" turns a one-line installation fault into a puzzling failure much
    further downstream.
    """
    if not os.path.isdir(PROMPTS_DIR):
        raise PromptAssetError(
            f"prompt directory not found at {PROMPTS_DIR}; the harness is "
            "installed without its prompts/*.json assets (packaging bug), or "
            "the directory was moved or renamed"
        )
    return _available()


def verify(names: Iterable[str] | None = None) -> list[str]:
    """One problem string per prompt asset that will not load (``[]`` = all good).

    The readiness gate's view of the prompt library: it reports *every* broken
    asset in one pass instead of stopping at the first, so a user fixes one list
    rather than re-running the gate per file. ``names`` defaults to
    :data:`REQUIRED_PROMPTS`.
    """
    try:
        list_prompts()
    except PromptAssetError as exc:
        return [str(exc)]        # the whole directory is gone: one problem, not 13
    problems: list[str] = []
    for name in (REQUIRED_PROMPTS if names is None else names):
        try:
            load(name)
        except PromptAssetError as exc:
            problems.append(str(exc))
    return problems
