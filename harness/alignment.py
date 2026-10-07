"""Optional deterministic annotations for admitted hypotheses.

D2 warns when an input field is unexplained by the hypothesis.
D3 warns when the declared sign contradicts causal in-sample rank IC.
Neither certifies an economic cause or changes mathematical admission.
The compiler owns the mandatory construction and dimensional checks."""
from __future__ import annotations

import ast
import re

import pandas as pd

from .factor.evaluate import evaluate
from .factor.operators import FIELDS as _PANEL_FIELDS


# --- D1: single-monomial structure (hard) ----------------------------------


def _fields_in(node: ast.AST) -> frozenset[str]:
    """The input fields referenced anywhere in ``node``'s subtree."""
    return frozenset(
        n.id for n in ast.walk(node)
        if isinstance(n, ast.Name) and n.id in _PANEL_FIELDS
    )


# --- D2: field <-> story mention (warn) -------------------------------------
# For each input field, the word STEMS that count as "mentioning" it. Matching is
# generous (prefix at a word boundary, so "move" matches "moves"/"movement") and
# the check is a non-fatal warning, so a missed mention is a soft miss, not an
# error --- the aim is only to catch a story that never once alludes to a field
# the maths depends on. Overlap between close/returns stems is intentional.
_FIELD_STEMS: dict[str, tuple[str, ...]] = {
    "open":    ("open",),
    "high":    ("high", "range", "intraday"),
    "low":     ("low", "range", "intraday"),
    "close":   ("clos", "price", "level", "move", "gain", "loss", "rally",
                "drop", "momentum", "trend", "winner", "loser"),
    "volume":  ("volume", "liquid", "turnover", "traded", "trade", "flow",
                "particip", "activ", "interest"),
    # (high+low+close)/3 --- an unweighted intraday average. Deliberately NO
    # volume vocabulary: this field has no volume term in it.
    "typical_price": ("typical price", "hlc", "midpoint", "mid-price",
                      "intraday average", "bar average", "fair value"),
    "vwap":    ("vwap", "volume-weight", "volume weight", "average price",
                "execution price"),
    "returns": ("return", "rever", "moment", "drift", "trend", "perform",
                "gain", "loss", "move", "winner", "loser", "volatil"),
}

# Phrases that claim the field carries a VOLUME term. Legitimate for a real
# source VWAP; a lie about the (high+low+close)/3 proxy, which is what the harness
# substitutes whenever the data source ships no VWAP of its own (see
# harness.data._panel_from_ohlcv). Kept narrow --- explicit volume-weighting or
# execution-benchmark claims only --- so an ordinary price story is not nagged.
_VOLUME_WEIGHT_CLAIMS: tuple[str, ...] = (
    "volume-weight", "volume weight", "weighted by volume", "volume weighted",
    "turnover-weight", "turnover weight", "dollar volume", "traded value",
    "execution price", "execution benchmark", "institutional flow",
)


def _mentions(stem: str, text: str) -> bool:
    if " " in stem or "-" in stem:            # multi-word phrase: plain substring
        return stem in text
    return re.search(r"\b" + re.escape(stem), text) is not None


def check_field_mentions(formula: str, rationale: str, mechanism: str,
                         panel=None) -> list[str]:
    """Warn for each formula field the rationale/mechanism never mentions.

    ``panel`` (optional) makes the ``vwap`` vocabulary honest. The panel's ``vwap``
    is a genuine volume-weighted price only when the data source supplied one
    (``panel.vwap_source == 'source'``); otherwise it is the (high+low+close)/3
    typical-price proxy, and a rationale that sells it as volume-weighted
    execution or institutional flow is describing an input that does not exist.
    That earns its own warning instead of being waved through by a stem match.
    """
    text = f"{rationale} {mechanism}".lower()
    proxy_vwap = getattr(panel, "vwap_source", "hlc3") != "source"
    out: list[str] = []
    field_names = (_fields_in(ast.parse(getattr(formula, "formula", formula), mode="eval").body))
    for fld in sorted(field_names):
        stems = _FIELD_STEMS.get(fld, (fld,))
        if fld == "vwap" and proxy_vwap:
            # the field is really typical price here, so judge the story against
            # typical-price vocabulary, not volume-weighted vocabulary
            stems = _FIELD_STEMS["typical_price"] + ("vwap",)
        if not any(_mentions(s, text) for s in stems):
            out.append(
                f"field '{fld}' drives the formula but is not mentioned or "
                "explained in the rationale/mechanism"
            )
    if "vwap" in field_names and proxy_vwap:
        claimed = [c for c in _VOLUME_WEIGHT_CLAIMS if c in text]
        if claimed:
            out.append(
                f"the story explains 'vwap' as a volume-weighted quantity "
                f"({', '.join(sorted(claimed))}), but this panel's vwap is the "
                "(high+low+close)/3 typical-price proxy --- it contains no volume "
                "term, so that mechanism is not in the data (supply a Vwap column "
                "via data_source='ctx', or tell a typical-price story)"
            )
    return out


# --- D3: expected_sign vs realised in-sample IC (warn) ----------------------
_IC_MIN_NAMES = 3        # need >=3 cross-sectional names to form a rank correlation
_IC_EPS = 0.02           # |IC| below this is "no reliable signal", not a contradiction


def in_sample_ic(formula: str, panel, config) -> float | None:
    """Mean cross-sectional Spearman IC of the factor vs the next-period return.

    Strictly point-in-time: a pair (signal at ``t``, forward return over
    ``t -> t+1``) is used only when BOTH ``t`` and its realised return bar ``t+1``
    fall on or before ``is_end`` (= T_n). So at the terminal IS date the pair whose
    forward return lands on the first post-T_n bar is dropped --- D3 never reads a
    return dated after the research date, keeping this diagnostic (which feeds back
    into the search) free of any look-ahead, exactly like the rest of the loop.
    The forward return is ``returns.shift(-1)``, matching the backtest's convention
    (a factor read at close ``t`` sizes the book for ``t+1``). Returns ``None`` when
    no IC can be formed (too few names, a constant/degenerate signal, or no dates).
    """
    panel = panel.slice(end=config.is_window()[1])
    signal = evaluate(formula, panel)
    if not isinstance(signal, pd.DataFrame):
        return None
    fwd = panel.fields["returns"].shift(-1)         # fwd[t] = return t -> t+1
    is_start, is_end = config.is_window()
    idx = signal.index
    # keep only dates whose forward-return bar (the NEXT bar) is still within IS,
    # so the terminal-date pair that would read the first post-T_n return is excluded.
    fwd_bar = idx.to_series().shift(-1)             # next-bar date for each t (NaT at end)
    in_is = (idx >= is_start) & (idx <= is_end)
    fwd_in_is = (fwd_bar <= is_end).to_numpy()      # NaT compares False -> last bar dropped
    dates = idx[in_is & fwd_in_is]
    ics: list[float] = []
    for d in dates:
        if d not in fwd.index:
            continue
        pair = pd.concat([signal.loc[d], fwd.loc[d]], axis=1, keys=["s", "r"]).dropna()
        if len(pair) < _IC_MIN_NAMES:
            continue
        if pair["s"].nunique() < 2 or pair["r"].nunique() < 2:
            continue                                # constant column -> undefined
        ic = pair["s"].corr(pair["r"], method="spearman")
        if ic == ic:                                # skip NaN
            ics.append(float(ic))
    if not ics:
        return None
    return sum(ics) / len(ics)


def check_sign(formula: str, panel, config, expected_sign: int) -> list[str]:
    """Warn if the realised in-sample IC sign contradicts ``expected_sign``."""
    ic = in_sample_ic(formula, panel, config)
    if ic is None or abs(ic) < _IC_EPS:
        return []                                   # can't tell / no reliable signal
    if (ic > 0) != (int(expected_sign) > 0):
        return [
            f"expected_sign={int(expected_sign):+d} but in-sample IC={ic:.4f}: the "
            "data's factor->forward-return direction contradicts the stated sign "
            "(possible post-hoc story)"
        ]
    return []


# --- orchestration ----------------------------------------------------------
def soft_warnings(proposal, panel, config) -> list[str]:
    """All non-fatal (warn) alignment findings for a computable candidate: D2 + D3."""
    factor = proposal.checked(panel)
    out = check_field_mentions(factor, proposal.rationale,
                               proposal.mechanism, panel)
    out += check_sign(factor, panel, config, proposal.expected_sign)
    return out
