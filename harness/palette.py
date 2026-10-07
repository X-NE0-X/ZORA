"""One source of truth for colour: **black / dark grey / pale gold**.

The whole UI speaks one hue. Hierarchy comes from *luminance*, not from picking a
new colour every time something needs to stand out:

    INK   -> SLATE -> STEEL      three greys: page, panel, raised chrome
    BRONZE -> BRASS -> GOLD      one gold, three brightnesses:
                                 recessive rule -> active border -> title/mark

Only four colours carry meaning rather than structure, and they are pulled toward
the same earth family so a log line never looks like it came from another app:

    OK / WARN / BAD / NOTE       sage / amber / terracotta / warm grey

Both surfaces read from here --- :mod:`harness.tui` builds its Textual theme from
these constants, and :func:`harness.preflight.format_readiness` tags its markup
with them --- so the TUI and the CLI cannot drift apart.
"""
from __future__ import annotations

# --- structure ---------------------------------------------------------------
INK = "#0b0b0c"          # page: near-black
SLATE = "#15151a"        # panel: rail, cards, log
STEEL = "#212129"        # raised chrome: buttons, input wells
PARCHMENT = "#d9d5cc"    # body text: warm light grey
MUTED = "#948d81"        # labels, hints: the same grey, dimmed (NOT a neutral one
                         # --- Textual's stock $text-muted is a cold #9d9d9d that
                         # reads as a different palette next to the gold)

BRONZE = "#7d6a44"       # recessive rules, frames, the sparkline
BRASS = "#c2a463"        # active border, focus, the Run bar
GOLD = "#e3c88b"         # wordmark, border titles --- the brightest thing shown

# --- meaning -----------------------------------------------------------------
OK = "#93ad74"           # PASS, saved, found
WARN = "#d9a441"         # needs setup, nothing found, refused
BAD = "#c2705f"          # FAIL, raised
NOTE = "#8c8578"         # asides, journal lines
