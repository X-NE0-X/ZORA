"""Render the TUI to plain text at a given terminal size --- a design eyeball.

Not a test (the runner only collects ``test_*``). It exists because TUI layout
regressions are invisible to assertions: this dumps what the compositor actually
paints, so a wrapped MISSION card or a clipped hint shows up as text you can read
in a diff.

    python tests/tuishot.py 120 40                # the Data tab, wide
    python tests/tuishot.py 100 30 tab-windows    # a narrow, short terminal

Writes ``artifacts/_tuishot_<w>x<h>_<tab>.txt`` (UTF-8 --- the output is full of
box-drawing characters a cp1252 console cannot print).
"""
from __future__ import annotations

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harness.tui import HarnessTUI  # noqa: E402


async def shot(width: int, height: int, tab: str | None = None) -> list[str]:
    app = HarnessTUI()
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()
        if tab:
            app.query_one("TabbedContent").active = tab
            await pilot.pause()
        await pilot.pause()
        return [s.text.rstrip() for s in app.screen._compositor.render_strips()]


def main(argv: list[str]) -> int:
    width = int(argv[1]) if len(argv) > 1 else 120
    height = int(argv[2]) if len(argv) > 2 else 40
    tab = argv[3] if len(argv) > 3 else None
    lines = asyncio.run(shot(width, height, tab))
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = os.path.join(root, "artifacts",
                       f"_tuishot_{width}x{height}_{tab or 'default'}.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(f"===== {width}x{height}  tab={tab or 'default'} =====\n")
        fh.write("\n".join(lines) + "\n")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
