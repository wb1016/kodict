"""Plain-ANSI terminal dashboard with one blinker per job.

Each job owns a status line whose blinker glyph **toggles ● ↔ ○ every time a
new token arrives** from the model, dims to · while no token has arrived for
>1 s, and turns into ✓ / ✗ when the job finishes. All 16 lines are redrawn as
one cursor-up rewrite of the block, coalesced so token bursts don't flicker.

No dependencies beyond the standard library; disabled automatically when the
output is not a TTY (or explicitly via ``enabled=False``).
"""

from __future__ import annotations

import sys
import time
from typing import IO

_HIDE_CURSOR = "\x1b[?25l"
_SHOW_CURSOR = "\x1b[?25h"

_STATUS_STYLE = {
    "idle": "\x1b[2m",      # dim
    "running": "\x1b[36m",  # cyan
    "retry": "\x1b[33m",    # yellow
    "done": "\x1b[32m",     # green
    "failed": "\x1b[31m",   # red
}
_RESET = "\x1b[0m"


class Blinker:
    """Status + blink state for a single job."""

    def __init__(self, index: int, clock=time.monotonic):
        self.index = index
        self.clock = clock
        self.status = "idle"
        self.task = "-"
        self.tokens = 0
        self.started_at = clock()
        self.last_token_at = 0.0
        self.last_text = ""
        self.blink_on = False

    def token(self, text: str) -> None:
        """Called for every received token: toggle the blinker."""
        self.blink_on = not self.blink_on
        self.tokens += 1
        self.last_token_at = self.clock()
        self.last_text = text

    def glyph(self, now: float) -> str:
        if self.status == "done":
            return "\x1b[1;32m✓" + _RESET
        if self.status == "failed":
            return "\x1b[1;31m✗" + _RESET
        if self.status == "retry":
            return "\x1b[33m↻" + _RESET
        if self.status == "running" and now - self.last_token_at <= 1.0:
            # blink: on when a token just arrived, off otherwise
            return ("\x1b[1;32m●" if self.blink_on else "\x1b[32m○") + _RESET
        if self.status == "running":
            return "\x1b[2m·" + _RESET  # running but silent for >1s
        return "\x1b[2m○" + _RESET  # idle

    def line(self, now: float) -> str:
        style = _STATUS_STYLE.get(self.status, "")
        elapsed = now - self.started_at
        last = self.last_text.replace("\n", " ").strip()[:12]
        return (
            f" job-{self.index:02d} {style}{self.status:<7}{_RESET} "
            f"{self.task:<16} tok={self.tokens:<5} {elapsed:6.1f}s "
            f"{self.glyph(now)} {last}"
        )


class Dashboard:
    """Renders all job lines plus a log line as one redrawn block."""

    def __init__(self, jobs: int, out: IO[str] | None = None,
                 enabled: bool | None = None, clock=time.monotonic):
        self.out = out if out is not None else sys.stdout
        self.blinkers = [Blinker(i, clock) for i in range(jobs)]
        self.log_message = ""
        self.clock = clock
        self.enabled = self.out.isatty() if enabled is None else enabled
        self._dirty = True
        self._started = False
        self._last_render_at = 0.0
        self._last_second = -1

    # -- lifecycle -----------------------------------------------------
    def start(self) -> None:
        if not self.enabled:
            return
        self.out.write(_HIDE_CURSOR)
        self._render()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._render()
        self.out.write(_SHOW_CURSOR + "\n")
        self.out.flush()
        self._started = False

    # -- updates -------------------------------------------------------
    def log(self, message: str) -> None:
        self.log_message = message
        self._dirty = True
        if not self.enabled:
            print(message)

    def tick(self) -> None:
        """Redraw if something changed (cheap: call at ~10 Hz)."""
        if not self.enabled or not self._started:
            return
        now = self.clock()
        second = int(now)
        if self._dirty or second != self._last_second:
            self._render()

    # -- rendering -----------------------------------------------------
    def _render(self) -> None:
        now = self.clock()
        lines = [b.line(now) for b in self.blinkers]
        log_line = f"\x1b[2K > {self.log_message}" if self.log_message else "\x1b[2K"
        out = ""
        if self._started:
            out += f"\x1b[{len(lines) + 1}A"  # back up over the whole block
        for line in lines:
            out += "\x1b[2K" + line + "\n"
        out += log_line + "\n"
        self.out.write(out)
        self.out.flush()
        self._started = True
        self._dirty = False
        self._last_render_at = now
        self._last_second = int(now)
