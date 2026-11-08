"""Tests for the ANSI blinker dashboard."""

import io

from kodict.blinker import Blinker, Dashboard


class Clock:
    def __init__(self, t: float = 100.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_start_hides_cursor_and_draws_16_lines():
    out = io.StringIO()
    d = Dashboard(16, out=out, enabled=True, clock=Clock())
    d.start()
    s = out.getvalue()
    assert s.startswith("\x1b[?25l")
    assert s.count(" job-") == 16
    assert " job-00 " in s and " job-15 " in s


def test_token_toggles_blinker():
    clk = Clock()
    b = Blinker(0, clock=clk)
    b.status = "running"
    b.token("사과")
    assert b.blink_on and b.tokens == 1 and b.last_text == "사과"
    b.token("나무")
    assert not b.blink_on and b.tokens == 2


def test_glyph_states():
    clk = Clock()
    b = Blinker(0, clock=clk)
    b.status = "running"
    b.token("x")
    assert "●" in b.glyph(clk())          # token just arrived: lit
    b.token("y")
    assert "○" in b.glyph(clk())          # toggled off, still recent
    clk.t += 2.0
    assert "·" in b.glyph(clk())          # running but silent >1s: dim
    b.status = "done"
    assert "✓" in b.glyph(clk())
    b.status = "failed"
    assert "✗" in b.glyph(clk())
    b.status = "retry"
    assert "↻" in b.glyph(clk())
    b.status = "idle"
    assert "○" in b.glyph(clk())


def test_block_redraw_moves_cursor_up_over_whole_block():
    out = io.StringIO()
    d = Dashboard(16, out=out, enabled=True, clock=Clock())
    d.start()
    out.seek(0)
    out.truncate(0)
    d._dirty = True
    d.tick()
    s = out.getvalue()
    assert "\x1b[17A" in s  # 16 job lines + 1 log line
    assert "\x1b[2K" in s


def test_tick_redraws_only_when_dirty_or_second_changes():
    clk = Clock()
    out = io.StringIO()
    d = Dashboard(2, out=out, enabled=True, clock=clk)
    d.start()
    out.seek(0)
    out.truncate(0)
    d.tick()  # same second, not dirty -> no redraw
    assert out.getvalue() == ""
    d.blinkers[0].status = "done"
    d._dirty = True
    d.tick()
    assert "✓" in out.getvalue()


def test_stop_restores_cursor():
    out = io.StringIO()
    d = Dashboard(4, out=out, enabled=True, clock=Clock())
    d.start()
    d.stop()
    assert out.getvalue().endswith("\x1b[?25h\n")


def test_disabled_writes_nothing():
    out = io.StringIO()
    d = Dashboard(16, out=out, enabled=False)
    d.start()
    d.tick()
    d.stop()
    assert out.getvalue() == ""


def test_log_line_rendered():
    out = io.StringIO()
    d = Dashboard(2, out=out, enabled=True, clock=Clock())
    d.start()
    d.log("job-00 → freq-000000")
    d.tick()
    assert "> job-00 → freq-000000" in out.getvalue()


def test_non_tty_defaults_to_disabled():
    out = io.StringIO()  # StringIO.isatty() is False
    d = Dashboard(2, out=out)
    assert d.enabled is False
