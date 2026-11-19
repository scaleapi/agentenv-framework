"""The ubuntu-24.04 base's clock-panel, run as it runs on a computer, against a clock page written as the follower
writes it and a stand-in for xfconf-query that records each call: when the clock is armed or jumps it has each of the
panel's clocks redraw at once and again at the universe's next minute, and at every universe minute while the clock
runs at another rate, at most once a second; a re-anchoring redraws nothing, nor does a clock that isn't armed; and a
clock given a zone of its own is left alone. The base installs it beside the page writer and runs it on real time."""
from __future__ import annotations

import configparser
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import agent_env

BASE = Path(agent_env.__file__).parent / "computers" / "artifacts" / "ubuntu-24.04"
CLOCK = Path(agent_env.__file__).parent / "utils" / "faketime_clock"
ZONE = "Europe/Paris"
#: The panel's plugins as xfconf-query lists them: two clocks, the second with a zone of its own.
LISTING = """/plugins/plugin-1                  applicationsmenu
/plugins/plugin-12                 clock
/plugins/plugin-13                 separator
/plugins/plugin-13/style           0
/plugins/plugin-20                 clock
/plugins/plugin-20/timezone        Asia/Tokyo
/plugins/plugin-22/base-directory  /home/user
"""
XFCONF = """#!{python}
import json, sys, time
with open({calls!r}, "a") as f:
    f.write(json.dumps({{"at": time.time(), "args": sys.argv[1:]}}) + "\\n")
if "-l" in sys.argv:
    sys.stdout.write({listing!r})
"""
REDRAW = [["-c", "xfce4-panel", "-p", "/plugins", "-l", "-v"],
          ["-c", "xfce4-panel", "-p", "/plugins/plugin-12/timezone", "-n", "-t", "string", "-s", ZONE],
          ["-c", "xfce4-panel", "-p", "/plugins/plugin-12/timezone", "-r"]]
MINUTE = 60.0


@pytest.fixture
def panel(tmp_path):
    """clock-panel started beside its own copy of the page writer, with the page, the zone and a recording
    xfconf-query under ``tmp_path``; yields a namespace with ``set(universe, rate)`` and ``redraws()``."""
    lib, bin_ = tmp_path / "lib", tmp_path / "bin"
    lib.mkdir()
    bin_.mkdir()
    shutil.copy(CLOCK / "clockpage.py", lib / "clockpage.py")
    shutil.copy(BASE / "clock-panel", lib / "clock-panel")
    calls = tmp_path / "calls.jsonl"
    xfconf = bin_ / "xfconf-query"
    xfconf.write_text(XFCONF.format(python=sys.executable, calls=str(calls), listing=LISTING))
    xfconf.chmod(0o755)
    (tmp_path / "timezone").write_text(ZONE + "\n")
    paths = {"AGENTENV_CLOCK_PAGE": tmp_path / "page", "AGENTENV_CLOCK_FILE": tmp_path / "clock",
             "AGENTENV_CLOCK_ZONE": tmp_path / "timezone"}
    env = {**os.environ, **{k: str(v) for k, v in paths.items()}, "PATH": f"{bin_}{os.pathsep}{os.environ['PATH']}"}
    spec = importlib.util.spec_from_file_location("clockpage_for_panel", lib / "clockpage.py")
    page = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(page)
    page.PAGE, page.FILE = str(paths["AGENTENV_CLOCK_PAGE"]), str(paths["AGENTENV_CLOCK_FILE"])
    process = subprocess.Popen([sys.executable, "-I", str(lib / "clock-panel")], env=env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    class Running:
        def set(self, universe: float, rate: float) -> float:
            """The page written as the follower writes it; when the write started."""
            started = time.time()
            page.write(1, universe, rate, time.time())
            return started

        def reanchor(self, by: float) -> None:
            with open(page.PAGE, "rb") as f:
                _, _, _, universe_ns, anchor_ns, rate, armed, *_ = page.LAYOUT.unpack(f.read(page.LAYOUT.size))
            now = time.time_ns()
            page.write(armed, (universe_ns + (now - anchor_ns) * rate) / 1e9 + by, rate, now / 1e9)

        def redraws(self) -> list[float]:
            """When each redraw began, each checked to be the one redraw the panel's clocks need."""
            try:
                recorded = [json.loads(line) for line in calls.read_text().splitlines()]
            except FileNotFoundError:
                return []
            starts = [n for n, call in enumerate(recorded) if "-l" in call["args"]]
            for n in starts[:-1] if starts and len(recorded) - starts[-1] < len(REDRAW) else starts:
                assert [call["args"] for call in recorded[n:n + len(REDRAW)]] == REDRAW, recorded
            return [recorded[n]["at"] for n in starts]

    yield Running()
    process.kill()
    output = process.communicate()[0]
    assert '"error"' not in output, output


def _until(check, within: float):
    deadline = time.monotonic() + within
    while time.monotonic() < deadline:
        if result := check():
            return result
        time.sleep(0.02)
    raise AssertionError(f"not within {within} s")


def _minute_ahead(seconds: float) -> float:
    """A universe ``seconds`` before a minute turns, a year on from now."""
    turn = (time.time() // MINUTE + 1) * MINUTE + 365 * 86400
    return turn - seconds


def test_an_unarmed_clock_redraws_nothing_and_arming_it_redraws_each_clock_without_a_zone_of_its_own_at_once(panel):
    time.sleep(1.0)
    assert panel.redraws() == []

    started = panel.set(_minute_ahead(30), 1)

    assert _until(panel.redraws, within=1.5)[0] - started < 1.0


def test_a_jump_redraws_at_once_and_again_as_the_universes_next_minute_turns_then_the_panels_own_timer_takes_over(
        panel):
    started = panel.set(_minute_ahead(2.5), 1)
    first, turned = _until(lambda: (shown := panel.redraws()) and len(shown) >= 2 and shown, within=5)[:2]
    time.sleep(1.5)

    assert first - started < 1.0
    assert 2.5 - 0.05 <= turned - started <= 2.5 + 0.75, turned - started
    assert len(panel.redraws()) == 2


def test_a_re_anchoring_by_less_than_a_second_redraws_nothing(panel):
    panel.set(_minute_ahead(30), 1)
    _until(panel.redraws, within=1.5)

    panel.reanchor(0.002)
    panel.reanchor(-0.4)
    time.sleep(1.0)

    assert len(panel.redraws()) == 1


def test_at_another_rate_it_redraws_at_every_universe_minute_at_most_once_a_second_and_frozen_not_at_all(panel):
    panel.set(_minute_ahead(30), 60)
    fast = _until(lambda: (shown := panel.redraws()) and len(shown) >= 4 and shown, within=6)
    panel.set(_minute_ahead(30), 0)
    frozen = _until(lambda: (shown := panel.redraws()) and len(shown) > len(fast) and shown, within=2.5)
    time.sleep(2.0)

    gaps = [b - a for a, b in zip(fast[1:], fast[2:])]
    assert all(0.95 <= gap <= 1.6 for gap in gaps), gaps
    assert len(panel.redraws()) == len(frozen)


def test_the_base_installs_it_beside_the_page_writer_and_its_session_runs_it_on_real_time():
    session = configparser.ConfigParser(interpolation=None)
    session.read_string((BASE / "supervisord.conf").read_text())
    program = session["program:clock-panel"]

    assert "COPY clock-panel /usr/local/lib/agentenv-clock/clock-panel\n" in (BASE / "Dockerfile").read_text()
    assert (program["command"], program["environment"]) == (
        "/usr/bin/python3 -I /usr/local/lib/agentenv-clock/clock-panel", 'AGENTENV_REAL_TIME="1"')
