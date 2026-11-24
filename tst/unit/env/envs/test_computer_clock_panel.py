"""The ubuntu-24.04 base's clock-panel, run as it runs on a computer, against a clock page written as the follower
writes it and a stand-in for xfconf-query that keeps the panel's config and records each call: when the clock is armed
or jumps it has each of the panel's clocks redraw at once and again at the universe's next minute, and at every universe
minute while the clock runs at another rate, at most once a second; a re-anchoring redraws nothing, nor does a clock
that isn't armed; a redraw leaves the config as it was, and one cut short leaves a clock the next redraws; and a clock
given another zone is left alone. The base installs it beside the page writer and runs it on real time."""
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
#: The panel's plugins: two clocks, the second with another zone.
PANEL = {"/plugins/plugin-1": "applicationsmenu", "/plugins/plugin-12": "clock", "/plugins/plugin-13": "separator",
         "/plugins/plugin-13/style": "0", "/plugins/plugin-20": "clock", "/plugins/plugin-20/timezone": "Asia/Tokyo",
         "/plugins/plugin-22/base-directory": "/home/user"}
#: xfconf-query on the panel's channel kept in a JSON file: it lists (``-l -v``), sets (``-s``) and resets (``-r``),
#: and fails the next reset while the failure marker is there, removing it. It records each call with when it began,
#: once the call is done, and replaces the file whole, so a test that has seen a call reads the config it left.
XFCONF = """#!@PYTHON@
import json, os, sys, time
args, at = sys.argv[1:], time.time()
try:
    with open("@STATE@") as f:
        state = json.load(f)
    path = args[args.index("-p") + 1]
    if "-l" in args:
        sys.stdout.write("".join(f"{key:<36}{value}\\n" for key, value in sorted(state.items())))
        sys.exit(0)
    if "-r" in args and os.path.exists("@FAIL@"):
        os.remove("@FAIL@")
        sys.exit("Failed to init libxfconf: the bus went away")
    if "-r" in args:
        state.pop(path, None)
    else:
        state[path] = args[args.index("-s") + 1]
    with open("@STATE@.tmp", "w") as f:
        json.dump(state, f)
    os.replace("@STATE@.tmp", "@STATE@")
finally:
    with open("@CALLS@", "a") as f:
        f.write(json.dumps({"at": at, "args": args}) + "\\n")
"""
TIMEZONE = "/plugins/plugin-12/timezone"
SET = ["-c", "xfce4-panel", "-p", TIMEZONE, "-n", "-t", "string", "-s", ZONE]
RESET = ["-c", "xfce4-panel", "-p", TIMEZONE, "-r"]
MINUTE = 60.0
#: How late a redraw's first recorded call may start under load: the fake xfconf-query records when its interpreter
#: is up, so one that starts late shortens the gap after it, though the panel spaced the redraws a second apart.
START_LAG_S = 0.4


@pytest.fixture
def panel(tmp_path):
    """clock-panel started beside its own copy of the page writer, with the page, the zone and a recording
    xfconf-query under ``tmp_path``; yields a namespace with ``set(universe, rate)`` and ``redraws()``."""
    lib, bin_ = tmp_path / "lib", tmp_path / "bin"
    lib.mkdir()
    bin_.mkdir()
    shutil.copy(CLOCK / "clockpage.py", lib / "clockpage.py")
    shutil.copy(BASE / "clock-panel", lib / "clock-panel")
    calls, state, fail = tmp_path / "calls.jsonl", tmp_path / "panel.json", tmp_path / "fail-reset"
    state.write_text(json.dumps(PANEL))
    xfconf = bin_ / "xfconf-query"
    xfconf.write_text(XFCONF.replace("@PYTHON@", sys.executable).replace("@CALLS@", str(calls))
                      .replace("@STATE@", str(state)).replace("@FAIL@", str(fail)))
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
        errors = 0

        def fail_next_reset(self) -> None:
            fail.touch()

        def config(self) -> dict:
            return json.loads(state.read_text())

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
            """When each redraw began."""
            return [call["at"] for call in self.calls() if "-l" in call["args"]]

        def writes(self) -> list[list[list[str]]]:
            """Each redraw's writes, in order."""
            redraws = []
            for call in self.calls():
                if "-l" in call["args"]:
                    redraws.append([])
                else:
                    redraws[-1].append(call["args"])
            return redraws

        def calls(self) -> list[dict]:
            try:
                return [json.loads(line) for line in calls.read_text().splitlines()]
            except FileNotFoundError:
                return []

    running = Running()
    yield running
    process.kill()
    output = process.communicate()[0]
    assert output.count('"event": "error"') == running.errors, output
    assert all(writes in ([SET, RESET], [RESET, SET], [SET]) for writes in running.writes()), running.writes()


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
    _until(lambda: len(panel.calls()) == 3, within=1.5)
    assert panel.writes() == [[SET, RESET]] and panel.config() == PANEL


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
    assert all(1 - START_LAG_S <= gap <= 1.6 for gap in gaps), gaps
    assert len(panel.redraws()) == len(frozen)


def test_the_base_installs_it_beside_the_page_writer_and_its_session_runs_it_on_real_time():
    session = configparser.ConfigParser(interpolation=None)
    session.read_string((BASE / "supervisord.conf").read_text())
    program = session["program:clock-panel"]

    assert "COPY clock-panel /usr/local/lib/agentenv-clock/clock-panel\n" in (BASE / "Dockerfile").read_text()
    assert (program["command"], program["environment"]) == (
        "/usr/bin/python3 -I /usr/local/lib/agentenv-clock/clock-panel", 'AGENTENV_REAL_TIME="1"')


def test_a_redraw_cut_short_between_its_changes_leaves_a_clock_on_the_local_zone_which_later_redraws_move_and_restore(
        panel):
    panel.fail_next_reset()
    panel.errors = 1
    panel.set(_minute_ahead(30), 1)
    _until(lambda: len(panel.calls()) == 3, within=1.5)
    left = panel.config()
    panel.set(_minute_ahead(20), 1)
    _until(lambda: len(panel.calls()) == 6, within=1.5)
    panel.set(_minute_ahead(10), 1)
    _until(lambda: len(panel.calls()) == 9, within=1.5)
    time.sleep(0.5)

    assert left == {**PANEL, TIMEZONE: ZONE}
    assert panel.writes() == [[SET, RESET], [RESET, SET], [RESET, SET]]
    assert panel.config() == {**PANEL, TIMEZONE: ZONE}
