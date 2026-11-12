"""agent-env's libfaketime on a real computer: the clock desktop test bundle's, the ubuntu-24.04 desktop with the
reference driver and agentenv's chromium, every process loading the library. Composed, it passes the conformance probe
at full. Deployed, with its clock page written as root, as the computer's follower will write it: a shell, Chrome's
JavaScript and the driver read universe time through 30 forward jumps of 8 h, a day or a week, a rate change and a
freeze; Chrome, Xvfb and the desktop's panel run on through them; and the driver's input reaches Chrome after every jump
and, frozen, is drawn. Needs Docker and the local registry on localhost:5000."""

import ast
import asyncio
import platform
import random
import re
import subprocess
import time

import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.clock.bounds import LATE_S
from tst.integration.computer.website_browser_test import _window_center

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

ARCH = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}[platform.machine().lower()]
DAY = 86400
JUMPS = 30
#: Chrome's start page: every 200 ms its title is "T<Date.now()>|<what was typed into its input>", which a click
#: anywhere focuses. (Chrome trims a title's trailing whitespace, so the separator isn't a space.)
PAGE = ("data:text/html,<title>T</title><input id=i autofocus><script>document.onclick=()=>i.focus();"
        "setInterval(()=>document.title='T'+Date.now()+'|'+i.value,200)</script>")
TITLE = re.compile(r'"T(\d+)\|(.*) - Google Chrome for Testing"')
WINDOWS = ('for w in $(xprop -root _NET_CLIENT_LIST | sed "s/.*# //; s/,//g"); do '
           'xprop -id "$w" _NET_WM_NAME; done')
#: The processes that must run on through the clock's changes: Xvfb, the desktop's panel and Chrome's browser process.
PIDS = "pgrep -o -x Xvfb; pgrep -o -x xfce4-panel; pgrep -o -f -- --remote-debugging-pipe"
PAGE_WRITER = "/opt/agentenv/clock/clockpage.py"
LIBRARY = "/opt/agentenv/clock/libfaketimeMT.so.1"


def test_composed_with_the_library_in_every_process_the_computer_passes_the_probe_at_full(clock_desktop):
    results = clock_desktop.metadata["composed"]["container"]

    assert results[ARCH]["status"] == "ready" and results[ARCH]["probe"]["passed"], results[ARCH]
    assert [app.environment_name for app in clock_desktop.composed_apps()] == ["chromium", "universe-clock"]


@pytest.mark.asyncio
async def test_the_desktop_follows_the_universe_through_jumps_a_rate_change_and_a_freeze(clock_desktop):
    rng = random.Random(5420)
    with namespace_routing():
        deployed = await Env.get(clock_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            clock = _Clock(deployed)
            opened = await _chrome_shows(deployed, {"universe_ns": 0}, seconds=90)
            assert opened is not None, "Chrome's page never opened"
            pids = await _pids(deployed)
            loaded = await _run(deployed, f"grep -c {LIBRARY} /proc/{pids[2]}/maps")
            clock.set(time.time() - 182 * DAY, 1)
            await asyncio.sleep(2)  # each program maps the page the next time it reads the clock, within a second
            async with streamablehttp_client(f"{deployed.environment_url}/mcp") as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    rounds = []
                    for n in range(JUMPS):
                        jump = rng.choice([8 * 3600, DAY, 7 * DAY])
                        written = clock.advance(jump)
                        rounds.append({"jump": jump, "shell": await clock.read(deployed),
                                       "chrome_s": await _chrome_shows(deployed, written),
                                       "typed": await _type(deployed, session, f"j{n}"),
                                       "screenshot": not (await session.call_tool("computer_screenshot", {})).isError})
                    clock.rate(60)
                    fast = [await clock.read(deployed)]
                    await asyncio.sleep(3)
                    fast.append(await clock.read(deployed))
                    frozen = clock.rate(0)
                    held = [await clock.read(deployed)]
                    await asyncio.sleep(2)
                    held.append(await clock.read(deployed))
                    before = await _screenshot(session)
                    typed_frozen = await _type(deployed, session, "frozen")
                    after = await _screenshot(session)
                    chrome_frozen = await _chrome_time(deployed)
                    clock.rate(1)
                    resumed = await clock.read(deployed)
            pids_after = await _pids(deployed)
        finally:
            report = await teardown_run(context)

    assert int(loaded) >= 1, "Chrome's browser process didn't load the library"
    assert [r for r in rounds
            if not (r["shell"] and r["chrome_s"] is not None and r["typed"] and r["screenshot"])] == []
    assert max(r["chrome_s"] for r in rounds) <= LATE_S + 0.2  # the page's title updates every 200 ms
    assert False not in fast and fast[1] - fast[0] > 60 * 2.5, fast  # 3 s at ×60
    assert all(value is not False and abs(value - frozen["universe_ns"] / 1e9) < 1e-6 for value in held), (frozen, held)
    assert typed_frozen and before != after, "the driver's input wasn't drawn while the clock was frozen"
    assert chrome_frozen == frozen["universe_ns"] // 1_000_000, (frozen, chrome_frozen)
    assert resumed is not False and resumed > frozen["universe_ns"] / 1e9, (frozen, resumed)
    assert pids_after == pids, (pids, pids_after)
    assert not report.failed


class _Clock:
    """The computer's clock page, written as root as its follower writes it, and what the session reads of it."""

    def __init__(self, deployed):
        self.container = _container(deployed)
        self.page = None

    def set(self, universe: float, rate: float) -> dict:
        """The page written to ``universe`` at ``rate``: the write's own answer, its rate, and when it started."""
        started = time.time()
        out = subprocess.run(["docker", "exec", "-u", "0", self.container, "python3", PAGE_WRITER, "set",
                              repr(universe), repr(rate)], capture_output=True, text=True, check=True).stdout
        self.page = ast.literal_eval(out.strip())
        self.page.update(rate=rate, started=started)
        return self.page

    def advance(self, seconds: float) -> dict:
        return self.set(self.now() + seconds, self.page["rate"])

    def rate(self, rate: float) -> dict:
        return self.set(self.now(), rate)

    def now(self, real: float | None = None) -> float:
        """The universe at real time ``real`` (now by default), as the page has it."""
        real_ns = (time.time() if real is None else real) * 1e9
        return (self.page["universe_ns"] + max(0.0, real_ns - self.page["anchor_ns"]) * self.page["rate"]) / 1e9

    async def read(self, deployed) -> float | bool:
        """The session's ``date``, when it's the universe between the moments before and after it ran (a few ms either
        side for the two machines' reads of the clock); False otherwise."""
        start = time.time()
        value = float(await _run(deployed, "date +%s.%N"))
        end = time.time()
        rate = self.page["rate"]
        return value if self.now(start) - 0.01 * max(rate, 1) <= value <= self.now(end) + 0.01 * max(rate, 1) else False


def _container(deployed) -> str:
    """The deployed computer's container, the one of its sandbox's (each named for the sandbox) holding the page's
    writer."""
    listed = subprocess.run(["docker", "ps", "-q", "--filter", f"name={deployed.sandbox_id}"],
                            capture_output=True, text=True, check=True).stdout.split()
    for container in listed:
        if subprocess.run(["docker", "exec", container, "test", "-e", PAGE_WRITER]).returncode == 0:
            return container
    raise AssertionError(f"no container of sandbox {deployed.sandbox_id} has {PAGE_WRITER}: {listed}")


async def _run(deployed, command: str) -> str:
    ran = await inside.run(deployed, ["sh", "-c", command], timeout_s=30)
    assert ran["exit_code"] == 0, ran
    return ran["stdout"].strip()


async def _pids(deployed) -> list[str]:
    return (await _run(deployed, PIDS)).split()


async def _title(deployed) -> tuple[int, str] | None:
    """Chrome's page's time (ms) and what was typed, from its window's title."""
    match = TITLE.search(await _run(deployed, WINDOWS))
    return (int(match.group(1)), match.group(2)) if match else None


async def _chrome_time(deployed) -> int | None:
    shown = await _title(deployed)
    return shown[0] if shown else None


async def _chrome_shows(deployed, written: dict, seconds: float = 10) -> float | None:
    """How long after the write started Chrome's page shows a time at or past it, or None if it doesn't within
    ``seconds``."""
    start = written.get("started", time.time())
    while time.time() - start < seconds:
        shown = await _title(deployed)
        if shown and shown[0] >= written["universe_ns"] // 1_000_000:
            return time.time() - start
        await asyncio.sleep(0.1)
    return None


async def _type(deployed, session: ClientSession, text: str, tries: int = 3) -> bool:
    """Whether ``text``, typed through the driver after a screenshot (what a click's point is in) and a click into
    Chrome's window, reaches the page's input, typed again (up to ``tries`` times) when a click that only focused the
    window took the keystrokes."""
    for _ in range(tries):
        x, y = await _window_center(deployed)
        await _screenshot(session)
        clicked = await session.call_tool("computer_click", {"x": x, "y": y})
        typed = await session.call_tool("computer_type", {"text": text})
        if clicked.isError or typed.isError:
            return False
        for _ in range(30):
            shown = await _title(deployed)
            if shown and shown[1].endswith(text):
                return True
            await asyncio.sleep(0.1)
    return False


async def _screenshot(session: ClientSession) -> bytes:
    shot = await session.call_tool("computer_screenshot", {})
    assert not shot.isError, shot
    return next(c.data for c in shot.content if getattr(c, "type", "") == "image").encode()
