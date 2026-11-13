"""agent-env's libfaketime on a real computer: the clock desktop test bundle's, the ubuntu-24.04 desktop with the
reference driver and agentenv's chromium, every process loading the library. Composed, it passes the conformance
probe at full. Deployed alone, with no gateway for its follower to follow, and its clock page written as the
follower's user, as the follower writes it: curl and
Python reach a live site, its certificate checked against real time by the TLS shim; a shell, Chrome's JavaScript and
the driver read universe time through 30 forward jumps of 8 h, a day or a week, a rate change and a freeze; Chrome,
Xvfb and the desktop's panel run on through them; and the driver's input reaches Chrome after every jump and, frozen,
is drawn. Routing a site, a fresh Chrome loads it over HTTPS with the clock six years back, six years on and past
2099, its certificate checked against the universe's time. Needs Docker and the local registry on localhost:5000."""

import asyncio
import platform
import random
import re
import time
from pathlib import PurePosixPath

import pytest
from agentenv_protocol.computer.clock import FOLLOWER
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.clock.bounds import LATE_S
from tst.integration.computer.conftest import UniverseClock, session_run
from tst.integration.computer.start_tabs import host_pages
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
#: A live site reached from the session, the clock months back, with no proxy: each exits 0 once its certificate
#: verifies.
REACH = {"curl": "curl -sS -o /dev/null --noproxy '*' --max-time 30 https://example.com/",
         "python": ("python3 -c \"import urllib.request; urllib.request.build_opener(urllib.request.ProxyHandler({}))"
                    ".open('https://example.com/', timeout=30)\"")}
LIBRARY = str(PurePosixPath(FOLLOWER).parent / "libfaketimeMT.so.1")
#: A fresh Chrome, as the session user, its own profile beside the desktop's, printing the DOM of the page it's given:
#: each run makes a TLS handshake of its own, which checks the site's certificate against the universe's time (Chrome's
#: TLS isn't OpenSSL, so the TLS shim doesn't reach it).
FRESH_CHROME = ('profile=$(mktemp -d) && /opt/agentenv/apps/chromium/chromium/chrome/chrome --headless=new '
                '--no-sandbox --user-data-dir="$profile" --dump-dom "$0"; rm -rf "$profile"')
ROUTED = "https://shop.test/landed"


def test_composed_with_the_library_in_every_process_the_computer_passes_the_probe_at_full(clock_desktop):
    results = clock_desktop.metadata["composed"]["container"]

    assert results[ARCH]["status"] == "ready" and results[ARCH]["probe"]["passed"], results[ARCH]
    assert [app.environment_name for app in clock_desktop.composed_apps()] == ["chromium", "curl"]


@pytest.mark.asyncio
async def test_the_desktop_follows_the_universe_through_jumps_a_rate_change_and_a_freeze(clock_desktop):
    rng = random.Random(5420)
    with namespace_routing():
        deployed = await Env.get(clock_desktop.id).deploy(
            env_params={"app_settings": {"chromium": {"start_urls": [PAGE]}}})
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            clock = UniverseClock(deployed)
            opened = await _chrome_shows(deployed, {"universe_ns": 0}, seconds=90)
            assert opened is not None, "Chrome's page never opened"
            pids = await _pids(deployed)
            loaded = await session_run(deployed, f"grep -c {LIBRARY} /proc/{pids[2]}/maps")
            clock.set(time.time() - 182 * DAY, 1)
            await asyncio.sleep(2)  # each program maps the page the next time it reads the clock, within a second
            reached = {client: await inside.run(deployed, ["sh", "-c", command], timeout_s=60)
                       for client, command in REACH.items()}
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
    # The TLS shim: a live site's certificate verifies against real time, the clock 182 days back.
    assert {client: ran["exit_code"] for client, ran in reached.items()} == {"curl": 0, "python": 0}, reached
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


@pytest.mark.asyncio
async def test_a_fresh_chrome_loads_a_routed_site_six_years_either_side_of_now_and_past_2099(clock_desktop):
    with host_pages() as port, namespace_routing():
        params = {"site_routing": True, "sites": [{"name": "shop", "url": f"http://127.0.0.1:{port}/website/shop/",
                                                   "domains": ["shop.test"]}]}
        deployed = await Env.get(clock_desktop.id).deploy(env_params=params)
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            clock, loaded = UniverseClock(deployed), {}
            for years in (-6, 6, 80):
                clock.set(time.time() + years * 365 * DAY, 1)
                await asyncio.sleep(2)  # each program maps the page the next time it reads the clock, within a second
                ran = await inside.run(deployed, ["sh", "-c", FRESH_CHROME, ROUTED], timeout_s=90)
                loaded[years] = {"universe": await clock.read(deployed),
                                 "landed": "<title>landed</title>" in ran["stdout"],
                                 "output": (ran["stdout"] + ran.get("stderr", ""))[-600:]}
        finally:
            report = await teardown_run(context)

    assert all(shown["universe"] for shown in loaded.values()), loaded
    assert {years: shown["landed"] for years, shown in loaded.items()} == {-6: True, 6: True, 80: True}, loaded
    assert not report.failed


async def _pids(deployed) -> list[str]:
    return (await session_run(deployed, PIDS)).split()


async def _title(deployed) -> tuple[int, str] | None:
    """Chrome's page's time (ms) and what was typed, from its window's title."""
    match = TITLE.search(await session_run(deployed, WINDOWS))
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
