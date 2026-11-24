"""agentenv's chromium on the ubuntu-24.04 desktop, on the universe clock: its boot browser, never launched again, goes
out through the computer's egress proxy by the base's policy and loads live sites it hasn't visited with the universe
400 days back, 730 days on and six years on, and right after a jump back; and X runs no screensaver. Needs Docker, the
local registry on localhost:5000, and the internet."""

import asyncio
import json
import re
import subprocess
import time
import warnings

import pytest
from agentenv_protocol import client as protocol_v1
from agentenv_protocol.types import WELL_KNOWN_PATH
from agentenv_protocol.website_browser import BROWSER

from agent_env.env import Env
from agent_env.env.envs.computer.utils import inside
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from tst.integration.computer.browser_state import _client
from tst.integration.computer.conftest import UniverseClock, computer_container
from tst.integration.computer.website_browser_test import _app, _browser_process

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

DAY = 86400
#: Live sites the browser hasn't visited, each with its page's title, opened in turn with the universe that many days
#: from now: the last right after the jump back. Each stop has a second site that stands in when the first doesn't load,
#: as a site refuses CI's runners for a while now and then (gnu.org did). None is behind a bot challenge, as
#: Cloudflare's "Just a moment..." meets CI's browsers at example.com now. Each date is outside any certificate valid
#: today, whose life is at most 398 days, so the egress proxy re-signs every site.
STOPS = [(-400, [("https://www.python.org/", "Welcome to Python.org"),
                 ("https://www.postgresql.org/", "PostgreSQL: The world's most advanced open source database")]),
         (730, [("https://www.openbsd.org/", "OpenBSD"), ("https://www.netbsd.org/", "The NetBSD Project")]),
         (6 * 365, [("https://www.kernel.org/", "The Linux Kernel Archives"),
                    ("https://www.ruby-lang.org/en/", "Ruby Programming Language")]),
         (-400, [("https://www.wikipedia.org/", "Wikipedia"), ("https://www.freebsd.org/", "The FreeBSD Project")])]
#: How long a page may take to load before its tab counts as not loaded.
LOAD_S = 45
POLICY = "/etc/opt/chrome_for_testing/policies/managed/agentenv-egress.json"


async def _first_loaded(call, ok, sites: list[tuple[str, str]]) -> tuple[str | None, list[str]]:
    """The first of ``sites`` whose title the active tab shows within ``LOAD_S`` of navigating to it, and why each one
    before it didn't; None when none did."""
    missed = []
    for url, title in sites:
        response = await call("navigate", {"url": url})
        if response.status_code != 200:
            missed.append(f"{url}: {response.status_code} {response.text}")
            continue
        tabs, deadline = [], time.monotonic() + LOAD_S
        while time.monotonic() < deadline:
            tabs = (await ok("tabs/list", method="GET"))["tabs"]
            if (active := next((tab for tab in tabs if tab["active"]), None)) and active["title"] == title:
                return url, missed
            await asyncio.sleep(0.5)
        missed.append(f"{url}: {json.dumps(tabs)}")
    return None, missed


def _egress_rows(deployed) -> list[dict]:
    log = subprocess.run(["docker", "exec", computer_container(deployed.sandbox_id), "cat",
                          "/var/log/agentenv-root/egress.log"], capture_output=True, text=True).stdout
    return [json.loads(line) for line in log.splitlines() if line.startswith("{")]


@pytest.mark.asyncio
async def test_its_boot_browser_loads_live_sites_at_any_universe_date_and_x_never_blanks(
        chromium_desktop):
    with namespace_routing():
        deployed = await Env.get(chromium_desktop.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            card = await _app(deployed, WELL_KNOWN_PATH, "chromium")
            call, ok = _client(deployed, lambda op: f"{deployed.environment_url}/svc/chromium"
                                                 f"{protocol_v1.find_extension_method(card, BROWSER, op)['endpoint']}")
            booted = await _browser_process(deployed)
            clock, loaded = UniverseClock(deployed), []
            for days, sites in STOPS:
                clock.set(time.time() + days * DAY, 1)
                await asyncio.sleep(2)  # each program maps the page the next time it reads the clock, within a second
                loaded.append(await _first_loaded(call, ok, sites))
            still = await _browser_process(deployed)
            rows = _egress_rows(deployed)
            xset = (await inside.run(deployed, ["xset", "q"], timeout_s=30))["stdout"]
            policy = json.loads((await inside.run(deployed, ["cat", POLICY], timeout_s=30))["stdout"])
        finally:
            report = await teardown_run(context)

    assert all(url for url, _ in loaded), loaded
    if stood_in := [missed for _, missed in loaded if missed]:
        warnings.warn(f"a stand-in site loaded: {stood_in}")
    assert still["pid"] == booted["pid"]
    hosts = {url.split("/")[2] for url, _ in loaded}
    intercepted = {row["host"] for row in rows if row.get("host") in hosts and row.get("mode") == "intercept"}
    assert intercepted == hosts, [row for row in rows if row.get("host") in hosts][-8:]
    assert re.search(r"Screen Saver:.*?timeout:\s+0\b", xset, re.S), xset
    assert "DPMS is Disabled" in xset or "does not have the DPMS Extension" in xset, xset
    assert policy["ProxySettings"]["ProxyServer"] == "http://127.0.0.1:3128"
    assert not report.failed
