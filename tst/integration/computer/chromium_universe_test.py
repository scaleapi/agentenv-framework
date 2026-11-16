"""agentenv's chromium on the ubuntu-24.04 desktop, on the universe clock: its boot browser, never launched again, goes
out through the computer's egress proxy by the base's policy and loads live sites it hasn't visited with the universe
182 days back, 730 days on and six years on, and right after a jump back; and X runs no screensaver. Needs Docker, the
local registry on localhost:5000, and the internet."""

import asyncio
import json
import re
import subprocess
import time

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
#: from now: the last right after the jump back.
STOPS = [(-182, "https://example.com/", "Example Domain"), (730, "https://example.org/", "Example Domain"),
         (6 * 365, "https://example.net/", "Example Domain"), (-182, "https://www.wikipedia.org/", "Wikipedia")]
#: How long a page may take to load before its tab counts as not loaded.
LOAD_S = 45
POLICY = "/etc/opt/chrome_for_testing/policies/managed/agentenv-egress.json"


async def _loaded(ok, url: str, title: str) -> str:
    """The title of the active tab once it's ``title`` after navigating to ``url``, else the tabs it last saw."""
    await ok("navigate", {"url": url})
    tabs, deadline = [], time.monotonic() + LOAD_S
    while time.monotonic() < deadline:
        tabs = (await ok("tabs/list", method="GET"))["tabs"]
        if (active := next((tab for tab in tabs if tab["active"]), None)) and active["title"] == title:
            return title
        await asyncio.sleep(0.5)
    return json.dumps(tabs)


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
            _, ok = _client(deployed, lambda op: f"{deployed.environment_url}/svc/chromium"
                                                 f"{protocol_v1.find_extension_method(card, BROWSER, op)['endpoint']}")
            booted = await _browser_process(deployed)
            clock, loaded = UniverseClock(deployed), []
            for days, url, title in STOPS:
                clock.set(time.time() + days * DAY, 1)
                await asyncio.sleep(2)  # each program maps the page the next time it reads the clock, within a second
                loaded.append(await _loaded(ok, url, title))
            still = await _browser_process(deployed)
            rows = _egress_rows(deployed)
            xset = (await inside.run(deployed, ["xset", "q"], timeout_s=30))["stdout"]
            policy = json.loads((await inside.run(deployed, ["cat", POLICY], timeout_s=30))["stdout"])
        finally:
            report = await teardown_run(context)

    assert loaded == [title for _, _, title in STOPS], loaded
    assert still["pid"] == booted["pid"]
    hosts = {url.split("/")[2] for _, url, _ in STOPS}
    intercepted = {row["host"] for row in rows if row.get("host") in hosts and row.get("mode") == "intercept"}
    assert intercepted == hosts, [row for row in rows if row.get("host") in hosts][-8:]
    assert re.search(r"Screen Saver:.*?timeout:\s+0\b", xset, re.S), xset
    assert "DPMS is Disabled" in xset or "does not have the DPMS Extension" in xset, xset
    assert policy["ProxySettings"]["ProxyServer"] == "http://127.0.0.1:3128"
    assert not report.failed
