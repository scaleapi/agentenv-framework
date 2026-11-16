"""agentenv's chromium as a multi env's website browser, on the universe clock: the browser computer follows the
multi env's gateway, so sync_env_clock arms it as it arms a desktop, and a page's Date.now() is the universe's, armed
two years back, after a jump of a week on, and after a reset; at that date a synthetic site loads by name, and a live
site through the computer's egress proxy, which hands the browser a certificate valid then. Deployed alone, the same
computer stays on real time and still loads a live site, the egress proxy relaying its TLS untouched. Needs Docker,
the local registry on localhost:5000, and the internet."""

import asyncio
import json
import subprocess
import time
import urllib.parse
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from agent_env.task_step.task_steps.sync_env_clock import SyncEnvClockTaskStep
from tst.integration.computer.browser_computer_test import _browser_op
from tst.integration.computer.browser_multi_env_test import MULTI_CHROMIUM, shop  # noqa: F401 - shop is a fixture
from tst.integration.computer.computer_clock_test import _gateway, _iso, _now, _universe_now, _write
from tst.integration.computer.conftest import computer_container
from tst.integration.computer.multi_env_computers_test import _image

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]

#: A page that names its own Date.now(), in milliseconds, in its title, five times a second.
CLOCK_PAGE = "data:text/html," + urllib.parse.quote(
    "<script>setInterval(() => document.title = 'now ' + Date.now(), 200)</script>")
#: How far a page's Date.now() may be from the gateway's universe: room for the page's title refresh and the reads.
MAX_SKEW_S = 3.0
LIVE = "https://github.com/"
SHOP = "https://shop.test/api/health"
LOAD_S = 90


@pytest.fixture(scope="module")
def live_github(shop):  # noqa: F811 - the fixture imported above
    """A live website env at github.com, under shop's local id."""
    with namespace_routing():
        return WebsiteEnv.put(id=f"{shop.id.rsplit('/', 1)[0]}/github", environment_name="github",
                              domains=["github.com"], live=True,
                              allow_hosts=["*.github.com", "*.githubassets.com", "*.githubusercontent.com"],
                              server_docker_image_artifact=_image("live_github_website",
                                                                  f"live-github-{uuid.uuid4().hex[:8]}"))


async def _page_now(browser: DeployedEnv) -> float:
    """What a fresh page in ``browser`` takes for now, in seconds, from its Date.now()."""
    opened = await _browser_op(browser, "tabs/open", {"url": CLOCK_PAGE})
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        tabs = (await _browser_op(browser, "tabs/list", {}))["tabs"]
        tab = next((tab for tab in tabs if tab["id"] == opened["tab"]), None)
        if tab and (tab.get("title") or "").startswith("now "):
            await _browser_op(browser, "tabs/close", {"tab": tab["id"]})
            return int(tab["title"].split()[1]) / 1000
        await asyncio.sleep(0.25)
    raise AssertionError(f"the clock page named no time within 30 s: {tabs}")


async def _skew(deployed, browser) -> float:
    """How far a page's Date.now() is from the gateway's universe, carried at its rate to when the page was read."""
    real_then, state = await _gateway(deployed)
    read = time.time()
    page = await _page_now(browser)
    rate = state["virtual_seconds_per_real_second"]
    return page - (_universe_now(state) + rate * (read - real_then))


async def _loads(browser: DeployedEnv, url: str, title: str) -> object:
    """True once a tab opened at ``url`` shows ``title`` in its own, else the tabs last listed."""
    await _browser_op(browser, "tabs/open", {"url": url})
    deadline, tabs = time.monotonic() + LOAD_S, []
    while time.monotonic() < deadline:
        tabs = (await _browser_op(browser, "tabs/list", {}))["tabs"]
        if any(tab["url"].startswith(url) and title in (tab.get("title") or "") for tab in tabs):
            return True
        await asyncio.sleep(1)
    return tabs


def _egress_rows(sandbox_id: str) -> list[dict]:
    log = subprocess.run(["docker", "exec", computer_container(sandbox_id), "cat", "/var/log/agentenv-root/egress.log"],
                         capture_output=True, text=True).stdout
    return [json.loads(line) for line in log.splitlines() if line.startswith("{")]


@pytest.mark.asyncio
async def test_a_multi_envs_browser_computer_shows_the_universe_through_a_jump_and_a_reset_and_loads_live_sites_then(
        browser_computer, shop, live_github):  # noqa: F811 - the fixture imported above
    with namespace_routing():
        office = MultiEnv.put(id=f"{shop.id.rsplit('/', 1)[0]}/office-clock", mcp_server_envs=[],
                              website_envs=[shop, live_github], browser_envs=[Env.get(browser_computer.id)],
                              name="office")
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": MULTI_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            kept = browser.metadata["computer"]["clock"]

            universe = datetime.now(timezone.utc) - timedelta(days=730)
            await SyncEnvClockTaskStep(id="clock", version=None, env_id=office.id,
                                       virtual_time=_iso(universe)).execute(context)
            synced = context.metadata["clock_configurations"][-1]["synced"]
            skews = {"armed two years back": await _skew(deployed, browser)}
            loaded = {"synthetic": await _loads(browser, SHOP, ""), "live": await _loads(browser, LIVE, "GitHub")}
            # Read before the reset, which boots the browser a fresh container.
            live = [row for row in _egress_rows(browser.sandbox_id) if row.get("host") == "github.com"]

            await _write(deployed, "advance", virtual_time=_iso(await _now(deployed) + timedelta(weeks=1)))
            skews["a week on"] = await _skew(deployed, browser)

            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            browser = DeployedEnv.from_dict(context.deployed_envs[0].child_envs["chromium"])
            skews["after a reset"] = await _skew(deployed, browser)
        finally:
            report = await teardown_run(context)

    assert kept["name"] == "chromium" and kept["watch_url"].endswith("/clock/watch"), kept
    # The browser computer's driver serves sync_time, and its follower follows the gateway under the computer's name.
    followers = [(entry["service"], entry["result"]["follower"]["name"], entry["result"]["follower"]["armed"])
                 for entry in synced if "follower" in (entry.get("result") or {})]
    assert followers == [("chromium-driver", "chromium", True)], synced
    assert all(abs(skew) < MAX_SKEW_S for skew in skews.values()), skews
    assert loaded == {"synthetic": True, "live": True}, loaded
    # Two years back, github.com's certificate isn't valid yet, so the egress proxy hands the browser one that is.
    assert any(row.get("mode") == "intercept" and row.get("decision") == "outside" for row in live), live[-6:]
    assert not report.failed


@pytest.mark.asyncio
async def test_alone_the_browser_computer_keeps_real_time_and_its_live_sites_pass_through_the_egress_proxy(
        browser_computer):
    with namespace_routing():
        deployed = await Env.get(browser_computer.id).deploy()
        context = TaskStepContext(deployed_envs=[deployed])
        try:
            read = time.time()
            page = await _page_now(deployed)
            loaded = await _loads(deployed, "https://example.com/", "Example Domain")
            rows = [row for row in _egress_rows(deployed.sandbox_id) if row.get("host") == "example.com"]
        finally:
            report = await teardown_run(context)

    assert abs(page - read) < MAX_SKEW_S, (page, read)
    assert loaded is True, loaded
    # Unarmed, the egress proxy relays the browser's TLS to the site untouched.
    assert rows and all((row["mode"], row["decision"]) == ("splice", "real") for row in rows), rows
    assert not report.failed
