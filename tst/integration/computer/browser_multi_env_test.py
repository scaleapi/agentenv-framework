"""agentenv's chromium as a multi env's website browser, ``browser_envs = [chromium]``: it deploys on a sandbox of
its own as one more computer child beside a synthetic site, with no playwright-mcp, and is reached through the multi
env's gateway. Its start tab loads the site by name through the gateway, an agent drives it through the gateway's /mcp
under ``chromium_``, its CDP address asked through ``/svc/chromium/`` names the gateway and reaches the site by name, a
later process reattaches it from the instance store, a universe loaded into the multi env sets its cookies, and
reset_env brings it back after its container is killed and after the gateway restarts; two leases are each signed in
to the site as a user of their own and open it by name, the default context having none of their cookies; a multi env
naming none takes the run's browser, chromium too, the same way, and ``browser_envs = []`` deploys none; and
browser_connect_cdp hands an agent an address to it at the multi env's gateway, whose relay takes the agent's page to
the site by name, with the multi env's tools hidden from the agent's role. A site chromium reaches by name, whether
its env stores the name or its card declares it, takes that name as its Host, so its redirects land on it and an agent
is given the site at it, while a client naming any other origin, or none, leaves the site its internal Host, through a
gateway restart too.
Needs Docker and the local registry on localhost:5000, and Node 20 or later and npm for the CDP clients."""

import asyncio
import itertools
import json
import subprocess
import time
import urllib.parse
import uuid

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from websockets.asyncio.client import connect

from agentenv_protocol import client as protocol_v1
from agentenv_protocol.website_browser import BROWSER

from agent_env.artifact import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.config import get_config
from agent_env.config.runtime import WEBSITE_BROWSER_OVERRIDE, run_env_ids
from agent_env.env import Env
from agent_env.env.env import DeployedEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.playwright_mcp import PLAYWRIGHT_MCP_ENVIRONMENT_NAME
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_BACKEND_PORT, AGENT_ENV_WEBSITE_BACKEND_SUFFIX
from agent_env.store.object_store.local.tls import LocalCA, local_ca
from agent_env.store.routing import namespace_routing
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.browser.connect_cdp import BrowserConnectCdpTaskStep
from agent_env.task_step.task_steps.browser.sign_in import BrowserSignInTaskStep
from agent_env.task_step.task_steps.deploy_agent import DeployAgentTaskStep
from agent_env.task_step.task_steps.deploy_env import DeployEnvTaskStep
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.reset_env import ResetEnvTaskStep
from tst.integration.computer.multi_env_computers_test import _containers, _gateway, _image
from tst.integration.computer.website_browser_test import _signed_in
from tst.integration.computer.start_tabs import tabs_of
from tst.integration.store.local_parallel_deploy_test import _seed_bootstrap_envs
from tst.util import local_sandboxes
from tst.util.a2a_test_agent import put_test_agent
from tst.util.tls_front import tls_front
from tst.integration.computer.browser_computer_test import POLL_S, WAIT_S, _browser_op, _cdp_agent_loads, _discovered, _har_entries, _opened_over_cdp, _tool_names, _trusting

pytestmark = [pytest.mark.integration, pytest.mark.int_test_slow]


MULTI_SITE = "https://shop.test/api/health"
MULTI_CHROMIUM = {"sandbox_type": "local",
            "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [MULTI_SITE]}}}}
#: How often a multi env's browser is asked through the gateway whether it answers again.
ANSWER_POLL_S = 2.0


@pytest.fixture(scope="module")
def shop(browser_computer):
    """The gateway's and service-db's envs, and a website whose backend is the small agentenv website, named
    shop.test, under a local id, since a multi env naming chromium's local id must be local too."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return WebsiteEnv.put(id=f"@local/~/browsing-{uid}/shop", environment_name="shop", domains=["shop.test"],
                              backend_docker_image_artifact=_image("agentenv_website", f"browsing-shop-{uid}"))


@pytest.fixture
def made(monkeypatch) -> list[str]:
    """The ids of the local sandboxes this test's process makes, so it finds and counts only its own."""
    return local_sandboxes.recorded_sandbox_ids(monkeypatch)


def _office(shop, browser_envs) -> MultiEnv:
    return MultiEnv.put(id=f"@local/~/browsing-{uuid.uuid4().hex[:8]}/office", mcp_server_envs=[],
                        website_envs=[shop], browser_envs=browser_envs, name="office")


async def _drive(deployed) -> dict:
    """What an agent sees through the gateway's /mcp: the tools it lists, a screenshot by chromium's name, and the
    tabs after it opens the site by name."""
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            shot = await session.call_tool("chromium_computer_screenshot", {})
            opened = await session.call_tool("chromium_computer_navigate", {"url": MULTI_SITE})
            tabs = await session.call_tool("chromium_computer_tabs", {"action": "list"})
    return {"tools": tools, "shot": not shot.isError and any(part.type == "image" for part in shot.content),
            "opened": not opened.isError,
            "tabs": " ".join(part.text for part in tabs.content if part.type == "text")}


async def _answers_again(deployed) -> bool:
    """Whether chromium's screenshot answers through the gateway, asked again until it does or ``WAIT_S`` passes."""
    deadline = asyncio.get_running_loop().time() + WAIT_S
    while asyncio.get_running_loop().time() < deadline:
        try:
            async with streamable_http_client(deployed.mcp_url) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    if not (await session.call_tool("chromium_computer_screenshot", {})).isError:
                        return True
        except Exception:  # noqa: BLE001 - the gateway or the browser restarting
            pass
        await asyncio.sleep(ANSWER_POLL_S)
    return False


async def _cdp_at_gateway(deployed, browser: DeployedEnv, *, at: str | None = None, user: str = "ada",
                          origin: str | None = None, ca: LocalCA | None = None, context: str | None = None) -> dict:
    """The CDP address the browser gives ``user``, asked through the gateway's ``/svc/chromium/`` route, at ``at``, a
    front before it, else its own address, with the browser's own control token, as agentenv asks a computer child,
    naming ``origin`` or not."""
    app = next(child for child in browser.environment_card["children_environments"] if child["name"] == "chromium")
    path = protocol_v1.find_extension_method(app, BROWSER, "cdp")["endpoint"]
    async with httpx.AsyncClient(timeout=180, verify=_trusting(ca) if ca else True) as http:
        answer = await http.post(f"{at or deployed.gateway_url}/svc/chromium{path}",
                                 json={"user": user, **({"origin": origin} if origin else {}),
                                       **({"context": context} if context else {})},
                                 headers=browser.control_headers)
    assert answer.status_code == 200, answer.text
    return answer.json()


def _browser_universe(env_id: str, cookies: list[dict]) -> EnvironmentUniverseArtifact:
    """An environment universe whose chromium environment holds a browser state setting ``cookies``."""
    run = uuid.uuid4().hex[:8]
    state = FileArtifact.put_bytes(id=f"{env_id}-browser-{run}", description="multi env browser e2e",
                                   filename="browser-state.json", content=json.dumps({"cookies": cookies}).encode(),
                                   content_type="application/json")
    environment = EnvironmentArtifact.put(id=f"{env_id}-browser-env-{run}", environment_name="chromium",
                                          file_artifacts={"browser-state.json": state})
    return EnvironmentUniverseArtifact.put(id=f"{env_id}-universe-{run}", environment_artifacts=[environment])


async def _discovery(http_url: str) -> dict:
    async with httpx.AsyncClient(timeout=60) as http:
        answer = await http.get(f"{http_url.rstrip('/')}/json/version")
    assert answer.status_code == 200, answer.text
    return answer.json()


@pytest.mark.asyncio
async def test_chromium_as_a_multi_envs_browser_runs_on_its_own_sandbox_reached_through_the_gateway(
        browser_computer, shop, made):
    with namespace_routing():
        office = _office(shop, [Env.get(browser_computer.id)])
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"chromium": MULTI_CHROMIUM})
        assert step.preflight() == []
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            stated = {child.environment_name for child in protocol_v1.children(deployed.environment_card)}
            driven = await _drive(deployed)
            address = await _cdp_at_gateway(deployed, browser)
            discovered = await _discovery(address["http_url"])
            seen = await _cdp_agent_loads(address["ws_url"], [MULTI_SITE])
            reattached = await Env.from_instance_id(deployed.instance_id)
            world = _browser_universe(office.id, [{"name": "sid", "value": "ada", "url": MULTI_SITE, "max_age": 3600}])
            await LoadArtifactTaskStep(id="seed", version=None, env_id=office.id, artifact_id=world.id).execute(context)
            loaded = await _browser_op(browser, "cookies/get", {"urls": [MULTI_SITE]})

            subprocess.run(["docker", "kill", *_containers(browser.sandbox_id)], capture_output=True)
            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            after_kill = await _answers_again(deployed)
            after_kill_tabs = (await _drive(deployed))["tabs"]

            subprocess.run(["docker", "restart", _gateway(deployed, made)], capture_output=True, check=True)
            through_restart = await _answers_again(deployed)
            await ResetEnvTaskStep(id="reset-2", version=None, env_id=office.id,
                                   fail_task_on_error=True).execute(context)
            after_restart = await _answers_again(deployed)
            up = _containers(browser.sandbox_id)
        finally:
            report = await teardown_run(context)

    assert not report.failed and up and _containers(browser.sandbox_id) == []
    # chromium is a computer child on a sandbox of its own, and no playwright-mcp runs.
    assert set(deployed.child_envs) == {"chromium"}
    assert deployed.sandbox_ids["local"] == {"chromium": browser.sandbox_id}
    assert "chromium" in stated and PLAYWRIGHT_MCP_ENVIRONMENT_NAME not in stated, stated
    computer = deployed.child_envs["chromium"]["metadata"]["computer"]
    assert computer["sites"]["names"]["routes"] == {"shop": ["shop.test"]}
    [start_tab] = computer["start_tabs"]["chromium"]
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (MULTI_SITE, "shop", "loaded"), start_tab
    # An agent drives it through the gateway, every tool under chromium_.
    assert {"chromium_computer_screenshot", "chromium_computer_navigate"} <= driven["tools"], driven
    assert not any(name.startswith("computer_") for name in driven["tools"]), driven
    assert driven["shot"] and driven["opened"] and MULTI_SITE in driven["tabs"], driven
    # Its CDP address names the gateway, its discovery answer too, and an agent there reaches the site by name.
    token = address["http_url"].split("/cdp/")[1].strip("/")
    assert address["http_url"] == f"{deployed.gateway_url}/browser/chromium/cdp/{token}/", address
    assert browser.environment_url.rstrip("/") not in json.dumps([address, discovered]), (address, discovered)
    assert discovered["webSocketDebuggerUrl"].startswith(
        f"ws{deployed.gateway_url.removeprefix('http')}/browser/chromium/cdp/"), discovered
    assert seen[MULTI_SITE]["error"] is None and seen[MULTI_SITE].get("ready") == "complete", seen
    # A later process reattaches it from the instance store, and a universe loaded into the multi env sets its cookies.
    assert reattached._computers.browser._deployed.sandbox_id == browser.sandbox_id
    assert [cookie["value"] for cookie in loaded["cookies"] if cookie["name"] == "sid"] == ["ada"], loaded
    # Through a kill and a gateway restart, a reset brings it back.
    assert after_kill and MULTI_SITE in after_kill_tabs, after_kill_tabs
    assert after_restart, through_restart


@pytest.mark.asyncio
async def test_two_leases_in_a_multi_envs_browser_each_sign_in_to_its_site_by_name_through_the_gateway(
        browser_computer, session_shop):
    """M8.15: in a multi env whose browser is chromium, two leases are each signed in to the env's site as a user of
    their own, and a tab of each opens the site by its name, over https through the gateway, as that user; the
    default context has none of their cookies."""
    with namespace_routing():
        office = _office(session_shop, [Env.get(browser_computer.id)])
        context = TaskStepContext()

        def step(type: str, **fields):
            return get_task_step_registry()[type].from_dict({
                "id": f"{type}-{fields.get('label') or fields.get('context')}", "version": None, "type": type,
                "env_id": office.id, **fields})

        try:
            [deployed] = (await DeployEnvTaskStep(
                id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                child_envs={"chromium": {"sandbox_type": "local", "env_params": {
                    "form": "container", "app_settings": {"chromium": {"start_urls": ["about:blank"]}}}}},
            ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            for label, user in (("buyer", "alex"), ("seller", "bo")):
                context = await step("browser_lease_context", label=label, grants=["ada"]).execute(context)
                context = await step("browser_sign_in", site=session_shop.id, user=user, context=label).execute(context)
                context = await step("browser_open_tab", url=SESSION_PAGE, context=label).execute(context)
            seen = {label: await _lease_page_text(browser, (await _cdp_at_gateway(deployed, browser, context=label))[
                "ws_url"], label) for label in ("buyer", "seller")}
            default_cookies = (await _browser_op(browser, "cookies/get", {"urls": ["https://shop.test/"]}))["cookies"]
        finally:
            report = await teardown_run(context)

    assert not report.failed
    assert '"alex"' in seen["buyer"] and '"bo"' in seen["seller"], seen
    assert default_cookies == [], default_cookies


async def _lease_page_text(browser: DeployedEnv, ws_url: str, label: str) -> str:
    """The text of the page the one tab of the lease ``label`` shows once it's on shop.test, read over the CDP
    address ``ws_url``; what it last read if that never says who's signed in."""
    text, deadline = "", time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        [tab] = (await _browser_op(browser, "tabs/list", {"context": label}))["tabs"]
        if tab["url"].startswith("https://shop.test/"):
            async with connect(ws_url, max_size=None, open_timeout=60) as ws:
                ids = itertools.count(1)

                async def call(method: str, params: dict, session: str | None = None) -> dict:
                    message_id = next(ids)
                    await ws.send(json.dumps({"id": message_id, "method": method, "params": params,
                                              **({"sessionId": session} if session else {})}))
                    while (reply := json.loads(await asyncio.wait_for(ws.recv(), 60))).get("id") != message_id:
                        pass
                    return reply

                session = (await call("Target.attachToTarget", {"targetId": tab["id"], "flatten": True}))[
                    "result"]["sessionId"]
                text = (await call("Runtime.evaluate", {"expression": "document.body ? document.body.innerText : ''",
                                                         "returnByValue": True}, session))["result"]["result"].get(
                    "value", "")
            if '"user"' in text:
                return text
        await asyncio.sleep(POLL_S)
    return text


@pytest.mark.asyncio
async def test_a_multi_env_naming_no_browser_takes_the_runs_and_chromium_runs_as_a_computer_child(browser_computer,
                                                                                                  shop):
    with namespace_routing():
        office = _office(shop, None)
        context = TaskStepContext()
        try:
            with run_env_ids({WEBSITE_BROWSER_OVERRIDE: browser_computer.id}):
                [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                      ttl_seconds=900, child_envs={"chromium": MULTI_CHROMIUM}
                                                      ).execute(context)).deployed_envs
            stated = {child.environment_name for child in protocol_v1.children(deployed.environment_card)}
        finally:
            report = await teardown_run(context)

    assert not report.failed and set(deployed.child_envs) == {"chromium"}
    assert PLAYWRIGHT_MCP_ENVIRONMENT_NAME not in stated, stated
    [start_tab] = deployed.child_envs["chromium"]["metadata"]["computer"]["start_tabs"]["chromium"]
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (MULTI_SITE, "shop", "loaded"), start_tab


@pytest.mark.asyncio
async def test_browser_connect_cdp_hands_an_agent_an_address_at_the_multi_envs_gateway_with_its_tools_hidden(
        browser_computer, shop):
    with namespace_routing():
        agent = put_test_agent(f"cdp-agent-{uuid.uuid4().hex[:8]}")
        office = _office(shop, [Env.get(browser_computer.id)])
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": MULTI_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            await DeployAgentTaskStep(id="deploy-solver", version=None, env_ids=[], a2a_agent_id=agent.id,
                                      a2a_agent_version=agent.version, agent_name="solver",
                                      sandbox_type="local").execute(context)
            await BrowserConnectCdpTaskStep(id="connect", version=None, env_id=office.id,
                                            agent_name="solver").execute(context)
            _, title = await _opened_over_cdp(context, "first", url=MULTI_SITE)
            tools = await _tool_names(deployed)
            async with httpx.AsyncClient(timeout=30) as http:
                lines = (await http.get(f"{deployed.gateway_url}/trajectory")).text.splitlines()
            relayed = [event for event in map(json.loads, lines) if event.get("event_type") == "browser_connection"]
        finally:
            report = await teardown_run(context)

    assert not title.startswith("failed"), title
    assert tools == [], tools
    assert [event["browser"] for event in relayed] and all(event["browser"] == "chromium" for event in relayed), relayed
    assert context.metadata["browser_connections"] == [{
        "step_id": "connect", "env_id": office.id, "agent_name": "solver", "browser": "chromium", "name": "chromium",
        "keep_tools": False, "role": "default", "tools_hidden": "*"}]
    assert not report.failed


@pytest.mark.asyncio
async def test_behind_a_front_that_ends_tls_the_multi_envs_gateway_hands_out_https_and_wss_and_an_agent_connects(
        browser_computer, shop):
    ca = local_ca()
    with namespace_routing():
        office = _office(shop, [Env.get(browser_computer.id)])
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": MULTI_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            gateway = urllib.parse.urlsplit(deployed.gateway_url)
            async with tls_front(ca, gateway.hostname, gateway.port) as front:
                named = await _cdp_at_gateway(deployed, browser, at=front, origin=front, ca=ca)
                unnamed = await _cdp_at_gateway(deployed, browser, at=front, user="bob", ca=ca)
                discovered = await _discovered(named["http_url"], ca)
                seen = await _cdp_agent_loads(named["ws_url"], [MULTI_SITE], tls=_trusting(ca))
        finally:
            report = await teardown_run(context)

    host = front.removeprefix("https://")
    assert named["http_url"].startswith(f"{front}/browser/chromium/cdp/"), named
    assert named["ws_url"].startswith(f"wss://{host}/browser/chromium/cdp/") and discovered == named["ws_url"], named
    assert unnamed["ws_url"].startswith(f"ws://{host}/browser/chromium/cdp/"), unnamed
    assert seen[MULTI_SITE]["error"] is None and seen[MULTI_SITE].get("ready") == "complete", seen
    assert not report.failed


#: A page the site that offers session serves, which says who a request signs in, and chromium starting there.
SESSION_PAGE = "https://shop.test/api/session"
SESSION_CHROMIUM = {**MULTI_CHROMIUM,
                    "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [SESSION_PAGE]}}}}


@pytest.fixture(scope="module")
def session_shop(browser_computer):
    """A site named shop.test that offers website/v1 session, under a local id, as a multi env naming chromium's local
    id must be."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return WebsiteEnv.put(id=f"@local/~/signin-{uid}/shop", environment_name="shop", domains=["shop.test"],
                              server_docker_image_artifact=_image("session_website", f"signin-shop-{uid}"))


@pytest.mark.asyncio
async def test_under_a_chromium_default_a_site_gets_chromium_unnamed_which_browser_sign_in_signs_in_after_each_reset(
        browser_computer, session_shop):
    config = get_config()
    with namespace_routing():
        office = MultiEnv.put(id=f"{session_shop.id.rsplit('/', 1)[0]}/office", mcp_server_envs=[],
                              website_envs=[session_shop], name="office")
        sign_in = BrowserSignInTaskStep(id="in", version=None, env_id=office.id, site=session_shop.id, user="alex")
        context = TaskStepContext()
        config.default_website_browser_env_id = browser_computer.id  # the config's default, not the run's override
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": SESSION_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            before = await _signed_in(browser)
            await sign_in.execute(context)
            signed = await _signed_in(browser)
            await ResetEnvTaskStep(id="reset", version=None, env_id=office.id, fail_task_on_error=True).execute(context)
            after_reset = await _signed_in(browser)
            await sign_in.execute(context)
            again = await _signed_in(browser)
        finally:
            config.default_website_browser_env_id = None
            report = await teardown_run(context)

    assert not report.failed
    assert office.browser_envs is None and set(deployed.child_envs) == {"chromium"}
    [start_tab] = deployed.child_envs["chromium"]["metadata"]["computer"]["start_tabs"]["chromium"]
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (SESSION_PAGE, "shop", "loaded"), start_tab
    assert (before, signed, after_reset, again) == (None, "alex", None, "alex")
    assert context.metadata["browser_steps"]["in"]["names"] == ["shop.test"]


FORWARDED_PAGE = "http://shop.localhost/admin/home"
FORWARDED_CHROMIUM = {**MULTI_CHROMIUM,
                      "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [FORWARDED_PAGE]}}}}


@pytest.fixture(scope="module")
def forwarding_shop(browser_computer):
    """A site named shop.localhost whose pages an app beside its server serves, signed in to by a header only the
    site's own request sends, under a local id, as a multi env naming chromium's local id must be."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        return WebsiteEnv.put(id=f"@local/~/forwarding-{uid}/shop", environment_name="shop", domains=["shop.localhost"],
                              server_docker_image_artifact=_image("forwarding_website", f"forwarding-shop-{uid}"))


async def _forwarded(browser: DeployedEnv, user: str | None) -> dict:
    """What the forwarding site's app saw of the browser's tab on its page, from the tab's title, once that says
    ``user`` is signed in, or what it last said once ``WAIT_S`` passes."""
    seen, deadline = {}, time.monotonic() + WAIT_S
    while time.monotonic() < deadline:
        for tab in (await _browser_op(browser, "tabs/list", {}))["tabs"]:
            if tab["url"] == FORWARDED_PAGE and tab.get("title", "").startswith("{"):
                seen = json.loads(tab["title"])
        if seen and seen["user"] == user:
            return seen
        await asyncio.sleep(POLL_S)
    return seen


@pytest.mark.asyncio
async def test_a_site_wrapping_an_app_signs_chromium_in_by_a_header_only_it_sends_and_the_app_serves_its_pages(
        browser_computer, forwarding_shop):
    """M8.35: a ForwardingSite's app serves a multi env's chromium its pages at the site's name, over HTTP for a
    .localhost name, seeing that name as its Host and the page's own path; browser_sign_in signs the browser in through
    the site's @session_state, the one request that sends the app its sign-in header, and the start tab loads again
    signed in; the browser's requests never carry the header, and network.har has the app's pages."""
    with namespace_routing():
        office = _office(forwarding_shop, [Env.get(browser_computer.id)])
        sign_in = BrowserSignInTaskStep(id="in", version=None, env_id=office.id, site=forwarding_shop.id, user="amy")
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900, child_envs={"chromium": FORWARDED_CHROMIUM}
                                                  ).execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            before = await _forwarded(browser, None)
            await sign_in.execute(context)
            after = await _forwarded(browser, "amy")
            entries = await _har_entries(browser)
        finally:
            report = await teardown_run(context)

    assert not report.failed
    [start_tab] = deployed.child_envs["chromium"]["metadata"]["computer"]["start_tabs"]["chromium"]
    assert (start_tab["url"], start_tab["site"], start_tab["outcome"]) == (FORWARDED_PAGE, "shop", "loaded"), start_tab
    page = {"host": "shop.localhost", "path": "/admin/home", "auto_login": False}
    assert (before, after) == ({"user": None, **page}, {"user": "amy", **page}), (before, after)
    loads = [entry for entry in entries if entry["request"]["url"] == FORWARDED_PAGE]
    assert len(loads) >= 2 and {entry["response"]["status"] for entry in loads} == {200}, loads
    assert not [entry for entry in entries for header in entry["request"]["headers"]
                if header["name"].lower() == "x-auto-login"]


@pytest.mark.asyncio
async def test_a_multi_env_naming_no_browser_deploys_none(browser_computer, shop):
    with namespace_routing():
        office = _office(shop, [])
        context = TaskStepContext()
        try:
            [deployed] = (await DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local",
                                                  ttl_seconds=900).execute(context)).deployed_envs
            stated = {child.environment_name for child in protocol_v1.children(deployed.environment_card)}
        finally:
            report = await teardown_run(context)

    assert not report.failed and not deployed.child_envs
    assert "shop" in stated and PLAYWRIGHT_MCP_ENVIRONMENT_NAME not in stated and "chromium" not in stated, stated


# --- a site reached by its names through a multi env's gateway ---

HOST_START = "https://shop.test/api/host"
HOST_CHROMIUM = {**MULTI_CHROMIUM,
                 "env_params": {"form": "container", "app_settings": {"chromium": {"start_urls": [HOST_START]}}}}
#: Where the gateway reaches the site, which is the site's Host when a client reaches it at any other name.
SITE_INTERNAL = f"shop-{AGENT_ENV_WEBSITE_BACKEND_SUFFIX}:{AGENT_ENV_WEBSITE_BACKEND_PORT}"


@pytest.fixture(scope="module")
def host_shops(browser_computer):
    """Two sites named shop.test that build their addresses from their Host, as many real apps do, under local ids:
    one whose env stores the name and that serves no card, and one that stores none and whose card declares it."""
    with namespace_routing():
        _seed_bootstrap_envs()
        uid = uuid.uuid4().hex[:8]
        image = _image("host_website", f"host-shop-{uid}")
        return {"stored": WebsiteEnv.put(id=f"@local/~/host-{uid}/stored/shop", environment_name="shop",
                                         domains=["shop.test"], server_docker_image_artifact=image),
                "card": WebsiteEnv.put(id=f"@local/~/host-{uid}/card/shop", environment_name="shop",
                                       settings={"card_domains": "shop.test"}, server_docker_image_artifact=image)}


async def _landed(deployed, browser: DeployedEnv, url: str) -> str:
    """Where chromium lands, driven through the gateway's /mcp to ``url``: the first tab at the site's /home, else
    every tab's URL."""
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            assert not (await session.call_tool("chromium_computer_navigate", {"url": url})).isError
    deadline = asyncio.get_running_loop().time() + WAIT_S
    while True:
        urls = [tab.get("url", "") for tab in await tabs_of(browser)]
        if (home := next((u for u in urls if u.endswith("/home")), None)) or \
                asyncio.get_running_loop().time() > deadline:
            return home or " ".join(urls)
        await asyncio.sleep(POLL_S)


async def _listed(deployed) -> list[str]:
    async with streamable_http_client(deployed.mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_website_urls", {})
    assert not result.isError and result.structuredContent, result
    return result.structuredContent["result"]


async def _seen(deployed, root: str | None) -> dict:
    """The Host and X-Forwarded-Host the site sees for a request straight to the gateway's site route naming ``root``
    as the origin it was reached at, or naming none."""
    async with httpx.AsyncClient(timeout=30) as http:
        answer = await http.get(f"{deployed.gateway_url}/website/shop/api/host",
                                headers={"AgentEnv-Site-Root": root} if root else {})
    assert answer.status_code == 200, answer.text
    return answer.json()


@pytest.mark.asyncio
@pytest.mark.parametrize("names", ["stored", "card"])
async def test_a_site_chromium_reaches_by_name_through_the_gateway_takes_that_name_as_its_host_and_no_forged_one(
        browser_computer, host_shops, made, names):
    with namespace_routing():
        office = _office(host_shops[names], [Env.get(browser_computer.id)])
        step = DeployEnvTaskStep(id="office", version=None, env_id=office.id, sandbox_type="local", ttl_seconds=900,
                                 child_envs={"chromium": HOST_CHROMIUM})
        context = TaskStepContext()
        try:
            [deployed] = (await step.execute(context)).deployed_envs
            browser = DeployedEnv.from_dict(deployed.child_envs["chromium"])
            landed, listed = await _landed(deployed, browser, "https://shop.test/login"), await _listed(deployed)
            seen = [await _seen(deployed, root) for root in ("https://Shop.Test", "https://evil.example", None)]
            subprocess.run(["docker", "restart", _gateway(deployed, made)], capture_output=True, check=True)
            again = await _answers_again(deployed)
            after_restart = (await _landed(deployed, browser, "https://shop.test/login"), await _listed(deployed),
                             await _seen(deployed, "https://evil.example"))
        finally:
            report = await teardown_run(context)

    assert not report.failed
    # chromium follows the site's redirect, built from its Host, to the name it reached it at.
    assert landed == "https://shop.test/home", landed
    # An agent beside chromium is given the site at the name chromium opens it at.
    assert listed == ["https://shop.test/"], listed
    # A client naming any other origin, or none, leaves the site its internal Host; X-Forwarded-Host is as it was.
    gateway_host = urllib.parse.urlsplit(deployed.gateway_url).netloc
    assert seen == [{"host": "Shop.Test", "forwarded_host": "Shop.Test"},
                    {"host": SITE_INTERNAL, "forwarded_host": "evil.example"},
                    {"host": SITE_INTERNAL, "forwarded_host": gateway_host}], seen
    # A restarted gateway, which reads a card's names again, keeps all of it.
    assert again and after_restart == ("https://shop.test/home", ["https://shop.test/"],
                                       {"host": SITE_INTERNAL, "forwarded_host": "evil.example"}), after_restart
