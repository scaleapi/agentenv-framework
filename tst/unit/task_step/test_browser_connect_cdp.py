"""browser_connect_cdp asks the env's website browser for the agent's CDP address under the env's URL as the agent's
sandbox reaches it, or, of a browser that takes no origin, rebases the address it answers onto that; a browser in one
of a multi env's computers is asked through the multi env's gateway, which names its own address, rebased the same
way; it adds the address to the agent's cdp-config and hides every tool the env serves from the agent's role unless
kept; it refuses an agent that takes no CDP address and an env whose browser it can't tell; and connecting under a
name already connected replaces its record. Only the HTTP boundary is mocked."""

from __future__ import annotations

import dataclasses
import json

import httpx
import pytest

from agent_env.env.env import DeployedGatewayEnv, DeployedSandboxEnv
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.browser.connect_cdp import BrowserConnectCdpTaskStep

_ENV = "http://localhost:41000"
_GATEWAY = "http://localhost:43000"
_AGENT = "http://localhost:42000"
_BROWSER = {"uri": "urn:agentenv:browser/v1", "params": {
    "endpoint": "/svc/chromium/agentenv/ext/browser", "methods": {"cdp": {"endpoint": "/svc/chromium/agentenv/ext/browser/cdp"}}}}
#: The browser's extension as one that takes an origin serves it.
_BROWSER_TAKING_ORIGIN = {**_BROWSER, "params": {**_BROWSER["params"], "methods": {"cdp": {
    **_BROWSER["params"]["methods"]["cdp"], "request": {"type": "object", "required": ["user"], "properties": {
        "user": {"type": "string"}, "origin": {"type": "string", "default": None}}}}}}}
_TOOL_ACCESS = {"uri": "urn:agentenv:disable-tool/v1", "params": {"endpoint": "/tools/disable",
                                                                  "methods": {"disable": {"method": "POST"}}}}
_CDP_CONFIG = {"uri": "urn:agentenv:cdp-config/v1", "params": {"endpoint": "/ext/cdp-config"}}


@pytest.mark.asyncio
async def test_the_agent_gets_the_browsers_address_as_its_sandbox_reaches_it_and_its_role_none_of_the_envs_tools(
        monkeypatch):
    sent = _mock(monkeypatch, tokens=["t1"])
    context = TaskStepContext(deployed_envs=[_env()], deployed_agents=[_agent()], metadata={})

    await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="computer", agent_name="solver").execute(
        context)

    asked, added, hidden = sent
    assert (str(asked.url), json.loads(asked.content)) == (f"{_ENV}/svc/chromium/agentenv/ext/browser/cdp",
                                                         {"user": "solver"})
    assert asked.headers["authorization"] == "Bearer control"
    assert (str(added.url), json.loads(added.content)) == (f"{_AGENT}/ext/cdp-config", {
        "name": "chromium", "http_url": "http://host.docker.internal:41000/svc/chromium/cdp/t1/",
        "ws_url": "ws://host.docker.internal:41000/svc/chromium/cdp/t1/devtools/browser",
        "metadata": {"browser": "chromium"}})
    assert (str(hidden.url), json.loads(hidden.content)) == (f"{_ENV}/tools/disable", {"role": "default", "tools": "*"})
    assert context.metadata["browser_connections"] == [{
        "step_id": "connect", "env_id": "computer", "agent_name": "solver", "browser": "chromium", "name": "chromium",
        "keep_tools": False, "role": "default", "tools_hidden": "*"}]


@pytest.mark.parametrize(("url", "sandbox_type", "origin"), [
    ("https://t.tunnel.test", "modal_vm", "https://t.tunnel.test"),
    (_ENV, "local", "http://host.docker.internal:41000")])
@pytest.mark.asyncio
async def test_a_browser_that_takes_an_origin_mints_the_agents_address_where_its_sandbox_reaches_the_env(
        monkeypatch, url, sandbox_type, origin):
    sent = _mock(monkeypatch, tokens=["t1"])
    deployed = _env([_app("chromium", _BROWSER_TAKING_ORIGIN)], url=url, sandbox_type=sandbox_type)
    context = TaskStepContext(deployed_envs=[deployed], deployed_agents=[_agent()], metadata={})

    await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="computer", agent_name="solver").execute(
        context)

    asked, added, _ = sent
    ws = "wss" + origin.removeprefix("https") if origin.startswith("https") else "ws" + origin.removeprefix("http")
    assert (str(asked.url), json.loads(asked.content)) == (f"{url}/svc/chromium/agentenv/ext/browser/cdp",
                                                         {"user": "solver", "origin": origin})
    assert json.loads(added.content) == {"name": "chromium", "http_url": f"{origin}/svc/chromium/cdp/t1/",
                                         "ws_url": f"{ws}/svc/chromium/cdp/t1/devtools/browser",
                                         "metadata": {"browser": "chromium"}}


@pytest.mark.asyncio
async def test_a_browser_in_a_multi_envs_computer_is_reached_through_its_gateway_and_its_tools_hidden_there(
        monkeypatch):
    sent = _mock(monkeypatch, tokens=["t1"], base=f"{_GATEWAY}/browser/chromium")
    context = TaskStepContext(deployed_envs=[_multi()], deployed_agents=[_agent()], metadata={})

    await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="office", agent_name="solver").execute(context)

    asked, added, hidden = sent
    assert (str(asked.url), json.loads(asked.content)) == (
        f"{_GATEWAY}/svc/chromium/svc/chromium/agentenv/ext/browser/cdp", {"user": "solver"})
    assert asked.headers["authorization"] == "Bearer control"  # the computer's own token, which the gateway passes on
    assert json.loads(added.content)["http_url"] == "http://host.docker.internal:43000/browser/chromium/cdp/t1/"
    assert json.loads(added.content)["ws_url"] == (
        "ws://host.docker.internal:43000/browser/chromium/cdp/t1/devtools/browser")
    assert (str(hidden.url), json.loads(hidden.content)) == (f"{_GATEWAY}/tools/disable",
                                                             {"role": "default", "tools": "*"})
    assert hidden.headers["authorization"] == "Bearer gateway-control"
    assert context.metadata["browser_connections"][0]["env_id"] == "office"


@pytest.mark.asyncio
async def test_behind_a_tunnel_that_ends_tls_the_gateway_is_asked_under_the_agents_origin_and_hands_out_https_and_wss(
        monkeypatch):
    sent = _mock(monkeypatch, tokens=["t1"], relay="gw.w.modal.host")  # the gateway sees plain http, as behind Modal
    context = TaskStepContext(deployed_envs=[_multi("https://gw.w.modal.host", "modal", _BROWSER_TAKING_ORIGIN)],
                              deployed_agents=[_agent(sandbox_type="modal")], metadata={})

    await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="office", agent_name="solver").execute(context)

    asked, added, _ = sent
    assert json.loads(asked.content) == {"user": "solver", "origin": "https://gw.w.modal.host"}
    assert (json.loads(added.content)["http_url"], json.loads(added.content)["ws_url"]) == (
        "https://gw.w.modal.host/browser/chromium/cdp/t1/",
        "wss://gw.w.modal.host/browser/chromium/cdp/t1/devtools/browser")


@pytest.mark.asyncio
async def test_behind_a_tunnel_a_browser_that_takes_no_origin_is_refused_with_word_to_compose_it_again(monkeypatch):
    _mock(monkeypatch, tokens=["t1"], relay="gw.w.modal.host")  # composed before #608: its cdp takes no origin
    context = TaskStepContext(deployed_envs=[_multi("https://gw.w.modal.host", "modal")],
                              deployed_agents=[_agent(sandbox_type="modal")], metadata={})

    with pytest.raises(RuntimeError, match="takes no origin.*compose it again on agentenv-framework 0.9.1505"):
        await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="office",
                                        agent_name="solver").execute(context)
    assert "browser_connections" not in context.metadata


@pytest.mark.asyncio
async def test_a_cdp_address_the_multi_envs_gateway_didnt_relay_is_refused_before_the_agent_gets_it(monkeypatch):
    sent = _mock(monkeypatch, tokens=["t1"])  # the computer's own address, as a gateway answers for several browsers
    context = TaskStepContext(deployed_envs=[_multi()], deployed_agents=[_agent()], metadata={})

    with pytest.raises(RuntimeError, match="gateway didn't put the CDP address .* at its relay"):
        await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="office",
                                        agent_name="solver").execute(context)
    assert [request.url.path for request in sent] == ["/svc/chromium/svc/chromium/agentenv/ext/browser/cdp"]
    assert "browser_connections" not in context.metadata


@pytest.mark.asyncio
async def test_a_role_kept_tools_a_name_and_a_gateways_address_are_the_tasks(monkeypatch):
    sent = _mock(monkeypatch, tokens=["t1"], base="https://gateway.test/browser/chromium")
    context = TaskStepContext(deployed_envs=[_env()], deployed_agents=[_agent(role="buyer")], metadata={})

    await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="computer", agent_name="solver",
                                    browser="chromium", name="shop", keep_tools=True).execute(context)

    _, added = sent
    assert json.loads(added.content)["name"] == "shop"
    assert json.loads(added.content)["http_url"] == "https://gateway.test/browser/chromium/cdp/t1/"
    assert context.metadata["browser_connections"][0] == {
        "step_id": "connect", "env_id": "computer", "agent_name": "solver", "browser": "chromium", "name": "shop",
        "keep_tools": True, "role": "buyer", "tools_hidden": None}


@pytest.mark.asyncio
@pytest.mark.parametrize(("env", "agent", "browser", "refusal"), [
    ("one", {}, None, "its card has no urn:agentenv:cdp-config/v1"),
    ("none", _CDP_CONFIG, None, "has no website browsers"),
    ("two", _CDP_CONFIG, None, "has 2 website browsers: name one as browser"),
    ("one", _CDP_CONFIG, "firefox", "has no website browser named 'firefox'"),
])
async def test_an_agent_without_cdp_config_or_a_browser_it_cant_tell_is_refused_before_any_request(
        monkeypatch, env, agent, browser, refusal):
    sent = _mock(monkeypatch, tokens=["t1"])
    children = {"one": [_app("chromium")], "none": [], "two": [_app("chromium"), _app("brave")]}[env]
    context = TaskStepContext(deployed_envs=[_env(children)], deployed_agents=[_agent(extensions=[agent] if agent
                                                                                      else [])], metadata={})

    with pytest.raises(RuntimeError, match=refusal):
        await BrowserConnectCdpTaskStep(id="connect", version=None, env_id="computer", agent_name="solver",
                                        browser=browser).execute(context)
    assert sent == []


@pytest.mark.asyncio
async def test_a_connection_under_a_name_already_connected_replaces_its_record(monkeypatch):
    sent = _mock(monkeypatch, tokens=["t1", "t2"])
    other = _env()
    other.env_id = "other"
    context = TaskStepContext(deployed_envs=[_env(), other], deployed_agents=[_agent()], metadata={})

    for env_id in ("computer", "other"):
        await BrowserConnectCdpTaskStep(id=f"connect-{env_id}", version=None, env_id=env_id,
                                        agent_name="solver").execute(context)

    assert len(sent) == 6
    assert [(record["env_id"], record["name"]) for record in context.metadata["browser_connections"]] == [
        ("other", "chromium")]


def test_it_round_trips_and_a_blank_name_is_a_config_problem():
    step = BrowserConnectCdpTaskStep(id="connect", version=None, env_id="computer", agent_name="solver",
                                     browser="chromium", name="shop", keep_tools=True)

    assert BrowserConnectCdpTaskStep.from_dict(step.to_dict()).to_dict() == step.to_dict()
    assert BrowserConnectCdpTaskStep(id="c", version=None, env_id="computer", name="").config_problems() == [
        "step 'c': name must be a non-empty string"]


def _app(name: str, browser: dict = _BROWSER) -> dict:
    return {"name": name, "capabilities": {"extensions": [browser]}}


def _env(children: list | None = None, *, url: str = _ENV, sandbox_type: str = "local") -> DeployedSandboxEnv:
    """A browser computer's deployment: a driver serving a data plane and tool access on its own card, named after
    its browser app among its children, as a browser computer's driver is."""
    return DeployedSandboxEnv(
        env_id="computer", env_version=1, sandbox_id="sb", sandbox_type=sandbox_type, control_token="control",
        environment_card_url=f"{url}/.well-known/agent-env.json",
        environment_card={"name": "chromium", "capabilities": {"extensions": [_TOOL_ACCESS], "operations": ["reset"]},
                          "children_environments": [_app("chromium")] if children is None else children})


def _multi(gateway: str = _GATEWAY, sandbox_type: str = "local", browser: dict = _BROWSER) -> DeployedGatewayEnv:
    """A multi env's deployment at ``gateway`` whose computer ``chromium`` is the browser computer ``_env`` gives,
    serving ``browser``, its gateway serving tool access on its own card."""
    return DeployedGatewayEnv(
        env_id="office", env_version=1, gateway_url=gateway, sandbox_id="gw", sandbox_type=sandbox_type,
        control_token="gateway-control", environment_card_url=f"{gateway}/.well-known/agent-env.json",
        environment_card={"name": "office", "capabilities": {"extensions": [_TOOL_ACCESS]},
                          "children_environments": [{"name": "chromium"}]},
        child_envs={"chromium": dataclasses.asdict(_env([_app("chromium", browser)], sandbox_type=sandbox_type))})


def _agent(role: str | None = None, extensions: list | None = None, sandbox_type: str = "local") -> DeployedAgent:
    return DeployedAgent(agent_name="solver", api_url=_AGENT, a2a_url=_AGENT, sandbox_type=sandbox_type, role=role,
                         a2a_card={"capabilities": {"extensions": [_CDP_CONFIG] if extensions is None else extensions}})


def _mock(monkeypatch, *, tokens: list[str], base: str = f"{_ENV}/svc/chromium",
          relay: str | None = None) -> list[httpx.Request]:
    """Answer the browser's cdp with each of ``tokens`` in turn, under the origin it names, else at ``base``, or, as a
    gateway at the host ``relay`` that sees plain http, at its relay in the origin's scheme; anything else as done;
    return the requests."""
    sent, real, left = [], httpx.AsyncClient, iter(tokens)

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/browser/cdp"):
            token, origin = next(left), json.loads(request.content).get("origin")
            at = f"{origin}/svc/chromium" if origin else base
            if relay:
                at = f"{'https' if (origin or '').startswith('https') else 'http'}://{relay}/browser/chromium"
            ws = "wss" + at.removeprefix("https") if at.startswith("https") else "ws" + at.removeprefix("http")
            return httpx.Response(200, json={"http_url": f"{at}/cdp/{token}/",
                                             "ws_url": f"{ws}/cdp/{token}/devtools/browser"})
        return httpx.Response(200, json={"status": "added"} if request.url.path == "/ext/cdp-config" else {"ok": True})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
