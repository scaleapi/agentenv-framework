"""browser_connect_cdp asks the env's website browser for the agent's CDP address, adds it to the agent's cdp-config as
the agent's sandbox reaches it, and hides every tool the env serves from the agent's role unless kept; it refuses an
agent that takes no CDP address and an env whose browser it can't tell; and connecting under a name already connected
replaces its record. Only the HTTP boundary is mocked."""

from __future__ import annotations

import json

import httpx
import pytest

from agent_env.env.env import DeployedSandboxEnv
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.task_step.task_steps.browser_connect_cdp import BrowserConnectCdpTaskStep

_ENV = "http://localhost:41000"
_AGENT = "http://localhost:42000"
_BROWSER = {"uri": "urn:agentenv:browser/v1", "params": {
    "endpoint": "/svc/chromium/agentenv/ext/browser", "methods": {"cdp": {"endpoint": "/svc/chromium/agentenv/ext/browser/cdp"}}}}
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


def _app(name: str) -> dict:
    return {"name": name, "capabilities": {"extensions": [_BROWSER]}}


def _env(children: list | None = None) -> DeployedSandboxEnv:
    """A browser computer's deployment: a driver serving a data plane and tool access on its own card, named after
    its browser app among its children, as a browser computer's driver is."""
    return DeployedSandboxEnv(
        env_id="computer", env_version=1, sandbox_id="sb", sandbox_type="local", control_token="control",
        environment_card_url=f"{_ENV}/.well-known/agent-env.json",
        environment_card={"name": "chromium", "capabilities": {"extensions": [_TOOL_ACCESS], "operations": ["reset"]},
                          "children_environments": [_app("chromium")] if children is None else children})


def _agent(role: str | None = None, extensions: list | None = None) -> DeployedAgent:
    return DeployedAgent(agent_name="solver", api_url=_AGENT, a2a_url=_AGENT, sandbox_type="local", role=role,
                         a2a_card={"capabilities": {"extensions": [_CDP_CONFIG] if extensions is None else extensions}})


def _mock(monkeypatch, *, tokens: list[str], base: str = f"{_ENV}/svc/chromium") -> list[httpx.Request]:
    """Answer the browser's cdp with each of ``tokens`` in turn, at ``base``, and anything else as done; return the
    requests."""
    sent, real, left = [], httpx.AsyncClient, iter(tokens)

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/browser/cdp"):
            token = next(left)
            ws = "wss" + base.removeprefix("https") if base.startswith("https") else "ws" + base.removeprefix("http")
            return httpx.Response(200, json={"http_url": f"{base}/cdp/{token}/",
                                             "ws_url": f"{ws}/cdp/{token}/devtools/browser"})
        return httpx.Response(200, json={"status": "added"} if request.url.path == "/ext/cdp-config" else {"ok": True})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
