"""browser_sign_in, and every browser step's reach into a multi env's computers: the site's session is asked through the
record that serves it, its cookies are set for each of the site's names, and the browser in the computer gets them
through data/add with the computer's own control token, the multi env's gateway listing the computer but not its apps.
Only the HTTP boundary and the store are faked."""

from __future__ import annotations

import dataclasses
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest

from agent_env.env.env import DeployedEnv, DeployedGatewayEnv, DeployedSandboxEnv, Env
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.env.envs.multi_env import MultiEnv
from agent_env.env.envs.website import WebsiteEnv
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step.task_steps.browser import sign_in
from agent_env.task_step.task_steps.browser.base import stored_browsers
from agent_env.task_step.task_steps.browser.cookies import BrowserSetCookiesTaskStep
from agent_env.task_step.task_steps.browser.sign_in import BrowserSignInTaskStep, scoped

_GATEWAY = "http://localhost:42000"
_DESKTOP = "http://localhost:42100"
_SITE = "http://localhost:42200"
_SESSION = {"name": "agentenv_session_shop", "value": "signed.alex", "path": "/", "max_age": 3600, "httpOnly": True,
            "sameSite": "Lax"}


def _website() -> WebsiteEnv:
    return WebsiteEnv(id="shop-site", version=1, server_docker_image_artifact=MagicMock(), environment_name="shop",
                      domains=["shop.test", "*.shop.test"])


def _browser_card(ops=("cookies/set", "cdp")) -> dict:
    prefix = "/svc/chromium/agentenv/ext/browser"
    return {"name": "chromium", "url": "/svc/chromium/agentenv",
            "capabilities": {"operations": ["data/reset", "data/add", "data/get"], "extensions": [
                {"uri": "urn:agentenv:browser/v1", "params": {"endpoint": prefix, "methods": {
                    op: {"endpoint": f"{prefix}/{op}"} for op in ops}}}]}}


def _office(*, session: bool = True, card_name: str = "shop") -> DeployedGatewayEnv:
    """A multi env's record: the shop website among its card's children, served as shop whatever its card names it,
    and the desktop computer, whose card lists chromium, kept whole in its child_envs, as the gateway's card lists the
    computer without its apps."""
    website = {"uri": "urn:agentenv:website/v1", "params": {
        "endpoint": "/svc/shop/agentenv/ext/website", "domains": ["shop.test", "*.shop.test"],
        "browse_url": "/website/shop/",
        "methods": {"session": {"endpoint": "/svc/shop/agentenv/ext/website/session"}} if session else {}}}
    desktop = DeployedSandboxEnv(
        env_id="desktop-computer", env_version=1, sandbox_id="sb-desktop", sandbox_type="local",
        control_token="desktop-token", environment_card_url=f"{_DESKTOP}/.well-known/agent-env.json",
        environment_card={"name": "desktop", "children_environments": [_browser_card()]})
    return DeployedGatewayEnv(
        env_id="office", env_version=1, sandbox_id="gw", sandbox_type="local", gateway_url=_GATEWAY,
        control_token="gateway-token", environment_card_url=f"{_GATEWAY}/.well-known/agent-env.json",
        environment_card={"name": "office", "children_environments": [
            {"name": card_name, "url": "/svc/shop/agentenv", "capabilities": {"extensions": [website]}},
            {"name": "desktop", "url": "/svc/desktop/agentenv", "capabilities": {"extensions": []}}]},
        child_envs={"desktop": dataclasses.asdict(desktop)})


@pytest.fixture
def served(monkeypatch) -> list[httpx.Request]:
    """The store holds the office multi env with its shop website; the site answers a session, the browser takes
    what it's sent. The requests, in order."""
    sent, real = [], httpx.AsyncClient
    office = SimpleNamespace(website_envs=[_website()], computer_envs=[])
    monkeypatch.setattr(Env, "get", classmethod(lambda cls, id, version=None: office))

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        if request.url.path.endswith("/website/session"):
            return httpx.Response(200, json={"cookies": [_SESSION], "local_storage": []})
        if request.url.path.endswith("/agentenv"):
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


@pytest.mark.asyncio
@pytest.mark.parametrize("card_name", ["shop", "Shop"], ids=["card-named-as-served", "card-named-otherwise"])
async def test_a_browser_in_a_multi_envs_computer_is_signed_in_with_the_sites_session_for_each_of_its_names(
        served, card_name):
    context = TaskStepContext(deployed_envs=[_office(card_name=card_name)], metadata={})

    await BrowserSignInTaskStep(id="in", version=None, env_id="office", site="shop-site", user="alex").execute(context)

    session, add = served
    assert str(session.url) == f"{_GATEWAY}/svc/shop/agentenv/ext/website/session"
    assert (json.loads(session.content), session.headers["authorization"]) == ({"user": "alex"}, "Bearer gateway-token")
    assert (str(add.url), add.headers["authorization"]) == (f"{_DESKTOP}/svc/chromium/agentenv", "Bearer desktop-token")
    body = json.loads(add.content)
    unplaced = {key: value for key, value in _SESSION.items() if key != "path"}
    assert body["method"] == "data/add" and body["params"]["parts"] == [{"kind": "data", "data": {
        "cookies": [{**unplaced, "url": "https://shop.test/"}, {**unplaced, "path": "/", "domain": ".shop.test"}],
        "local_storage": []}}]
    assert context.metadata["browser_steps"]["in"] == {
        "env_id": "office", "browser": None, "site": "shop-site", "user": "alex", "cookies": ["agentenv_session_shop"],
        "names": ["shop.test", "*.shop.test"]}


@pytest.mark.asyncio
@pytest.mark.parametrize(("stored", "names"), [(["shop.test"], ["shop.test"]), ([], ["shop.example"])],
                         ids=["the-names-it-stores", "the-names-its-card-declares"])
async def test_a_website_a_plugin_deployed_alone_is_signed_in_by_the_names_it_stores_else_those_its_card_declares(
        served, monkeypatch, stored, names):
    """A website deployed alone, its card its own and no frontend, which deployed_sites doesn't route, beside the
    desktop deployed alone."""
    website = {"uri": "urn:agentenv:website/v1", "params": {
        "endpoint": "/agentenv/ext/website", "domains": ["shop.example"],
        "methods": {"session": {"endpoint": "/agentenv/ext/website/session"}}}}
    site = DeployedEnv(env_id="shop-site", env_version=1, env_provider_type="plugin", instance_id="inst-shop",
                       environment_card_url=f"{_SITE}/.well-known/agent-env.json",
                       environment_card={"name": "shop", "url": "/agentenv", "capabilities": {"extensions": [website]}})
    desktop = DeployedSandboxEnv(
        env_id="desktop-computer", env_version=1, sandbox_id="sb-desktop", sandbox_type="local",
        control_token="desktop-token", environment_card_url=f"{_DESKTOP}/.well-known/agent-env.json",
        environment_card={"name": "desktop", "children_environments": [_browser_card()]})
    envs = {"shop-site": WebsiteEnv(id="shop-site", version=1, server_docker_image_artifact=MagicMock(),
                                    environment_name="shop", domains=stored),
            "desktop-computer": SimpleNamespace(website_envs=[])}
    monkeypatch.setattr(Env, "get", classmethod(lambda cls, id, version=None: envs[id]))
    context = TaskStepContext(deployed_envs=[desktop, site], metadata={})

    await BrowserSignInTaskStep(id="in", version=None, env_id="desktop-computer", site="shop-site",
                                user="alex").execute(context)

    session, add = served
    assert str(session.url) == f"{_SITE}/agentenv/ext/website/session"
    assert (str(add.url), add.headers["authorization"]) == (f"{_DESKTOP}/svc/chromium/agentenv", "Bearer desktop-token")
    [cookie] = json.loads(add.content)["params"]["parts"][0]["data"]["cookies"]
    assert cookie["url"] == f"https://{names[0]}/"
    assert context.metadata["browser_steps"]["in"]["names"] == names


@pytest.mark.asyncio
async def test_any_browser_step_reaches_a_browser_in_a_multi_envs_computer_with_the_computers_token(served):
    cookie = {"name": "theme", "value": "dark", "url": "https://shop.test/"}

    await BrowserSetCookiesTaskStep(id="c", version=None, env_id="office", cookies=[cookie]).execute(
        TaskStepContext(deployed_envs=[_office()], metadata={}))

    (request,) = served
    assert str(request.url) == f"{_DESKTOP}/svc/chromium/agentenv/ext/browser/cookies/set"
    assert request.headers["authorization"] == "Bearer desktop-token"


@pytest.mark.asyncio
@pytest.mark.parametrize(("office", "site", "problem"), [
    (_office(session=False), "shop-site", "site 'shop-site' serves no website/v1 session"),
    (_office(), "blog-site", "site 'blog-site' isn't deployed in this run"),
], ids=["no-session", "not-deployed"])
async def test_a_site_that_cant_sign_in_fails_the_step_before_the_browser_is_touched(served, office, site, problem):
    with pytest.raises(RuntimeError, match=problem):
        await BrowserSignInTaskStep(id="in", version=None, env_id="office", site=site, user="alex").execute(
            TaskStepContext(deployed_envs=[office], metadata={}))
    assert served == []


def test_each_cookie_is_placed_on_each_name_the_site_is_reached_by():
    cookie = {"name": "s", "value": "v", "path": "/app", "max_age": 60, "domain": "ignored"}
    assert scoped([cookie], ["shop.test", "*.shop.test", "notes.localhost"]) == [
        {"name": "s", "value": "v", "max_age": 60, "url": "https://shop.test/app"},
        {"name": "s", "value": "v", "path": "/app", "max_age": 60, "domain": ".shop.test"},
        {"name": "s", "value": "v", "max_age": 60, "url": "http://notes.localhost/app"}]


def test_the_step_round_trips_and_checks_its_fields(monkeypatch):
    step = BrowserSignInTaskStep(id="in", version=None, env_id="office", site="shop-site", user="alex", browser="chromium")
    data = step.to_dict()
    assert get_task_step_registry()["browser_sign_in"].from_dict(data).to_dict() == data
    assert BrowserSignInTaskStep(id="in", version=None, env_id="office", site="", user="").config_problems() == [
        "step 'in': site must be a website env's id",
        "step 'in': user must be a non-empty string; browser_delete_cookies signs out"]
    monkeypatch.setattr(sign_in, "stored_env", lambda id: ComputerEnv.__new__(ComputerEnv))
    monkeypatch.setattr("agent_env.task_step.task_steps.browser.base.stored_env", lambda id: None)
    assert "site 'shop-site' is a ComputerEnv, not a website env" in step.preflight()[-1]


def test_a_multi_env_deploys_its_computers_browsers():
    app = ComputerAppEnv.__new__(ComputerAppEnv)
    app.environment_name, app.declaration = "chromium", SimpleNamespace(browser=True)
    computer = ComputerEnv.__new__(ComputerEnv)
    computer.composed_apps = lambda: [app]
    office = MultiEnv.__new__(MultiEnv)
    office._computers, office.browser_envs, office.website_envs = SimpleNamespace(envs=[computer]), None, []
    assert stored_browsers(office) == ["chromium"]
