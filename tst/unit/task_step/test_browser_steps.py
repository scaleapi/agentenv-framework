"""The browser task steps: each calls its browser/v1 operation on the env's browser with the control token, the tab
steps pick the tab by its URL as the browser shows it, a browser whose card lacks the operation is refused before any
call, preflight refuses an env with no browser or none of the name given, and fields that don't fit are config
problems naming no cookie's value. Only the HTTP boundary is mocked."""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import httpx
import pytest

from agent_env.env.env import DeployedSandboxEnv
from agent_env.env.envs.computer.app_env import ComputerAppEnv
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.store.base import NotFoundError
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.registry import get_task_step_registry
from agent_env.task_step import preflight
from agent_env.task_step.task_steps.browser.contexts import BrowserLeaseContextTaskStep, BrowserReleaseContextTaskStep
from agent_env.task_step.task_steps.browser.cookies import BrowserDeleteCookiesTaskStep, BrowserSetCookiesTaskStep
from agent_env.task_step.task_steps.browser.local_storage import (BrowserClearLocalStorageTaskStep,
                                                                   BrowserSetLocalStorageTaskStep)
from agent_env.task_step.task_steps.browser.tabs import (BrowserActivateTabTaskStep, BrowserCloseTabTaskStep,
                                                         BrowserOpenTabTaskStep, normalized)

_ENV = "http://localhost:41000"
_PREFIX = "/svc/chromium/agentenv/ext/browser"
_OPS = ("cookies/set", "cookies/delete", "local_storage/set", "local_storage/clear", "tabs/open", "tabs/list",
        "tabs/close", "tabs/activate", "contexts/lease", "contexts/grant", "contexts/release")
_TABS = [{"id": "A", "url": "https://shop.test/", "active": True}, {"id": "B", "url": "https://shop.test/orders?page=2",
                                                                      "active": False}]


@pytest.mark.asyncio
@pytest.mark.parametrize(("step", "op", "body", "recorded"), [
    (BrowserSetCookiesTaskStep(id="s", version=None, env_id="computer", cookies=[
        {"name": "sid", "value": "ada", "url": "https://shop.test/", "max_age": 3600}]),
     "cookies/set", {"cookies": [{"name": "sid", "value": "ada", "url": "https://shop.test/", "max_age": 3600}]}, None),
    (BrowserDeleteCookiesTaskStep(id="s", version=None, env_id="computer", name="sid", domain=".shop.test"),
     "cookies/delete", {"name": "sid", "domain": ".shop.test"}, {"deleted": 2}),
    (BrowserDeleteCookiesTaskStep(id="s", version=None, env_id="computer"), "cookies/delete", {}, {"deleted": 2}),
    (BrowserSetLocalStorageTaskStep(id="s", version=None, env_id="computer", origin="https://shop.test",
                                    items={"cart": "3", "theme": "dark"}),
     "local_storage/set", {"origins": [{"origin": "https://shop.test", "localStorage": [
         {"name": "cart", "value": "3"}, {"name": "theme", "value": "dark"}]}]}, None),
    (BrowserClearLocalStorageTaskStep(id="s", version=None, env_id="computer", origin="https://shop.test"),
     "local_storage/clear", {"origins": ["https://shop.test"]}, None),
    (BrowserClearLocalStorageTaskStep(id="s", version=None, env_id="computer"), "local_storage/clear", {}, None),
    (BrowserOpenTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/help", background=True),
     "tabs/open", {"url": "https://shop.test/help", "background": True}, {"tab": "C"}),
    (BrowserOpenTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/help"),
     "tabs/open", {"url": "https://shop.test/help"}, {"tab": "C"}),
])
async def test_each_step_calls_its_operation_on_the_browser_with_the_control_token(monkeypatch, step, op, body,
                                                                                   recorded):
    sent = _mock(monkeypatch)
    context = TaskStepContext(deployed_envs=[_env()], metadata={})

    await step.execute(context)

    (request,) = sent
    assert (str(request.url), json.loads(request.content)) == (f"{_ENV}{_PREFIX}/{op}", body)
    assert request.headers["authorization"] == "Bearer control"
    assert context.metadata.get("browser_steps", {}).get("s") == (
        None if recorded is None else {"env_id": "computer", "browser": None, **recorded})


@pytest.mark.asyncio
@pytest.mark.parametrize(("step_class", "op"), [(BrowserCloseTabTaskStep, "tabs/close"),
                                                (BrowserActivateTabTaskStep, "tabs/activate")])
@pytest.mark.parametrize(("url", "tab"), [("HTTPS://Shop.test:443", "A"), ("https://shop.test/orders?page=2", "B")])
async def test_a_tab_step_acts_on_the_one_tab_at_its_url_as_the_browser_shows_it(monkeypatch, step_class, op, url,
                                                                                tab):
    sent = _mock(monkeypatch)
    context = TaskStepContext(deployed_envs=[_env()], metadata={})

    await step_class(id="s", version=None, env_id="computer", url=url).execute(context)

    listed, acted = sent
    assert str(listed.url) == f"{_ENV}{_PREFIX}/tabs/list"
    assert (str(acted.url), json.loads(acted.content)) == (f"{_ENV}{_PREFIX}/{op}", {"tab": tab})
    assert context.metadata["browser_steps"]["s"]["tab"] == tab


@pytest.mark.asyncio
@pytest.mark.parametrize(("tabs", "url", "says"), [
    (_TABS, "https://shop.test/orders", "no tabs are at https://shop.test/orders in the browser of env 'computer'; "
                                        "the tabs open are at ['https://shop.test/', 'https://shop.test/orders?page=2']"),
    ([_TABS[0], {**_TABS[0], "id": "C"}], "https://shop.test/", "2 tabs are at https://shop.test/ in the browser"),
])
async def test_no_tab_or_several_at_a_tab_steps_url_fail_it_naming_the_tabs_open(monkeypatch, tabs, url, says):
    sent = _mock(monkeypatch, tabs=tabs)

    with pytest.raises(RuntimeError, match=re.escape(says)):
        await BrowserCloseTabTaskStep(id="s", version=None, env_id="computer", url=url).execute(
            TaskStepContext(deployed_envs=[_env()], metadata={}))
    assert [str(request.url) for request in sent] == [f"{_ENV}{_PREFIX}/tabs/list"]


@pytest.mark.asyncio
async def test_a_lease_step_leases_and_grants_and_a_release_step_ends_it(monkeypatch):
    sent = _mock(monkeypatch)
    context = TaskStepContext(deployed_envs=[_env()], metadata={})

    await BrowserLeaseContextTaskStep(id="l", version=None, env_id="computer", label="buyer",
                                      grants=["ada", "bob"]).execute(context)
    await BrowserReleaseContextTaskStep(id="r", version=None, env_id="computer", label="buyer").execute(context)

    assert [(request.url.path.removeprefix(f"{_PREFIX}/"), json.loads(request.content)) for request in sent] == [
        ("contexts/lease", {"label": "buyer"}), ("contexts/grant", {"context": "buyer", "user": "ada"}),
        ("contexts/grant", {"context": "buyer", "user": "bob"}), ("contexts/release", {"context": "buyer"})]
    assert context.metadata["browser_steps"] == {
        "l": {"env_id": "computer", "browser": None, "label": "buyer", "id": "C1", "grants": ["ada", "bob"]},
        "r": {"env_id": "computer", "browser": None, "label": "buyer", "released": True}}


@pytest.mark.asyncio
async def test_a_step_that_names_a_context_acts_in_it_and_a_browser_that_leases_none_is_refused(monkeypatch):
    sent = _mock(monkeypatch)
    context = TaskStepContext(deployed_envs=[_env()], metadata={})

    await BrowserSetCookiesTaskStep(id="s", version=None, env_id="computer", context="buyer", cookies=[
        {"name": "sid", "value": "ada", "url": "https://shop.test/"}]).execute(context)
    await BrowserCloseTabTaskStep(id="c", version=None, env_id="computer", url="https://shop.test/",
                                  context="buyer").execute(context)
    old = _env(ops=[op for op in _OPS if not op.startswith("contexts/")])
    with pytest.raises(RuntimeError, match="serves no browser/v1 contexts/lease"):
        await BrowserOpenTabTaskStep(id="o", version=None, env_id="computer", url="https://shop.test/",
                                     context="buyer").execute(TaskStepContext(deployed_envs=[old], metadata={}))

    assert json.loads(sent[0].content)["context"] == "buyer"
    assert sent[1].url.params.get("context") == "buyer" and json.loads(sent[2].content) == {"tab": "A",
                                                                                           "context": "buyer"}
    assert len(sent) == 3 and context.metadata["browser_steps"]["c"]["context"] == "buyer"


@pytest.mark.asyncio
async def test_a_browser_whose_card_lacks_the_operation_or_isnt_there_is_refused_before_any_call(monkeypatch):
    sent = _mock(monkeypatch)
    old = _env(ops=[op for op in _OPS if op != "tabs/open"])
    step = BrowserOpenTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/")

    with pytest.raises(RuntimeError, match="'chromium' in env 'computer' serves no browser/v1 tabs/open"):
        await step.execute(TaskStepContext(deployed_envs=[old], metadata={}))
    with pytest.raises(RuntimeError, match="has no website browser named 'firefox'"):
        await BrowserOpenTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/",
                                     browser="firefox").execute(TaskStepContext(deployed_envs=[_env()], metadata={}))
    with pytest.raises(RuntimeError, match="isn't in context.deployed_envs"):
        await step.execute(TaskStepContext(deployed_envs=[], metadata={}))
    assert sent == []


_NO_BROWSER = "env 'computer' has no website browser: browser steps act on a computer env's browser app, or a browser deployed alone"


@pytest.mark.parametrize(("stored", "browser", "problem"), [
    ("desktop", None, None),
    ("alone", None, None),
    ("alone", "chromium", None),
    ("two", "brave", None),
    ("missing", None, None),
    ("plain", None, _NO_BROWSER),
    ("website", None, _NO_BROWSER),
    ("desktop", "firefox", "env 'computer' has no website browser named 'firefox' (it has ['chromium'])"),
    ("two", None, "env 'computer' has 2 website browsers, ['chromium', 'brave']: name one as browser"),
])
def test_preflight_refuses_an_env_with_no_browser_none_of_its_name_or_several_unnamed(monkeypatch, stored, browser,
                                                                                     problem):
    def get(env_id, version=None):
        if stored == "missing":
            raise NotFoundError(env_id)
        return _stored(stored)

    monkeypatch.setattr(preflight, "Env", SimpleNamespace(get=get))
    step = BrowserActivateTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/",
                                      browser=browser)

    assert step.preflight() == ([] if problem is None else [f"step 's': {problem}"])


@pytest.mark.parametrize(("step", "problems"), [
    (BrowserSetCookiesTaskStep(id="s", version=None, env_id="", cookies=[{"name": "sid", "value": "hunter2",
                                                                          "colour": "red"}]),
     ["step 's': env_id must be a non-empty string", "step 's': cookies[0]: colour: Extra inputs are not permitted"]),
    (BrowserSetCookiesTaskStep(id="s", version=None, env_id="computer", cookies=[]),
     ["step 's': cookies must be a non-empty list"]),
    (BrowserDeleteCookiesTaskStep(id="s", version=None, env_id="computer", domain=""),
     ["step 's': domain, when given, must be a non-empty string"]),
    (BrowserSetLocalStorageTaskStep(id="s", version=None, env_id="computer", origin="https://shop.test/cart",
                                    items={"cart": 3}),
     ["step 's': origin: 'https://shop.test/cart' is a URL, not an origin: give only its scheme, host and port",
      "step 's': items must be a non-empty map of keys to string values"]),
    (BrowserClearLocalStorageTaskStep(id="s", version=None, env_id="computer", origin="shop.test"),
     ["step 's': origin: 'shop.test' isn't an http or https origin"]),
    (BrowserOpenTabTaskStep(id="s", version=None, env_id="computer", url="/orders", background="yes"),
     ["step 's': url '/orders' isn't absolute: give its scheme and host, such as https://shop.test/",
      "step 's': background must be true or false"]),
    (BrowserCloseTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test:port/", browser=""),
     ["step 's': browser, when given, must be a non-empty string",
      "step 's': url 'https://shop.test:port/' isn't a URL: Port could not be cast to integer value as 'port'"]),
    (BrowserOpenTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/", context="Buyer"),
     ["step 's': context: a context's label is lowercase letters, digits and '-', not at either end, at most 40 "
      "characters, not 'Buyer'"]),
    (BrowserLeaseContextTaskStep(id="s", version=None, env_id="computer", label="agent-ada-1", grants=[""]),
     ["step 's': label: a label starting 'agent-' names a context an agent made, so a task's can't: 'agent-ada-1'",
      "step 's': grants, when given, must be a list of agent names"]),
    (BrowserReleaseContextTaskStep(id="s", version=None, env_id="computer", label=""),
     ["step 's': label: a context's label is lowercase letters, digits and '-', not at either end, at most 40 "
      "characters, not ''"]),
])
def test_fields_that_dont_fit_are_config_problems_and_a_cookies_value_never_shows(step, problems):
    found = step.config_problems()

    assert found == problems
    assert "hunter2" not in " ".join(found)


@pytest.mark.parametrize("step", [
    BrowserSetCookiesTaskStep(id="s", version=None, env_id="computer", browser="chromium",
                              cookies=[{"name": "sid", "value": "ada", "url": "https://shop.test/"}]),
    BrowserDeleteCookiesTaskStep(id="s", version=None, env_id="computer", name="sid", domain="shop.test", path="/a"),
    BrowserSetLocalStorageTaskStep(id="s", version=None, env_id="computer", origin="https://shop.test",
                                   items={"cart": "3"}),
    BrowserClearLocalStorageTaskStep(id="s", version=None, env_id="computer"),
    BrowserOpenTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/", background=True),
    BrowserCloseTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/"),
    BrowserActivateTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/"),
    BrowserActivateTabTaskStep(id="s", version=None, env_id="computer", url="https://shop.test/", context="buyer"),
    BrowserLeaseContextTaskStep(id="s", version=None, env_id="computer", label="buyer", grants=["ada"]),
    BrowserReleaseContextTaskStep(id="s", version=None, env_id="computer", label="buyer"),
])
def test_each_step_is_registered_and_round_trips(step):
    registered = get_task_step_registry()[step.type]

    assert registered is type(step) and registered.from_dict(step.to_dict()).to_dict() == step.to_dict()
    assert ("context" in step.to_dict()) == (getattr(step, "context_label", None) is not None)


@pytest.mark.parametrize(("url", "shown"), [
    ("HTTPS://Shop.TEST", "https://shop.test/"),
    ("http://shop.test:80/a?b=1#c", "http://shop.test/a?b=1#c"),
    ("https://shop.test:8443", "https://shop.test:8443/"),
    ("http://[::1]:8080", "http://[::1]:8080/"),
    ("https://Bücher.test/a", "https://xn--bcher-kva.test/a"),
    ("https://faß.test/", "https://xn--fa-hia.test/"),
    ("https://fass.test/", "https://fass.test/"),
    ("https://☃.net/", "https://xn--n3h.net/"),
    ("https://ｂüｃher．test/", "https://xn--bcher-kva.test/"),
    ("about:blank", "about:blank"),
])
def test_a_url_is_normalized_as_the_browser_shows_it(url, shown):
    assert normalized(url) == shown


def _env(ops=_OPS) -> DeployedSandboxEnv:
    """A browser computer's deployment, its browser app chromium among its children, serving ``ops``."""
    browser = {"uri": "urn:agentenv:browser/v1", "params": {"endpoint": _PREFIX, "methods": {
        op: {"endpoint": f"{_PREFIX}/{op}", **({"method": "GET"} if op == "tabs/list" else {})} for op in ops}}}
    return DeployedSandboxEnv(
        env_id="computer", env_version=1, sandbox_id="sb", sandbox_type="local", control_token="control",
        environment_card_url=f"{_ENV}/.well-known/agent-env.json",
        environment_card={"name": "chromium-driver", "capabilities": {"operations": ["reset"]},
                          "children_environments": [{"name": "chromium",
                                                     "capabilities": {"extensions": [browser]}}]})


def _stored(kind: str):
    """A stored env without its constructor: a desktop with chromium, the browser alone, a desktop with two browsers
    or none, or a website env."""
    def app(name, browser=True):
        made = ComputerAppEnv.__new__(ComputerAppEnv)
        made.environment_name, made.declaration = name, SimpleNamespace(browser=browser)
        return made

    if kind == "alone":
        return app("chromium")
    if kind == "website":
        return SimpleNamespace(type="website")
    desktop = ComputerEnv.__new__(ComputerEnv)
    desktop.apps = {"desktop": [app("chromium"), app("notes", browser=None)], "two": [app("chromium"), app("brave")],
                    "plain": [app("notes", browser=None)]}[kind]
    return desktop


def _mock(monkeypatch, *, tabs=_TABS) -> list[httpx.Request]:
    """Answer the browser's operations as a browser would, ``tabs`` open; return the requests."""
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        op = request.url.path.removeprefix(f"{_PREFIX}/")
        return httpx.Response(200, json={"cookies/delete": {"deleted": 2}, "local_storage/clear": {"origins": []},
                                         "tabs/open": {"tab": "C"}, "tabs/list": {"tabs": tabs},
                                         "contexts/lease": {"context": "buyer", "id": "C1"},
                                         "contexts/release": {"released": True}}.get(op, {}))

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent
