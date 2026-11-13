"""browser/v1's contexts: a lease by its label, the same one when leased again, its downloads in its own folder; labels
a task can and can't use; a context an agent makes leased to it as agent-<user>-<n>, and one made unseen to nobody;
each context's tabs, cookies, local storage and downloads its own; a release that closes its tabs as agentenv's and
deletes its folder; and what a load may name."""
from __future__ import annotations

import asyncio
from pathlib import PurePosixPath

import pytest
from starlette.requests import Request

from agentenv_protocol import website_browser
from agentenv_protocol.computer import Launch, Ready, app, linux
from agentenv_protocol.computer.errors import InvalidArgument, PermissionDenied, TargetUnavailable
from agentenv_protocol.computer.platform import Platform
from agentenv_protocol.website_browser import BrowserControl, CookieToSet, WebsiteBrowserEnvironment, data
from agentenv_protocol.website_browser.contexts import Contexts, label_problem
from agentenv_protocol.website_browser.environment import _check_contexts
from agentenv_protocol.website_browser.shapes import StateToLoad


@linux(launch=Launch(command=["chromium/chrome"], window={"class": "Chromium"}, ready=Ready(window=True)))
class Browser(WebsiteBrowserEnvironment):
    version = "156.0.8078.4"


@pytest.fixture(autouse=True)
def browser_home(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux"))
    monkeypatch.setattr(website_browser.environment, "_shm_bytes", lambda path="/dev/shm": 1 << 30)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.setenv("ENVIRONMENT_NAME", "chromium")
    return tmp_path / "home"


class StubDevTools:
    """A browser connection that makes a context ``C<n>`` for each one asked for, answers each other call from
    ``answers`` by method, records every call, and hands the control the events a test sends."""

    def __init__(self, answers=None):
        self.calls, self.answers, self.listeners, self.made = [], answers or {}, [], 0

    def listen(self, listener):
        self.listeners.append(listener)

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        self.calls.append((method, params or {}, session_id))
        if method == "Target.createBrowserContext":
            self.made += 1
            return {"browserContextId": f"C{self.made}"}
        answer = self.answers.get(method, {})
        return answer(params, session_id) if callable(answer) else answer

    def send(self, method, params, session=None):
        for listener in self.listeners:
            listener({"method": method, "params": params, **({"sessionId": session} if session else {})})


def _sent(devtools, method):
    return [params for called, params, _ in devtools.calls if called == method]


def _contexts(home, answers=None):
    devtools = StubDevTools(answers)
    control = BrowserControl(devtools, "agent")
    contexts = Contexts(devtools, control, home=lambda: home)
    control.download_folders = contexts.holding
    return devtools, control, contexts


def _tab(devtools, target, context, url="https://shop.test/"):
    devtools.send("Target.targetCreated", {"targetInfo": {"targetId": target, "type": "page", "url": url,
                                                          "title": target, "browserContextId": context}})


@pytest.mark.asyncio
async def test_a_lease_is_a_context_of_its_own_by_its_label_saving_downloads_in_its_folder_and_found_again(
        browser_home):
    devtools, control, contexts = _contexts(browser_home)

    lease = await contexts.lease("buyer")
    again = await contexts.lease("buyer")

    assert (lease.label, lease.id, lease.made_by, lease.grants) == ("buyer", "C1", "task", [])
    assert again is lease and _sent(devtools, "Target.createBrowserContext") == [{"disposeOnDetach": False}]
    assert _sent(devtools, "Browser.setDownloadBehavior") == [{
        "behavior": "allow", "browserContextId": "C1", "eventsEnabled": True,
        "downloadPath": str(browser_home / "Contexts" / "buyer" / "Downloads")}]
    assert (browser_home / "Contexts" / "buyer" / "Downloads").is_dir()
    assert control.other_contexts == {"C1": "buyer"}
    assert contexts.id_of("buyer") == "C1" and contexts.id_of(None) is None
    with pytest.raises(TargetUnavailable, match="no context 'seller' is leased"):
        contexts.id_of("seller")


@pytest.mark.parametrize("label, by_task, problem", [
    ("buyer-2", True, None), ("b", True, None), ("x" * 40, True, None), ("agent-ada-1", False, None),
    ("Buyer", True, "lowercase"), ("-buyer", True, "lowercase"), ("buyer-", True, "lowercase"),
    ("buy_er", True, "lowercase"), ("", True, "lowercase"), ("x" * 41, True, "lowercase"), (None, True, "lowercase"),
    ("agent-ada-1", True, "names a context an agent made"),
])
def test_a_label_is_lowercase_letters_digits_and_dashes_and_a_tasks_never_starts_agent(label, by_task, problem):
    found = label_problem(label, by_task=by_task)
    assert found is None if problem is None else problem in found


@pytest.mark.asyncio
async def test_a_task_cant_lease_an_agents_label_and_grants_are_recorded_once_each(browser_home):
    _, _, contexts = _contexts(browser_home)
    with pytest.raises(InvalidArgument, match="names a context an agent made"):
        await contexts.lease("agent-ada-1")
    await contexts.lease("buyer", grants=("ada",))

    contexts.grant("buyer", "bob")
    contexts.grant("buyer", "ada")

    assert contexts.get("buyer").record() == {"context": "buyer", "id": "C1", "made_by": "task",
                                              "grants": ["ada", "bob"]}
    with pytest.raises(TargetUnavailable):
        contexts.grant("seller", "ada")
    with pytest.raises(InvalidArgument, match="non-empty"):
        contexts.grant("buyer", "")


@pytest.mark.asyncio
async def test_a_context_an_agent_makes_is_leased_to_it_and_one_made_unseen_to_nobody(browser_home):
    devtools, control, contexts = _contexts(browser_home, {
        "Target.getBrowserContexts": {"browserContextIds": ["X1", "X2", "X9"]}})

    first, second = contexts.made_by_agent("ada", "X1"), contexts.made_by_agent("ada", "X2")
    named = contexts.made_by_agent("Ada Lovelace!", "X3")
    long = contexts.made_by_agent("a-very-long-user-name-that-will-not-fit-in-a-label", "X4")
    [unseen] = await contexts.find_unseen()

    assert [first.label, second.label, named.label] == ["agent-ada-1", "agent-ada-2", "agent-ada-lovelace-1"]
    assert (first.made_by, first.grants) == ("ada", ["ada"]) and contexts.made_by_agent("ada", "X1") is first
    assert len(long.label) <= 40 and long.label.startswith("agent-a-very-long") and long.label.endswith("-1")
    assert (unseen.label, unseen.id, unseen.made_by, unseen.grants) == ("agent-unknown-1", "X9", "unknown", [])
    assert set(control.other_contexts) == {"X1", "X2", "X3", "X4", "X9"}


@pytest.mark.asyncio
async def test_only_the_agent_that_made_a_context_may_dispose_it(browser_home):
    _, _, contexts = _contexts(browser_home)
    await contexts.lease("buyer")
    contexts.made_by_agent("ada", "X1")

    assert contexts.may_dispose("ada", "X1") is None and contexts.may_dispose("ada", "elsewhere") is None
    assert "another agent's" in contexts.may_dispose("bob", "X1")
    assert "the task's" in contexts.may_dispose("ada", "C1")
    contexts.disposed("X1")
    assert contexts.get("agent-ada-1") is None


@pytest.mark.asyncio
async def test_each_contexts_tabs_cookies_and_local_storage_are_its_own(browser_home):
    devtools, control, contexts = _contexts(browser_home, {"Storage.getCookies": {"cookies": []}})
    await contexts.lease("buyer")
    _tab(devtools, "D1", "DEFAULT")
    _tab(devtools, "L1", "C1", "https://cart.test/")

    default, leased = await control.tab_list(), await control.tab_list("C1")
    every = await control.tab_list(every=True)
    await control.cookies(context="C1")
    await control.set_cookies([CookieToSet(name="sid", value="v", domain="cart.test")], "C1")

    assert [(tab["id"], tab["context"]) for tab in default] == [("D1", None)]
    assert [(tab["id"], tab["context"]) for tab in leased] == [("L1", "buyer")]
    assert [tab["id"] for tab in every] == ["D1", "L1"]
    assert _sent(devtools, "Storage.getCookies") == [{"browserContextId": "C1"}]
    assert _sent(devtools, "Storage.setCookies")[0]["browserContextId"] == "C1"
    assert control.origins == {"https://shop.test"} and control.origins_of("C1") == {"https://cart.test"}
    with pytest.raises(InvalidArgument, match="no tab D1 is open in that context"):
        await control.activate("D1", "C1")
    with pytest.raises(InvalidArgument, match="no tab L1 is open; list the tabs"):
        await control.close("L1")


@pytest.mark.asyncio
async def test_a_leases_downloads_are_its_own_by_the_tab_that_began_them_or_the_folder_they_land_in(browser_home):
    devtools, control, contexts = _contexts(browser_home)
    lease = await contexts.lease("buyer")
    _tab(devtools, "L1", "C1")
    landed = lease.downloads(browser_home) / "invoice.pdf"
    landed.write_bytes(b"%PDF")

    devtools.send("Browser.downloadWillBegin", {"guid": "g1", "url": "https://shop.test/i", "frameId": "L1",
                                                "suggestedFilename": "invoice.pdf"})
    devtools.send("Browser.downloadWillBegin", {"guid": "g2", "url": "https://shop.test/j", "frameId": "iframe",
                                                "suggestedFilename": "invoice.pdf"})
    devtools.send("Browser.downloadProgress", {"guid": "g2", "state": "completed", "filePath": str(landed)})

    assert control.download_list() == []
    assert [(d["guid"], d["context"]) for d in control.download_list("C1")] == [("g1", "buyer"), ("g2", "buyer")]
    assert control.download("g1", "C1")["context"] == "buyer"
    with pytest.raises(InvalidArgument, match="no download g1 has begun; list the downloads"):
        control.download("g1")
    assert data.finished_downloads(browser_home, lease.under()) == ["Contexts/buyer/Downloads/invoice.pdf"]


@pytest.mark.asyncio
async def test_a_release_closes_its_tabs_as_agentenvs_deletes_its_folder_and_does_nothing_again(browser_home):
    devtools, control, contexts = _contexts(browser_home)
    lease = await contexts.lease("buyer")
    _tab(devtools, "L1", "C1")

    assert await contexts.release("buyer") is True
    devtools.send("Target.targetDestroyed", {"targetId": "L1"})

    assert _sent(devtools, "Target.disposeBrowserContext") == [{"browserContextId": "C1"}]
    assert not lease.folder(browser_home).exists() and contexts.get("buyer") is None
    assert [(tab["id"], tab["closed_by"]) for tab in (await control.record())["closed_tabs"]] == [("L1", "agentenv")]
    assert (await control.record())["closed_tabs"][0]["context"] == "buyer"
    assert await contexts.release("buyer") is False
    assert await control.tab_list() == []


@pytest.mark.asyncio
async def test_a_label_released_and_leased_again_at_once_keeps_the_new_leases_folder(browser_home):
    _, _, contexts = _contexts(browser_home)
    await contexts.lease("buyer")

    released, again = await asyncio.gather(contexts.release("buyer"), contexts.lease("buyer"))

    assert released is True and again.id == "C2" and again.downloads(browser_home).is_dir()


def _attached(devtools, target, session):
    devtools.send("Target.attachedToTarget", {"sessionId": session, "targetInfo": {"targetId": target, "type": "page"}})


@pytest.mark.asyncio
async def test_the_pages_operations_follow_a_leased_tab_brought_to_the_front_and_one_left_alone(browser_home):
    devtools, control, contexts = _contexts(browser_home)
    await contexts.lease("buyer")
    _tab(devtools, "D1", "DEFAULT")
    _tab(devtools, "L1", "C1")
    _attached(devtools, "D1", "s-d1")
    _attached(devtools, "L1", "s-l1")

    await control.activate("L1", "C1")
    page, default = await control.page_tab(), await control.active_tab()
    await control.close("D1")
    devtools.send("Target.targetDestroyed", {"targetId": "D1"})
    devtools.send("Target.detachedFromTarget", {"sessionId": "s-d1"})

    assert (page, default) == ("L1", None) and await control.active_session() == "s-l1"


@pytest.mark.asyncio
async def test_every_contexts_tabs_list_together_each_naming_its_context_the_pages_tab_the_active_one(
        browser_home, monkeypatch):
    browser = Browser()
    devtools = StubDevTools()
    devtools.listen(browser.control._event)
    browser.devtools = browser.contexts.devtools = browser.control.devtools = devtools
    monkeypatch.setattr(browser.contexts, "home", lambda: browser_home)
    await browser.browser_contexts_lease("buyer")
    _tab(devtools, "D1", "DEFAULT")
    _tab(devtools, "L1", "C1")
    await browser.control.activate("L1", "C1")

    listed = (await browser.browser_tabs_list(every=True))["tabs"]

    assert [(tab["id"], tab["context"], tab["active"]) for tab in listed] == [("D1", None, False), ("L1", "buyer", True)]
    with pytest.raises(InvalidArgument, match="a context or every, not both"):
        await browser.browser_tabs_list(context="buyer", every=True)


@pytest.mark.asyncio
async def test_a_frame_attached_to_a_leases_page_is_its_tab_s_though_it_navigates_in_a_process_of_its_own(
        browser_home):
    devtools, control, contexts = _contexts(browser_home)
    await contexts.lease("buyer")
    _tab(devtools, "L1", "C1")
    _attached(devtools, "L1", "s-l1")

    devtools.send("Page.frameAttached", {"frameId": "OOPIF", "parentFrameId": "L1"}, "s-l1")

    assert control.context_of_frame("OOPIF") == "C1" and control.context_of_frame("L1") == "C1"
    assert control.context_of_frame("elsewhere") is None


@pytest.mark.asyncio
async def test_a_label_an_agent_disposed_and_a_load_leases_again_at_once_keeps_the_new_folder(browser_home):
    _, _, contexts = _contexts(browser_home)
    old = contexts.made_by_agent("ada", "X1")
    (old.downloads(browser_home) / "old.txt").write_text("old")

    contexts.disposed("X1")
    again = await contexts.lease("agent-ada-1", made_by="ada")
    (again.downloads(browser_home) / "new.txt").write_text("new")
    await asyncio.gather(*contexts._tasks)

    assert again.id == "C1" and [path.name for path in again.downloads(browser_home).iterdir()] == ["new.txt"]
    assert sorted(path.name for path in (browser_home / "Contexts").iterdir()) == ["agent-ada-1"]


@pytest.mark.asyncio
async def test_a_load_into_an_agents_lease_finds_it_though_a_task_couldnt_lease_its_label(browser_home):
    _, _, contexts = _contexts(browser_home)
    made = contexts.made_by_agent("ada", "X1")

    assert await contexts.lease("agent-ada-1") is made and made.made_by == "ada"


@pytest.mark.asyncio
async def test_a_load_waiting_on_an_agents_lease_that_ends_meanwhile_cant_lease_its_label_as_the_tasks(browser_home):
    _, _, contexts = _contexts(browser_home)
    contexts.made_by_agent("ada", "X1")

    async with contexts._lock:
        waiting = asyncio.ensure_future(contexts.lease("agent-ada-1"))
        await asyncio.sleep(0)
        contexts.disposed("X1")
    with pytest.raises(InvalidArgument, match="names a context an agent made"):
        await waiting

    assert contexts.get("agent-ada-1") is None


@pytest.mark.asyncio
async def test_a_context_made_unseen_keeps_the_origins_its_pages_loaded_before_it_was_found(browser_home):
    devtools, control, contexts = _contexts(browser_home, {
        "Target.getBrowserContexts": {"browserContextIds": ["X9"]}})
    _tab(devtools, "U1", "X9", "https://cart.test/")

    [unseen] = await contexts.find_unseen()

    assert control.origins_of(unseen.id) == {"https://cart.test"}


@pytest.mark.asyncio
async def test_a_download_an_iframe_began_is_its_tabs_contexts(browser_home):
    devtools, control, contexts = _contexts(browser_home)
    await contexts.lease("buyer")
    _tab(devtools, "L1", "C1")
    _attached(devtools, "L1", "s-l1")
    devtools.send("Page.frameNavigated", {"frame": {"id": "F1", "parentId": "L1", "url": "https://pay.test/"}},
                  "s-l1")

    devtools.send("Browser.downloadWillBegin", {"guid": "g1", "url": "https://pay.test/r", "frameId": "F1",
                                                "suggestedFilename": "receipt.pdf"})

    assert control.download_list() == [] and [d["guid"] for d in control.download_list("C1")] == ["g1"]


@pytest.mark.asyncio
async def test_the_browser_ending_ends_every_lease_and_deletes_their_folders(browser_home):
    _, _, contexts = _contexts(browser_home)
    await contexts.lease("buyer")
    contexts.made_by_agent("ada", "X1")

    contexts.ended()

    assert list(contexts) == [] and not (browser_home / "Contexts").exists()
    assert (await contexts.lease("buyer")).id == "C2"


def test_a_load_refuses_a_label_no_lease_can_have_and_downloads_for_a_context_it_doesnt_lease():
    state = StateToLoad.model_validate({"contexts": {"buyer": {}, "agent-ada-1": {"made_by": "ada"}}})
    files = data.Load(context_files={"buyer": {}, "seller": {}})

    _check_contexts(state, data.Load(context_files={"buyer": {}}), set())
    _check_contexts(state, files, {"seller"})
    with pytest.raises(InvalidArgument, match="seller"):
        _check_contexts(state, files, set())
    with pytest.raises(InvalidArgument, match="names a context an agent made"):
        _check_contexts(StateToLoad.model_validate({"contexts": {"agent-ada-1": {}}}), data.Load(), set())


def test_a_loads_files_under_a_leases_downloads_are_its_and_any_other_under_contexts_none():
    assert data.context_of(PurePosixPath("Contexts/buyer/Downloads/a.pdf")) == "buyer"
    assert data.context_of(PurePosixPath("Contexts/buyer/Downloads/sub/a.pdf")) == "buyer"
    assert data.context_of(PurePosixPath("Contexts/buyer/Downloads")) is None
    assert data.context_of(PurePosixPath("Contexts/buyer/profile/a")) is None
    assert data.context_of(PurePosixPath("Downloads/a.pdf")) is None


@pytest.mark.asyncio
async def test_the_ops_lease_grant_list_and_release_and_name_a_context_by_its_label(browser_home, monkeypatch):
    browser = Browser()
    devtools = StubDevTools({"Storage.getCookies": {"cookies": []},
                             "Target.getBrowserContexts": {"browserContextIds": ["C1"]}})
    devtools.listen(browser.control._event)
    browser.devtools = browser.contexts.devtools = browser.control.devtools = devtools
    monkeypatch.setattr(browser.contexts, "home", lambda: browser_home)

    leased = await browser.browser_contexts_lease("buyer")
    await browser.browser_contexts_grant("buyer", "ada")
    _tab(devtools, "L1", "C1")
    listed = await browser.browser_contexts_list()
    tabs = await browser.browser_tabs_list(context="buyer")

    assert leased == {"context": "buyer", "id": "C1"}
    assert listed == {"contexts": [{"context": "buyer", "id": "C1", "made_by": "task", "grants": ["ada"]}]}
    assert [tab["id"] for tab in tabs["tabs"]] == ["L1"] and (await browser.browser_tabs_list())["tabs"] == []
    with pytest.raises(TargetUnavailable):
        await browser.browser_cookies_get(context="seller")
    assert await browser.browser_contexts_release("seller") == {"released": False}


@pytest.mark.asyncio
async def test_a_scoped_cdp_address_is_only_for_a_user_its_lease_is_granted_to_and_ends_with_the_lease(
        browser_home, monkeypatch):
    browser = Browser()
    devtools = StubDevTools()
    devtools.listen(browser.control._event)
    browser.devtools = browser.contexts.devtools = browser.control.devtools = devtools
    monkeypatch.setattr(browser.contexts, "home", lambda: browser_home)

    async def ready():
        return None

    monkeypatch.setattr(browser, "_until_ready", ready)
    request = Request({"type": "http", "method": "POST", "scheme": "http", "path": "/", "query_string": b"",
                       "server": ("computer.test", 80), "headers": [(b"host", b"computer.test")]})
    await browser.browser_contexts_lease("buyer")
    await browser.browser_contexts_grant("buyer", "ada")

    with pytest.raises(PermissionDenied, match="isn't granted to 'bob'"):
        await browser.browser_cdp("bob", request, context="buyer")
    with pytest.raises(TargetUnavailable, match="no context 'seller' is leased"):
        await browser.browser_cdp("ada", request, context="seller")
    scoped = await browser.browser_cdp("ada", request, context="buyer")
    token = scoped["ws_url"].split("/cdp/")[1].split("/")[0]
    found = browser.cdp_tokens.address(token)
    await browser.browser_contexts_release("buyer")

    assert (found.user, found.context) == ("ada", "buyer")
    assert browser.cdp_tokens.address(token) is None

