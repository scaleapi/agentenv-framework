"""browser/v1's typed operations on a browser's state: cookies as Playwright's storageState has them, picked for URLs as
Playwright picks them, set with a max_age on this machine's clock and deleted by name, domain and path; local storage
read and written through a page agentenv serves, so the site gets no request, for the origins named or every origin its
pages and frames loaded; downloads with their file; and input refused without its values."""
from __future__ import annotations

import asyncio
import base64
import hashlib

import pytest

from agentenv_protocol import website_browser
from agentenv_protocol.computer import Launch, Ready, app, linux
from agentenv_protocol.computer.errors import InvalidArgument
from agentenv_protocol.computer.platform import Platform
from agentenv_protocol.website_browser import (BrowserControl, CookieToSet, DevToolsError, OriginStorage,
                                               WebsiteBrowserEnvironment)
from agentenv_protocol.website_browser.control import ARRIVED, LEAVING, SERVED_PAGE
from agentenv_protocol.website_browser.environment import _parsed


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
    """A browser connection that answers each call from ``answers`` by method, records it, and hands the control the
    events a test sends."""

    def __init__(self, answers=None):
        self.calls, self.answers, self.listener = [], answers or {}, None

    def listen(self, listener):
        self.listener = listener

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        self.calls.append((method, params or {}, session_id))
        answer = self.answers.get(method, {})
        return answer(params, session_id) if callable(answer) else answer

    def send(self, method, params, session=None):
        self.listener({"method": method, "params": params, **({"sessionId": session} if session else {})})


def _sent(devtools, method):
    return [(params, session) for called, params, session in devtools.calls if called == method]


CDP_COOKIES = [
    {"name": "sid", "value": "s1", "domain": "shop.test", "path": "/", "expires": -1, "size": 5, "httpOnly": True,
     "secure": False, "session": True, "sameSite": "Strict", "priority": "Medium", "sourceScheme": "NonSecure"},
    {"name": "pref", "value": "dark", "domain": ".wiki.test", "path": "/app", "expires": 1900000000.5, "size": 8,
     "httpOnly": False, "secure": False, "session": False, "priority": "Medium"},
    {"name": "token", "value": "t1", "domain": "bank.test", "path": "/", "expires": -1, "httpOnly": False,
     "secure": True, "session": True, "sameSite": "None"},
    {"name": "token", "value": "t2", "domain": "localhost", "path": "/", "expires": -1, "httpOnly": False,
     "secure": True, "session": True, "sameSite": "Lax"},
]


@pytest.mark.asyncio
async def test_its_cookies_are_storage_states_and_a_url_gets_those_a_request_to_it_would_send():
    control = BrowserControl(StubDevTools({"Storage.getCookies": {"cookies": CDP_COOKIES}}), "agent")

    every = await control.cookies()
    assert every[0] == {"name": "sid", "value": "s1", "domain": "shop.test", "path": "/", "expires": -1,
                        "httpOnly": True, "secure": False, "sameSite": "Strict"}
    assert every[1]["sameSite"] == "Lax"
    assert [cookie["name"] for cookie in await control.cookies(["https://en.wiki.test/app/page"])] == ["pref"]
    assert await control.cookies(["https://en.wiki.test/other"]) == []
    assert [c["value"] for c in await control.cookies(["http://bank.test/", "http://localhost:8080/"])] == ["t2"]
    assert [c["value"] for c in await control.cookies(["https://bank.test/"])] == ["t1"]


@pytest.mark.asyncio
async def test_a_max_age_becomes_an_expiry_on_this_machines_clock_and_a_domain_cookie_gets_its_path(monkeypatch):
    devtools = StubDevTools()
    control = BrowserControl(devtools, "agent")
    monkeypatch.setattr(website_browser.control.time, "time", lambda: 1000.0)

    await control.set_cookies([CookieToSet(name="sid", value="v", domain="shop.test", max_age=60),
                               CookieToSet(name="s", value="v", url="https://shop.test/cart", expires=-1),
                               CookieToSet(name="e", value="v", domain=".shop.test", path="/a", expires=5000,
                                           sameSite="None", secure=True)])

    [(params, _)] = _sent(devtools, "Storage.setCookies")
    sent = params["cookies"]
    assert sent[0] == {"name": "sid", "value": "v", "domain": "shop.test", "path": "/", "expires": 1060.0,
                       "httpOnly": False, "secure": False}
    assert sent[1] == {"name": "s", "value": "v", "url": "https://shop.test/cart", "httpOnly": False, "secure": False}
    assert (sent[2]["path"], sent[2]["expires"], sent[2]["sameSite"]) == ("/a", 5000, "None")


@pytest.mark.asyncio
async def test_cookies_are_set_in_one_call_so_one_the_browser_refuses_leaves_none_set():
    def set_cookies(params, session):
        if any(cookie["name"] == "bad" for cookie in params["cookies"]):
            raise DevToolsError("Storage.setCookies: Invalid cookie fields")
        return {}

    devtools = StubDevTools({"Storage.setCookies": set_cookies})
    control = BrowserControl(devtools, "agent")
    with pytest.raises(InvalidArgument, match="refused the cookies, so it set none of them: .*Invalid cookie fields") \
            as refused:
        await control.set_cookies([CookieToSet(name="ok", value="v", domain="a.test"),
                                   CookieToSet(name="bad", value="v", domain="a.test")])
    assert refused.value.applied == "no"
    assert [[c["name"] for c in params["cookies"]] for params, _ in _sent(devtools, "Storage.setCookies")] == [
        ["ok", "bad"]]


@pytest.mark.asyncio
async def test_a_partitioned_cookie_keeps_its_key_read_set_and_deleted_beside_an_unpartitioned_one_of_its_name():
    key = {"topLevelSite": "https://top.test", "hasCrossSiteAncestor": True}
    cdp = [{"name": "chip", "value": "1", "domain": "embed.test", "path": "/", "expires": -1, "httpOnly": False,
            "secure": True, "sameSite": "None", "partitionKey": key},
           {"name": "chip", "value": "2", "domain": "embed.test", "path": "/", "expires": -1, "httpOnly": False,
            "secure": True, "sameSite": "None"}]
    devtools = StubDevTools({"Storage.getCookies": {"cookies": cdp}, "Target.createTarget": {"targetId": "h"},
                             "Target.attachToTarget": {"sessionId": "s-h"}})
    control = BrowserControl(devtools, "agent")

    read = await control.cookies()
    assert read[0]["partitionKey"] == key and "partitionKey" not in read[1]
    await control.set_cookies([CookieToSet.model_validate(read[0])])
    assert _sent(devtools, "Storage.setCookies")[0][0]["cookies"][0]["partitionKey"] == key
    assert await control.delete_cookies(name="chip") == 2
    assert [params.get("partitionKey") for params, _ in _sent(devtools, "Network.deleteCookies")] == [key, None]


def test_a_cookie_with_no_place_both_expiries_or_a_field_it_doesnt_take_is_refused_without_its_value():
    for cookie, reason in (({"name": "sid", "value": "SECRET-1"}, "names neither a url nor a domain"),
                           ({"name": "sid", "value": "SECRET-2", "domain": "a.test", "expires": 5, "max_age": 5},
                            "both expires and max_age"),
                           ({"name": "sid", "value": "SECRET-3", "domain": "a.test", "maxAge": 5}, "maxAge")):
        with pytest.raises(InvalidArgument, match=reason) as refused:
            _parsed(CookieToSet, [cookie], "cookies")
        assert "SECRET" not in str(refused.value) and "cookies[0]" in str(refused.value)
    with pytest.raises(InvalidArgument, match="cookies is a list"):
        _parsed(CookieToSet, {"name": "sid"}, "cookies")


@pytest.mark.asyncio
async def test_cookies_matching_every_field_given_are_deleted_by_their_exact_domain_on_a_page_of_agentenvs_own():
    devtools = StubDevTools({"Storage.getCookies": {"cookies": CDP_COOKIES}, "Target.createTarget": {"targetId": "h"},
                             "Target.attachToTarget": {"sessionId": "s-h"}})
    control = BrowserControl(devtools, "agent")

    assert await control.delete_cookies(name="token", domain="bank.test") == 1
    assert _sent(devtools, "Network.deleteCookies") == [({"name": "token", "domain": "bank.test", "path": "/"}, "s-h")]
    assert await control.delete_cookies() == 4
    assert await control.delete_cookies(name="nothing") == 0


def _serving_browser(storage, names=None):
    """A browser whose pages go where they're sent once agentenv answers their request, each origin named as ``names``
    has it, and whose local storage is ``storage``, by storage key."""
    devtools = StubDevTools()
    at, left = {}, {}

    def navigate(params, session):
        devtools.send("Fetch.requestPaused", {"requestId": f"r-{params['url']}", "request": {"url": params["url"]}},
                      session)
        asked = params["url"].rstrip("/")
        at[session], left[session] = (names or {}).get(asked, asked), False
        return {"frameId": "f"}

    def evaluate(params, session):
        if params["expression"] == LEAVING:
            left[session] = True
            return {"result": {}}
        if params["expression"] == ARRIVED:
            return {"result": {"value": "" if left.get(session, True) else at[session]}}
        return {"result": {"value": ["hidden", False]}}

    devtools.answers.update({
        "Target.createTarget": {"targetId": "h"}, "Target.attachToTarget": {"sessionId": "s-h"},
        "Page.navigate": navigate, "Runtime.evaluate": evaluate,
        "DOMStorage.getDOMStorageItems": lambda params, session: {
            "entries": [[k, v] for k, v in storage.get(params["storageId"]["storageKey"], {}).items()]},
        "DOMStorage.setDOMStorageItem": lambda params, session: storage.setdefault(
            params["storageId"]["storageKey"], {}).update({params["key"]: params["value"]}) or {},
        "DOMStorage.clear": lambda params, session: storage.pop(params["storageId"]["storageKey"], None) and {}})
    return devtools


@pytest.mark.asyncio
async def test_local_storage_goes_through_a_page_agentenv_serves_so_the_site_gets_no_request():
    storage = {"https://shop.test/": {"cart": "3"}}
    devtools = _serving_browser(storage)
    control = BrowserControl(devtools, "agent")

    assert await control.local_storage(["https://shop.test"]) == [
        {"origin": "https://shop.test", "localStorage": [{"name": "cart", "value": "3"}]}]
    await control.set_local_storage([OriginStorage.model_validate(
        {"origin": "http://wiki.test:8080/", "localStorage": [{"name": "theme", "value": "dark"}]})])
    await asyncio.sleep(0)

    assert storage["http://wiki.test:8080/"] == {"theme": "dark"}
    first_calls = [method for method, _, session in devtools.calls if session == "s-h"][:4]
    assert first_calls == ["Network.enable", "Network.setBypassServiceWorker", "Fetch.enable", "DOMStorage.enable"]
    assert _sent(devtools, "Network.setBypassServiceWorker")[0] == ({"bypass": True}, "s-h")
    assert ({"patterns": [{"urlPattern": "*"}]}, "s-h") in _sent(devtools, "Fetch.enable")
    fulfilled = _sent(devtools, "Fetch.fulfillRequest")
    assert [params["requestId"] for params, _ in fulfilled] == ["r-https://shop.test/", "r-http://wiki.test:8080/"]
    assert all(params["body"] == SERVED_PAGE and session == "s-h" for params, session in fulfilled)
    assert base64.b64decode(SERVED_PAGE) == b"<!doctype html>"
    assert ("Target.closeTarget", {"targetId": "h"}, None) in devtools.calls

    devtools.send("Fetch.requestPaused", {"requestId": "an-agents"}, "s-agent")
    await asyncio.sleep(0)
    assert len(_sent(devtools, "Fetch.fulfillRequest")) == 2


@pytest.mark.asyncio
async def test_with_no_origins_named_it_reads_and_clears_every_origin_its_pages_and_frames_loaded():
    storage = {"https://a.test/": {"k": "1"}, "https://b.test:8443/": {}, "http://frame.test/": {"f": "2"}}
    devtools = _serving_browser(storage)
    control = BrowserControl(devtools, "agent")
    await control.attach()
    devtools.send("Target.targetCreated", {"targetInfo": {"targetId": "t1", "type": "page", "url": "https://a.test/x"}})
    devtools.send("Target.targetCreated", {"targetInfo": {"targetId": "t2", "type": "page", "url": "about:blank"}})
    devtools.send("Target.targetInfoChanged",
                  {"targetInfo": {"targetId": "t2", "type": "page", "url": "https://b.test:8443/q?x=1"}})
    devtools.send("Target.attachedToTarget",
                  {"sessionId": "s1", "targetInfo": {"targetId": "t1", "type": "page", "url": "https://a.test/x"}})
    devtools.send("Page.frameNavigated", {"frame": {"url": "http://frame.test:80/embed"}}, "s1")
    devtools.send("Page.frameNavigated", {"frame": {"url": "data:text/html,hi"}}, "s1")
    devtools.send("Page.frameNavigated", {"frame": {"url": "https://unfollowed.test/"}}, "s-not-ours")
    await asyncio.sleep(0)

    read = await control.local_storage()
    origins = ["http://frame.test", "https://a.test", "https://b.test:8443"]
    assert [origin["origin"] for origin in read] == origins
    assert read[0]["localStorage"] == [{"name": "f", "value": "2"}]
    assert [tab["id"] for tab in await control.tab_list()] == ["t1", "t2"]
    assert await control.clear_local_storage() == origins
    assert storage == {}


@pytest.mark.asyncio
async def test_an_origin_is_kept_as_the_browser_names_it_once_its_page_is_there():
    storage = {"https://xn--bcher-kva.test/": {"k": "v"}}
    devtools = _serving_browser(storage, names={"https://bücher.test": "https://xn--bcher-kva.test"})
    control = BrowserControl(devtools, "agent")

    assert await control.local_storage(["https://bücher.test", "https://a.test"]) == [
        {"origin": "https://xn--bcher-kva.test", "localStorage": [{"name": "k", "value": "v"}]},
        {"origin": "https://a.test", "localStorage": []}]
    assert control.origins == {"https://xn--bcher-kva.test", "https://a.test"}


@pytest.mark.asyncio
async def test_a_url_that_isnt_an_origin_is_refused_and_an_origin_is_kept_as_the_browser_names_it():
    control = BrowserControl(_serving_browser({}), "agent")
    for url in ("https://a.test/path", "https://a.test/?q=1", "ftp://a.test", "a.test"):
        with pytest.raises(InvalidArgument):
            await control.local_storage([url])
    assert OriginStorage(origin="HTTPS://A.test:443/", localStorage=[]).origin == "https://a.test"
    assert OriginStorage(origin="http://[::1]:8080", localStorage=[]).origin == "http://[::1]:8080"


@pytest.mark.asyncio
async def test_a_download_comes_with_its_file_inline_once_completed_and_a_guid_it_hasnt_seen_is_refused(browser_home):
    browser = Browser()
    landed = browser_home / "Downloads" / "report.csv"
    landed.write_bytes(b"a,b\n1,2\n")
    browser.control.downloads["g1"] = {"guid": "g1", "url": "https://shop.test/report", "filename": "report.csv",
                                       "state": "completed", "started_at": "t0", "ended_at": "t1",
                                       "path": str(landed), "size": 8,
                                       "sha256": hashlib.sha256(b"a,b\n1,2\n").hexdigest()}
    browser.control.downloads["g2"] = {"guid": "g2", "url": "https://shop.test/big", "filename": "big.zip",
                                       "state": "inProgress", "started_at": "t0"}

    got = await browser.browser_downloads_get("g1")
    assert got["download"]["sha256"] == hashlib.sha256(b"a,b\n1,2\n").hexdigest()
    [part] = got["parts"]
    assert base64.b64decode(part["file"]["bytes"]) == b"a,b\n1,2\n" and part["file"]["name"] == "report.csv"
    assert (await browser.browser_downloads_get("g2"))["parts"] == []
    assert [d["guid"] for d in (await browser.browser_downloads_list())["downloads"]] == ["g1", "g2"]
    with pytest.raises(InvalidArgument, match="no download nope"):
        await browser.browser_downloads_get("nope")
    with pytest.raises(InvalidArgument, match=r"write_namespace is not a valid namespace grant \(") as refused:
        await browser.browser_downloads_get("g1", write_namespace={"signature": "SIGNED-VALUE"})
    assert "SIGNED-VALUE" not in str(refused.value)


@pytest.mark.asyncio
async def test_its_record_and_tab_list_have_the_shapes_and_name_each_tabs_opener():
    devtools = StubDevTools()
    control = BrowserControl(devtools, "agent")
    await control.attach()
    devtools.send("Target.targetCreated", {"targetInfo": {"targetId": "a", "type": "page", "url": "https://a.test/"}})
    devtools.send("Target.targetCreated",
                  {"targetInfo": {"targetId": "popup", "type": "page", "url": "https://a.test/p", "openerId": "a"}})
    devtools.send("Target.targetDestroyed", {"targetId": "popup"})

    record = await control.record()
    assert set(record["tabs"][0]) == {"id", "url", "title", "opened_at", "opener", "active"}
    assert record["tabs"][0]["opener"] is None
    assert (record["closed_tabs"][0]["opener"], record["closed_tabs"][0]["closed_by"]) == ("a", "unknown")
    assert (await control.tab_list())[0]["id"] == "a"
