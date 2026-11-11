"""A website browser's CDP addresses: a user's token minted once, kept by hash only, so it still works after the server
restarts and the computer's session can't use the file; the relay giving each WebSocket connection its own
browser-level session over one pipe, its answers and events and nobody else's, until the browser goes; the guard on
each command; and the cdp operation and discovery answered at the address the call reached, or under the origin its
caller names, which behind a tunnel that ends TLS is the only way to an https address."""
from __future__ import annotations

import asyncio
import contextlib
import json
import socket

import httpx
import pytest
import uvicorn

pytest.importorskip("mcp.server.fastmcp")

from starlette.applications import Starlette  # noqa: E402
from starlette.routing import WebSocketRoute  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK  # noqa: E402

from agentenv_protocol import browser_cdp  # noqa: E402
from agentenv_protocol.browser_cdp import NO_SESSION, Relay, Tokens, guard  # noqa: E402
from agentenv_protocol.computer import Launch, Platform, Ready, app, computer_app, linux  # noqa: E402
from agentenv_protocol.computer.app import APP_SETTINGS_ENV  # noqa: E402
from agentenv_protocol.computer.app_runtime import AppReadiness  # noqa: E402
from agentenv_protocol.computer.errors import DeviceUnavailable  # noqa: E402
from agentenv_protocol.website_browser import SITE_PROXY_ENV, WebsiteBrowserEnvironment  # noqa: E402

FORWARDED = {"x-forwarded-host": "computer.test:443", "x-forwarded-proto": "https",
             "x-forwarded-prefix": "/svc/chromium"}
#: What the driver forwards behind a tunnel that ends TLS: the tunnel's host, and http, how the tunnel reached it.
BEHIND_TLS = {"x-forwarded-host": "t.tunnel.test", "x-forwarded-proto": "http", "x-forwarded-prefix": "/svc/chromium"}


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11", "none"]}, "arch": ["amd64"]}])
@linux(launch=Launch(command=["chromium/chrome"], window={"class": "Chromium"}, ready=Ready(window=True, timeout_s=30)))
class Browser(WebsiteBrowserEnvironment):
    version = "156.0.8078.4"


class Pipe:
    """A browser's pipe: each connection's browser-level session ``b<n>``, an attach on a session making its child
    ``<session>-c`` and saying so on that session, ``Boom`` failing, ``Hang`` never answered, anything else echoed;
    every request recorded with the last params of each method, and each ``Hang`` given up."""

    def __init__(self) -> None:
        self.listeners, self.sent, self.sessions, self.given_up, self.params = [], [], 0, [], {}

    def listen(self, listener) -> None:
        self.listeners.append(listener)

    def emit(self, message: dict) -> None:
        for listener in self.listeners:
            listener(message)

    async def call(self, method, params=None, *, session_id=None, timeout_s=None):
        self.sent.append((method, session_id))
        if method == "Target.attachToBrowserTarget":
            self.sessions += 1
            return {"sessionId": f"b{self.sessions}"}
        return {}

    async def request(self, method, params=None, *, session_id=None):
        self.sent.append((method, session_id))
        self.params[method] = params
        if method == "Target.attachToTarget":
            child = f"{session_id}-c"
            self.emit({"method": "Target.attachedToTarget", "sessionId": session_id,
                       "params": {"sessionId": child, "targetInfo": {"targetId": params["targetId"]}}})
            return {"result": {"sessionId": child}}
        if method == "Boom":
            return {"error": {"code": -32000, "message": "boom"}}
        if method == "Hang":
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.given_up.append(method)
                raise
        return {"result": {"method": method, "session": session_id}}


class SlowClient:
    """A client's WebSocket that takes in nothing sent to it until ``reading`` is set, and records what it got and how
    it was closed."""

    def __init__(self) -> None:
        self.reading, self.incoming = asyncio.Event(), asyncio.Queue()
        self.received, self.closed = [], None

    async def accept(self) -> None:
        pass

    async def receive(self) -> dict:
        return await self.incoming.get()

    async def send_text(self, text: str) -> None:
        await self.reading.wait()
        self.received.append(text)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.closed = (code, reason)
        self.incoming.put_nowait({"type": "websocket.disconnect", "code": code})


@contextlib.asynccontextmanager
async def _relay(dialogs_answered: bool = False):
    """The relay over a fake pipe, served on loopback with each connection's user its path's: the pipe, the relay and
    the server's ws:// base."""
    pipe = Pipe()
    relay = Relay(pipe, dialogs_answered=dialogs_answered)

    async def endpoint(websocket):
        await relay.serve(websocket, websocket.path_params["user"])

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(Starlette(routes=[WebSocketRoute("/{user}", endpoint)]), log_level="warning",
                                           lifespan="off"))
    serving = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        yield pipe, relay, f"ws://127.0.0.1:{sock.getsockname()[1]}"
    finally:
        server.should_exit = True
        await serving


async def _ask(socket, message: dict) -> dict:
    await socket.send(json.dumps(message))
    while (answer := json.loads(await socket.recv())).get("id") != message["id"]:
        pass
    return answer


def test_a_users_token_is_minted_once_kept_by_hash_and_still_known_after_a_restart(tmp_path):
    path = tmp_path / ".agentenv" / "cdp-tokens.json"
    tokens = Tokens(path)
    ada = tokens.mint("ada", "https://computer.test/svc/chromium")
    again, bob = tokens.mint("ada", "https://elsewhere.test"), tokens.mint("bob", "https://computer.test/svc/chromium")
    cy = tokens.mint("cy", "https://t.tunnel.test/svc/chromium", named=True)
    token, cys = (address.http_url.split("/cdp/")[1].strip("/") for address in (ada, cy))
    restarted = Tokens(path)

    assert again == ada and bob != ada and ada.http_url == f"https://computer.test/svc/chromium/cdp/{token}/"
    assert ada.ws_url == f"wss://computer.test/svc/chromium/cdp/{token}/devtools/browser"
    assert token not in path.read_text() and restarted.address(token) == ada and tokens.address("guess") is None
    assert restarted.mint("ada", "https://computer.test").http_url != ada.http_url
    assert not ada.named and restarted.address(cys) == cy and cy.named


@pytest.mark.asyncio
async def test_each_connection_gets_its_own_browser_session_its_own_answers_and_events_and_its_commands_end_with_it():
    async with _relay() as (pipe, relay, base):
        async with connect(f"{base}/ada") as ada, connect(f"{base}/bob") as bob:
            mine = await _ask(ada, {"id": 1, "method": "Browser.getVersion"})
            theirs = await _ask(bob, {"id": 1, "method": "Browser.getVersion"})
            await ada.send(json.dumps({"id": 2, "method": "Target.attachToTarget",
                                       "params": {"targetId": "t1", "flatten": True}}))
            attached = json.loads(await ada.recv())
            await ada.recv()
            child = await _ask(ada, {"id": 3, "method": "Runtime.evaluate", "sessionId": "b1-c"})
            refused = await _ask(bob, {"id": 4, "method": "Runtime.evaluate", "sessionId": "b1-c"})
            failed = await _ask(ada, {"id": 5, "method": "Boom"})
            await ada.send(json.dumps({"id": 6, "method": "Hang"}))
            while ("Hang", "b1") not in pipe.sent:
                await asyncio.sleep(0.01)
            pipe.emit({"method": "Page.loadEventFired", "sessionId": "b1-c", "params": {}})
            pipe.emit({"method": "Target.targetCreated", "sessionId": "b2", "params": {}})
            pipe.emit({"method": "Page.frameNavigated", "params": {}})
            heard = json.loads(await ada.recv()), json.loads(await bob.recv())
            relay.ended()
            with pytest.raises(ConnectionClosedError) as closed:
                await ada.recv()

    assert mine == {"id": 1, "result": {"method": "Browser.getVersion", "session": "b1"}}
    assert theirs == {"id": 1, "result": {"method": "Browser.getVersion", "session": "b2"}}
    assert attached == {"method": "Target.attachedToTarget", "params": {"sessionId": "b1-c",
                                                                        "targetInfo": {"targetId": "t1"}}}
    assert child == {"id": 3, "result": {"method": "Runtime.evaluate", "session": "b1-c"}, "sessionId": "b1-c"}
    assert refused == {"id": 4, "error": NO_SESSION, "sessionId": "b1-c"}
    assert ("Runtime.evaluate", "b2") not in pipe.sent
    assert failed == {"id": 5, "error": {"code": -32000, "message": "boom"}}
    assert heard == ({"method": "Page.loadEventFired", "sessionId": "b1-c", "params": {}},
                     {"method": "Target.targetCreated", "params": {}})
    assert (closed.value.rcvd.code, closed.value.rcvd.reason) == browser_cdp.RELAUNCHED
    assert [(c["user"], "closed_at" in c) for c in relay.connections] == [("ada", True), ("bob", True)]
    assert ("Target.detachFromTarget", None) in pipe.sent
    assert pipe.given_up == ["Hang"]


@pytest.mark.parametrize(("method", "params", "action"), [
    ("Page.navigate", {"url": "https://site.test/"}, "send"),
    ("Page.navigate", {"url": "about:blank"}, "send"),
    ("Page.navigate", {"url": "about:settings"}, "refuse"),
    ("Page.navigate", {"url": " \tFI\nLE:///etc/passwd"}, "refuse"),
    ("Target.createTarget", {"url": "chrome://settings"}, "refuse"),
    ("Target.createTarget", {"url": "View-Source:https://site.test/"}, "refuse"),
    ("Target.createBrowserContext", {"proxyBypassList": "*"}, "refuse"),
    ("Target.createBrowserContext", {"disposeOnDetach": True}, "send"),
    ("Page.handleJavaScriptDialog", {"accept": True}, "send"),
    ("Browser.crash", {}, "close"),
    ("Target.sendMessageToTarget", {"sessionId": "s", "message": "{}"}, "refuse"),
])
def test_the_guard_reads_a_url_as_the_browser_does_and_lets_through_what_it_doesnt_guard(method, params, action):
    assert guard(method, params, dialogs_answered=False).action == action


@pytest.mark.asyncio
async def test_the_guard_keeps_the_browser_up_its_traffic_and_downloads_its_own_and_its_dialogs_agentenvs():
    async with _relay(dialogs_answered=True) as (pipe, relay, base):
        async with connect(f"{base}/ada") as ada, connect(f"{base}/bob") as bob:
            navigated = await _ask(ada, {"id": 1, "method": "Page.navigate", "params": {"url": "file:///etc/passwd"}})
            exposed = await _ask(ada, {"id": 2, "method": "Target.exposeDevToolsProtocol",
                                       "params": {"targetId": "t1"}})
            proxied = await _ask(ada, {"id": 3, "method": "Target.createBrowserContext",
                                       "params": {"proxyServer": "http://elsewhere.test:8080"}})
            context = await _ask(ada, {"id": 4, "method": "Target.createBrowserContext", "params": {}})
            downloads = await _ask(ada, {"id": 5, "method": "Browser.setDownloadBehavior", "params": {
                "behavior": "allowAndName", "downloadPath": "/tmp/elsewhere", "eventsEnabled": True}})
            older = await _ask(ada, {"id": 6, "method": "Page.setDownloadBehavior",
                                     "params": {"behavior": "allow", "downloadPath": "/tmp/elsewhere"}})
            dialog = await _ask(ada, {"id": 7, "method": "Page.handleJavaScriptDialog", "params": {"accept": False}})
            wrapped = await _ask(ada, {"id": 9, "method": "Target.sendMessageToTarget", "params": {
                "sessionId": "b1-c", "message": json.dumps({"id": 1, "method": "Page.navigate",
                                                            "params": {"url": "file:///etc/passwd"}})}})
            malformed = [await _ask(ada, {"id": 10, "method": "Page.navigate", "params": [1]}),
                         await _ask(ada, {"id": 11, "params": {}}),
                         await _ask(ada, {"id": 12, "method": "Browser.getVersion", "sessionId": ["b1"]})]
            await ada.send("[1]")
            not_a_command = json.loads(await ada.recv())
            closed = await _ask(bob, {"id": 1, "method": "Browser.close"})
            with pytest.raises(ConnectionClosedOK) as ended:
                await bob.recv()
            still = await _ask(ada, {"id": 8, "method": "Browser.getVersion"})
    sent = {method for method, _ in pipe.sent}

    assert navigated["error"] == {"code": -32000, "message": "agentenv refuses Page.navigate: file: URLs open the "
                                                             "browser's own files, settings and tools"}
    assert exposed["error"]["message"].startswith("agentenv refuses Target.exposeDevToolsProtocol: ")
    assert proxied["error"]["message"].startswith("agentenv refuses Target.createBrowserContext: ")
    assert "result" in context and "result" in downloads and "result" in older
    assert pipe.params["Browser.setDownloadBehavior"] == {"behavior": "default", "eventsEnabled": True}
    assert pipe.params["Page.setDownloadBehavior"] == {"behavior": "default"}
    assert dialog == {"id": 7, "result": {}} and closed == {"id": 1, "result": {}}
    assert (ended.value.rcvd.code, ended.value.rcvd.reason) == browser_cdp.CLOSED
    assert still["result"] == {"method": "Browser.getVersion", "session": "b1"}
    assert wrapped["error"]["message"].startswith("agentenv refuses Target.sendMessageToTarget: ")
    assert [answer["error"]["code"] for answer in malformed] == [-32600] * 3
    assert not_a_command == {"id": None, "error": browser_cdp.MALFORMED}
    assert not {"Page.navigate", "Target.exposeDevToolsProtocol", "Target.sendMessageToTarget",
                "Page.handleJavaScriptDialog", "Browser.close"} & sent
    assert [(entry["method"], entry["did"]) for entry in relay.connections[0]["guarded"]] == [
        ("Page.navigate", "refused"), ("Target.exposeDevToolsProtocol", "refused"),
        ("Target.createBrowserContext", "refused"), ("Browser.setDownloadBehavior", "changed"),
        ("Page.setDownloadBehavior", "changed"), ("Page.handleJavaScriptDialog", "answered"),
        ("Target.sendMessageToTarget", "refused")]
    assert [(entry["method"], entry["did"]) for entry in relay.connections[1]["guarded"]] == [
        ("Browser.close", "closed")]


@pytest.mark.asyncio
async def test_a_client_that_falls_too_far_behind_is_closed_and_what_waited_for_it_dropped(monkeypatch):
    monkeypatch.setattr(browser_cdp, "OUTBOX_LIMIT_BYTES", 10_000)
    pipe, slow = Pipe(), SlowClient()
    relay = Relay(pipe)
    serving = asyncio.ensure_future(relay.serve(slow, "ada"))
    while not relay.connections:
        await asyncio.sleep(0)
    for n in range(20):
        pipe.emit({"method": "Network.dataReceived", "sessionId": "b1", "params": {"n": n, "data": "x" * 1000}})
    slow.reading.set()
    await asyncio.wait_for(serving, 5)

    assert slow.closed == browser_cdp.TOO_SLOW and slow.received == []
    assert "closed_at" in relay.connections[0] and ("Target.detachFromTarget", None) in pipe.sent


def _browser(monkeypatch, tmp_path) -> Browser:
    """The browser alone, in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux", display="none"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ENVIRONMENT_NAME", "chromium")
    monkeypatch.setenv(SITE_PROXY_ENV, "http://127.0.0.1:3128")
    monkeypatch.delenv(APP_SETTINGS_ENV, raising=False)
    return Browser()


@pytest.mark.asyncio
async def test_cdp_waits_for_what_computer_app_ready_reports_up_to_its_declared_ready_timeout(monkeypatch, tmp_path):
    browser = _browser(monkeypatch, tmp_path)
    answers = iter([False, False, True])

    async def readiness():
        return AppReadiness(ready=next(answers, False), problems=[], launched=True, waited_s=0, timeout_s=30)

    monkeypatch.setattr(browser._computer_app, "readiness", readiness)
    monkeypatch.setattr("agentenv_protocol.website_browser.environment.READY_POLL_S", 0.01)
    timeout_s = browser._computer_app.timeout_s
    await browser._until_ready()
    browser._computer_app.timeout_s = 0.05
    with pytest.raises(DeviceUnavailable):
        await browser._until_ready()

    assert timeout_s == 30


@pytest.mark.asyncio
async def test_the_cdp_operation_and_discovery_answer_at_the_address_reached_and_refuse_other_tokens_or_a_gone_browser(
        monkeypatch, tmp_path):
    browser = _browser(monkeypatch, tmp_path)

    async def ready():
        return None

    async def call(method, params=None, *, session_id=None, timeout_s=None):
        assert method == "Browser.getVersion"
        return {"product": "HeadlessChrome/156.0.8078.4", "protocolVersion": "1.3", "userAgent": "UA",
                "jsVersion": "15.6", "revision": "@abc"}

    async def gone(method, params=None, *, session_id=None, timeout_s=None):
        raise ConnectionError("the browser closed its debugging pipe")

    async def tabs():
        return [{"id": "T1", "url": "https://site.test/", "title": "Site", "active": True}]

    monkeypatch.setattr(browser, "_until_ready", ready)
    monkeypatch.setattr(browser.devtools, "call", call)
    monkeypatch.setattr(browser.control, "tab_list", tabs)
    asgi = browser.create_app().streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://app") as http:
        ada = (await http.post("/agentenv/ext/browser/cdp", json={"user": "ada"}, headers=FORWARDED)).json()
        again = (await http.post("/agentenv/ext/browser/cdp", json={"user": "ada"})).json()
        root = ada["http_url"].removeprefix("https://computer.test:443/svc/chromium")
        version = (await http.get(f"{root}json/version", headers={**FORWARDED, "x-forwarded-host": "seen.test"})).json()
        listed = (await http.get(f"{root}json/list")).json()
        refused = await http.get("/cdp/guess/json/version")
        card = (await http.get("/.well-known/agent-env.json")).json()
        monkeypatch.setattr(browser.devtools, "call", gone)
        unanswered = await http.get(f"{root}json/list")

    assert again == ada and ada["ws_url"].startswith("wss://computer.test:443/svc/chromium/cdp/")
    assert version["Browser"] == "HeadlessChrome/156.0.8078.4"
    assert version["webSocketDebuggerUrl"] == f"wss://seen.test/svc/chromium{root}devtools/browser"
    assert listed == [{"id": "T1", "type": "page", "title": "Site", "url": "https://site.test/", "description": "",
                       "devtoolsFrontendUrl": ""}]
    assert refused.status_code == 403 and unanswered.status_code == 503
    computer_app_params = next(e["params"] for e in card["capabilities"]["extensions"]
                               if e["uri"] == "urn:agentenv:computer-app/v1")
    browser_methods = next(e["params"]["methods"] for e in card["capabilities"]["extensions"]
                           if e["uri"] == "urn:agentenv:browser/v1")
    assert computer_app_params["agent_routes"] == ["/cdp/"]
    assert browser_methods["cdp"]["request"]["properties"] == {"user": {"type": "string"},
                                                               "origin": {"type": "string", "default": None}}


@pytest.mark.asyncio
async def test_under_an_origin_its_caller_names_an_address_is_https_and_discovery_answers_wss_though_reached_over_http(
        monkeypatch, tmp_path):
    browser = _browser(monkeypatch, tmp_path)

    async def ready():
        return None

    async def call(method, params=None, *, session_id=None, timeout_s=None):
        return {"product": "Chrome/156.0.8078.4", "protocolVersion": "1.3"}

    monkeypatch.setattr(browser, "_until_ready", ready)
    monkeypatch.setattr(browser.devtools, "call", call)
    asgi = browser.create_app().streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://app") as http:
        ada = (await http.post("/agentenv/ext/browser/cdp", json={"user": "ada", "origin": "https://t.tunnel.test/"},
                               headers=BEHIND_TLS)).json()
        bob = (await http.post("/agentenv/ext/browser/cdp", json={"user": "bob"}, headers=BEHIND_TLS)).json()
        cy = (await http.post("/agentenv/ext/browser/cdp", json={"user": "cy", "origin": "HTTPS://t.tunnel.test"},
                              headers=BEHIND_TLS)).json()
        paths = [address["http_url"].split("/svc/chromium", 1)[1] for address in (ada, bob)]
        seen = [(await http.get(f"{path}json/version", headers=BEHIND_TLS)).json()["webSocketDebuggerUrl"]
                for path in paths]

    token = ada["http_url"].split("/cdp/")[1].strip("/")
    assert ada == {"http_url": f"https://t.tunnel.test/svc/chromium/cdp/{token}/",
                   "ws_url": f"wss://t.tunnel.test/svc/chromium/cdp/{token}/devtools/browser"}
    assert seen[0] == ada["ws_url"]
    assert bob["ws_url"].startswith("ws://t.tunnel.test/") and seen[1] == bob["ws_url"]
    assert cy["http_url"].startswith("https://t.tunnel.test/svc/chromium/cdp/")
    assert cy["ws_url"].startswith("wss://t.tunnel.test/svc/chromium/cdp/")


@pytest.mark.asyncio
@pytest.mark.parametrize("origin", ["ftp://t.test", "t.test", "https://", "https://t.test/?q=1", "https://t.test/#top",
                                    "https://ada:pw@t.test", "https://t.test:0", "https://t.test:99999",
                                    "https://t.test/\n", 7])
async def test_an_origin_that_isnt_an_http_or_https_url_is_refused_and_mints_nothing(monkeypatch, tmp_path, origin):
    browser = _browser(monkeypatch, tmp_path)
    asgi = browser.create_app().streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://app") as http:
        answer = await http.post("/agentenv/ext/browser/cdp", json={"user": "ada", "origin": origin})

    assert answer.status_code == 400 and "origin is an http or https URL" in answer.text, answer.text
    assert not browser.cdp_tokens.path.exists()
