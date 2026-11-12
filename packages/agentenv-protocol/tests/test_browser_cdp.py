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

from agentenv_protocol.website_browser import cdp as browser_cdp  # noqa: E402
from agentenv_protocol.website_browser import log as browser_log  # noqa: E402
from agentenv_protocol.website_browser.cdp import NO_SESSION, Relay, Tokens, guard  # noqa: E402
from agentenv_protocol.website_browser.log import REDACTED, TOOL_CALL_HEADER, BrowserLog, Causes, tool_call_header  # noqa: E402
from agentenv_protocol.computer import Launch, Platform, Ready, app, computer_app, linux  # noqa: E402
from agentenv_protocol.computer.app import APP_SETTINGS_ENV  # noqa: E402
from agentenv_protocol.computer.app_runtime import AppReadiness  # noqa: E402
from agentenv_protocol.computer.errors import DeviceUnavailable, InvalidArgument  # noqa: E402
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
    ``<session>-c`` and saying so on that session, ``Boom`` failing, ``Hang`` never answered, ``Ordered`` answered and
    then followed at once by an event, as the real pipe reads them when both came in one read, anything else echoed;
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

    async def request(self, method, params=None, *, session_id=None, answered=None):
        self.sent.append((method, session_id))
        self.params[method] = params
        if method == "Ordered":
            answer = {"result": {"method": method}}
            answered(answer)
            self.emit({"method": "Page.after", "sessionId": session_id, "params": {}})
            return answer
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
async def _relay(dialogs_answered: bool = False, **logging):
    """The relay over a fake pipe, served on loopback with each connection's user its path's: the pipe, the relay and
    the server's ws:// base."""
    pipe = Pipe()
    relay = Relay(pipe, dialogs_answered=dialogs_answered, **logging)

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
async def test_an_answer_reaches_its_client_before_the_events_the_browser_sent_after_it():
    async with _relay() as (pipe, relay, base):
        async with connect(f"{base}/ada") as ada:
            await ada.send(json.dumps({"id": 1, "method": "Ordered"}))
            first, second = json.loads(await ada.recv()), json.loads(await ada.recv())

    assert first == {"id": 1, "result": {"method": "Ordered"}} and second == {"method": "Page.after", "params": {}}


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


class Control:
    """agentenv's control as the relay asks it: the URL of tab T1, a password field focused wherever it's asked unless
    ``focused`` says not, and the causes of tab events; each check made with how many commands ``pipe`` had been sent
    by then."""

    def __init__(self, pipe=None) -> None:
        self.causes, self.asked, self.pipe, self.focused = Causes(), [], pipe, True

    def url_of(self, target):
        return "https://a.test/page" if target == "T1" else None

    async def password_focused(self, target) -> bool:
        self.asked.append((target, len(self.pipe.sent) if self.pipe is not None else None))
        return self.focused


@pytest.mark.asyncio
async def test_every_relayed_command_is_an_entry_on_its_sessions_tab_an_acting_one_with_its_parameters_kept(tmp_path):
    log, control = BrowserLog(tmp_path / "browser-chromium.jsonl"), Control()
    async with _relay(log=log, control=control) as (pipe, relay, base):
        async with connect(f"{base}/ada") as ada:
            attached = await _ask(ada, {"id": 1, "method": "Target.attachToTarget",
                                        "params": {"targetId": "T1", "flatten": True}})
            child = attached["result"]["sessionId"]
            for n, (method, params) in enumerate([
                    ("DOM.getDocument", {"depth": 1}),
                    ("Network.setCookie", {"name": "sid", "value": "s3cret", "url": "https://a.test/?token=t"}),
                    ("Input.insertText", {"text": "hunter2"}), ("Boom", {})], start=2):
                await _ask(ada, {"id": n, "method": method, "params": params, "sessionId": child})
            await _ask(ada, {"id": 6, "method": "Target.exposeDevToolsProtocol", "params": {"targetId": "T1"}})
            await _ask(ada, {"id": 7, "method": "Page.navigate", "params": {"url": "https://a.test/"},
                             "sessionId": "theirs"})
        await asyncio.sleep(0.05)

    common = {"via": "cdp", "who": "ada", "connection": 1}
    page = {**common, "tab": "T1", "url": "https://a.test/page"}
    assert [line["action"] for line in log.read()["lines"]] == [
        {**page, "method": "Target.attachToTarget", "acts": False, "ok": True},
        {**page, "method": "DOM.getDocument", "acts": False, "ok": True},
        {**page, "method": "Network.setCookie", "acts": True, "ok": True,
         "params": {"name": "sid", "value": REDACTED, "url": f"https://a.test/?token={REDACTED}"}},
        {**page, "method": "Input.insertText", "acts": True, "ok": True, "params": {"text": {"length": 7}}},
        {**page, "method": "Boom", "acts": False, "ok": False, "error": "boom"},
        {**page, "method": "Target.exposeDevToolsProtocol", "acts": False, "ok": False, "guard": "refused",
         "error": "agentenv refuses Target.exposeDevToolsProtocol: it would give a page's script the browser's own "
                  "CDP channel"},
        {**common, "method": "Page.navigate", "acts": True, "tab": None, "url": None, "ok": False,
         "error": NO_SESSION["message"], "params": {"url": "https://a.test/"}},
    ]
    assert [target for target, _ in control.asked] == ["T1"]


async def _attached(socket) -> str:
    return (await _ask(socket, {"id": 0, "method": "Target.attachToTarget",
                                "params": {"targetId": "T1", "flatten": True}}))["result"]["sessionId"]


def _call(tool: str, call: str, args: dict) -> dict:
    return {"tool": tool, "call": call, "args": args}


@pytest.mark.asyncio
async def test_a_users_commands_are_credited_to_its_open_tool_call_or_to_every_one_open_when_calls_overlap(tmp_path):
    log, control = BrowserLog(tmp_path / "browser-chromium.jsonl"), Control()
    async with _relay(log=log, control=control) as (pipe, relay, base):
        async with connect(f"{base}/playwright-mcp") as mcp, connect(f"{base}/ada") as ada:
            child = await _attached(mcp)
            click = relay.begin_call("playwright-mcp", "agent", _call("browser_click", "c1", {"target": "e3"}),
                                     acts=True, within_s=60)
            await _ask(mcp, {"id": 1, "method": "Input.dispatchMouseEvent", "params": {"type": "mousePressed"},
                             "sessionId": child})
            await _ask(ada, {"id": 1, "method": "Browser.getVersion"})
            snapshot = relay.begin_call("playwright-mcp", "agent", _call("browser_snapshot", "c2", {}), acts=False,
                                        within_s=60)
            await _ask(mcp, {"id": 2, "method": "DOM.getDocument", "sessionId": child})
            ended = relay.end_call(click, True), relay.end_call(snapshot, False, "Playwright MCP ended during it")
            await _ask(mcp, {"id": 3, "method": "DOM.getDocument", "sessionId": child})
            again = relay.end_call(click, True)
        await asyncio.sleep(0.05)

    mine = {"via": "cdp", "who": "playwright-mcp", "connection": 1, "tab": "T1", "url": "https://a.test/page"}
    assert [line["action"] for line in log.read()["lines"]] == [
        {**mine, "method": "Target.attachToTarget", "acts": False, "ok": True},
        {"via": "mcp", "who": "agent", "tool": "browser_click", "call": "c1", "acts": True, "tab": "T1",
         "url": "https://a.test/page", "ok": True, "args": {"target": "e3"}},
        {**mine, "method": "Input.dispatchMouseEvent", "acts": True, "for": click, "ok": True,
         "params": {"type": "mousePressed"}},
        {"via": "cdp", "who": "ada", "connection": 2, "method": "Browser.getVersion", "acts": False, "tab": None,
         "url": None, "ok": True},
        {"via": "mcp", "who": "agent", "tool": "browser_snapshot", "call": "c2", "acts": False, "tab": None,
         "url": None, "ok": False, "error": "Playwright MCP ended during it"},
        {**mine, "method": "DOM.getDocument", "acts": False, "for": [click, snapshot], "ok": True},
        {**mine, "method": "DOM.getDocument", "acts": False, "ok": True},
    ]
    assert ended == (True, True) and again is False and not relay.calls


@pytest.mark.asyncio
async def test_a_tool_calls_typed_text_is_kept_only_once_its_own_typing_found_no_password_field(tmp_path):
    log, control = BrowserLog(tmp_path / "browser-chromium.jsonl"), Control()
    async with _relay(log=log, control=control) as (pipe, relay, base):
        async with connect(f"{base}/playwright-mcp") as mcp:
            child = await _attached(mcp)
            for n, (focused, types) in enumerate([(False, True), (True, True), (False, False)], start=1):
                control.focused = focused
                entry = relay.begin_call("playwright-mcp", "agent",
                                         _call("browser_type", f"c{n}", {"target": "e3", "text": "hunter2"}),
                                         acts=True, within_s=60)
                if types:
                    await _ask(mcp, {"id": n, "method": "Input.insertText", "params": {"text": "hunter2"},
                                     "sessionId": child})
                relay.end_call(entry, True)
        await asyncio.sleep(0.05)

    typed = {line["action"]["call"]: line["action"]["args"]["text"] for line in log.read()["lines"]
             if line["action"]["via"] == "mcp"}
    assert typed == {"c1": "hunter2", "c2": {"length": 7}, "c3": {"length": 7}}


@pytest.mark.asyncio
async def test_a_tool_call_never_said_to_end_is_written_so_once_its_bound_passes_and_is_credited_no_more(tmp_path):
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    async with _relay(log=log) as (pipe, relay, base):
        async with connect(f"{base}/playwright-mcp") as mcp:
            entry = relay.begin_call("playwright-mcp", "agent", _call("browser_click", "c1", {}), acts=True,
                                     within_s=0.05)
            await asyncio.sleep(0.1)
            await _ask(mcp, {"id": 1, "method": "Browser.getVersion"})
            ended = relay.end_call(entry, True)
        await asyncio.sleep(0.05)

    call, command = [line["action"] for line in log.read()["lines"]]
    assert (call["ok"], call["error"]) == (False, "its driver didn't say it ended within 0.05 s")
    assert "for" not in command and ended is False


@pytest.mark.asyncio
async def test_a_command_is_checked_for_a_password_before_it_is_sent_and_logged_though_its_client_leaves(tmp_path):
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    async with _relay(log=log) as (pipe, relay, base):
        control = Control(pipe)
        relay.control = control
        async with connect(f"{base}/ada") as ada:
            child = (await _ask(ada, {"id": 1, "method": "Target.attachToTarget",
                                      "params": {"targetId": "T1", "flatten": True}}))["result"]["sessionId"]
            await _ask(ada, {"id": 2, "method": "Input.insertText", "params": {"text": "hunter2"}, "sessionId": child})
            sent_before = next(n for n, (method, _) in enumerate(pipe.sent) if method == "Input.insertText")
            await ada.send(json.dumps({"id": 3, "method": "Hang", "params": {}, "sessionId": child}))
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)
        after_leaving = log.read()
        assert pipe.given_up == ["Hang"]

    assert control.asked == [("T1", sent_before)]
    entries = [line["action"] for line in after_leaving["lines"]]
    assert [entry["method"] for entry in entries] == ["Target.attachToTarget", "Input.insertText", "Hang"]
    assert entries[1]["params"] == {"text": {"length": 7}}
    assert entries[2]["ok"] is False and entries[2]["error"] == browser_cdp.ENDED_UNANSWERED
    assert after_leaving["last"] == 3


def _browser(monkeypatch, tmp_path) -> Browser:
    """The browser alone, in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux", display="none"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ENVIRONMENT_NAME", "chromium")
    monkeypatch.setenv(SITE_PROXY_ENV, "http://127.0.0.1:3128")
    monkeypatch.delenv(APP_SETTINGS_ENV, raising=False)
    monkeypatch.setattr(browser_log, "LOG_DIR", tmp_path / "logs")
    return Browser()


@pytest.mark.asyncio
async def test_an_op_for_an_agents_tool_call_and_agentenvs_own_acting_ops_are_entries_and_its_own_reads_are_not(
        monkeypatch, tmp_path):
    (tmp_path / "logs").mkdir()
    browser = _browser(monkeypatch, tmp_path)

    async def navigate(url):
        browser.control._acted_on("T1")

    async def tabs():
        return []

    monkeypatch.setattr(browser.control, "navigate", navigate)
    monkeypatch.setattr(browser.control, "tab_list", tabs)
    monkeypatch.setattr(browser.control, "url_of", lambda target: "https://a.test/" if target else None)
    asgi = browser.served_asgi(browser.create_app().streamable_http_app())

    def tool(name: str, call: str, args: dict) -> dict:
        return {TOOL_CALL_HEADER: tool_call_header(name, call, args)}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://app") as http:
        url = "https://a.test/?token=t"
        await http.post("/agentenv/ext/browser/navigate", json={"url": url},
                        headers={**tool("computer_navigate", "c1", {"url": url}), "AgentEnv-Role": "ada"})
        await http.get("/agentenv/ext/browser/tabs/list")
        await http.get("/agentenv/ext/browser/tabs/list", headers=tool("computer_tabs", "c2", {"action": "list"}))
        await http.post("/agentenv/ext/browser/sites/set", json={"sites": []})
        closed = await http.post("/agentenv/ext/browser/tabs/close", json={"tab": "nope"},
                                 headers=tool("computer_tabs", "c3", {"action": "close", "tab": "nope"}))
        read = (await http.get("/agentenv/ext/browser/log", params={"after": 0})).json()

    redacted = f"https://a.test/?token={REDACTED}"
    assert closed.status_code >= 400
    assert (tmp_path / "logs" / "browser-chromium.jsonl").is_file()
    assert [line["action"] for line in read["lines"]] == [
        {"via": "mcp", "op": "navigate", "acts": True, "who": "ada", "tool": "computer_navigate", "call": "c1",
         "tab": "T1", "url": "https://a.test/", "ok": True, "params": {"url": redacted}, "args": {"url": redacted}},
        {"via": "mcp", "op": "tabs/list", "acts": False, "who": "default", "tool": "computer_tabs", "call": "c2",
         "tab": None, "url": None, "ok": True},
        {"via": "agentenv", "op": "sites/set", "acts": True, "tab": None, "url": None, "ok": True,
         "params": {"sites": []}},
        {"via": "mcp", "op": "tabs/close", "acts": True, "who": "default", "tool": "computer_tabs", "call": "c3",
         "tab": None, "url": None, "ok": False, "error": "no tab nope is open; list the tabs for their ids",
         "params": {"tab": "nope"}, "args": {"action": "close", "tab": "nope"}},
    ]
    assert read["last"] == 4 and not read["more"]


@pytest.mark.asyncio
async def test_calls_begin_and_end_write_a_tool_calls_entry_from_its_header_and_are_no_entries_themselves(
        monkeypatch, tmp_path):
    (tmp_path / "logs").mkdir()
    browser = _browser(monkeypatch, tmp_path)
    asgi = browser.served_asgi(browser.create_app().streamable_http_app())
    url = "https://a.test/?token=t"
    called = {TOOL_CALL_HEADER: tool_call_header("browser_navigate", "c1", {"url": url}), "AgentEnv-Role": "ada"}

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://app") as http:
        begun = await http.post("/agentenv/ext/browser/calls/begin", json={"user": "playwright-mcp"}, headers=called)
        ended = await http.post("/agentenv/ext/browser/calls/end", json={"entry": begun.json()["entry"], "ok": True})
        headerless = await http.post("/agentenv/ext/browser/calls/begin", json={"user": "playwright-mcp"})
        unknown = await http.post("/agentenv/ext/browser/calls/end", json={"entry": 99, "ok": True})
        read = (await http.get("/agentenv/ext/browser/log", params={"after": 0})).json()

    assert [line["action"] for line in read["lines"]] == [
        {"via": "mcp", "who": "ada", "tool": "browser_navigate", "call": "c1", "acts": True, "tab": None, "url": None,
         "ok": True, "args": {"url": f"https://a.test/?token={REDACTED}"}}]
    assert ended.status_code == 200 and headerless.status_code == 400 and unknown.status_code == 400
    assert "AgentEnv-Tool-Call" in headerless.json()["error"]["message"]


@pytest.mark.asyncio
async def test_a_commands_wait_for_its_password_check_lets_none_after_it_be_sent_first(tmp_path):
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    async with _relay(log=log) as (pipe, relay, base):
        control = Control(pipe)

        async def slowly(target) -> bool:
            await asyncio.sleep(0.2)
            return False

        control.password_focused = slowly
        relay.control = control
        async with connect(f"{base}/ada") as ada:
            child = (await _ask(ada, {"id": 1, "method": "Target.attachToTarget",
                                      "params": {"targetId": "T1", "flatten": True}}))["result"]["sessionId"]
            await ada.send(json.dumps({"id": 2, "method": "Input.insertText", "params": {"text": "desk"},
                                       "sessionId": child}))
            await ada.send(json.dumps({"id": 3, "method": "Input.dispatchMouseEvent",
                                       "params": {"type": "mousePressed", "x": 1, "y": 1}, "sessionId": child}))
            answered = [json.loads(await ada.recv())["id"] for _ in range(2)]

    sent = [method for method, _ in pipe.sent if method.startswith("Input.")]
    assert sent == ["Input.insertText", "Input.dispatchMouseEvent"] and sorted(answered) == [2, 3]


@pytest.mark.asyncio
async def test_typing_still_waiting_for_its_check_when_its_client_leaves_is_logged_withheld(tmp_path):
    log = BrowserLog(tmp_path / "browser-chromium.jsonl")
    async with _relay(log=log) as (pipe, relay, base):
        control = Control(pipe)

        async def never(target) -> bool:
            await asyncio.Event().wait()

        control.password_focused = never
        relay.control = control
        async with connect(f"{base}/ada") as ada:
            child = (await _ask(ada, {"id": 1, "method": "Target.attachToTarget",
                                      "params": {"targetId": "T1", "flatten": True}}))["result"]["sessionId"]
            for n, text in ((2, "hunter2"), (3, "hunter3")):
                await ada.send(json.dumps({"id": n, "method": "Input.insertText", "params": {"text": text},
                                           "sessionId": child}))
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.2)

    typed = [line["action"] for line in log.read()["lines"] if line["action"]["method"] == "Input.insertText"]
    assert [entry["params"] for entry in typed] == [{"text": {"length": 7}}, {"text": {"length": 7}}]
    assert all(entry["ok"] is False for entry in typed) and "hunter" not in json.dumps(log.read())


@pytest.mark.asyncio
async def test_typing_that_fails_before_its_check_and_a_failed_navigation_keep_no_secrets(monkeypatch, tmp_path):
    browser = _browser(monkeypatch, tmp_path)

    async def no_tab(events):
        raise DeviceUnavailable("the browser has no tab open")

    async def unreachable(url):
        raise InvalidArgument(f"the browser couldn't open {url}: net::ERR_NAME_NOT_RESOLVED")

    monkeypatch.setattr(browser.control, "dispatch", no_tab)
    monkeypatch.setattr(browser.control, "navigate", unreachable)
    asgi = browser.served_asgi(browser.create_app().streamable_http_app())
    typed = [{"method": "Input.insertText", "params": {"text": "hunter2"}}]
    url = "https://a.test/?token=secret"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://app") as http:
        await http.post("/agentenv/ext/browser/page/input", json={"events": typed},
                        headers={TOOL_CALL_HEADER: tool_call_header("computer_type", "c1", {"text": "hunter2"})})
        await http.post("/agentenv/ext/browser/navigate", json={"url": url},
                        headers={TOOL_CALL_HEADER: tool_call_header("computer_navigate", "c2", {"url": url})})
        read = (await http.get("/agentenv/ext/browser/log")).json()

    typing, navigating = (line["action"] for line in read["lines"])
    assert typing["ok"] is False and typing["args"] == {"text": {"length": 7}}
    assert typing["params"] == {"events": [{"method": "Input.insertText", "params": {"text": {"length": 7}}}]}
    assert navigating["ok"] is False and "secret" not in navigating["error"] and REDACTED in navigating["error"]
    assert "hunter2" not in json.dumps(read) and "secret" not in json.dumps(read)


@pytest.mark.asyncio
async def test_an_op_body_too_long_to_keep_is_logged_by_its_length(monkeypatch, tmp_path):
    browser = _browser(monkeypatch, tmp_path)
    monkeypatch.setattr("agentenv_protocol.website_browser.environment.LOGGED_BODY_BYTES", 100)

    async def dispatch(events):
        return None

    monkeypatch.setattr(browser.control, "dispatch", dispatch)
    asgi = browser.served_asgi(browser.create_app().streamable_http_app())
    events = [{"method": "Input.insertText", "params": {"text": "x" * 200}}]
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://app") as http:
        answered = await http.post("/agentenv/ext/browser/page/input", json={"events": events},
                                   headers={TOOL_CALL_HEADER: tool_call_header("computer_type", "c1", {"text": "x"})})
        read = (await http.get("/agentenv/ext/browser/log")).json()

    assert answered.status_code == 200
    assert read["lines"][0]["action"]["params"] == {"too_long": True, "length": len(answered.request.content)}


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
