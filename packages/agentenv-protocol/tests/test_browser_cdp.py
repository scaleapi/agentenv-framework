"""A website browser's CDP addresses: a user's token minted once, kept by hash only, so it still works after the server
restarts and the computer's session can't use the file; the relay giving each WebSocket connection its own
browser-level session over one pipe, its answers and events and nobody else's, until the browser goes; and the cdp
operation and discovery answered at the address the call reached."""
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
from websockets.exceptions import ConnectionClosedError  # noqa: E402

from agentenv_protocol import browser_cdp  # noqa: E402
from agentenv_protocol.browser_cdp import NO_SESSION, Relay, Tokens  # noqa: E402
from agentenv_protocol.computer import Launch, Platform, Ready, app, computer_app, linux  # noqa: E402
from agentenv_protocol.computer.app import APP_SETTINGS_ENV  # noqa: E402
from agentenv_protocol.website_browser import SITE_PROXY_ENV, WebsiteBrowserEnvironment  # noqa: E402

FORWARDED = {"x-forwarded-host": "computer.test:443", "x-forwarded-proto": "https",
             "x-forwarded-prefix": "/svc/chromium"}


@computer_app(supports=[{"os": {"type": "linux", "display": ["x11", "none"]}, "arch": ["amd64"]}])
@linux(launch=Launch(command=["chromium/chrome"], window={"class": "Chromium"}, ready=Ready(window=True, timeout_s=30)))
class Browser(WebsiteBrowserEnvironment):
    version = "156.0.8078.4"


class Pipe:
    """A browser's pipe: each connection's browser-level session ``b<n>``, an attach on a session making its child
    ``<session>-c`` and saying so on that session, ``Boom`` failing, anything else echoed; every request recorded."""

    def __init__(self) -> None:
        self.listeners, self.sent, self.sessions = [], [], 0

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
        if method == "Target.attachToTarget":
            child = f"{session_id}-c"
            self.emit({"method": "Target.attachedToTarget", "sessionId": session_id,
                       "params": {"sessionId": child, "targetInfo": {"targetId": params["targetId"]}}})
            return {"result": {"sessionId": child}}
        if method == "Boom":
            return {"error": {"code": -32000, "message": "boom"}}
        return {"result": {"method": method, "session": session_id}}


@contextlib.asynccontextmanager
async def _relay():
    """The relay over a fake pipe, served on loopback with each connection's user its path's: the pipe, the relay and
    the server's ws:// base."""
    pipe = Pipe()
    relay = Relay(pipe)

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
    token = ada.http_url.split("/cdp/")[1].strip("/")
    restarted = Tokens(path)

    assert again == ada and bob != ada and ada.http_url == f"https://computer.test/svc/chromium/cdp/{token}/"
    assert ada.ws_url == f"wss://computer.test/svc/chromium/cdp/{token}/devtools/browser"
    assert token not in path.read_text() and restarted.address(token) == ada and tokens.address("guess") is None
    assert restarted.mint("ada", "https://computer.test").http_url != ada.http_url


@pytest.mark.asyncio
async def test_each_connection_gets_its_own_browser_session_and_only_its_own_answers_and_events():
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


@pytest.mark.asyncio
async def test_the_cdp_operation_and_discovery_answer_at_the_address_reached_and_refuse_other_tokens(monkeypatch,
                                                                                                      tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(app, "detect", lambda: Platform(os="linux", display="none"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("ENVIRONMENT_NAME", "chromium")
    monkeypatch.setenv(SITE_PROXY_ENV, "http://127.0.0.1:3128")
    monkeypatch.delenv(APP_SETTINGS_ENV, raising=False)
    browser = Browser()

    async def ready():
        return None

    async def call(method, params=None, *, session_id=None, timeout_s=None):
        assert method == "Browser.getVersion"
        return {"product": "HeadlessChrome/156.0.8078.4", "protocolVersion": "1.3", "userAgent": "UA",
                "jsVersion": "15.6", "revision": "@abc"}

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

    assert again == ada and ada["ws_url"].startswith("wss://computer.test:443/svc/chromium/cdp/")
    assert version["Browser"] == "HeadlessChrome/156.0.8078.4"
    assert version["webSocketDebuggerUrl"] == f"wss://seen.test/svc/chromium{root}devtools/browser"
    assert listed == [{"id": "T1", "type": "page", "title": "Site", "url": "https://site.test/", "description": "",
                       "devtoolsFrontendUrl": ""}]
    assert refused.status_code == 403
    computer_app_params = next(e["params"] for e in card["capabilities"]["extensions"]
                               if e["uri"] == "urn:agentenv:computer-app/v1")
    browser_methods = next(e["params"]["methods"] for e in card["capabilities"]["extensions"]
                           if e["uri"] == "urn:agentenv:browser/v1")
    assert computer_app_params["agent_routes"] == ["/cdp/"]
    assert browser_methods["cdp"]["request"]["properties"] == {"user": {"type": "string"}}
