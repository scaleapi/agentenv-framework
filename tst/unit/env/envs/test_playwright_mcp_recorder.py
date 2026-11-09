"""The Playwright MCP browser's recorder: it finds the browser Playwright MCP launched, follows the page the agent is on
through DevTools target events, holds a still page's frame so the recording doesn't split, and serves
computer-recording/v1 on a card of its own that the gateway nests as the playwright-mcp child env. It fronts the MCP
port: the card and recording there, every other request passed to Playwright MCP behind it, streamed, and 502 when
Playwright MCP doesn't answer."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import itertools
import re
import socket
from pathlib import Path

import httpx
import pytest
import uvicorn

av = pytest.importorskip("av")

from PIL import Image as PILImage  # noqa: E402
from starlette.applications import Starlette  # noqa: E402
from starlette.responses import JSONResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402
from agentenv_protocol import client as protocol_client  # noqa: E402
from agentenv_protocol.computer.card import COMPUTER_RECORDING  # noqa: E402
from agentenv_protocol.computer.recording import RecordHints  # noqa: E402

from agent_env.env.envs import playwright_mcp  # noqa: E402
from agent_env.env.envs.playwright_mcp import recorder  # noqa: E402
from agent_env.env.gateway.constants import GATEWAY_EXTENSIONS  # noqa: E402
from agent_env.env.gateway.rest_proxy import rewrite_child_card  # noqa: E402

START, END = "/agentenv/ext/computer-recording/start", "/agentenv/ext/computer-recording/end"


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _jpeg(color: tuple, size: tuple = (320, 200)) -> str:
    out = io.BytesIO()
    PILImage.new("RGB", size, color).save(out, "JPEG")
    return base64.b64encode(out.getvalue()).decode()


class _Browser:
    """A DevTools connection standing in for Chromium's: each attach is a new session, and what's emitted is read in
    order until ``close``."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, str | None]] = []
        self.gone: set[str] = set()  # pages that close before an attach reaches them
        self._events: asyncio.Queue = asyncio.Queue()
        self._sessions = itertools.count(1)

    async def call(self, method, params=None, session=None):
        self.calls.append((method, params or {}, session))
        if method == "Target.attachToTarget":
            if params["targetId"] in self.gone:
                raise recorder.CdpError(f"No target with given id found: {params['targetId']}")
            return {"sessionId": f"{params['targetId']}-{next(self._sessions)}"}
        return {}

    async def events(self):
        while (event := await self._events.get()) is not None:
            yield event

    def emit(self, method: str, params: dict, session: str | None = None) -> None:
        self._events.put_nowait((method, params, session))

    def page(self, target: str, url: str, created: bool = True, **info) -> None:
        method = "Target.targetCreated" if created else "Target.targetInfoChanged"
        self.emit(method, {"targetInfo": {"targetId": target, "type": "page", "url": url, **info}})

    def frame(self, session: str, data: str) -> None:
        self.emit("Page.screencastFrame", {"data": data, "sessionId": 7, "metadata": {}}, session)

    def close(self) -> None:
        self._events.put_nowait(None)

    def attached(self) -> list[str]:
        return [params["targetId"] for method, params, _ in self.calls if method == "Target.attachToTarget"]


def _proc(root: Path, pid: int, argv: list[str], started: int | None) -> None:
    entry = root / str(pid)
    entry.mkdir()
    (entry / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    if started is not None:
        (entry / "stat").write_text(f"{pid} (chrome) S " + "0 " * 18 + f"{started} 0 0")


def test_the_browser_is_the_newest_chromium_with_a_debugging_port(tmp_path):
    _proc(tmp_path, 1, ["node", "cli.js", "--port", "18765"], 10)
    _proc(tmp_path, 40, ["chrome", "--remote-debugging-port=1111", "--headless"], 100)
    _proc(tmp_path, 41, ["chrome", "--type=renderer", "--remote-debugging-port=3333"], 300)
    _proc(tmp_path, 50, ["chrome", "--headless", "--remote-debugging-port=2222"], 200)
    _proc(tmp_path, 51, ["chrome", "--remote-debugging-port=0"], 400)
    _proc(tmp_path, 60, ["chrome", "--remote-debugging-port=4444"], None)  # exited while it was read
    (tmp_path / "self").mkdir()
    assert recorder.find_browser(tmp_path) == 2222
    assert recorder.find_browser(tmp_path / "1") is None


@pytest.mark.asyncio
async def test_the_screencast_follows_new_and_navigated_pages_and_falls_back_when_one_closes():
    browser, red, blue = _Browser(), _jpeg((255, 0, 0)), _jpeg((0, 0, 255))
    screencast = recorder.Screencast(640, find=lambda: None, connect=None)
    browser.page("A", "about:blank")
    browser.emit("Target.targetCreated", {"targetInfo": {"targetId": "F", "type": "iframe", "url": "x"}})
    browser.frame("A-1", red)
    browser.page("B", "about:blank", openerId="A")
    browser.frame("A-1", blue)  # A's last frame, after the switch: acknowledged, not shown
    watching = asyncio.create_task(screencast.watch(browser))
    await asyncio.sleep(0.05)
    assert (screencast.page, screencast.latest, screencast.frames) == ("B", base64.b64decode(red), 1)

    browser.frame("B-2", blue)
    browser.page("B", "about:blank", created=False, attached=True)  # nothing about it changed that matters
    browser.page("A", "https://shop.test/cart", created=False)
    await asyncio.sleep(0.05)
    assert (screencast.page, screencast.latest) == ("A", base64.b64decode(blue))

    browser.emit("Target.targetDestroyed", {"targetId": "A"})
    await asyncio.sleep(0.05)
    assert screencast.page == "B"
    browser.emit("Target.targetDestroyed", {"targetId": "B"})
    browser.close()
    await watching

    assert browser.attached() == ["A", "B", "A", "B"]
    assert (screencast.page, screencast.latest) == (None, None)
    started = [(params, session) for method, params, session in browser.calls if method == "Page.startScreencast"]
    assert started[0] == ({"format": "jpeg", "quality": 80, "everyNthFrame": 1, "maxWidth": 640, "maxHeight": 640},
                          "A-1")
    assert [session for method, _, session in browser.calls if method == "Page.screencastFrameAck"] == ["A-1", "A-1",
                                                                                                        "B-2"]
    assert [params["sessionId"] for method, params, _ in browser.calls if method == "Target.detachFromTarget"] == [
        "A-1", "B-2", "A-3", "B-4"]


@pytest.mark.asyncio
async def test_a_page_that_closes_before_its_screencast_starts_hands_back_to_the_page_before_it():
    browser, red = _Browser(), _jpeg((255, 0, 0))
    screencast = recorder.Screencast(None, find=lambda: None, connect=None)
    browser.gone.add("B")
    browser.page("A", "https://shop.test/")
    browser.frame("A-1", red)
    browser.page("B", "about:blank", openerId="A")  # a popup that closes while the attach is on its way
    browser.emit("Target.targetDestroyed", {"targetId": "B"})
    browser.close()
    await screencast.watch(browser)

    assert browser.attached() == ["A", "B", "A"]
    assert (screencast.page, screencast.latest) == ("A", base64.b64decode(red))
    assert screencast._session == "A-2"


def _recorder(browser: _Browser) -> recorder.BrowserRecorder:
    class _Connection:
        async def __aenter__(self):
            return browser

        async def __aexit__(self, *exc):
            return False

    return recorder.BrowserRecorder(find=lambda: 9222, connect=lambda port: _Connection())


@pytest.mark.asyncio
async def test_a_still_page_is_filmed_again_after_the_hold_and_nothing_is_filmed_without_a_page(monkeypatch, home):
    monkeypatch.setattr(recorder, "HOLD_S", 0.2)
    browser = _Browser()
    frames = _recorder(browser).frames(RecordHints("display", 20, None, False, 15, home))
    first = asyncio.ensure_future(anext(frames))
    await asyncio.sleep(0.2)
    assert not first.done()  # no page yet
    browser.page("A", "https://shop.test/")
    browser.frame("A-1", _jpeg((0, 255, 0)))
    one = await asyncio.wait_for(first, 1)
    two = await asyncio.wait_for(anext(frames), 1)
    assert 0.15 <= (two.captured_at - one.captured_at).total_seconds() < 0.5
    assert two.image.convert("RGB").getpixel((5, 5))[1] > 200 and two.image.size == (320, 200)
    await frames.aclose()


@pytest.mark.asyncio
async def test_a_recording_through_the_card_is_an_mp4_of_the_page(home):
    browser = _Browser()
    app = recorder.create_app(_recorder(browser))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://recorder") as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
        started = (await http.post(START, json={"request_id": "run-1:start", "fps": 10, "cursor": True})).json()
        browser.page("A", "https://shop.test/")
        for shade in range(0, 250, 25):
            browser.frame("A-1", _jpeg((shade, 0, 255 - shade)))
            await asyncio.sleep(0.1)
        manifest = (await http.post(END, json={"recording_id": started["recording_id"]})).json()

    assert card["name"] == "playwright-mcp" and card["capabilities"]["operations"] == []
    assert [e["uri"] for e in card["capabilities"]["extensions"]] == [COMPUTER_RECORDING]
    params = card["capabilities"]["extensions"][0]["params"]
    assert params["cursor"] is False and params["source"] == "stream" and set(params["methods"]) == {"start", "end"}
    assert (manifest["ended_by"], manifest["cursor"], manifest["source"]) == ("end", False, "stream")
    (segment,) = manifest["segments"]
    assert (segment["width"], segment["height"], segment["uploaded"]) == (320, 200, False)
    with av.open(str(home / ".agentenv-recordings" / segment["path"])) as video:
        decoded = list(video.decode(video=0))
    assert len(decoded) == segment["frames"] >= 8 and decoded[0].width == 320


@pytest.mark.asyncio
async def test_the_gateway_nests_the_card_as_a_child_env_whose_recording_is_invoked_under_its_proxy(monkeypatch):
    card = recorder.create_app(_recorder(_Browser()))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=card), base_url="http://recorder") as http:
        served = (await http.get("/.well-known/agent-env.json")).json()
    composed = {"name": "env", "capabilities": {"extensions": GATEWAY_EXTENSIONS},
                "children_environments": [rewrite_child_card("mcp-playwright-mcp", served, GATEWAY_EXTENSIONS)]}
    child = protocol_client.find_child(composed, playwright_mcp.PLAYWRIGHT_MCP_ENVIRONMENT_NAME)
    methods = protocol_client.find_extension(child, COMPUTER_RECORDING)["params"]["methods"]
    assert {op: m["endpoint"] for op, m in methods.items()} == {
        "start": "/svc/mcp-playwright-mcp/agentenv/ext/computer-recording/start",
        "end": "/svc/mcp-playwright-mcp/agentenv/ext/computer-recording/end"}


def test_the_image_fronts_the_mcp_port_with_the_recorder_and_serves_the_card_where_deploys_read_it():
    entrypoint = (Path(playwright_mcp.__file__).parent / "entrypoint.sh").read_text()

    def default(name: str) -> int:
        return int(re.search(rf'{name}="\$\{{{name}:-(\d+)\}}"', entrypoint).group(1))

    assert default("RECORDER_PORT") == recorder.DEFAULT_PORT == playwright_mcp.PLAYWRIGHT_MCP_RECORDER_PORT
    assert default("MCP_PORT") == recorder.DEFAULT_MCP_PORT
    assert default("PLAYWRIGHT_MCP_PORT") == recorder.DEFAULT_PLAYWRIGHT_MCP_PORT
    assert '--host 127.0.0.1' in entrypoint and '--port "${PLAYWRIGHT_MCP_PORT}"' in entrypoint
    assert "--isolated" in entrypoint  # the remote debugging port the recorder finds comes with it


def _behind(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://playwright-mcp")


@pytest.mark.asyncio
async def test_the_front_serves_the_card_and_recording_and_passes_everything_else_to_playwright_mcp():
    passed = []

    async def mcp(request):
        passed.append((request.method, request.url.path, request.url.query, await request.body(),
                       request.headers.get("mcp-session-id"), request.headers.get("host")))
        return JSONResponse({"jsonrpc": "2.0", "id": 1, "result": {}}, headers={"Mcp-Session-Id": "s-2"})

    playwright = Starlette(routes=[Route("/mcp", mcp, methods=["POST", "DELETE"])])
    async with _behind(playwright) as behind:
        front = recorder.create_app(_recorder(_Browser()), behind)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=front), base_url="http://front") as http:
            card = (await http.get("/.well-known/agent-env.json")).json()
            answer = await http.post("/mcp?x=1", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                                     headers={"Mcp-Session-Id": "s-1"})
            closed = await http.delete("/mcp", headers={"Mcp-Session-Id": "s-2"})

    assert card["name"] == "playwright-mcp"
    assert [e["uri"] for e in card["capabilities"]["extensions"]] == [COMPUTER_RECORDING]
    assert (answer.status_code, answer.json()["result"], answer.headers["mcp-session-id"]) == (200, {}, "s-2")
    assert passed[0][:3] == ("POST", "/mcp", "x=1") and b'"tools/list"' in passed[0][3] and passed[0][4] == "s-1"
    assert passed[0][5] == "front"  # the Host the caller sent, not Playwright MCP's own address
    assert closed.status_code == 200 and (passed[1][0], passed[1][4]) == ("DELETE", "s-2")


@pytest.mark.asyncio
async def test_the_front_answers_502_naming_playwright_mcp_when_it_doesnt_answer():
    def refuse(request):
        raise httpx.ConnectError("Connection refused", request=request)

    behind = httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://playwright-mcp")
    front = recorder.create_app(_recorder(_Browser()), behind)
    async with behind, httpx.AsyncClient(transport=httpx.ASGITransport(app=front), base_url="http://front") as http:
        answer = await http.post("/mcp", json={})
        card = await http.get("/.well-known/agent-env.json")

    assert answer.status_code == 502 and answer.text.startswith("Playwright MCP isn't answering: ConnectError")
    assert card.status_code == 200


async def _serve(app) -> tuple[uvicorn.Server, int]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    config = uvicorn.Config(app, lifespan="off", log_level="critical", access_log=False)
    config.load()
    server = uvicorn.Server(config)
    server.lifespan = config.lifespan_class(config)
    await server.startup(sockets=[sock])
    return server, sock.getsockname()[1]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def _accepts(port: int) -> bool:
    try:
        _, writer = await asyncio.open_connection("127.0.0.1", port)
    except OSError:
        return False
    writer.close()
    return True


@pytest.mark.asyncio
@pytest.mark.usefixtures("socket_enabled")
async def test_the_card_is_served_at_once_and_the_mcp_port_opens_only_once_playwright_mcp_listens(monkeypatch):
    mcp, card, behind = _free_port(), _free_port(), _free_port()
    for name, port in (("MCP_PORT", mcp), ("RECORDER_PORT", card), ("PLAYWRIGHT_MCP_PORT", behind)):
        monkeypatch.setenv(name, str(port))
    monkeypatch.setenv("MCP_HOST", "127.0.0.1")
    serving = asyncio.create_task(recorder.serve())
    playwright = None
    try:
        for _ in range(100):
            if await _accepts(card):
                break
            await asyncio.sleep(0.05)
        await asyncio.sleep(3 * recorder.LISTENING_POLL_S)
        card_first, mcp_first = await _accepts(card), await _accepts(mcp)
        playwright = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", behind)
        for _ in range(100):
            if await _accepts(mcp):
                break
            await asyncio.sleep(0.05)
        async with httpx.AsyncClient() as http:
            served = (await http.get(f"http://127.0.0.1:{mcp}/.well-known/agent-env.json")).json()
    finally:
        serving.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await serving
        if playwright is not None:
            playwright.close()

    assert (card_first, mcp_first) == (True, False)
    assert served["name"] == "playwright-mcp"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec_version", ["2.0", "2.4"], ids=["disconnect-message", "send-raises"])
async def test_an_answer_is_closed_when_its_caller_leaves_before_the_first_chunk(spec_version):
    """Under ASGI 2.4 a caller that left shows as ``send`` raising; before it, as a disconnect message."""
    closed = asyncio.Event()

    class Held(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.Event().wait()
            yield b""

        async def aclose(self) -> None:
            closed.set()

    answer = httpx.Response(200, headers=[("Set-Cookie", "a=1"), ("Set-Cookie", "b=2"), ("Connection", "close")],
                            stream=Held())
    passed = recorder.PassedAnswer(answer)

    async def receive():
        return {"type": "http.disconnect"}

    async def send(message):
        if spec_version == "2.4":
            raise OSError("the caller left")

    with contextlib.suppress(Exception):
        await asyncio.wait_for(passed({"type": "http", "asgi": {"spec_version": spec_version}}, receive, send), 5)

    assert closed.is_set()
    assert [value for key, value in passed.raw_headers if key == b"set-cookie"] == [b"a=1", b"b=2"]
    assert b"connection" not in dict(passed.raw_headers)


@pytest.mark.asyncio
@pytest.mark.usefixtures("socket_enabled")
async def test_the_front_streams_an_event_stream_as_playwright_mcp_sends_it():
    release = asyncio.Event()

    async def stream(reader, writer):
        await reader.readuntil(b"\r\n\r\n")
        event = b"event: message\ndata: 1\n\n"
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                     + f"{len(event):x}\r\n".encode() + event + b"\r\n")
        await writer.drain()
        await release.wait()
        writer.write(b"0\r\n\r\n")
        await writer.drain()
        writer.close()

    playwright = await asyncio.start_server(stream, "127.0.0.1", 0)
    behind = httpx.AsyncClient(base_url=f"http://127.0.0.1:{playwright.sockets[0].getsockname()[1]}")
    server, port = await _serve(recorder.create_app(_recorder(_Browser()), behind))
    try:
        async with httpx.AsyncClient() as http, http.stream("GET", f"http://127.0.0.1:{port}/mcp") as answer:
            lines = answer.aiter_lines()
            first = await asyncio.wait_for(anext(lines), 5)
            release.set()
            rest = [line async for line in lines]
    finally:
        await server.shutdown()
        playwright.close()
        await behind.aclose()

    assert answer.headers["content-type"] == "text/event-stream" and first == "event: message"
    assert rest[:1] == ["data: 1"]
