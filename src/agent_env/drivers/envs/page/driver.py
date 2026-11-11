"""The page driver, for a computer with no display whose one website browser runs alone: its screen is the browser's
active tab. A screenshot is the page, and input reaches the page as CDP input events, both through the browser's own
server on its socket (``browser/v1``'s ``page/*`` operations), since only that server holds the browser's debugging pipe.
The browser ending on its own is an infrastructure failure, since nothing here launches it again. Beside its own tools it
serves Playwright MCP's ``browser_*``, from a Playwright MCP it runs attached to the browser over a CDP address the
browser's relay issues to it. Needs ``agentenv-framework-protocol[computer]`` where it runs."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import functools
import json
import logging
import math
import os
import socket
import string
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from agentenv_protocol import EnvironmentTool, ForwardedTool, ToolContext, client
from agentenv_protocol.computer import (
    ComputerDriverEnvironment,
    DeviceUnavailable,
    Display,
    FileChooser,
    Image,
    InvalidArgument,
    PageDialog,
    PageState,
    ReadyCheck,
    Recording,
    Tab,
    Unsupported,
    capture,
    click,
    computer_driver,
    dialog,
    display,
    drag,
    history,
    key_down,
    key_up,
    navigate,
    page_state,
    pointer_down,
    pointer_move,
    pointer_position,
    pointer_up,
    press,
    scroll,
    tabs,
    type_text,
    upload,
)
from agentenv_protocol.computer import apps
from agentenv_protocol.computer.served import served
from agentenv_protocol.website_browser import BROWSER

BUTTONS = {"left": 1, "right": 2, "middle": 4, "back": 8, "forward": 16}
MODIFIERS = {"Alt": 1, "Control": 2, "Meta": 4, "Shift": 8}
#: Keys that aren't characters: each one's DOM code, its Windows virtual key code, and the text a press of it types.
NAMED = {
    "Enter": ("Enter", 13, "\r"), "Tab": ("Tab", 9, ""), "Escape": ("Escape", 27, ""),
    "Backspace": ("Backspace", 8, ""), "Delete": ("Delete", 46, ""), "Insert": ("Insert", 45, ""),
    "ArrowUp": ("ArrowUp", 38, ""), "ArrowDown": ("ArrowDown", 40, ""), "ArrowLeft": ("ArrowLeft", 37, ""),
    "ArrowRight": ("ArrowRight", 39, ""), "Home": ("Home", 36, ""), "End": ("End", 35, ""),
    "PageUp": ("PageUp", 33, ""), "PageDown": ("PageDown", 34, ""), "Control": ("ControlLeft", 17, ""),
    "Shift": ("ShiftLeft", 16, ""), "Alt": ("AltLeft", 18, ""), "Meta": ("MetaLeft", 91, ""),
    " ": ("Space", 32, " "), **{f"F{n}": (f"F{n}", 111 + n, "") for n in range(1, 13)},
}
#: The US layout's punctuation: each key's DOM code and virtual key code, and the character Shift makes of it.
PUNCTUATION = {"-": ("Minus", 189, "_"), "=": ("Equal", 187, "+"), "[": ("BracketLeft", 219, "{"),
               "]": ("BracketRight", 221, "}"), "\\": ("Backslash", 220, "|"), ";": ("Semicolon", 186, ":"),
               "'": ("Quote", 222, '"'), ",": ("Comma", 188, "<"), ".": ("Period", 190, ">"),
               "/": ("Slash", 191, "?"), "`": ("Backquote", 192, "~")}
SHIFTED_DIGITS = dict(zip(")!@#$%^&*(", string.digits))
SHIFTED = {**{shifted: key for key, (_, _, shifted) in PUNCTUATION.items()}, **SHIFTED_DIGITS}
TYPED = {"\n": "Enter", "\t": "Tab"}
#: How far one scroll step moves, and how far apart a drag's pointer moves are, in pixels.
WHEEL_PX = 100
DRAG_STEP_PX = 20
#: A pixel of a tab with nothing in it.
BLANK = b"\xff\xff\xff"
#: How long one call to the browser's server may take.
CALL_TIMEOUT_S = 30.0
#: Playwright MCP, as its install lays it out beside the driver, and its tools as 0.0.84 lists them with ``--caps
#: vision``, which an integration test checks against the one installed.
HERE = Path(__file__).resolve().parent
NODE = HERE / "node" / "bin" / "node"
PLAYWRIGHT_MCP = HERE / "node_modules" / "@playwright" / "mcp" / "cli.js"
PLAYWRIGHT_MCP_TOOLS = HERE / "playwright-mcp-tools.json"
#: Playwright MCP's tools that run the agent's own code, in the page and in Playwright MCP's own process: hidden until
#: a task's tool access names them.
HIDDEN_TOOLS = frozenset({"browser_evaluate", "browser_run_code_unsafe"})
#: Whom the browser issues Playwright MCP's CDP address to, as its record names the connections.
PLAYWRIGHT_MCP_USER = "playwright-mcp"
#: What Playwright MCP's process keeps of the driver's environment, which holds the deploy's tokens and settings.
KEPT_ENVIRONMENT = ("PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ", "TMPDIR")
#: Where Playwright MCP writes what it names itself, and its log.
PLAYWRIGHT_MCP_OUTPUT = Path(tempfile.gettempdir()) / "agentenv-playwright-mcp"
#: How long Playwright MCP may take to listen or to open a session, and one of its tools to answer.
PLAYWRIGHT_MCP_START_S = 30.0
PLAYWRIGHT_MCP_CALL_TIMEOUT_S = 300.0
#: How often a start checks whether Playwright MCP listens yet, and how long a stop waits for it to end on SIGTERM
#: before it's killed.
PLAYWRIGHT_MCP_POLL_S = 0.1
PLAYWRIGHT_MCP_STOP_S = 10.0
#: How long a caller's session on Playwright MCP is kept with no call: a caller's own MCP session ends with no word to
#: the driver, and each holds a connection to the browser.
PLAYWRIGHT_MCP_SESSION_IDLE_S = 3600.0

logger = logging.getLogger(__name__)


@computer_driver(name="page", supports=[{"os": {"type": "linux", "display": ["none"]}, "arch": ["amd64", "arm64"]}])
class PageDriver(ComputerDriverEnvironment):
    buttons = ("left", "right", "middle", "back", "forward")
    max_click_count = 3
    max_drag_points = 50
    keys = sorted({*NAMED, *string.ascii_letters, *string.digits, *PUNCTUATION, *SHIFTED})
    recording = Recording(max_fps=10)
    screen_kind = "page"
    notes = ("This computer is one web page: the browser's active tab, with no address bar, tab strip or desktop "
             "around it. Its screen is the page's viewport. Go to a URL with computer_navigate, back, forward or "
             "reload with computer_history, and list, open, switch or close tabs with computer_tabs. Each screenshot "
             "names the page, and an open dialog, which computer_dialog answers, or file chooser, which "
             "computer_upload gives files in ~/Uploads. Its browser_* tools, Playwright MCP's, act on the same "
             "browser by the page's accessibility snapshot; browser_file_upload takes files in ~/Uploads too.")

    def __init__(self) -> None:
        super().__init__()
        self._browser: Optional[tuple[Path, dict]] = None
        self._pointer = (0.0, 0.0)
        self._buttons = 0
        self._held: list[str] = []
        self._playwright = _PlaywrightMcp(self._playwright_endpoint)

    def forwarded_tools(self) -> list[ForwardedTool]:
        """Playwright MCP's tools, as its manifest lists them, each called in the caller's own session there; those
        that run the agent's own code hidden until a task's tool access names them."""
        return [ForwardedTool(EnvironmentTool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]),
                              functools.partial(self._playwright_call, t["name"]), t.get("annotations"),
                              t.get("outputSchema"), hidden_by_default=t["name"] in HIDDEN_TOOLS)
                for t in json.loads(PLAYWRIGHT_MCP_TOOLS.read_text(encoding="utf-8"))["tools"]]

    async def _playwright_call(self, tool: str, /, **arguments: Any) -> CallToolResult:
        return await self._playwright.call(tool, **arguments)

    async def _playwright_endpoint(self) -> str:
        """The CDP address the browser issues Playwright MCP, at this driver's own loopback address, where its relay is
        reached as any agent reaches it: the same one until the browser's data is reset, which ends it."""
        folder, _ = await self._found()
        origin = f"http://127.0.0.1:{os.environ.get('MCP_PORT', '18765')}/svc/{folder.name}"
        return (await self._call("cdp", {"user": PLAYWRIGHT_MCP_USER, "origin": origin}))["ws_url"]

    @display
    async def viewport(self) -> Display:
        size = await self._call("page/viewport")
        return Display(size["width"], size["height"], 1.0)

    @capture
    async def page(self) -> Image:
        """The page, or a blank one where the browser shows nothing, as a desktop browser shows a tab with nothing
        in it; with the page it shows, read with it."""
        shot = await self._call("page/capture")
        page = _page_state(shot["page"]) if shot.get("page") else None
        if shot.get("blank"):
            size = await self.viewport()
            return Image(size.width, size.height, BLANK * (size.width * size.height), "RGB", page)
        data = base64.b64decode(shot["png"])
        return Image(int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big"), data, "png", page)

    async def ready_checks(self) -> list[ReadyCheck]:
        try:
            folder, _ = await self._found()
        except DeviceUnavailable as e:
            return [ReadyCheck("browser", False, e.message)]
        failed = apps.ended_on_its_own(folder)
        return [ReadyCheck("browser", failed is None, failed or f"the page of the browser in {folder.name}",
                           final=failed is not None)]

    @click
    async def click_at(self, x, y, button, count, modifiers) -> None:
        mask = _modifiers(modifiers) | self._mask()
        events = [_mouse("mouseMoved", x, y, modifiers=mask)]
        for n in range(1, count + 1):
            events += [_mouse("mousePressed", x, y, button, BUTTONS[button], n, mask),
                       _mouse("mouseReleased", x, y, button, 0, n, mask)]
        self._pointer = (x, y)
        await self._input(events)

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers) -> None:
        mask, held = _modifiers(modifiers) | self._mask(), BUTTONS[button]
        steps = _steps(path)
        pause = duration_ms / 1000 / len(steps)
        await self._input([_mouse("mouseMoved", *path[0], modifiers=mask),
                           _mouse("mousePressed", *path[0], button, held, 1, mask)])
        try:
            for x, y in steps:
                await self._input([_mouse("mouseMoved", x, y, "none", held, 0, mask)])
                await asyncio.sleep(pause)
        finally:
            await self._input([_mouse("mouseReleased", *path[-1], button, 0, 1, mask)])
        self._pointer = tuple(path[-1])

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers) -> None:
        """One wheel event a notch, as a mouse wheel sends them."""
        dx = {"left": -1, "right": 1}.get(direction, 0) * WHEEL_PX
        dy = {"up": -1, "down": 1}.get(direction, 0) * WHEEL_PX
        self._pointer = (x, y)
        await self._input([{"method": "Input.dispatchMouseEvent", "params": {
            "type": "mouseWheel", "x": x, "y": y, "deltaX": dx, "deltaY": dy,
            "modifiers": _modifiers(modifiers) | self._mask()}}] * amount)

    @pointer_move
    async def move(self, x, y) -> None:
        self._pointer = (x, y)
        await self._input([_mouse("mouseMoved", x, y, "none", self._buttons, 0, self._mask())])

    @pointer_down
    async def down(self, button) -> None:
        self._buttons |= BUTTONS[button]
        await self._input([_mouse("mousePressed", *self._pointer, button, self._buttons, 1, self._mask())])

    @pointer_up
    async def up(self, button) -> None:
        self._buttons &= ~BUTTONS[button]
        await self._input([_mouse("mouseReleased", *self._pointer, button, self._buttons, 1, self._mask())])

    @pointer_position
    async def where(self) -> tuple[int, int]:
        return round(self._pointer[0]), round(self._pointer[1])

    @type_text
    async def type_in(self, text) -> None:
        """Each character as the keystroke that types it, with the keys held down still held: Shift held types ``a`` as
        ``A``, and Control held makes it a shortcut; a character no key types is inserted as text."""
        held, events = self._mask(), []
        for ch in text:
            key = TYPED.get(ch, ch)
            if key in NAMED or _printable(key):
                mask = held | (MODIFIERS["Shift"] if key in SHIFTED or key.isupper() else 0)
                events += _keystroke(_shifted(key) if mask & MODIFIERS["Shift"] else key, mask)
            else:
                events.append({"method": "Input.insertText", "params": {"text": ch}})
        await self._input(events)

    @press
    async def press_keys(self, keys) -> None:
        events, mask = [], self._mask()
        for key in keys:
            mask |= MODIFIERS.get(key, 0)
            events.append(_key("keyDown", key, mask))
        for key in reversed(keys):
            mask &= ~MODIFIERS.get(key, 0)
            events.append(_key("keyUp", key, mask))
        await self._input(events)

    @key_down
    async def keys_down(self, keys) -> None:
        self._held += [key for key in keys if key not in self._held]
        await self._input([_key("keyDown", key, self._mask()) for key in keys])

    @key_up
    async def keys_up(self, keys) -> None:
        self._held = [key for key in self._held if key not in keys]
        await self._input([_key("keyUp", key, self._mask()) for key in keys])

    @navigate
    async def go_to(self, url) -> None:
        await self._call("navigate", {"url": url})

    @history
    async def move_in_history(self, direction) -> None:
        await self._call("history", {"direction": direction})

    @tabs
    async def tab_action(self, action, tab, url) -> list[Tab]:
        if action == "open":
            await self._call("tabs/open", {"url": url} if url else {})
        elif action in ("switch", "close"):
            await self._call("tabs/activate" if action == "switch" else "tabs/close", {"tab": tab})
        listed = (await self._call("tabs/list"))["tabs"]
        return [Tab(t["id"], t["url"], t["title"], t["active"]) for t in listed]

    @dialog
    async def answer_dialog(self, accept, text) -> None:
        await self._call("dialog/answer", {"accept": accept, **({"text": text} if text is not None else {})})

    @upload
    async def choose_files(self, paths) -> None:
        await self._call("files/choose", {"paths": paths})

    @page_state
    async def page_now(self) -> PageState:
        return _page_state(await self._call("page/state"))

    def _mask(self) -> int:
        return _modifiers(self._held)

    async def _input(self, events: list[dict]) -> None:
        await self._call("page/input", {"events": events})

    async def _call(self, op: str, params: Optional[dict] = None) -> Any:
        """``browser/v1``'s ``op`` on the browser's server; DeviceUnavailable when it can't answer or says why not, and
        once the browser has ended on its own, why, even after its server has launched another."""
        folder, card = await self._found()
        if (failed := _ended(folder)) is not None:
            raise DeviceUnavailable(failed)
        method = client.find_extension_method(card, BROWSER, op)
        if method is None:
            raise Unsupported(f"the browser in {folder.name} serves no {op}")
        try:
            async with _http(folder) as http:
                answer = await http.request(method["method"], method["endpoint"],
                                            **({"json": params} if method["method"] != "GET" else {}))
        except (httpx.HTTPError, OSError) as e:
            raise DeviceUnavailable(_ended(folder) or f"the browser's server isn't answering: {type(e).__name__}: {e}"
                                    ) from None
        if answer.status_code >= 400:
            if (refused := _refusal(answer)) is not None:
                raise InvalidArgument(refused)
            raise DeviceUnavailable(_ended(folder) or f"the browser couldn't {op}: {answer.text[:300]}")
        return answer.json()

    async def _found(self) -> tuple[Path, dict]:
        """The folder and card of the computer's one website browser, found once its server answers."""
        if self._browser is None:
            for folder in self._computer.apps:
                card = await apps.served_card(folder)
                if card is not None and client.find_extension_method(card, BROWSER, "page/capture"):
                    self._browser = (folder, card)
                    break
            else:
                raise DeviceUnavailable("no website browser on this computer answers yet")
        return self._browser


def _page_state(state: dict) -> PageState:
    open_dialog, chooser = state.get("dialog"), state.get("file_chooser")
    return PageState(state["tab"], state["url"], state["title"], PageDialog(**open_dialog) if open_dialog else None,
                     FileChooser(**chooser) if chooser else None)


def _refusal(answer: httpx.Response) -> Optional[str]:
    """Why the browser's server refused a call it was given wrong, which the agent can put right."""
    try:
        error = answer.json().get("error") or {}
    except ValueError:
        return None
    return error.get("message") if error.get("code") == "invalid_argument" else None


def _ended(folder: Path) -> Optional[str]:
    if (failed := apps.ended_on_its_own(folder)) is not None:
        return f"the browser in {folder.name} has failed: {failed}"
    return None


def _http(folder: Path) -> httpx.AsyncClient:
    return served(folder).client(CALL_TIMEOUT_S)


def _mouse(kind: str, x: float, y: float, button: str = "none", buttons: int = 0, count: int = 0,
           modifiers: int = 0) -> dict:
    return {"method": "Input.dispatchMouseEvent", "params": {
        "type": kind, "x": x, "y": y, "button": button, "buttons": buttons, "clickCount": count, "modifiers": modifiers}}


def _keystroke(key: str, modifiers: int) -> list[dict]:
    return [_key("keyDown", key, modifiers), _key("keyUp", key, modifiers)]


def _key(kind: str, key: str, modifiers: int) -> dict:
    """A CDP key event for the W3C key ``key`` with ``modifiers`` held: one with Control, Alt or Meta held types
    nothing, so it reaches the page as a shortcut."""
    code, vk, text = _codes(key)
    if modifiers & (MODIFIERS["Control"] | MODIFIERS["Alt"] | MODIFIERS["Meta"]):
        text = ""
    params = {"type": ("keyDown" if text else "rawKeyDown") if kind == "keyDown" else "keyUp", "key": key,
              "code": code, "windowsVirtualKeyCode": vk, "nativeVirtualKeyCode": vk, "modifiers": modifiers}
    if kind == "keyDown" and text:
        params.update(text=text, unmodifiedText=text)
    return {"method": "Input.dispatchKeyEvent", "params": params}


def _codes(key: str) -> tuple[str, int, str]:
    """The DOM code, virtual key code and text of ``key`` on a US layout."""
    if key in NAMED:
        return NAMED[key]
    if len(key) == 1 and key in string.ascii_letters:
        return f"Key{key.upper()}", ord(key.upper()), key
    if len(key) == 1 and key in string.digits:
        return f"Digit{key}", ord(key), key
    if key in PUNCTUATION:
        code, vk, _ = PUNCTUATION[key]
        return code, vk, key
    if key in SHIFTED:
        code, vk, _ = _codes(SHIFTED[key])
        return code, vk, key
    raise Unsupported(f"this driver has no key {key!r}")


def _shifted(key: str) -> str:
    """The key Shift makes of ``key`` on a US layout: a letter's capital, a digit's or punctuation's symbol."""
    if len(key) == 1 and key in string.ascii_lowercase:
        return key.upper()
    if key in PUNCTUATION:
        return PUNCTUATION[key][2]
    return {digit: symbol for symbol, digit in SHIFTED_DIGITS.items()}.get(key, key)


def _printable(key: str) -> bool:
    return len(key) == 1 and (key in string.ascii_letters or key in string.digits or key in PUNCTUATION
                              or key in SHIFTED)


def _modifiers(keys) -> int:
    mask = 0
    for key in keys:
        mask |= MODIFIERS.get(key, 0)
    return mask


def _steps(path: list) -> list[tuple[float, float]]:
    """The points a drag moves through after its start, about every DRAG_STEP_PX pixels."""
    steps = []
    for (x0, y0), (x1, y1) in zip(path, path[1:]):
        n = max(2, int(math.hypot(x1 - x0, y1 - y0) / DRAG_STEP_PX))
        steps += [(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(1, n + 1)]
    return steps


class _PlaywrightMcp:
    """Playwright MCP, attached to the browser over the CDP address ``endpoint`` gives, and its sessions there, one per
    caller of the driver's tools, so each agent keeps its own snapshots and tab, each closed once idle for
    PLAYWRIGHT_MCP_SESSION_IDLE_S. It starts on the first call, again once it has exited or failed to start, and again on
    a fresh address once the browser's data reset has ended the last; it ends with the driver, so a driver started again
    finds none of its predecessor's."""

    def __init__(self, endpoint: Callable[[], Awaitable[str]]) -> None:
        self._endpoint = endpoint
        self._lock = asyncio.Lock()
        self._process: Optional[asyncio.subprocess.Process] = None
        self._running_on: Optional[str] = None
        self._url = ""
        self._sessions: dict[Optional[str], _Session] = {}

    async def call(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """``tool`` called on Playwright MCP as it answers, or refused saying why it couldn't be. A call Playwright MCP
        or its session ends under fails then, and isn't sent again, since it may have acted."""
        try:
            held, process = await self._session(ToolContext.current().caller.session)
        except DeviceUnavailable as e:
            return _refused(e.message)
        except Exception as e:
            return _refused(f"Playwright MCP didn't answer {tool}: {type(e).__name__}: {e}")
        called = asyncio.ensure_future(held.session.call_tool(
            tool, arguments, read_timeout_seconds=timedelta(seconds=PLAYWRIGHT_MCP_CALL_TIMEOUT_S)))
        exited = asyncio.ensure_future(process.wait())
        try:
            await asyncio.wait({called, exited, held.owner}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            called.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await called
            raise
        finally:
            exited.cancel()
        if not called.done():
            called.cancel()
            held.close.set()
            return _refused(f"Playwright MCP ended during {tool}, which may have acted before it did")
        try:
            return called.result()
        except Exception as e:
            held.close.set()
            return _refused(f"Playwright MCP didn't answer {tool}: {type(e).__name__}: {e}")
        finally:
            held.used_at = asyncio.get_running_loop().time()

    async def _session(self, caller: Optional[str]) -> tuple[_Session, asyncio.subprocess.Process]:
        async with self._lock:
            url = await self._running()
            now = asyncio.get_running_loop().time()
            for idle in [key for key, held in self._sessions.items()
                         if key != caller and now - held.used_at > PLAYWRIGHT_MCP_SESSION_IDLE_S]:
                self._sessions.pop(idle).close.set()
            held = self._sessions.get(caller)
            if held is None or held.url != url or held.owner.done() or held.close.is_set():
                if held is not None:
                    held.close.set()
                try:
                    held = self._sessions[caller] = await _Session.open(url)
                except DeviceUnavailable:
                    await self._stop()
                    raise
            held.used_at = now
            return held, self._process

    async def _running(self) -> str:
        """Playwright MCP's MCP URL, started on the browser's current address for it unless it runs there already."""
        endpoint = await self._endpoint()
        if self._process is not None and self._process.returncode is None and endpoint == self._running_on:
            return self._url
        await self._stop()
        port = _free_port()
        PLAYWRIGHT_MCP_OUTPUT.mkdir(parents=True, exist_ok=True)
        uploads = Path.home() / "Uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        with (PLAYWRIGHT_MCP_OUTPUT / "playwright-mcp.log").open("ab") as log:
            self._process = await asyncio.create_subprocess_exec(
                *_command(endpoint, port), cwd=uploads, env=_environment(), stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL, stderr=log)
        try:
            await _listening(port, self._process)
        except BaseException:
            await self._stop()
            raise
        self._running_on, self._url = endpoint, f"http://127.0.0.1:{port}/mcp"
        return self._url

    async def _stop(self) -> None:
        for held in self._sessions.values():
            held.close.set()
        self._sessions.clear()
        if self._process is not None and self._process.returncode is None:
            self._process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._process.wait(), PLAYWRIGHT_MCP_STOP_S)
            if self._process.returncode is None:
                self._process.kill()
                await self._process.wait()
        self._process, self._running_on = None, None


class _Session:
    """One MCP session on Playwright MCP, held open by a task of its own until ``close`` is set, since mcp's client
    contexts are anyio task groups, which only the task that entered one can leave."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.close = asyncio.Event()
        self.session: Optional[ClientSession] = None
        self.owner: Optional[asyncio.Task] = None
        self.used_at = 0.0

    @classmethod
    async def open(cls, url: str) -> _Session:
        """A session on Playwright MCP at ``url``; DeviceUnavailable when it doesn't open one in time, wedged."""
        held = cls(url)
        ready: asyncio.Future = asyncio.get_running_loop().create_future()
        held.owner = asyncio.create_task(held._own(ready))
        try:
            held.session = await asyncio.wait_for(ready, PLAYWRIGHT_MCP_START_S)
        except asyncio.TimeoutError:
            held.close.set()
            held.owner.cancel()
            raise DeviceUnavailable(f"Playwright MCP didn't open a session within {PLAYWRIGHT_MCP_START_S:.0f} s")
        return held

    async def _own(self, ready: asyncio.Future) -> None:
        try:
            timeout = httpx.Timeout(PLAYWRIGHT_MCP_CALL_TIMEOUT_S, connect=PLAYWRIGHT_MCP_START_S)
            async with httpx.AsyncClient(timeout=timeout, trust_env=False) as http, \
                    streamable_http_client(self.url, http_client=http) as (read, write, _), \
                    ClientSession(read, write) as session:
                await session.initialize()
                if ready.done():
                    return
                ready.set_result(session)
                await self.close.wait()
        except BaseException as e:
            if not ready.done():
                ready.set_exception(e if isinstance(e, Exception) else RuntimeError(repr(e)))
            elif not isinstance(e, asyncio.CancelledError):
                logger.warning(f"a session on Playwright MCP ended: {type(e).__name__}: {e}")
            if not isinstance(e, Exception):
                raise


def _command(endpoint: str, port: int) -> list[str]:
    """Playwright MCP on loopback ``port``, attached to the browser at ``endpoint``, never a browser of its own, with
    its file access confined to its working folder, the session user's ~/Uploads; ``setpriv`` ends it with the driver."""
    return ["setpriv", "--pdeathsig", "TERM", "--", str(NODE), str(PLAYWRIGHT_MCP), "--cdp-endpoint", endpoint,
            "--host", "127.0.0.1", "--port", str(port), "--caps", "vision", "--ignore-roots",
            "--output-dir", str(PLAYWRIGHT_MCP_OUTPUT)]


def _environment() -> dict[str, str]:
    return {name: os.environ[name] for name in KEPT_ENVIRONMENT if name in os.environ}


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def _listening(port: int, process: asyncio.subprocess.Process) -> None:
    """Once Playwright MCP listens on ``port``; DeviceUnavailable when it exits first or takes too long."""
    deadline = asyncio.get_running_loop().time() + PLAYWRIGHT_MCP_START_S
    while asyncio.get_running_loop().time() < deadline:
        if process.returncode is not None:
            raise DeviceUnavailable(f"Playwright MCP exited at start with {process.returncode}; its log is in "
                                    f"{PLAYWRIGHT_MCP_OUTPUT}")
        with contextlib.suppress(OSError):
            _, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            return
        await asyncio.sleep(PLAYWRIGHT_MCP_POLL_S)
    raise DeviceUnavailable(f"Playwright MCP didn't listen within {PLAYWRIGHT_MCP_START_S:.0f} s")


def _refused(message: str) -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])


if __name__ == "__main__":
    PageDriver().serve()
