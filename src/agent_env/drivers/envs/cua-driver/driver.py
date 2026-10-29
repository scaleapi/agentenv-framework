"""cua-driver as a computer driver, for an X11 desktop on Linux: trycua's cua-driver runs on the computer as a daemon
(``daemon.sh``), agents get its own tools as its pinned version lists them (``tools.json``), beside the ``computer_*``
tools, and the computer/v1 operations are its desktop-scope calls. The daemon takes line-delimited JSON on a Unix
socket; the framework's calls and each caller's run on connections of their own, the framework's in a session of its
own. When a connection drops, as when the daemon restarts, a read is sent once more on a new one; input is not, and
input whose reply is lost fails with its outcome unknown, since it may have arrived. It needs
``agentenv-framework-protocol[computer]`` and ``cua-driver`` where it runs."""
from __future__ import annotations

import asyncio
import base64
import functools
import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent

from agentenv_protocol import EnvironmentTool, ForwardedTool, ToolContext
from agentenv_protocol.computer import (
    ComputerDriverEnvironment,
    DeliveryFailed,
    DeviceUnavailable,
    Display,
    Image,
    ReadyCheck,
    Unsupported,
    capture,
    click,
    computer_driver,
    display,
    drag,
    press,
    scroll,
    type_text,
)

#: Where ``daemon.sh`` serves cua-driver's daemon.
SOCKET = Path("/run/agentenv/cua-driver.sock")
#: cua-driver's tools as its pinned version lists them, which the running daemon must list the same.
MANIFEST = Path(__file__).with_name("tools.json")
#: The sha256 of the cua-driver executable each arch's pinned wheel installs.
EXECUTABLES = {
    "amd64": "bf57b2f3778a0a7daa9388e241f20ce948b32a7a2f4dd102e497f34e54137934",
    "arm64": "26e9eee8165e9ce7b614b5e3d68e5d963231385a82b65236406718b5ce1ff936",
}
#: The session the framework's own calls run in, apart from every caller's.
FRAMEWORK_SESSION = "agentenv"
#: W3C modifier keys and the names cua-driver holds them by.
MODIFIERS = {"Control": "ctrl", "Shift": "shift", "Alt": "alt", "Meta": "super", "Super": "super"}
#: W3C key values and the names cua-driver presses them by; any printable ASCII character is pressed as itself.
NAMED = {
    "Enter": "enter", "Tab": "tab", "Escape": "escape", " ": "space", "Backspace": "backspace", "Delete": "delete",
    "Insert": "insert", "Home": "home", "End": "end", "PageUp": "pageup", "PageDown": "pagedown", "ArrowUp": "up",
    "ArrowDown": "down", "ArrowLeft": "left", "ArrowRight": "right", **{f"F{n}": f"f{n}" for n in range(1, 13)},
}
PRINTABLE = tuple(chr(code) for code in range(0x21, 0x7F))
#: How long the skill pack's writing may take.
SKILL_TIMEOUT_S = 60
#: The largest reply the daemon sends: a screenshot of a large display, base64.
REPLY_LIMIT = 64 << 20


@computer_driver(name="cua-driver",
                 supports=[{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64", "arm64"]}])
class CuaDriver(ComputerDriverEnvironment):
    buttons = ("left", "right", "middle")
    max_click_count = 3
    max_scroll_amount = 50
    keys = (*MODIFIERS, *NAMED, *PRINTABLE)

    def __init__(self) -> None:
        super().__init__()
        self._framework = _Connection()
        self._callers: dict[str, _Connection] = {}
        self._pinned: list[ReadyCheck] | None = None
        self._skill: tuple[str, str, dict[str, str]] | None = None

    def forwarded_tools(self) -> list[ForwardedTool]:
        return [ForwardedTool(EnvironmentTool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]),
                              functools.partial(self._forward, t["name"]), t["annotations"] or None)
                for t in _manifest()]

    async def ready_checks(self) -> list[ReadyCheck]:
        """That the daemon answers as the pinned cua-driver does: its executable's sha256 and its tools, checked
        until they pass once."""
        if self._pinned is None or not all(check.ok for check in self._pinned):
            self._pinned = await self._pinned_checks()
        return self._pinned

    async def bundled_skill(self) -> tuple[str, str, dict[str, str]]:
        if self._skill is None:
            self._skill = await asyncio.to_thread(_skill_pack)
        return self._skill

    @display
    async def screen(self) -> Display:
        size = (await self._call("get_screen_size", DeviceUnavailable))["structuredContent"]
        return Display(size["width"], size["height"], size.get("scale_factor") or 1.0)

    @capture
    async def grab(self) -> Image:
        result = await self._call("get_desktop_state", DeviceUnavailable)
        shot, state = next(c for c in result["content"] if c["type"] == "image"), result["structuredContent"]
        return Image(state["screenshot_width"], state["screenshot_height"], base64.b64decode(shot["data"]), "png")

    @click
    async def click_at(self, x, y, button, count, modifiers) -> None:
        await self._send("click", x=x, y=y, button=button, count=count, modifier=_modifiers(modifiers))

    @drag
    async def drag_along(self, path, button, duration_ms, modifiers) -> None:
        (x0, y0), (x1, y1) = path[0], path[-1]
        await self._send("drag", from_x=x0, from_y=y0, to_x=x1, to_y=y1, button=button, duration_ms=duration_ms,
                         modifier=_modifiers(modifiers))

    @scroll
    async def scroll_at(self, x, y, direction, amount, modifiers) -> None:
        if modifiers:
            raise Unsupported("cua-driver scrolls with no keys held")
        await self._send("scroll", x=x, y=y, direction=direction, amount=amount)

    @type_text
    async def type_in(self, text) -> None:
        await self._send("type_text", text=text)

    @press
    async def press_keys(self, keys) -> None:
        *held, key = keys
        await self._send("press_key", modifiers=_modifiers(held), key=MODIFIERS.get(key) or NAMED.get(key, key))

    async def _forward(self, tool: str, /, **arguments: Any) -> CallToolResult:
        """A call to one of cua-driver's own tools, on its caller's connection, answered as cua-driver answers it."""
        connection = self._callers.setdefault(self.caller_identity(ToolContext.current()), _Connection())
        try:
            reply = await connection.request({"method": "call", "name": tool, "args": arguments},
                                             retry=tool in _read_only())
        except DeviceUnavailable as e:
            return _refused(e.message)
        if not reply.get("ok"):
            return _refused(reply.get("error") or f"cua-driver refused {tool}")
        return CallToolResult.model_validate(reply["result"])

    async def _send(self, name: str, **args: Any) -> None:
        await self._call(name, DeliveryFailed, scope="desktop", **args)

    async def _call(self, name: str, failure: type[Exception], **args: Any) -> dict:
        """The result of one of cua-driver's tools called in the framework's session, a read (``DeviceUnavailable``
        when it fails) sent again on a new connection when this one dropped; ``failure`` names what went wrong when
        it fails, refuses or loses its reply after it was sent."""
        try:
            reply = await self._framework.request({"method": "call", "name": name,
                                                   "args": {**args, "session": FRAMEWORK_SESSION}},
                                                  retry=failure is DeviceUnavailable)
        except _ReplyLost as e:
            raise failure(e.message) from None
        if not reply.get("ok"):
            raise failure(f"cua-driver {name}: {reply.get('error')}")
        result = reply["result"]
        if result.get("isError"):
            raise failure(f"cua-driver {name}: {_text(result)}")
        return result

    async def _pinned_checks(self) -> list[ReadyCheck]:
        try:
            listed = (await self._framework.request({"method": "list"}, retry=True))["result"]["tools"]
        except DeviceUnavailable:
            return [ReadyCheck("cua-driver", False, "its daemon isn't answering yet")]
        digest, expected = await asyncio.to_thread(_executable_sha256), EXECUTABLES.get(self.platform.arch or "")
        running = {t["name"]: t["input_schema"] for t in listed}
        pinned = {t["name"]: t["inputSchema"] for t in _manifest()}
        differ = sorted(name for name in running.keys() | pinned.keys() if running.get(name) != pinned.get(name))
        return [ReadyCheck("cua-driver", digest == expected, "the pinned executable" if digest == expected else
                           f"its executable's sha256 is {digest}, not the pinned {expected}"),
                ReadyCheck("cua-driver tools", not differ, f"{len(running)} tools, as pinned" if not differ else
                           f"these tools differ from tools.json: {', '.join(differ)}")]


class _ReplyLost(DeviceUnavailable):
    """The connection dropped after a request was sent, so the daemon may have acted on it."""


class _Connection:
    """One connection to cua-driver's daemon, opened on first use and again after it drops, taking one request at a
    time."""

    def __init__(self) -> None:
        self._stream: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
        self._lock = asyncio.Lock()

    async def request(self, payload: dict, *, retry: bool = False) -> dict:
        """The daemon's reply to ``payload``; with ``retry``, sent once more on a new connection when an open one has
        dropped."""
        async with self._lock:
            was_open = self._stream is not None
            try:
                return await self._exchange(payload)
            except DeviceUnavailable:
                if not (retry and was_open):
                    raise
            return await self._exchange(payload)

    async def _exchange(self, payload: dict) -> dict:
        """One request and its reply; a request cancelled in flight closes the connection, so its reply can't be read
        as the next one's."""
        sent = False
        try:
            if self._stream is None:
                self._stream = await asyncio.open_unix_connection(str(SOCKET), limit=REPLY_LIMIT)
            reader, writer = self._stream
            sent = True
            writer.write((json.dumps(payload) + "\n").encode())
            await writer.drain()
            line = await reader.readline()
            if not line:
                raise ConnectionError("the daemon closed the connection")
        except (OSError, ValueError) as e:
            self._close()
            if sent:
                raise _ReplyLost(f"cua-driver's daemon stopped answering after the request was sent, which it may "
                                 f"have acted on: {e}") from None
            raise DeviceUnavailable(f"cua-driver's daemon isn't answering: {e}") from None
        except asyncio.CancelledError:
            self._close()
            raise
        return json.loads(line)

    def _close(self) -> None:
        if self._stream is not None:
            self._stream[1].close()
        self._stream = None


@functools.cache
def _manifest() -> list[dict]:
    return json.loads(MANIFEST.read_text())


@functools.cache
def _read_only() -> frozenset[str]:
    return frozenset(t["name"] for t in _manifest() if (t["annotations"] or {}).get("readOnlyHint"))


def _modifiers(keys) -> list[str]:
    if held := [key for key in keys if key not in MODIFIERS]:
        raise Unsupported(f"cua-driver holds only modifier keys, not {held}")
    return [MODIFIERS[key] for key in keys]


def _refused(message: str) -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text=message)])


def _text(result: dict) -> str:
    return " ".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")


def _executable() -> Path:
    """cua-driver's executable in the driver's venv, found without importing its package, which loads its SDK."""
    return Path(importlib.util.find_spec("cua_driver").submodule_search_locations[0]) / "bin" / "cua-driver"


def _executable_sha256() -> str:
    return hashlib.sha256(_executable().read_bytes()).hexdigest()


def _skill_pack() -> tuple[str, str, dict[str, str]]:
    """cua-driver's skill pack as its executable embeds it, written offline by ``mcp-config`` into a home of its own:
    its description, its SKILL.md body and its other files."""
    with tempfile.TemporaryDirectory(prefix="cua-driver-skill-") as home:
        (Path(home) / ".claude").mkdir()
        subprocess.run([str(_executable()), "mcp-config", "--client", "claude"], capture_output=True, check=True,
                       timeout=SKILL_TIMEOUT_S, env={**os.environ, "HOME": home, "DO_NOT_TRACK": "1",
                                                     "CUA_DRIVER_RS_TELEMETRY_ENABLED": "false",
                                                     "CUA_DRIVER_RS_UPDATE_CHECK": "false"})
        folder = Path(home) / ".cua-driver" / "skills" / "cua-driver"
        files = {path.name: path.read_text() for path in sorted(folder.iterdir()) if path.is_file()}
    _, frontmatter, body = files.pop("SKILL.md").split("---\n", 2)
    description = next(line.partition(":")[2].strip() for line in frontmatter.splitlines()
                       if line.startswith("description:"))
    return description, body.lstrip("\n"), files
