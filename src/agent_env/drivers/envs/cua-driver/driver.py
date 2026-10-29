"""cua-driver as a computer driver, for an X11 desktop on Linux: trycua's cua-driver runs on the computer as a daemon,
which this file starts when run as a script, agents get what its own MCP server serves at its pinned version
(``mcp.json``): its instructions, its tools beside the ``computer_*`` tools, and its skill resources, read from it. The
computer/v1 operations are its desktop-scope calls. The daemon takes
line-delimited JSON on a Unix socket, and each call goes on a connection of its own, as cua-driver's own MCP server sends
them. The framework and each caller, told apart by MCP session, have an unnamed cua-driver session of their own, which
cua-driver starts again after ending it for idling, where a named one stays ended. A call the daemon can't take, as
while it restarts, fails as unavailable; input whose reply is lost fails with its outcome unknown, since it may have
arrived. It needs ``agentenv-framework-protocol[computer]`` and ``cua-driver`` where it runs."""
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
import time
import uuid
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent
from pydantic import AnyUrl

from agentenv_protocol import EnvironmentTool, ForwardedResource, ForwardedTool, ToolContext
from agentenv_protocol.computer import (
    PRINTABLE_KEYS,
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

#: Where cua-driver's daemon listens.
SOCKET = Path("/run/agentenv/cua-driver.sock")
#: What cua-driver's own MCP server (``cua-driver mcp``) says at connect and lists at the pinned version: its
#: instructions, its tools, which the running daemon must list the same, and its skill resources.
MANIFEST = Path(__file__).with_name("mcp.json")
#: The sha256 of the cua-driver executable each arch's pinned wheel installs.
EXECUTABLES = {
    "amd64": "bf57b2f3778a0a7daa9388e241f20ce948b32a7a2f4dd102e497f34e54137934",
    "arm64": "26e9eee8165e9ce7b614b5e3d68e5d963231385a82b65236406718b5ce1ff936",
}
#: What keeps cua-driver from reaching trycua: no telemetry and no update checks.
QUIET = {"DO_NOT_TRACK": "1", "CUA_DRIVER_RS_TELEMETRY_ENABLED": "false", "CUA_DRIVER_RS_UPDATE_CHECK": "false"}
#: W3C modifier keys and the names cua-driver holds them by.
MODIFIERS = {"Control": "ctrl", "Shift": "shift", "Alt": "alt", "Meta": "super", "Super": "super"}
#: W3C key values and the names cua-driver presses them by; any printable ASCII character is pressed as itself.
NAMED = {
    "Enter": "enter", "Tab": "tab", "Escape": "escape", " ": "space", "Backspace": "backspace", "Delete": "delete",
    "Insert": "insert", "Home": "home", "End": "end", "PageUp": "pageup", "PageDown": "pagedown", "ArrowUp": "up",
    "ArrowDown": "down", "ArrowLeft": "left", "ArrowRight": "right", **{f"F{n}": f"f{n}" for n in range(1, 13)},
}
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
    keys = (*MODIFIERS, *NAMED, *PRINTABLE_KEYS)

    def __init__(self) -> None:
        super().__init__()
        self._session = uuid.uuid4().hex
        self._callers: dict[str | None, str] = {}
        self._pinned: list[ReadyCheck] | None = None
        self._skill: tuple[str, str, dict[str, str]] | None = None
        self._texts: dict[str, str] | None = None

    def forwarded_tools(self) -> list[ForwardedTool]:
        return [ForwardedTool(EnvironmentTool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"]),
                              functools.partial(self._forward, t["name"]), t["annotations"] or None, t.get("outputSchema"))
                for t in _manifest()["tools"]]

    def forwarded_resources(self) -> list[ForwardedResource]:
        return [ForwardedResource(r["uri"], r["name"], r.get("description"), r["mimeType"],
                                  functools.partial(self._resource, r["uri"]))
                for r in _manifest()["resources"]]

    def instructions(self) -> str:
        return _manifest()["instructions"]

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
        """A call to one of cua-driver's own tools in its caller's session, answered as cua-driver answers it."""
        session = self._callers.setdefault(ToolContext.current().caller.session, uuid.uuid4().hex)
        try:
            reply = await _request({"method": "call", "name": tool, "args": arguments, "session_id": session})
        except DeviceUnavailable as e:
            return _refused(e.message)
        if not reply.get("ok"):
            return _refused(reply.get("error") or "daemon reported failure", reply.get("exit_code") or 1)
        return CallToolResult.model_validate(reply["result"])

    async def _resource(self, uri: str) -> str:
        """One of cua-driver's skill resources as its own MCP server serves it, all of them read on first use."""
        if self._texts is None:
            self._texts = await _read_resources([r["uri"] for r in _manifest()["resources"]])
        return self._texts[uri]

    async def _send(self, name: str, **args: Any) -> None:
        await self._call(name, DeliveryFailed, scope="desktop", **args)

    async def _call(self, name: str, failure: type[Exception], **args: Any) -> dict:
        """The result of one of cua-driver's tools called in the framework's session; ``failure`` names what went wrong
        when it fails, refuses or loses its reply after it was sent."""
        try:
            reply = await _request({"method": "call", "name": name, "args": args, "session_id": self._session})
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
            listed = (await _request({"method": "list"}))["result"]["tools"]
        except DeviceUnavailable:
            return [ReadyCheck("cua-driver", False, "its daemon isn't answering yet")]
        digest, expected = await asyncio.to_thread(_executable_sha256), EXECUTABLES.get(self.platform.arch or "")
        running = {t["name"]: t["input_schema"] for t in listed}
        pinned = {t["name"]: t["inputSchema"] for t in _manifest()["tools"]}
        differ = sorted(name for name in running.keys() | pinned.keys() if running.get(name) != pinned.get(name))
        return [ReadyCheck("cua-driver", digest == expected, "the pinned executable" if digest == expected else
                           f"its executable's sha256 is {digest}, not the pinned {expected}"),
                ReadyCheck("cua-driver tools", not differ, f"{len(running)} tools, as pinned" if not differ else
                           f"these tools differ from mcp.json: {', '.join(differ)}")]


class _ReplyLost(DeviceUnavailable):
    """The connection dropped after a request was sent, so the daemon may have acted on it."""


async def _request(payload: dict) -> dict:
    """The daemon's reply to ``payload``, sent on a connection of its own; a cancelled request closes it, so nothing is
    left to read its reply."""
    try:
        reader, writer = await asyncio.open_unix_connection(str(SOCKET), limit=REPLY_LIMIT)
    except OSError as e:
        raise DeviceUnavailable(f"cua-driver's daemon isn't answering: {e}") from None
    try:
        writer.write((json.dumps(payload) + "\n").encode())
        await writer.drain()
        line = await reader.readline()
        if not line:
            raise ConnectionError("the daemon closed the connection")
        return json.loads(line)
    except (OSError, ValueError) as e:
        raise _ReplyLost(f"cua-driver's daemon stopped answering after the request was sent, which it may have acted "
                         f"on: {e}") from None
    finally:
        writer.close()


async def _read_resources(uris: list[str]) -> dict[str, str]:
    """cua-driver's skill resources as its own MCP server serves them, on the running daemon."""
    server = StdioServerParameters(command=str(_executable()), args=["mcp", "--socket", str(SOCKET)],
                                   env={**os.environ, **QUIET})
    async with stdio_client(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()
        return {uri: (await session.read_resource(AnyUrl(uri))).contents[0].text for uri in uris}


@functools.cache
def _manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def _modifiers(keys) -> list[str]:
    if held := [key for key in keys if key not in MODIFIERS]:
        raise Unsupported(f"cua-driver holds only modifier keys, not {held}")
    return [MODIFIERS[key] for key in keys]


def _refused(message: str, exit_code: int = 1) -> CallToolResult:
    """A failed call as cua-driver's own MCP server answers one, which every tool's output schema admits."""
    return CallToolResult(isError=True, content=[TextContent(type="text", text=message)],
                          structuredContent={"code": "tool_invocation_failed", "exit_code": exit_code})


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
                       timeout=SKILL_TIMEOUT_S, env={**os.environ, "HOME": home, **QUIET})
        folder = Path(home) / ".cua-driver" / "skills" / "cua-driver"
        files = {path.name: path.read_text() for path in sorted(folder.iterdir()) if path.is_file()}
    _, frontmatter, body = files.pop("SKILL.md").split("---\n", 2)
    description = next(line.partition(":")[2].strip() for line in frontmatter.splitlines()
                       if line.startswith("description:"))
    return description, body.lstrip("\n"), files


def _daemon() -> None:
    """Run cua-driver's daemon once X answers: from the session user's home, where the apps it launches start too, on
    the socket the driver reaches it at, with approvals off, since nobody is at the computer to give them, and nothing
    sent to trycua."""
    display = os.environ["DISPLAY"].lstrip(":").partition(".")[0]
    while not Path(f"/tmp/.X11-unix/X{display}").is_socket():
        time.sleep(0.2)
    os.chdir(Path.home())
    executable = str(_executable())
    os.execve(executable, [executable, "serve", "--socket", str(SOCKET), "--dangerously-bypass-approvals", "--no-overlay"],
              {**os.environ, **QUIET})


if __name__ == "__main__":
    _daemon()
