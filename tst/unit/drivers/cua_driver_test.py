"""The cua-driver driver against a fake cua-driver daemon on a Unix socket: the computer/v1 operations become its
desktop-scope calls in the framework's own session; its instructions, tools and skill resources are served as its own
MCP server serves them, its tools forwarded in their caller's own session as they came, readiness checks the pinned executable and tools, and the skill leads with cua-driver's pack."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import re
import socket
import sys
import types
import stat
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image as PILImage

import agent_env
from agent_env.env.envs.computer.utils.install import for_os, installs
from agentenv_protocol import SESSION_META_KEY
from agentenv_protocol.computer import DeliveryFailed, DeviceUnavailable, Platform, Unsupported
from agentenv_protocol.computer.platform import supported

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "cua-driver" / "driver.py"
#: cua-driver's Linux release tarball, its version the group.
RELEASE_TARBALL = (r"https://github\.com/trycua/cua/releases/download/cua-driver-rs-v([\d.]+)/"
                   r"cua-driver-rs-\1-linux-x86_64-binary\.tar\.gz")


def _png(width: int, height: int) -> str:
    out = io.BytesIO()
    PILImage.new("RGB", (width, height), (255, 0, 0)).save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


class FakeDaemon:
    """cua-driver's daemon as the driver sees it, on ``path``: each request by the connection it came on, ``refuse``
    naming tools whose results are errors, ``fail`` tools it can't call, ``slow`` tools it answers after a second,
    ``drop`` tools whose connection it closes without answering, ``grab`` a layer such as a menu holding the keyboard,
    from which it refuses a key aimed at a window, ``unlaid`` the seat's keymap changed, for which it refuses one too
    until a chord at the desktop of the layout's keys puts it back, which ``relaid`` false keeps it from, and ``tools``
    what ``list`` lists."""

    def __init__(self, manifest: list[dict], path: Path) -> None:
        self.path = path
        self.requests: list[tuple[int, dict]] = []
        self.refuse: set[str] = set()
        self.fail: set[str] = set()
        self.slow: set[str] = set()
        self.drop: set[str] = set()
        self.grab = self.unlaid = False
        self.relaid = True
        self.tools = [{"name": t["name"], "input_schema": t["inputSchema"]} for t in manifest]
        self.connections = 0
        self.writers: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        self.server = await asyncio.start_unix_server(self.serve, path=str(self.path))

    async def stop(self) -> None:
        """As the daemon dying: its socket gone and every connection closed."""
        self.server.close()
        for writer in self.writers:
            writer.close()
        await self.server.wait_closed()
        self.path.unlink(missing_ok=True)

    async def serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connections += 1
        connection = self.connections
        self.writers.append(writer)
        while line := await reader.readline():
            request = json.loads(line)
            self.requests.append((connection, request))
            if request.get("name") in self.drop:
                break
            if request.get("name") in self.slow:
                await asyncio.sleep(1)
            writer.write((json.dumps(self.reply(request)) + "\n").encode())
            await writer.drain()
        writer.close()

    def reply(self, request: dict) -> dict:
        if request["method"] == "list":
            return {"ok": True, "result": {"tools": self.tools}}
        name = request["name"]
        if name in self.fail:
            return {"ok": False, "error": f"unknown tool {name}", "exit_code": 64}
        if name in self.refuse:
            return {"ok": True, "result": {"isError": True, "content": [{"type": "text", "text": f"{name} refused"}]}}
        if name == "press_key" and "pid" in request["args"] and (self.grab or self.unlaid):
            why = "(primary_target_busy): foreground_grab" if self.grab else "(unsupported_layout): unsupported_layout"
            return {"ok": True, "result": {"isError": True, "content": [
                {"type": "text", "text": f"foreground_unavailable {why}"}]}}
        if name == "press_key" and "super" in request["args"]["modifiers"] and self.relaid:
            self.unlaid = False
        if name == "list_windows":
            return {"ok": True, "result": {"content": [{"type": "text", "text": "Found 0 windows"}],
                                           "structuredContent": {"windows": []}}}
        if name == "get_screen_size":
            return {"ok": True, "result": {"content": [{"type": "text", "text": "1280x800"}],
                                           "structuredContent": {"width": 1280, "height": 800, "scale_factor": 1.0}}}
        if name == "get_desktop_state":
            return {"ok": True, "result": {
                "content": [{"type": "image", "data": _png(1280, 800), "mimeType": "image/png"}],
                "structuredContent": {"screenshot_width": 1280, "screenshot_height": 800}}}
        return {"ok": True, "result": {"content": [{"type": "text", "text": f"did {name}"}],
                                       "structuredContent": {"tool": name}}}

    def calls(self, session: str | None = None) -> list[tuple[str, dict]]:
        return [(r["name"], r["args"]) for _, r in self.requests if r["method"] == "call"
                and (session is None or r["session_id"] == session)]


class FakeKeymap:
    """An X display's keymap: printable ASCII on keycodes 8 to 102, and 103 and 104 spare."""

    def __init__(self):
        self.display = types.SimpleNamespace(info=types.SimpleNamespace(min_keycode=8, max_keycode=104))
        self.keys = {8 + i: (0x20 + i, 0x20 + i) for i in range(0x5F)} | {103: (0, 0), 104: (0, 0)}

    def get_keyboard_mapping(self, first, count):
        return [self.keys[code] for code in range(first, first + count)]

    def change_keyboard_mapping(self, code, keysyms):
        self.keys[code] = tuple(keysyms[0])

    def sync(self):
        pass


@pytest.fixture
def module(monkeypatch):
    """driver.py loaded as it is on an X11 desktop on Linux, whatever runs the tests, with X stood in for; a test of
    another platform sets that platform's flags itself."""
    monkeypatch.delenv("HYPRLAND_INSTANCE_SIGNATURE", raising=False)
    return _loaded(monkeypatch)


def _loaded(monkeypatch):
    """driver.py loaded on Linux, in a Hyprland session when ``HYPRLAND_INSTANCE_SIGNATURE`` is set."""
    monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    xlib = types.ModuleType("Xlib")
    xlib.display = types.ModuleType("Xlib.display")
    xlib.display.Display = FakeKeymap
    xlib.error = types.ModuleType("Xlib.error")
    xlib.error.ConnectionClosedError = type("ConnectionClosedError", (Exception,), {})
    xlib.error.DisplayConnectionError = type("DisplayConnectionError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "Xlib", xlib)
    monkeypatch.setitem(sys.modules, "Xlib.display", xlib.display)
    monkeypatch.setitem(sys.modules, "Xlib.error", xlib.error)
    monkeypatch.syspath_prepend(str(DRIVER.parent))
    monkeypatch.delitem(sys.modules, "hyprland", raising=False)
    spec = importlib.util.spec_from_file_location("cua_driver_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as linux:
        linux.setattr(sys, "platform", "linux")
        spec.loader.exec_module(loaded)
    return loaded


@pytest_asyncio.fixture
async def daemon(module, monkeypatch):
    """A fake daemon on a socket the driver is pointed at, under a short /tmp path for macOS's socket path limit."""
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="cd-") as folder:
        fake = FakeDaemon(module._manifest()["tools"], Path(folder) / "cua-driver.sock")
        monkeypatch.setattr(module, "SOCKET", fake.path)
        await fake.start()
        yield fake
        await fake.stop()


@contextlib.asynccontextmanager
async def mcp_session(driver):
    async with create_connected_server_and_client_session(driver.mcp or driver.create_app()) as session:
        yield session


@pytest.mark.asyncio
async def test_the_operations_become_desktop_scope_calls_in_the_frameworks_own_session_and_rest_the_pointer_on_macos(
        module, daemon, monkeypatch):
    driver = module.CuaDriver()

    size, shot = await driver.screen(), await driver.grab()
    await driver.click_at(100, 200, "right", 2, ["Control"])
    await driver.drag_along([(10, 20), (30, 40)], "left", 300, ["Shift"])
    await driver.scroll_at(5, 6, "down", 3, [])
    await driver.type_in("Hello, Wörld! 東京")
    await driver.press_keys(["Control", "/"])
    await driver.press_keys(["Enter"])

    assert (size.width, size.height, shot.width, shot.height, shot.mode) == (1280, 800, 1280, 800, "png")
    assert daemon.calls() == [
        ("get_screen_size", {}), ("get_desktop_state", {"max_image_dimension": 0}),
        ("click", {"scope": "desktop", "x": 100, "y": 200, "button": "right", "count": 2, "modifier": ["ctrl"]}),
        ("drag", {"scope": "desktop", "from_x": 10, "from_y": 20, "to_x": 30, "to_y": 40, "button": "left",
                  "duration_ms": 300, "modifier": ["shift"]}),
        ("scroll", {"scope": "desktop", "x": 5, "y": 6, "direction": "down", "amount": 3}),
        ("type_text", {"scope": "desktop", "text": "Hello, Wörld! 東京"}),
        ("press_key", {"scope": "desktop", "key": "/", "modifiers": ["ctrl"]}),
        ("press_key", {"scope": "desktop", "key": "enter", "modifiers": []}),
    ]
    assert len({r["session_id"] for _, r in daemon.requests}) == 1
    for name, value in (("MACOS", True), ("FOREGROUND", {"delivery_mode": "foreground"}), ("SETTLE_S", 0)):
        monkeypatch.setattr(module, name, value)
    await driver.click_at(100, 200, "left", 1, [])
    await driver.drag_along([(10, 20), (30, 40)], "left", 300, [])
    assert daemon.calls()[-4:] == [
        ("move_cursor", {"scope": "desktop", "x": 100, "y": 200, "delivery_mode": "foreground"}),
        ("click", {"scope": "desktop", "x": 100, "y": 200, "button": "left", "count": 1, "modifier": [],
                   "delivery_mode": "foreground"}),
        ("move_cursor", {"scope": "desktop", "x": 10, "y": 20, "delivery_mode": "foreground"}),
        ("drag", {"scope": "desktop", "from_x": 10, "from_y": 20, "to_x": 30, "to_y": 40, "button": "left",
                  "duration_ms": 300, "modifier": [], "steps": 20, "delivery_mode": "foreground"}),
    ]


@pytest.mark.asyncio
async def test_characters_off_x11s_keymap_are_kept_on_spare_keycodes_before_cua_driver_types_them(
        module, daemon, monkeypatch):
    monkeypatch.setattr(module, "MACOS", False)
    monkeypatch.setattr(module, "x_display", sys.modules["Xlib.display"], raising=False)
    for name in ("ConnectionClosedError", "DisplayConnectionError"):
        monkeypatch.setattr(module, name, getattr(sys.modules["Xlib.error"], name), raising=False)
    driver = module.CuaDriver()
    oe, east, capital = 0xF6, 0x1000000 | ord("東"), 0x1000000 | ord("京")

    await driver.type_in("Wörld")
    keymap = driver._keymap.d
    assert keymap.keys[103] == (oe, oe) and keymap.keys[104] == (0, 0)
    await driver.type_in("東")
    await driver.type_in("京")
    assert keymap.keys[103] == (capital, capital) and keymap.keys[104] == (east, east)
    await driver.type_in("東京ö")
    assert keymap.keys[103] == keymap.keys[104] == (0, 0) and not driver._keymap.ours

    def dropped(*args):
        raise sys.modules["Xlib.error"].ConnectionClosedError("X restarted")
    monkeypatch.setattr(keymap, "get_keyboard_mapping", dropped)
    await driver.type_in("ö")
    assert driver._keymap.d is not keymap and driver._keymap.d.keys[103] == (oe, oe)
    assert [r["text"] for name, r in daemon.calls() if name == "type_text"] == ["Wörld", "東", "京", "東京ö", "ö"]


@pytest.mark.asyncio
async def test_what_cua_driver_cant_do_or_refuses_fails_naming_it(module, daemon):
    driver = module.CuaDriver()
    daemon.refuse.add("click")

    with pytest.raises(Unsupported, match="holds only modifier keys"):
        await driver.press_keys(["Control", "a", "b"])
    with pytest.raises(Unsupported, match="holds only modifier keys, not \\['a'\\]"):
        await driver.click_at(1, 1, "left", 1, ["Shift", "a"])
    with pytest.raises(Unsupported, match="no keys held"):
        await driver.scroll_at(1, 1, "up", 1, ["Shift"])
    with pytest.raises(Exception, match="cua-driver click: click refused") as refused:
        await driver.click_at(1, 1, "left", 1, [])
    assert type(refused.value).__name__ == "DeliveryFailed"


@pytest.mark.asyncio
async def test_cua_drivers_tools_are_listed_as_pinned_and_forwarded_in_their_callers_session(module, daemon):
    driver = module.CuaDriver()
    async with mcp_session(driver) as session:
        listed = {t.name: t for t in (await session.list_tools()).tools}
        await session.call_tool("computer_screenshot", {})
        answered = await session.call_tool("get_accessibility_tree", {"pid": 42, "window_id": 7})
        launched = await session.call_tool("launch_app", {"name": "thunar"})

    manifest = {t["name"]: t for t in module._manifest()["tools"]}
    assert set(manifest) <= set(listed) and {"computer_click", "computer_screenshot"} <= set(listed)
    assert listed["get_window_state"].inputSchema == manifest["get_window_state"]["inputSchema"]
    assert listed["get_window_state"].annotations.readOnlyHint == manifest["get_window_state"]["annotations"]["readOnlyHint"]
    assert {name: listed[name].outputSchema for name in manifest} == {
        name: tool.get("outputSchema") for name, tool in manifest.items()}
    assert driver.mcp.instructions == module._manifest()["instructions"]
    framework, *_, caller = [r["session_id"] for _, r in daemon.requests if r["method"] == "call"]
    assert caller != framework and daemon.calls(caller) == [
        ("get_accessibility_tree", {"pid": 42, "window_id": 7}), ("launch_app", {"name": "thunar"})]
    assert not launched.isError
    assert answered.structuredContent == {"tool": "get_accessibility_tree"}
    assert answered.content[0].text == "did get_accessibility_tree"


@pytest.mark.asyncio
async def test_each_caller_gets_a_cua_driver_session_of_its_own(module, daemon):
    async with mcp_session(module.CuaDriver()) as session:
        for caller in ("a", "b", "a"):
            await session.call_tool("list_windows", {}, meta={SESSION_META_KEY: caller})
        await session.call_tool("get_accessibility_tree", {"session": "mine"}, meta={SESSION_META_KEY: "b"})

    a, b, again, named = [(r["session_id"], r["args"]) for _, r in daemon.requests if r["method"] == "call"]
    assert a[0] != b[0] and again == a and named == (b[0], {"session": "mine"})


@pytest.mark.asyncio
async def test_a_failed_call_to_one_of_cua_drivers_tools_reaches_its_caller_as_an_error(module, daemon):
    daemon.fail.add("list_apps")
    async with mcp_session(module.CuaDriver()) as session:
        failed = await session.call_tool("list_apps", {})
        await daemon.stop()
        down = await session.call_tool("list_windows", {})
    assert failed.isError and failed.content[0].text == "unknown tool list_apps"
    assert failed.structuredContent == {"code": "tool_invocation_failed", "exit_code": 64}
    assert down.isError and "cua-driver's daemon isn't answering" in down.content[0].text
    assert down.structuredContent == {"code": "tool_invocation_failed", "exit_code": 1}


@pytest.mark.asyncio
async def test_readiness_checks_the_pinned_executable_and_tools_until_they_pass_once(module, daemon, monkeypatch):
    driver = module.CuaDriver()
    pinned = module.EXECUTABLES[driver.platform.arch]
    digests = iter(["0" * 64, pinned])
    monkeypatch.setattr(module, "_executable_sha256", lambda: next(digests))
    daemon.tools[0] = {**daemon.tools[0], "input_schema": {"type": "object"}}

    wrong = await driver.ready_checks()
    daemon.tools[0] = {"name": daemon.tools[0]["name"], "input_schema": module._manifest()["tools"][0]["inputSchema"]}
    right = await driver.ready_checks()
    lists = sum(1 for _, r in daemon.requests if r["method"] == "list")
    again = await driver.ready_checks()

    assert [(c.name, c.ok) for c in wrong] == [("cua-driver", False), ("cua-driver tools", False)]
    assert wrong[0].detail.startswith("its executable's sha256 is 0000") and module._manifest()["tools"][0]["name"] in wrong[1].detail
    assert [(c.name, c.ok) for c in right] == [("cua-driver", True), ("cua-driver tools", True)]
    assert again == right and sum(1 for _, r in daemon.requests if r["method"] == "list") == lists


@pytest.mark.asyncio
async def test_readiness_waits_for_a_daemon_that_isnt_answering(module, monkeypatch):
    monkeypatch.setattr(module, "SOCKET", Path("/tmp/cd-nothing-here.sock"))
    assert [(c.name, c.ok, c.detail) for c in await module.CuaDriver().ready_checks()] == [
        ("cua-driver", False, "its daemon isn't answering yet")]


@pytest.mark.asyncio
async def test_while_the_daemon_restarts_calls_fail_as_unavailable_and_reach_it_once_it_is_back(module, daemon):
    driver = module.CuaDriver()
    await driver.screen()
    await daemon.stop()
    with pytest.raises(DeviceUnavailable, match="cua-driver's daemon isn't answering"):
        await driver.screen()
    with pytest.raises(DeviceUnavailable) as unsent:
        await driver.click_at(1, 1, "left", 1, [])
    await daemon.start()

    assert (await driver.screen()).width == 1280
    await driver.click_at(2, 2, "left", 1, [])
    assert type(unsent.value) is DeviceUnavailable and unsent.value.applied == "no"
    assert [args["x"] for name, args in daemon.calls() if name == "click"] == [2]


@pytest.mark.asyncio
async def test_input_whose_reply_is_lost_fails_with_its_outcome_unknown(module, daemon):
    daemon.drop.add("type_text")
    with pytest.raises(DeliveryFailed, match="may have acted on") as lost:
        await module.CuaDriver().type_in("hello")
    assert lost.value.applied == "unknown" and [name for name, _ in daemon.calls()] == ["type_text"]


@pytest.mark.asyncio
async def test_a_request_cancelled_in_flight_leaves_no_reply_for_the_next_one(module, daemon):
    driver = module.CuaDriver()
    daemon.slow.add("get_screen_size")
    screen = asyncio.ensure_future(driver.screen())
    await asyncio.sleep(0.2)
    screen.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await screen
    daemon.slow.clear()

    shot = await driver.grab()
    assert (shot.width, shot.mode) == (1280, "png") and daemon.connections == 2


@pytest.mark.asyncio
async def test_cua_drivers_skill_resources_are_listed_as_pinned_and_read_once_from_its_own_mcp_server(module, monkeypatch):
    reads = []

    async def read_resources(uris):
        reads.append(uris)
        return {uri: f"text of {uri}" for uri in uris}

    monkeypatch.setattr(module, "_read_resources", read_resources)
    async with mcp_session(module.CuaDriver()) as session:
        listed = (await session.list_resources()).resources
        skill = await session.read_resource("skill://cua-driver/SKILL.md")
        linux = await session.read_resource("skill://cua-driver/LINUX.md")

    pinned = module._manifest()["resources"]
    assert [(str(r.uri), r.name, r.description, r.mimeType) for r in listed] == [
        (r["uri"], r["name"], r.get("description"), r["mimeType"]) for r in pinned]
    assert skill.contents[0].text == "text of skill://cua-driver/SKILL.md" and linux.contents[0].text.endswith("LINUX.md")
    assert reads == [[r["uri"] for r in pinned]]


def test_the_skill_pack_is_written_offline_from_the_executable(module, monkeypatch, tmp_path):
    executable = tmp_path / "cua-driver"
    executable.write_text("""#!/bin/sh
[ "$1 $2 $3" = "mcp-config --client claude" ] && [ "$DO_NOT_TRACK" = 1 ] && [ -d "$HOME/.claude" ] || exit 9
mkdir -p "$HOME/.cua-driver/skills/cua-driver"
printf -- '---\\nname: cua-driver\\ndescription: Drive a native GUI app.\\nversion: 0.38.0\\n---\\n\\n# cua-driver\\n\\nSee LINUX.md.\\n' > "$HOME/.cua-driver/skills/cua-driver/SKILL.md"
printf '# Linux\\n' > "$HOME/.cua-driver/skills/cua-driver/LINUX.md"
""")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(module, "_executable", lambda: executable)

    assert module._skill_pack() == ("Drive a native GUI app.", "# cua-driver\n\nSee LINUX.md.\n", {"LINUX.md": "# Linux\n"})


@pytest.mark.asyncio
async def test_the_computer_skill_leads_with_cua_drivers_pack(module, daemon, monkeypatch):
    calls = []
    monkeypatch.setattr(module, "_skill_pack", lambda: calls.append(1) or (
        "Drive a native GUI app.", "# cua-driver\n", {"LINUX.md": "# Linux\n"}))
    driver = module.CuaDriver()
    asgi = driver.create_app().streamable_http_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=asgi), base_url="http://driver") as http:
        first = (await http.get("/agentenv/ext/skill/get", params={"name": "desktop"})).json()
        again = (await http.get("/agentenv/ext/skill/get", params={"name": "desktop"})).json()

    assert first == again and calls == [1] and first["files"] == {"LINUX.md": "# Linux\n"}
    assert first["skill_md"].partition("---\n\n")[2].startswith("# cua-driver\n\n# This computer\n")
    assert first["description"].startswith("Drive a native GUI app. This computer: ")


class FakePipe(io.BytesIO):
    """The daemon's named pipe on Windows, answering one request with ``reply``."""

    def __init__(self, reply: dict):
        super().__init__()
        self.reply, self.sent = (json.dumps(reply) + "\n").encode(), b""

    def write(self, data):
        self.sent += data
        return len(data)

    def read(self, size=-1):
        reply, self.reply = self.reply, b""
        return reply


def test_on_windows_a_call_waits_out_a_busy_or_briefly_absent_pipe_and_a_daemon_down_fails_soon(module, monkeypatch):
    opened = []
    errors = iter([OSError(22, "Invalid argument"), OSError(2, "No such file or directory"), None])

    def fake_open(path, mode, buffering):
        opened.append(path)
        if (error := next(errors)) is not None:
            raise error
        return FakePipe({"ok": True, "result": {"tools": []}})

    monkeypatch.setattr(module, "open", fake_open, raising=False)
    monkeypatch.setattr(module.time, "sleep", lambda s: None)

    assert module._piped({"method": "list"}) == {"ok": True, "result": {"tools": []}}
    assert opened == [module.SOCKET] * 3

    clock = iter(range(0, 100, 1))
    monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
    def daemon_down(path, mode, buffering):
        raise OSError(2, "No such file or directory")

    monkeypatch.setattr(module, "open", daemon_down, raising=False)
    with pytest.raises(DeviceUnavailable, match="isn't answering"):
        module._piped({"method": "list"})


#: What ``hyprctl -j cua:status`` answers while cua-driver's plugin is loaded, enabled, built for that Hyprland and
#: taking input.
PLUGIN_LOADED = {"configured": True, "abi": {"match": True}, "transport": {"ready": True, "last_error": ""}}
#: Its answer once it's loaded again into a running Hyprland, which gives it no input seat until the desktop restarts.
PLUGIN_RELOADED = {**PLUGIN_LOADED, "state": "discovery_only", "transport": {
    "ready": False,
    "last_error": "input seat lifetime unavailable; restart the desktop before loading a replacement plugin"}}


class FakeHyprland:
    """Hyprland's IPC as ``hyprctl -j`` answers: an answer by request, a list of them answered one per ask, None for a
    request it has no answer to, and every request asked, in order; ``binds`` what it lists as bound, and ``lua`` the
    Lua it was given (``hyprctl eval``), a bind in which it then lists unless ``binding`` is false."""

    def __init__(self, answers: dict):
        self.answers, self.asked, self.binds, self.lua, self.binding = answers, [], [], [], True

    def __call__(self, request: str) -> dict | list | None:
        self.asked.append(request)
        if request == "binds":
            return self.binds
        answer = self.answers.get(request)
        return answer.pop(0) if isinstance(answer, list) else answer

    def run_lua(self, code: str) -> None:
        self.lua.append(code)
        if self.binding and code.startswith('hl.bind("SUPER + CTRL + ALT + SHIFT + F12"'):
            self.binds.append({"modmask": 77, "key": "F12", "dispatcher": "__lua", "arg": str(300 + len(self.lua)),
                               "non_consuming": False})


@pytest.fixture
def hyprland(module, monkeypatch) -> FakeHyprland:
    """The driver as it runs in a Hyprland session, whatever OS the tests run on, with cua-driver's plugin taking
    input and nothing focused; the fake IPC, whose answers a test changes."""
    for name, value in (("MACOS", False), ("WINDOWS", False), ("FOREGROUND", {}), ("HYPRLAND", True)):
        monkeypatch.setattr(module, name, value)
    fake = FakeHyprland({"cua:status": PLUGIN_LOADED, "activeworkspace": {"id": 1}})
    monkeypatch.setattr(module.hyprland, "hyprctl", fake)
    monkeypatch.setattr(module.hyprland, "lua", fake.run_lua)
    return fake


@pytest.mark.asyncio
async def test_on_hyprland_text_goes_to_the_desktop_and_keys_to_the_focused_window_as_on_a_us_keyboard(
        module, daemon, hyprland):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    driver = module.CuaDriver()

    await driver.type_in("Hello, Wörld")
    for keys in (["Control", "a"], ["Backspace"], ["Control", "+"], ["Shift", "?"], ["A"], ["/"]):
        await driver.press_keys(keys)
    await driver.click_at(10, 20, "left", 1, [])

    focused = {"pid": 42, "window_id": 0x5557E0F44AD0, "delivery_mode": "foreground"}
    assert daemon.calls() == [
        ("type_text", {"scope": "desktop", "text": "Hello, Wörld", "delivery_mode": "foreground"}),
        ("press_key", {"scope": "desktop", "key": "a", "modifiers": ["ctrl"], **focused}),
        ("press_key", {"scope": "desktop", "key": "backspace", "modifiers": [], **focused}),
        ("press_key", {"scope": "desktop", "key": "=", "modifiers": ["ctrl", "shift"], **focused}),
        ("press_key", {"scope": "desktop", "key": "/", "modifiers": ["shift"], **focused}),
        ("press_key", {"scope": "desktop", "key": "a", "modifiers": ["shift"], **focused}),
        ("press_key", {"scope": "desktop", "key": "/", "modifiers": [], **focused}),
        ("click", {"scope": "desktop", "x": 10, "y": 20, "button": "left", "count": 1, "modifier": []}),
    ]
    assert driver._keymap is None
    assert hyprland.asked == ["cua:status"] + ["activewindow", "cua:status"] * 6 + ["cua:status"]


@pytest.mark.asyncio
async def test_on_hyprland_super_and_keys_held_with_it_go_to_the_desktop_as_hyprlands_own(module, daemon, hyprland):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    driver = module.CuaDriver()

    await driver.press_keys(["Meta", "2"])
    await driver.press_keys(["Super"])

    assert daemon.calls() == [
        ("press_key", {"scope": "desktop", "key": "2", "modifiers": ["super"], "delivery_mode": "foreground"}),
        ("press_key", {"scope": "desktop", "key": "super", "modifiers": [], "delivery_mode": "foreground"})]
    assert "activewindow" not in hyprland.asked


@pytest.mark.asyncio
async def test_on_hyprland_with_nothing_focused_keys_go_to_the_desktop_punctuation_by_its_keysym_name(
        module, daemon, hyprland):
    for keys in (["Escape"], ["Control", "/"], ["!"]):
        await module.CuaDriver().press_keys(keys)

    assert daemon.calls() == [
        ("press_key", {"scope": "desktop", "key": "escape", "modifiers": [], "delivery_mode": "foreground"}),
        ("press_key", {"scope": "desktop", "key": "slash", "modifiers": ["ctrl"], "delivery_mode": "foreground"}),
        ("press_key", {"scope": "desktop", "key": "exclam", "modifiers": [], "delivery_mode": "foreground"})]


@pytest.mark.asyncio
async def test_on_hyprland_a_key_refused_while_a_menu_holds_the_keyboard_goes_to_the_desktop(module, daemon, hyprland):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    daemon.grab = True

    for keys in (["?"], ["Escape"]):
        await module.CuaDriver().press_keys(keys)

    focused = {"pid": 42, "window_id": 0x5557E0F44AD0, "delivery_mode": "foreground"}
    assert daemon.calls() == [
        ("press_key", {"scope": "desktop", "key": "/", "modifiers": ["shift"], **focused}),
        ("press_key", {"scope": "desktop", "key": "question", "modifiers": [], "delivery_mode": "foreground"}),
        ("press_key", {"scope": "desktop", "key": "escape", "modifiers": [], **focused}),
        ("press_key", {"scope": "desktop", "key": "escape", "modifiers": [], "delivery_mode": "foreground"})]


RESET = ("press_key", {"scope": "desktop", "key": "f12", "modifiers": ["super", "ctrl", "alt", "shift"],
                       "delivery_mode": "foreground"})


@pytest.mark.asyncio
async def test_on_hyprland_a_key_refused_for_the_seats_keymap_goes_again_once_a_chord_bound_to_nothing_puts_it_back(
        module, daemon, hyprland):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    driver = module.CuaDriver()

    for keys in (["Backspace"], ["Control", "a"]):
        daemon.unlaid = True
        await driver.press_keys(keys)
    hyprland.binds.clear()
    daemon.unlaid = True
    await driver.press_keys(["Backspace"])

    window = {"scope": "desktop", "pid": 42, "window_id": 0x5557E0F44AD0, "delivery_mode": "foreground"}
    backspace = ("press_key", {"key": "backspace", "modifiers": [], **window})
    select_all = ("press_key", {"key": "a", "modifiers": ["ctrl"], **window})
    assert daemon.calls() == [backspace, RESET, backspace, select_all, RESET, select_all, backspace, RESET, backspace]
    assert hyprland.lua == ['hl.bind("SUPER + CTRL + ALT + SHIFT + F12", function() end)'] * 2


@pytest.mark.asyncio
async def test_on_hyprland_a_key_refused_for_the_seats_keymap_fails_without_the_chord_while_another_bind_has_it(
        module, daemon, hyprland):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    driver = module.CuaDriver()
    daemon.unlaid = True
    await driver.press_keys(["Backspace"])
    for theirs in ([{"modmask": 77, "key": "F12", "dispatcher": "__lua", "arg": "7", "non_consuming": False}],
                   [*hyprland.binds, {"modmask": 77, "key": "F12", "dispatcher": "exec", "arg": "reboot"}]):
        hyprland.binds[:] = theirs
        daemon.unlaid = True
        with pytest.raises(DeliveryFailed, match="unsupported_layout") as refused:
            await driver.press_keys(["Backspace"])
        assert "bound to something this driver didn't bind" in refused.value.hint

    assert [args.get("pid") for _, args in daemon.calls()] == [42, None, 42, 42, 42] and len(hyprland.lua) == 1


@pytest.mark.asyncio
async def test_on_hyprland_a_key_refused_for_the_seats_keymap_fails_without_the_chord_when_hyprland_wont_bind_it(
        module, daemon, hyprland):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    hyprland.binding, daemon.unlaid = False, True

    with pytest.raises(DeliveryFailed, match="unsupported_layout"):
        await module.CuaDriver().press_keys(["Backspace"])

    assert [args.get("pid") for _, args in daemon.calls()] == [42] and len(hyprland.lua) == 1


@pytest.mark.asyncio
async def test_on_hyprland_a_key_still_refused_for_the_seats_keymap_after_the_chord_fails_in_time(
        module, daemon, hyprland, monkeypatch):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    daemon.unlaid, daemon.relaid = True, False
    monkeypatch.setattr(module.hyprland, "RESET_S", 0.2)

    with pytest.raises(DeliveryFailed, match="unsupported_layout"):
        await module.CuaDriver().press_keys(["Backspace"])

    calls = daemon.calls()
    assert calls[1] == RESET and len(calls) > 3 and all(args.get("pid") == 42 for _, args in calls[:1] + calls[2:])


@pytest.mark.asyncio
async def test_on_hyprland_a_key_the_focused_window_refuses_otherwise_fails_without_going_to_the_desktop(
        module, daemon, hyprland):
    hyprland.answers["activewindow"] = {"address": "0x5557e0f44ad0", "pid": 42}
    daemon.refuse.add("press_key")

    with pytest.raises(DeliveryFailed, match="cua-driver press_key: press_key refused"):
        await module.CuaDriver().press_keys(["Backspace"])

    assert [args.get("pid") for _, args in daemon.calls()] == [42]


@pytest.mark.asyncio
async def test_on_hyprland_input_is_refused_and_not_sent_once_hyprland_runs_without_cua_drivers_plugin(
        module, daemon, hyprland):
    driver = module.CuaDriver()
    await driver.click_at(10, 20, "left", 1, [])

    del hyprland.answers["cua:status"]
    for act in (driver.click_at(10, 20, "left", 1, []), driver.press_keys(["Escape"]), driver.type_in("hi")):
        with pytest.raises(module.DeviceUnavailable, match="Hyprland hasn't loaded cua-driver's plugin"):
            await act
    async with mcp_session(driver) as session:
        clicked = await session.call_tool("click", {"pid": 42, "window_id": 7, "x": 1, "y": 2})
        listed = await session.call_tool("list_windows", {})

    assert clicked.isError and clicked.content[0].text == "Hyprland hasn't loaded cua-driver's plugin"
    assert not listed.isError
    assert [name for name, _ in daemon.calls()] == ["click", "list_windows"]


@pytest.mark.asyncio
async def test_on_hyprland_readiness_checks_its_own_pin_and_every_time_that_hyprland_has_cua_drivers_plugin(
        module, daemon, hyprland, monkeypatch):
    driver = module.CuaDriver()
    driver.platform = Platform(os="linux", arch="amd64", display="wayland")
    digests = iter([module.EXECUTABLES["amd64"], module.EXECUTABLES["hyprland/amd64"]])
    monkeypatch.setattr(module, "_executable_sha256", lambda: next(digests))
    x11 = await driver.ready_checks()
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "ours")
    hyprland.answers.update({
        "cua:status": [PLUGIN_LOADED, None, {**PLUGIN_LOADED, "configured": False},
                       {**PLUGIN_LOADED, "abi": {"match": False}}, PLUGIN_RELOADED, None, None],
        "activeworkspace": [{"id": 1}, None, None],
        "instances": [[{"instance": "ours"}], [{"instance": "after-a-crash"}]]})

    checks = [await driver.ready_checks() for _ in range(7)]

    assert (x11[0].name, x11[0].ok) == ("cua-driver", False)
    assert [(c.name, c.ok) for c in checks[0]] == [("cua-driver", True), ("cua-driver tools", True),
                                                    ("cua-driver plugin", True)]
    assert [(c[-1].ok, c[-1].detail) for c in checks[1:]] == [
        (False, "Hyprland hasn't loaded cua-driver's plugin"),
        (False, "cua-driver's plugin is loaded but not enabled"),
        (False, "cua-driver's plugin was built for another Hyprland"),
        (False, "cua-driver's plugin isn't taking input: input seat lifetime unavailable; restart the desktop before "
                "loading a replacement plugin"),
        (False, "Hyprland doesn't answer on its IPC"),
        (False, module.hyprland.REPLACED)]
    assert [c[-1].final for c in checks] == [False] * 6 + [True]
    assert module.hyprland.REPLACED.endswith("an infrastructure failure")
    assert sum(1 for _, r in daemon.requests if r["method"] == "list") == 2


@pytest.mark.usefixtures("hyprland")
def test_on_hyprland_the_daemon_waits_for_the_compositor_and_runs_cua_drivers_wayland_backend(module, monkeypatch):
    started = []

    def execve(path, argv, env):
        started.append((argv, env))
        raise SystemExit
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="cd-") as runtime:
        monkeypatch.setenv("XDG_RUNTIME_DIR", runtime)
        monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-1")
        monkeypatch.setattr(module, "_executable", lambda: Path("/opt/cua-driver"))
        monkeypatch.setattr(module.os, "execve", execve)
        with socket.socket(socket.AF_UNIX) as compositor:
            compositor.bind(str(Path(runtime) / "wayland-1"))
            with pytest.raises(SystemExit):
                module._daemon()

    (argv, env), = started
    assert argv[:2] == ["/opt/cua-driver", "serve"] and env["CUA_DRIVER_RS_ENABLE_WAYLAND"] == "1"
    assert env["DO_NOT_TRACK"] == "1"


@pytest.mark.asyncio
async def test_on_hyprland_input_once_another_hyprland_runs_in_place_of_the_drivers_fails_as_an_infrastructure_failure(
        module, daemon, hyprland, monkeypatch):
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "ours")
    hyprland.answers.update({"cua:status": None, "activeworkspace": None,
                             "instances": [[{"instance": "after-a-crash"}]] * 2})

    with pytest.raises(DeviceUnavailable, match="an infrastructure failure"):
        await module.CuaDriver().press_keys(["Escape"])

    assert daemon.calls() == []


def test_in_a_hyprland_session_the_driver_pins_hyprlands_manifest_and_tells_agents_how_keys_go(module, monkeypatch):
    monkeypatch.setenv("HYPRLAND_INSTANCE_SIGNATURE", "ours")
    monkeypatch.setenv("XDG_RUNTIME_DIR", "/run/user/1000")
    loaded = _loaded(monkeypatch)

    assert loaded.MANIFEST.name == "mcp-hyprland.json" and loaded.CuaDriver.notes == loaded.hyprland.NOTES
    assert loaded._executable() == DRIVER.with_name("cua-driver") / "cua-driver"
    assert module.MANIFEST.name == "mcp.json" and module.CuaDriver.notes == ""


def test_an_omarchy_computer_installs_cua_driver_at_hyprlands_pin_and_the_driver_supports_omarchy(module):
    found, problems = installs(tomllib.loads((DRIVER.parent / "env.toml").read_text())["install"])
    omarchy, ubuntu = (SimpleNamespace(type="linux", distribution=name) for name in ("omarchy", "ubuntu"))

    assert not problems
    assert str(for_os(found, omarchy).python) == "requirements-omarchy.txt"
    assert str(for_os(found, ubuntu).python) == "requirements.txt"
    assert for_os(found, omarchy).services == for_os(found, ubuntu).services
    assert [(d.to, d.arch) for d in for_os(found, omarchy).downloads] == [("cua-driver.tar.gz", "amd64")]
    assert re.fullmatch(RELEASE_TARBALL, for_os(found, omarchy).downloads[0].url)
    assert [run.script for run in for_os(found, omarchy).run] == ["install-hyprland.sh"]
    assert not any(line.startswith("cua-driver") for line in
                   (DRIVER.parent / "requirements-omarchy.txt").read_text().splitlines())
    platforms = [platform for entry in module.CuaDriver.supports for platform in supported(entry)]
    assert Platform(os="linux", distribution="omarchy", arch="amd64", display="wayland") in platforms


def test_omarchys_base_builds_the_hyprland_plugin_of_the_cua_driver_release_its_computers_install():
    found, _ = installs(tomllib.loads((DRIVER.parent / "env.toml").read_text())["install"])
    pin = re.fullmatch(RELEASE_TARBALL, for_os(found, SimpleNamespace(type="linux", distribution="omarchy"))
                       .downloads[0].url)[1]
    recipe = (Path(agent_env.__file__).parent / "computers" / "disk" / "omarchy" / "recipe.sh").read_text()

    release = re.search(r"^plugin_release=\S+/cua-driver-rs-v(\S+)$", recipe, re.MULTILINE)[1]
    source = re.search(r"^plugin=cua-hyprland-plugin-(\S+)-[0-9a-f]{40}$", recipe, re.MULTILINE)[1]
    assert release == source == pin
