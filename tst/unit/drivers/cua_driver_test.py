"""The cua-driver driver against a fake cua-driver daemon on a Unix socket: the computer/v1 operations become its
desktop-scope calls in the framework's session, its own tools are listed as pinned and forwarded on their caller's
connection as they came, readiness checks the pinned executable and tools, and the skill leads with cua-driver's pack."""
from __future__ import annotations

import asyncio
import base64
import contextlib
import importlib.util
import io
import json
import stat
import tempfile
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from mcp.shared.memory import create_connected_server_and_client_session
from PIL import Image as PILImage

import agent_env
from agentenv_protocol.computer import DeviceUnavailable, Unsupported

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "cua-driver" / "driver.py"


def _png(width: int, height: int) -> str:
    out = io.BytesIO()
    PILImage.new("RGB", (width, height), (255, 0, 0)).save(out, "PNG")
    return base64.b64encode(out.getvalue()).decode()


class FakeDaemon:
    """cua-driver's daemon as the driver sees it, on ``path``: each request by the connection it came on, ``refuse``
    naming tools whose results are errors, ``fail`` tools it can't call, and ``tools`` what ``list`` lists."""

    def __init__(self, manifest: list[dict], path: Path) -> None:
        self.path = path
        self.requests: list[tuple[int, dict]] = []
        self.refuse: set[str] = set()
        self.fail: set[str] = set()
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
            writer.write((json.dumps(self.reply(request)) + "\n").encode())
            await writer.drain()
        writer.close()

    def reply(self, request: dict) -> dict:
        if request["method"] == "list":
            return {"ok": True, "result": {"tools": self.tools}}
        name = request["name"]
        if name in self.fail:
            return {"ok": False, "error": f"unknown tool {name}", "exit_code": 1}
        if name in self.refuse:
            return {"ok": True, "result": {"isError": True, "content": [{"type": "text", "text": f"{name} refused"}]}}
        if name == "get_screen_size":
            return {"ok": True, "result": {"content": [{"type": "text", "text": "1280x800"}],
                                           "structuredContent": {"width": 1280, "height": 800, "scale_factor": 1.0}}}
        if name == "get_desktop_state":
            return {"ok": True, "result": {
                "content": [{"type": "image", "data": _png(1280, 800), "mimeType": "image/png"}],
                "structuredContent": {"screenshot_width": 1280, "screenshot_height": 800}}}
        return {"ok": True, "result": {"content": [{"type": "text", "text": f"did {name}"}],
                                       "structuredContent": {"tool": name}}}

    def calls(self, connection: int | None = None) -> list[tuple[str, dict]]:
        return [(r["name"], r["args"]) for c, r in self.requests if r["method"] == "call"
                and (connection is None or c == connection)]


@pytest.fixture
def module(monkeypatch):
    monkeypatch.delenv("ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    spec = importlib.util.spec_from_file_location("cua_driver_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest_asyncio.fixture
async def daemon(module, monkeypatch):
    """A fake daemon on a socket the driver is pointed at, under a short /tmp path for macOS's socket path limit."""
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="cd-") as folder:
        fake = FakeDaemon(module._manifest(), Path(folder) / "cua-driver.sock")
        monkeypatch.setattr(module, "SOCKET", fake.path)
        await fake.start()
        yield fake
        await fake.stop()


@contextlib.asynccontextmanager
async def mcp_session(driver):
    async with create_connected_server_and_client_session(driver.mcp or driver.create_app()) as session:
        yield session


@pytest.mark.asyncio
async def test_the_operations_become_desktop_scope_calls_in_the_frameworks_session(module, daemon):
    driver = module.CuaDriver()

    size, shot = await driver.screen(), await driver.grab()
    await driver.click_at(100, 200, "right", 2, ["Control"])
    await driver.drag_along([(10, 20), (30, 40)], "left", 300, ["Shift"])
    await driver.scroll_at(5, 6, "down", 3, [])
    await driver.type_in("Hello, Wörld! 東京")
    await driver.press_keys(["Control", "/"])
    await driver.press_keys(["Enter"])

    assert (size.width, size.height, shot.width, shot.height, shot.mode) == (1280, 800, 1280, 800, "png")
    framework = {"session": "agentenv"}
    assert daemon.calls() == [
        ("get_screen_size", framework), ("get_desktop_state", framework),
        ("click", {"scope": "desktop", "x": 100, "y": 200, "button": "right", "count": 2, "modifier": ["ctrl"], **framework}),
        ("drag", {"scope": "desktop", "from_x": 10, "from_y": 20, "to_x": 30, "to_y": 40, "button": "left",
                  "duration_ms": 300, "modifier": ["shift"], **framework}),
        ("scroll", {"scope": "desktop", "x": 5, "y": 6, "direction": "down", "amount": 3, **framework}),
        ("type_text", {"scope": "desktop", "text": "Hello, Wörld! 東京", **framework}),
        ("press_key", {"scope": "desktop", "key": "/", "modifiers": ["ctrl"], **framework}),
        ("press_key", {"scope": "desktop", "key": "enter", "modifiers": [], **framework}),
    ]
    assert daemon.connections == 1


@pytest.mark.asyncio
async def test_what_cua_driver_cant_do_or_refuses_fails_naming_it(module, daemon):
    driver = module.CuaDriver()
    daemon.refuse.add("click")

    with pytest.raises(Unsupported, match="holds only modifier keys"):
        await driver.press_keys(["Control", "a", "b"])
    with pytest.raises(Unsupported, match="no keys held"):
        await driver.scroll_at(1, 1, "up", 1, ["Shift"])
    with pytest.raises(Exception, match="cua-driver click: click refused") as refused:
        await driver.click_at(1, 1, "left", 1, [])
    assert type(refused.value).__name__ == "DeliveryFailed"


@pytest.mark.asyncio
async def test_cua_drivers_tools_are_listed_as_pinned_and_forwarded_on_their_callers_connection(module, daemon):
    driver = module.CuaDriver()
    async with mcp_session(driver) as session:
        listed = {t.name: t for t in (await session.list_tools()).tools}
        await session.call_tool("computer_screenshot", {})
        answered = await session.call_tool("get_window_state", {"pid": 42, "window_id": 7, "query": "OK"})

    manifest = {t["name"]: t for t in module._manifest()}
    assert set(manifest) <= set(listed) and {"computer_click", "computer_screenshot"} <= set(listed)
    assert listed["get_window_state"].inputSchema == manifest["get_window_state"]["inputSchema"]
    assert listed["get_window_state"].annotations.readOnlyHint == manifest["get_window_state"]["annotations"]["readOnlyHint"]
    framework, caller = 1, max(c for c, _ in daemon.requests)
    assert caller != framework and daemon.calls(caller) == [("get_window_state", {"pid": 42, "window_id": 7, "query": "OK"})]
    assert answered.structuredContent == {"tool": "get_window_state"} and answered.content[0].text == "did get_window_state"


@pytest.mark.asyncio
async def test_a_failed_call_to_one_of_cua_drivers_tools_reaches_its_caller_as_an_error(module, daemon):
    daemon.fail.add("list_apps")
    async with mcp_session(module.CuaDriver()) as session:
        failed = await session.call_tool("list_apps", {})
        await daemon.stop()
        down = await session.call_tool("list_windows", {})
    assert failed.isError and failed.content[0].text == "unknown tool list_apps"
    assert down.isError and "cua-driver's daemon isn't answering" in down.content[0].text


@pytest.mark.asyncio
async def test_readiness_checks_the_pinned_executable_and_tools_until_they_pass_once(module, daemon, monkeypatch):
    driver = module.CuaDriver()
    pinned = module.EXECUTABLES[driver.platform.arch]
    digests = iter(["0" * 64, pinned])
    monkeypatch.setattr(module, "_executable_sha256", lambda: next(digests))
    daemon.tools[0] = {**daemon.tools[0], "input_schema": {"type": "object"}}

    wrong = await driver.ready_checks()
    daemon.tools[0] = {"name": daemon.tools[0]["name"], "input_schema": module._manifest()[0]["inputSchema"]}
    right = await driver.ready_checks()
    lists = sum(1 for _, r in daemon.requests if r["method"] == "list")
    again = await driver.ready_checks()

    assert [(c.name, c.ok) for c in wrong] == [("cua-driver", False), ("cua-driver tools", False)]
    assert wrong[0].detail.startswith("its executable's sha256 is 0000") and module._manifest()[0]["name"] in wrong[1].detail
    assert [(c.name, c.ok) for c in right] == [("cua-driver", True), ("cua-driver tools", True)]
    assert again == right and sum(1 for _, r in daemon.requests if r["method"] == "list") == lists


@pytest.mark.asyncio
async def test_readiness_waits_for_a_daemon_that_isnt_answering(module, monkeypatch):
    monkeypatch.setattr(module, "SOCKET", Path("/tmp/cd-nothing-here.sock"))
    assert [(c.name, c.ok, c.detail) for c in await module.CuaDriver().ready_checks()] == [
        ("cua-driver", False, "its daemon isn't answering yet")]


@pytest.mark.asyncio
async def test_a_dropped_daemon_connection_is_device_unavailable_and_reopened(module, daemon):
    driver = module.CuaDriver()
    await driver.screen()
    await daemon.stop()

    with pytest.raises(DeviceUnavailable, match="cua-driver's daemon isn't answering"):
        await driver.screen()
    await daemon.start()
    assert (await driver.screen()).width == 1280 and daemon.connections == 2


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
