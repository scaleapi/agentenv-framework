"""HCU's Linux and Windows MCP contracts, isolated server environment, retained session, failures and readiness,
without a desktop or native Windows imports.
The installer runs against a local tarball with its dconf paths redirected to the test directory."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
import tarfile
import textwrap
import tomllib
import types
from pathlib import Path

import pytest
import pytest_asyncio
from mcp.types import CallToolResult, TextContent
from mcp.client import stdio

import agent_env
from agentenv_protocol import computer
from agentenv_protocol.computer import ComputerDriverEnvironment, Platform, ReadyCheck

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "hcu" / "driver.py"
ROOT = Path(__file__).resolve().parents[3]
PLATFORM = Platform(os="linux", distribution="ubuntu", version="24.04", arch="amd64", display="x11")
WINDOWS_PLATFORM = Platform(os="windows", version="11", arch="amd64", display="dwm")
FAKE = textwrap.dedent('''\
    import json
    import os
    import sys
    from pathlib import Path

    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    state = Path(os.environ.get("USERPROFILE", str(Path.home()))) / "fake"
    with (state / "starts.jsonl").open("a") as output:
        output.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ), "pid": os.getpid()}) + "\\n")
    if (state / "die").exists():
        sys.exit(1)
    manifest = json.loads((state / "mcp.json").read_text())
    server = Server("hcu", version="1.0.1", instructions=manifest["instructions"])
    calls = 0

    @server.list_tools()
    async def tools():
        return [types.Tool(**tool) for tool in manifest["tools"]]

    @server.call_tool(validate_input=False)
    async def call(name, arguments):
        global calls
        calls += 1
        with (state / "calls.jsonl").open("a") as output:
            output.write(json.dumps({"tool": name, "arguments": arguments, "pid": os.getpid()}) + "\\n")
        if arguments.get("mode") == "exit":
            os._exit(1)
        if arguments.get("mode") == "stall":
            await server.request_context.session.send_log_message(level="info", data="stalled")
            await anyio.sleep_forever()
        if arguments.get("mode") == "error":
            return types.CallToolResult(isError=True, content=[types.TextContent(type="text", text="ref expired")])
        answer = {"tool": name, "arguments": arguments, "pid": os.getpid(), "calls": calls}
        content = [types.TextContent(type="text", text=json.dumps(answer))]
        if name == "GUI":
            content.append(types.ImageContent(type="image", mimeType="image/png", data="cG5n"))
        return types.CallToolResult(content=content)

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
''')


def _fake_x11(monkeypatch) -> None:
    def module(name: str, **attributes) -> types.ModuleType:
        loaded = types.ModuleType(name)
        loaded.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, loaded)
        return loaded

    constants = module("Xlib.X", Button1Mask=1 << 8, Button2Mask=1 << 9, Button3Mask=1 << 10)
    xtest = module("Xlib.ext.xtest")
    module("Xlib", X=constants, XK=module("Xlib.XK"), display=module("Xlib.display"))
    module("Xlib.error", ConnectionClosedError=OSError, DisplayConnectionError=OSError)
    module("Xlib.ext", xtest=xtest)
    module("mss")
    monkeypatch.delitem(sys.modules, "agentenv_protocol.computer._x11", raising=False)


async def _xtest(self) -> list[ReadyCheck]:
    return [ReadyCheck("xtest", True, "XTEST available")]


def _fake_windows(monkeypatch) -> None:
    class WindowsDriver(ComputerDriverEnvironment):
        @computer.display
        def screen(self):
            raise AssertionError("no Windows desktop in this test")

        @computer.capture
        def capture_screen(self):
            raise AssertionError("no Windows desktop in this test")

        @computer.click
        def click_at(self, *args):
            raise AssertionError("no Windows desktop in this test")

        @computer.drag
        def drag_along(self, *args):
            raise AssertionError("no Windows desktop in this test")

        @computer.scroll
        def scroll_at(self, *args):
            raise AssertionError("no Windows desktop in this test")

        @computer.type_text
        def type_in(self, text):
            raise AssertionError("no Windows desktop in this test")

        @computer.press
        def press_keys(self, keys):
            raise AssertionError("no Windows desktop in this test")

        async def ready_checks(self) -> list[ReadyCheck]:
            return [ReadyCheck("input_desktop", True, "the input desktop is this session's")]

    loaded = types.ModuleType("agentenv_protocol.computer._windows")
    loaded.WindowsDriver = WindowsDriver
    monkeypatch.setitem(sys.modules, loaded.__name__, loaded)


@pytest.fixture(params=["linux", "windows"])
def module(request, monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "C.UTF-8")
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    monkeypatch.setenv("DISPLAY", ":42")
    monkeypatch.setenv("XAUTHORITY", str(tmp_path / "Xauthority"))
    monkeypatch.setenv("AGENTENV_CONTROL_TOKEN", "must-not-reach-hcu")
    monkeypatch.setenv("PYTHONPATH", "must-not-reach-hcu")
    monkeypatch.setenv("SHELL", "/private/shell")
    monkeypatch.setenv("TERM", "must-not-reach-hcu")
    for name in ("AGENTENV_APP_SETTINGS", "AGENTENV_MODEL_API_KEY",
                 "COMPUTER_USE_WINDOWS_PROVIDER", "COMPUTER_USE_WINDOWS_PROVIDER_TRACE",
                 "COMPUTER_USE_CHANNELS", "HCU_SOCKET", "HCU_SESSION", "DOTNET_STARTUP_HOOKS", "LD_PRELOAD"):
        monkeypatch.setenv(name, "must-not-reach-hcu")
    monkeypatch.delenv("AGENTENV_TOOL_PREFIX", raising=False)
    monkeypatch.delenv("USERPROFILE", raising=False)
    for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        monkeypatch.delenv(name, raising=False)
    platform = WINDOWS_PLATFORM if request.param == "windows" else PLATFORM
    monkeypatch.setattr("agentenv_protocol.computer.driver.detect", lambda: platform)
    if request.param == "windows":
        _fake_windows(monkeypatch)
        for name, value in {"SYSTEMROOT": r"C:\Windows", "WINDIR": r"C:\Windows", "SYSTEMDRIVE": "C:",
                            "USERNAME": "user", "USERPROFILE": str(tmp_path), "TEMP": str(tmp_path),
                            "TMP": str(tmp_path), "APPDATA": str(tmp_path / "AppData" / "Roaming"),
                            "LOCALAPPDATA": str(tmp_path / "AppData" / "Local"),
                            "COMSPEC": r"C:\Windows\System32\cmd.exe", "PATHEXT": ".COM;.EXE;.BAT;.CMD"}.items():
            monkeypatch.setenv(name, value)
        monkeypatch.setattr(stdio, "DEFAULT_INHERITED_ENV_VARS", [
            "APPDATA", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "PATH", "PATHEXT", "PROCESSOR_ARCHITECTURE",
            "SYSTEMDRIVE", "SYSTEMROOT", "TEMP", "USERNAME", "USERPROFILE"])
    else:
        _fake_x11(monkeypatch)
    spec = importlib.util.spec_from_file_location("hcu_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as importing:
        importing.setattr(sys, "platform", "win32" if request.param == "windows" else "linux")
        spec.loader.exec_module(loaded)
    if request.param == "linux":
        monkeypatch.setattr(loaded.OSDriver, "ready_checks", _xtest)
    monkeypatch.setattr(loaded.tempfile, "tempdir", str(tmp_path))
    state = tmp_path / "fake"
    state.mkdir()
    (state / "mcp.json").write_text(loaded.MANIFEST.read_text())
    executable = tmp_path / loaded.EXECUTABLE.name
    executable.write_text(f"#!{sys.executable}\n{FAKE}")
    executable.chmod(0o755)
    monkeypatch.setattr(loaded, "EXECUTABLE", executable)
    provider = tmp_path / "computer-use-windows-provider.exe"
    provider.write_bytes(b"Windows provider from the release")
    monkeypatch.setattr(loaded, "PROVIDER", provider)
    input_ack = tmp_path / "computer-use-input-ack.dll"
    input_ack.write_bytes(b"Windows input acknowledgement helper")
    monkeypatch.setattr(loaded, "INPUT_ACK", input_ack)
    bus = tmp_path / "bus"
    bus.touch()
    monkeypatch.setattr(loaded, "SESSION_BUS", bus)
    return loaded


async def _stop(driver) -> None:
    task = driver._server._task
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@pytest_asyncio.fixture
async def driver(module):
    instance = module.HybridComputerUse()
    yield instance
    await _stop(instance)


def _starts(home: Path) -> list[dict]:
    return [json.loads(line) for line in (home / "fake" / "starts.jsonl").read_text().splitlines()]


def _pin_fake(module, monkeypatch) -> None:
    platform = "windows/amd64" if module.WINDOWS else "linux/amd64"
    monkeypatch.setitem(module.EXECUTABLES, platform, hashlib.sha256(module.EXECUTABLE.read_bytes()).hexdigest())
    if module.WINDOWS:
        monkeypatch.setattr(module, "PROVIDER_SHA256", hashlib.sha256(module.PROVIDER.read_bytes()).hexdigest())
        monkeypatch.setattr(module, "INPUT_ACK_SHA256", hashlib.sha256(module.INPUT_ACK.read_bytes()).hexdigest())


def test_install_downloads_the_version_and_platform_the_driver_pins(module):
    configuration = tomllib.loads((DRIVER.parent / "env.toml").read_text())
    assert configuration["module"] == "driver.py:HybridComputerUse"
    assert module.HybridComputerUse.supports == [
        {"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64"]},
        {"os": {"type": "windows", "display": ["dwm"]}, "arch": ["amd64"]}]
    assert set(configuration["install"]) == {"linux", "windows"}
    downloads = configuration["install"]["linux"]["downloads"]
    assert set(module.EXECUTABLES) == {"linux/amd64", "windows/amd64"}
    assert len(downloads) == 1
    download = downloads[0]
    assert download["url"] == ("https://github.com/microsoft/Hybrid-Computer-Use/releases/download/"
                               f"v{module.VERSION}/computer-use-{module.VERSION}-linux-{download['arch']}.tar.gz")
    assert download["to"] == "computer-use.tar.gz"
    assert download["sha256"] == "f078cb073b8284e158b91dcc202d47e38acf5f1ecd99245328f9b15f0a48c2fc"
    assert module.EXECUTABLES["linux/amd64"] == "3a4243b3801b4e49c3fb50d0191e31be5bcc590cc700d2c921c3ff864a7f3923"
    windows = configuration["install"]["windows"]
    for digest in (module.EXECUTABLES["windows/amd64"], module.PROVIDER_SHA256, module.INPUT_ACK_SHA256):
        assert len(digest) == 64 and int(digest, 16) > 0
    assert windows["python"] == "requirements-windows.txt"
    assert (DRIVER.parent / windows["python"]).read_text().strip() == (
        "agentenv-framework-protocol[computer,windows]>=0.1.503")
    assert windows["run"] == [{"script": "install-windows.ps1"}]
    assert windows["services"] == [{"name": "driver", "command": (
        "venv/Scripts/python.exe -m agentenv_protocol.computer.serve driver.py:HybridComputerUse")}]
    assert (DRIVER.parent / windows["run"][0]["script"]).is_file()
    assert not configuration.get("known_failures")
    assert not (DRIVER.parent / "bin").exists()


def test_platform_selects_the_desktop_base_executable_and_manifest(module):
    expected = ("WindowsDriver", "computer-use.exe", "mcp-windows.json", WINDOWS_PLATFORM) if module.WINDOWS else (
        "X11Driver", "computer-use", "mcp.json", PLATFORM)
    assert (module.OSDriver.__name__, module.EXECUTABLE.name, module.MANIFEST.name,
            module.HybridComputerUse().platform) == expected
    assert module.HybridComputerUse.__bases__ == (module.OSDriver,)


def test_instructions_and_every_tool_field_are_the_pinned_servers(module):
    instance = module.HybridComputerUse()
    manifest = json.loads(module.MANIFEST.read_text())
    assert instance.hcu_instructions() == manifest["instructions"]
    assert manifest["instructions"]
    forwarded = instance.forwarded_tools()
    assert [tool.tool.name for tool in forwarded] == ["AX", "GUI"]
    for tool, pinned in zip(forwarded, manifest["tools"], strict=True):
        assert tool.tool.name == pinned["name"]
        assert tool.tool.description == pinned["description"]
        assert tool.tool.inputSchema == pinned["inputSchema"]
        assert tool.annotations == pinned.get("annotations")
        assert tool.output_schema == pinned.get("outputSchema")


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["linux"], indirect=True)
async def test_calls_keep_refs_in_one_server_with_only_the_kept_environment(module, driver, tmp_path):
    ax, gui = driver.forwarded_tools()
    first = await ax.call(mode="snapshot", app="Text Editor")
    second = await ax.call(mode="act", actions=[{"type": "click", "ref": "r1"}])
    screenshot = await gui.call(action="screenshot")

    first_data, second_data = (json.loads(answer.content[0].text) for answer in (first, second))
    assert not first.isError and not second.isError and not screenshot.isError
    assert first_data["arguments"] == {"mode": "snapshot", "app": "Text Editor"}
    assert second_data["arguments"] == {"mode": "act", "actions": [{"type": "click", "ref": "r1"}]}
    assert first_data["pid"] == second_data["pid"] and second_data["calls"] == 2
    assert screenshot.content[1].type == "image" and screenshot.content[1].data == "cG5n"
    (started,) = _starts(tmp_path)
    assert started["argv"] == ["mcp"]
    runtime = Path(started["env"]["XDG_RUNTIME_DIR"])
    assert runtime.is_dir() and stat.S_IMODE(runtime.stat().st_mode) == 0o700
    expected = {name: os.environ[name] for name in module.KEPT if name in os.environ}
    assert started["env"] == {**expected, "PATH": module.PATH, "XDG_RUNTIME_DIR": str(runtime),
                               "DBUS_SESSION_BUS_ADDRESS": f"unix:path={module.SESSION_BUS}"}
    await _stop(driver)
    assert not runtime.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["windows"], indirect=True)
async def test_windows_calls_use_stdio_directly_and_keep_only_the_windows_environment(module, driver, tmp_path,
                                                                                     monkeypatch):
    servers = []
    original = module.stdio_client

    @contextlib.asynccontextmanager
    async def client(server):
        servers.append(server)
        async with original(server) as streams:
            yield streams

    def no_runtime_directory(**kwargs):
        raise AssertionError("Windows must not create an X11 runtime directory")

    monkeypatch.setattr(module, "stdio_client", client)
    monkeypatch.setattr(module.tempfile, "TemporaryDirectory", no_runtime_directory)
    ax, gui = driver.forwarded_tools()
    snapshot = await ax.call(mode="apps")
    action = await ax.call(mode="act", actions=[{"type": "click", "ref": "r1"}])
    screenshot = await gui.call(actions=[{"type": "screenshot"}])
    assert not snapshot.isError and not action.isError and not screenshot.isError
    assert json.loads(snapshot.content[0].text)["arguments"] == {"mode": "apps"}
    action_data = json.loads(action.content[0].text)
    assert action_data["arguments"] == {"mode": "act", "actions": [{"type": "click", "ref": "r1"}]}
    assert action_data["pid"] == json.loads(snapshot.content[0].text)["pid"] and action_data["calls"] == 2
    assert screenshot.content[1].type == "image" and screenshot.content[1].data == "cG5n"
    assert json.loads(screenshot.content[0].text)["arguments"] == {"actions": [{"type": "screenshot"}]}
    (server,) = servers
    assert (server.command, server.args) == (str(module.EXECUTABLE), ["mcp"])
    expected = {name: os.environ[name] for name in module.WINDOWS_KEPT if name in os.environ}
    expected["PATH"] = (r"C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;"
                        r"C:\Windows\System32\WindowsPowerShell\v1.0")
    assert server.env == expected
    (started,) = _starts(tmp_path)
    assert started["argv"] == ["mcp"]
    actual = started["env"]
    assert actual.pop("LC_CTYPE", "C.UTF-8") == "C.UTF-8"
    assert actual == expected
    assert expected["USERPROFILE"] == str(tmp_path)
    assert expected["APPDATA"] == str(tmp_path / "AppData" / "Roaming")
    assert expected["LOCALAPPDATA"] == str(tmp_path / "AppData" / "Local")
    assert expected["COMSPEC"] == r"C:\Windows\System32\cmd.exe"
    assert "must-not-reach-hcu" not in actual.values()
    assert not {"DISPLAY", "XAUTHORITY", "XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS", "HOME", "SHELL", "TERM"} & actual.keys()


def test_deploy_secrets_and_provider_overrides_never_reach_hcu(module, monkeypatch):
    for name in ("AGENTENV_COMPUTER_SETTINGS", "AGENTENV_CONTROL_TOKEN", "AGENTENV_APP_SETTINGS",
                 "AGENTENV_TOOL_PREFIX", "COMPUTER_USE_WINDOWS_PROVIDER", "COMPUTER_USE_WINDOWS_PROVIDER_TRACE",
                 "COMPUTER_USE_CHANNELS", "HCU_SOCKET", "HCU_SESSION", "PYTHONPATH", "DOTNET_STARTUP_HOOKS"):
        monkeypatch.setenv(name, "must-not-reach-hcu")
    monkeypatch.setenv("PATH", "/untrusted/bin")
    environment = module._environment()
    assert "must-not-reach-hcu" not in environment.values()
    assert "/untrusted/bin" not in environment["PATH"]


def test_the_manifest_is_read_as_utf8(module, monkeypatch, tmp_path):
    manifest = json.loads(module.MANIFEST.read_text(encoding="utf-8"))
    manifest["instructions"] += "\n读取桌面"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    original = Path.read_text

    def read_text(self, encoding=None, errors=None):
        return original(self, encoding=encoding or "cp1252", errors=errors)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(module, "MANIFEST", path)
    module._manifest.cache_clear()
    assert module.HybridComputerUse().hcu_instructions() == manifest["instructions"]


@pytest.mark.asyncio
async def test_another_driver_has_its_own_server(module, driver):
    other = module.HybridComputerUse()
    try:
        first = await driver._forward("AX", mode="apps")
        second = await other._forward("AX", mode="apps")
        assert json.loads(first.content[0].text)["pid"] != json.loads(second.content[0].text)["pid"]
        assert json.loads(second.content[0].text)["calls"] == 1
    finally:
        await _stop(other)


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["linux"], indirect=True)
async def test_an_existing_runtime_and_bus_are_preserved(module, driver, monkeypatch, tmp_path):
    runtime = tmp_path / "existing-runtime"
    runtime.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", "unix:abstract=existing-session")
    await driver._forward("AX", mode="apps")
    (started,) = _starts(tmp_path)
    assert started["env"]["XDG_RUNTIME_DIR"] == str(runtime)
    assert started["env"]["DBUS_SESSION_BUS_ADDRESS"] == "unix:abstract=existing-session"
    await _stop(driver)
    assert runtime.is_dir()


@pytest.mark.parametrize("module", ["linux"], indirect=True)
def test_an_absent_session_bus_is_not_advertised(module):
    module.SESSION_BUS.unlink()
    assert "DBUS_SESSION_BUS_ADDRESS" not in module._environment()


@pytest.mark.asyncio
async def test_server_errors_are_forwarded_without_restarting(driver, tmp_path):
    result = await driver._forward("AX", mode="error")
    assert result.isError and result.content[0].text == "ref expired"
    next_result = await driver._forward("AX", mode="apps")
    assert not next_result.isError and json.loads(next_result.content[0].text)["calls"] == 2
    assert len(_starts(tmp_path)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["startup", "call"])
async def test_a_failed_server_fails_pending_calls_and_the_next_call_starts_fresh(driver, tmp_path, when):
    marker = tmp_path / "fake" / "die"
    if when == "startup":
        marker.touch()
    answers = await asyncio.wait_for(asyncio.gather(
        driver._forward("AX", mode="exit"), driver._forward("GUI", action="screenshot")), timeout=10)
    assert all(answer.isError for answer in answers)
    assert "so AX may have run" in answers[0].content[0].text
    assert "so GUI may have run" in answers[1].content[0].text
    await driver._server._task
    marker.unlink(missing_ok=True)
    result = await driver._forward("AX", mode="apps")
    assert not result.isError and json.loads(result.content[0].text)["calls"] == 1
    assert len(_starts(tmp_path)) == 2


@pytest.mark.asyncio
async def test_concurrent_calls_are_serial_and_a_cancelled_queued_call_is_not_sent(module, driver, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    seen = []

    class Session:
        async def call_tool(self, tool, arguments):
            seen.append(arguments["number"])
            if arguments["number"] == 1:
                entered.set()
                await release.wait()
            return CallToolResult(content=[TextContent(type="text", text=tool)])

    @contextlib.asynccontextmanager
    async def session():
        yield Session(), None

    monkeypatch.setattr(module, "_session", session)
    first = asyncio.create_task(driver._forward("AX", number=1))
    await entered.wait()
    queued = asyncio.Event()
    original_put = driver._server._calls.put

    async def put(item):
        await original_put(item)
        queued.set()

    monkeypatch.setattr(driver._server._calls, "put", put)
    cancelled = asyncio.create_task(driver._forward("AX", number=2))
    await queued.wait()
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert seen == [1]
    queued.clear()
    last = asyncio.create_task(driver._forward("GUI", number=3))
    await queued.wait()
    assert seen == [1]
    release.set()
    assert all(not result.isError for result in await asyncio.gather(first, last))
    assert seen == [1, 3]


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_caller", [False, True])
async def test_a_stalled_call_expires_closes_its_session_and_does_not_replay_queued_input(
        module, driver, monkeypatch, tmp_path, cancel_caller):
    entered, queued = asyncio.Event(), asyncio.Event()
    deadlines = []
    real_timeout = asyncio.timeout
    real_client = module.ClientSession

    def deadline(delay):
        limit = real_timeout(None)
        deadlines.append(limit)
        return limit

    async def logging(params):
        assert params.data == "stalled"
        entered.set()

    def client(*args, **kwargs):
        return real_client(*args, logging_callback=logging, **kwargs)

    monkeypatch.setattr(module, "ClientSession", client)
    monkeypatch.setattr(module.asyncio, "timeout", deadline)
    first = asyncio.create_task(driver._forward("AX", mode="stall"))
    await asyncio.wait_for(entered.wait(), timeout=10)
    put = driver._server._calls.put

    async def enqueue(item):
        await put(item)
        queued.set()

    monkeypatch.setattr(driver._server._calls, "put", enqueue)
    next_call = asyncio.create_task(driver._forward("GUI", action="screenshot"))
    await queued.wait()
    if cancel_caller:
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
    deadlines[0].reschedule(asyncio.get_running_loop().time() - 1)
    answer = await asyncio.wait_for(next_call, timeout=10)
    await driver._server._task

    assert answer.isError and "TimeoutError: HCU did not answer within" in answer.content[0].text
    (old,) = _starts(tmp_path)
    if module.WINDOWS:
        assert "XDG_RUNTIME_DIR" not in old["env"]
    else:
        assert not Path(old["env"]["XDG_RUNTIME_DIR"]).exists()
    with pytest.raises(ProcessLookupError):
        os.kill(old["pid"], 0)
    if not cancel_caller:
        failed = await first
        assert failed.isError and "so AX may have run" in failed.content[0].text
    fresh = await driver._forward("AX", mode="apps")
    assert not fresh.isError and json.loads(fresh.content[0].text)["calls"] == 1
    old, new = _starts(tmp_path)
    assert old["pid"] != new["pid"]
    called = [json.loads(line) for line in (tmp_path / "fake" / "calls.jsonl").read_text().splitlines()]
    assert called == [
        {"tool": "AX", "arguments": {"mode": "stall"}, "pid": old["pid"]},
        {"tool": "AX", "arguments": {"mode": "apps"}, "pid": new["pid"]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["schema", "missing", "extra", "instructions"])
async def test_readiness_checks_the_executable_then_instructions_and_tools(module, driver, tmp_path, monkeypatch, change):
    unpinned = await driver.ready_checks()
    assert not unpinned[0].ok and unpinned[-1].ok
    if module.WINDOWS:
        assert not unpinned[1].ok and unpinned[1].name == "hcu provider"
    assert not (tmp_path / "fake" / "starts.jsonl").exists()
    _pin_fake(module, monkeypatch)
    ready = await driver.ready_checks()
    names = ["hcu", *(["hcu provider", "hcu input acknowledgement"] if module.WINDOWS else []), "hcu tools",
             "input_desktop" if module.WINDOWS else "xtest"]
    assert [(check.name, check.ok) for check in ready] == [(name, True) for name in names]
    assert all(check.ok for check in await driver.ready_checks())
    assert len(_starts(tmp_path)) == 1
    manifest = json.loads(module.MANIFEST.read_text())
    if change == "schema":
        manifest["tools"][0]["inputSchema"] = {"type": "object", "properties": {}}
    elif change == "missing":
        manifest["tools"].pop(0)
    elif change == "extra":
        manifest["tools"].append({"name": "extra", "inputSchema": {"type": "object"}})
    else:
        manifest["instructions"] += " changed"
    (tmp_path / "fake" / "mcp.json").write_text(json.dumps(manifest))
    driver._pinned = None
    changed = (await driver.ready_checks())[-2]
    expected = {"schema": "AX", "missing": "AX", "extra": "extra", "instructions": "instructions"}[change]
    assert not changed.ok and changed.detail == f"these differ from {module.MANIFEST.name}: {expected}"


@pytest.mark.asyncio
async def test_readiness_refuses_the_other_platforms_contract(module, driver, tmp_path, monkeypatch):
    _pin_fake(module, monkeypatch)
    other = DRIVER.parent / ("mcp.json" if module.WINDOWS else "mcp-windows.json")
    (tmp_path / "fake" / "mcp.json").write_bytes(other.read_bytes())
    checks = await driver.ready_checks()
    failed, = [check for check in checks if not check.ok]
    assert failed.name == "hcu tools"
    assert "AX" in failed.detail and module.MANIFEST.name in failed.detail


@pytest.mark.asyncio
async def test_missing_executable_and_unpinned_platform_fail_readiness(module, driver, monkeypatch, tmp_path):
    _pin_fake(module, monkeypatch)
    platform = driver.platform
    monkeypatch.setattr(driver, "platform", Platform(os=platform.os, arch="arm64", display=platform.display))
    unsupported = (await driver.ready_checks())[0]
    assert not unsupported.ok and unsupported.detail.endswith("not the pinned None")
    monkeypatch.setattr(driver, "platform", platform)
    module.EXECUTABLE.unlink()
    assert module._digest() is None
    assert not (await driver.ready_checks())[0].ok
    assert not (tmp_path / "fake" / "starts.jsonl").exists()


@pytest.mark.asyncio
async def test_readiness_reports_a_server_that_cannot_list_tools(module, driver, monkeypatch, tmp_path):
    _pin_fake(module, monkeypatch)
    (tmp_path / "fake" / "die").touch()
    checks = await driver.ready_checks()
    assert all(check.ok for check in checks[:-2]) and checks[-1].ok and not checks[-2].ok
    assert checks[-2].detail.startswith("its server didn't list them:")


@pytest.mark.asyncio
@pytest.mark.parametrize("module", ["windows"], indirect=True)
@pytest.mark.parametrize("binary", ["EXECUTABLE", "PROVIDER", "INPUT_ACK"])
@pytest.mark.parametrize("change", ["missing", "digest"])
async def test_windows_checks_all_binaries_before_starting_and_retries_after_repair(module, driver, monkeypatch,
                                                                                  tmp_path, binary, change):
    _pin_fake(module, monkeypatch)
    path = getattr(module, binary)
    original = path.read_bytes()
    if change == "missing":
        path.unlink()
    else:
        path.write_bytes(b"untrusted executable")
    checks = await driver.ready_checks()
    expected = {"EXECUTABLE": "hcu", "PROVIDER": "hcu provider", "INPUT_ACK": "hcu input acknowledgement"}[binary]
    failed, = [check for check in checks if not check.ok]
    assert failed.name == expected
    assert path.name in failed.detail and "not the pinned" in failed.detail
    assert not (tmp_path / "fake" / "starts.jsonl").exists()
    assert (module._digest(path) is None) == (change == "missing")
    path.write_bytes(original)
    path.chmod(0o755)
    assert all(check.ok for check in await driver.ready_checks())
    assert len(_starts(tmp_path)) == 1


@pytest.mark.asyncio
async def test_successful_pins_are_cached_but_desktop_readiness_is_checked_every_time(module, driver, monkeypatch,
                                                                                  tmp_path):
    _pin_fake(module, monkeypatch)
    checked = []
    name = "input_desktop" if module.WINDOWS else "xtest"

    async def desktop(self):
        checked.append(self)
        return [ReadyCheck(name, len(checked) == 1, "current desktop state")]

    monkeypatch.setattr(module.OSDriver, "ready_checks", desktop)
    assert all(check.ok for check in await driver.ready_checks())
    checks = await driver.ready_checks()
    assert all(check.ok for check in checks[:-1])
    assert not checks[-1].ok and checks[-1].name == name
    assert checked == [driver, driver]
    assert len(_starts(tmp_path)) == 1


def test_install_unpacks_the_archive_and_sets_accessibility_before_apps_start(tmp_path):
    source = tmp_path / "computer-use"
    source.write_bytes(b"executable from the release")
    source.chmod(0o755)
    archive = tmp_path / "computer-use.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(source, arcname="computer-use")
    script = (DRIVER.parent / "install.sh").read_text()
    dconf = tmp_path / "etc" / "dconf"
    installer = tmp_path / "install.sh"
    installer.write_text(script.replace("/etc/dconf", str(dconf)))
    commands = tmp_path / "commands"
    commands.mkdir()
    command = commands / "dconf"
    command.write_text('#!/bin/sh\n[ "$1" = update ]\nprintf updated > dconf-updated\n')
    command.chmod(0o755)
    result = subprocess.run(["sh", str(installer)], cwd=tmp_path, env={"PATH": f"{commands}:/usr/bin:/bin"},
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    installed = tmp_path / "bin" / "computer-use"
    assert installed.read_bytes() == source.read_bytes() and os.access(installed, os.X_OK)
    assert not archive.exists()
    assert (dconf / "profile" / "user").read_text() == "user-db:user\nsystem-db:local\n"
    assert (dconf / "db" / "local.d" / "00-hcu-accessibility").read_text() == (
        "[org/gnome/desktop/interface]\ntoolkit-accessibility=true\n")
    assert (tmp_path / "dconf-updated").read_text() == "updated"


@pytest.mark.parametrize("version", [None, "1.0.1"])
def test_manifest_helper_reads_the_servers_initialize_and_list_tools(module, tmp_path, version):
    output = tmp_path / "regenerated.json"
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "dump_hcu_mcp.py"),
                             str(module.EXECUTABLE), str(output), *(["--version", version] if version else [])], env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == json.loads(module.MANIFEST.read_text())
    assert output.read_bytes().endswith(b"\n") and b"\r\n" not in output.read_bytes()


@pytest.mark.parametrize("existing", [False, True])
def test_manifest_helper_refuses_a_different_version_without_writing(module, tmp_path, existing):
    output = tmp_path / "regenerated.json"
    original = b"existing manifest\n"
    if existing:
        output.write_bytes(original)
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "dump_hcu_mcp.py"),
                             str(module.EXECUTABLE), str(output), "--version", "0.0.0"], env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "Expected HCU 0.0.0, got 1.0.1" in result.stderr
    assert output.exists() == existing
    if existing:
        assert output.read_bytes() == original
