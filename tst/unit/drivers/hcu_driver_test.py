"""HCU's pinned MCP contract, isolated server environment, retained session, failures and readiness, without a desktop.
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

import agent_env
from agentenv_protocol.computer import Platform, ReadyCheck

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "hcu" / "driver.py"
ROOT = Path(__file__).resolve().parents[3]
PLATFORM = Platform(os="linux", distribution="ubuntu", version="24.04", arch="amd64", display="x11")
FAKE = textwrap.dedent('''\
    import json
    import os
    import sys
    from pathlib import Path

    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    state = Path.home() / "fake"
    with (state / "starts.jsonl").open("a") as output:
        output.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ), "pid": os.getpid()}) + "\\n")
    if (state / "die").exists():
        sys.exit(1)
    manifest = json.loads((state / "mcp.json").read_text())
    server = Server("hcu", instructions=manifest["instructions"])
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


@pytest.fixture
def module(monkeypatch, tmp_path):
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
    for name in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("agentenv_protocol.computer.driver.detect", lambda: PLATFORM)
    _fake_x11(monkeypatch)
    spec = importlib.util.spec_from_file_location("hcu_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    monkeypatch.setattr(loaded.X11Driver, "ready_checks", _xtest)
    monkeypatch.setattr(loaded.tempfile, "tempdir", str(tmp_path))
    state = tmp_path / "fake"
    state.mkdir()
    (state / "mcp.json").write_text(loaded.MANIFEST.read_text())
    executable = tmp_path / "computer-use"
    executable.write_text(f"#!{sys.executable}\n{FAKE}")
    executable.chmod(0o755)
    monkeypatch.setattr(loaded, "EXECUTABLE", executable)
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
    monkeypatch.setitem(module.EXECUTABLES, "linux/amd64", hashlib.sha256(module.EXECUTABLE.read_bytes()).hexdigest())


def test_install_downloads_the_version_and_platform_the_driver_pins(module):
    configuration = tomllib.loads((DRIVER.parent / "env.toml").read_text())
    assert configuration["module"] == "driver.py:HybridComputerUse"
    assert module.HybridComputerUse.supports == [{"os": {"type": "linux", "display": ["x11"]}, "arch": ["amd64"]}]
    assert set(configuration["install"]) == {"linux"}
    downloads = configuration["install"]["linux"]["downloads"]
    assert {f"linux/{download['arch']}" for download in downloads} == set(module.EXECUTABLES)
    assert len(downloads) == 1
    download = downloads[0]
    assert download["url"] == ("https://github.com/microsoft/Hybrid-Computer-Use/releases/download/"
                               f"v{module.VERSION}/computer-use-{module.VERSION}-linux-{download['arch']}.tar.gz")
    assert download["to"] == "computer-use.tar.gz"
    assert download["sha256"] == "f078cb073b8284e158b91dcc202d47e38acf5f1ecd99245328f9b15f0a48c2fc"
    assert module.EXECUTABLES["linux/amd64"] == "3a4243b3801b4e49c3fb50d0191e31be5bcc590cc700d2c921c3ff864a7f3923"
    assert not (DRIVER.parent / "bin").exists()


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
    assert not Path(old["env"]["XDG_RUNTIME_DIR"]).exists()
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
    wrong, xtest = await driver.ready_checks()
    assert not wrong.ok and xtest.ok
    assert not (tmp_path / "fake" / "starts.jsonl").exists()
    _pin_fake(module, monkeypatch)
    ready = await driver.ready_checks()
    assert [(check.name, check.ok) for check in ready] == [("hcu", True), ("hcu tools", True), ("xtest", True)]
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
    _, changed, _ = await driver.ready_checks()
    expected = {"schema": "AX", "missing": "AX", "extra": "extra", "instructions": "instructions"}[change]
    assert not changed.ok and changed.detail == f"these differ from mcp.json: {expected}"


@pytest.mark.asyncio
async def test_missing_executable_and_unpinned_platform_fail_readiness(module, driver, monkeypatch, tmp_path):
    module.EXECUTABLE.unlink()
    assert module._digest() is None
    assert not (await driver.ready_checks())[0].ok
    monkeypatch.setattr(driver, "platform", Platform(os="linux", arch="arm64", display="x11"))
    assert not (await driver.ready_checks())[0].ok
    assert not (tmp_path / "fake" / "starts.jsonl").exists()


@pytest.mark.asyncio
async def test_readiness_reports_a_server_that_cannot_list_tools(module, driver, monkeypatch, tmp_path):
    _pin_fake(module, monkeypatch)
    (tmp_path / "fake" / "die").touch()
    executable, tools, xtest = await driver.ready_checks()
    assert executable.ok and xtest.ok and not tools.ok
    assert tools.detail.startswith("its server didn't list them:")


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


def test_manifest_helper_reads_the_servers_initialize_and_list_tools(module, tmp_path):
    output = tmp_path / "regenerated.json"
    environment = {name: value for name, value in os.environ.items() if name != "PYTHONPATH"}
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "dump_hcu_mcp.py"),
                             str(module.EXECUTABLE), str(output)], env=environment,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == json.loads(module.MANIFEST.read_text())
