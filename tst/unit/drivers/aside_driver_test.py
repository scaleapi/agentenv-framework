"""The aside driver against a fake aside CLI: agents get its repl as pinned, with its instructions, marked as acting on
apps; allowed_apps becomes one --app each, none for every app, and never empty; each agent session keeps a REPL of
its own across calls, one that stops starting afresh, and a call its caller gave up while it waited never reaches
Aside; the CLI gets the session user's environment, Aside's home and
no update check, and none of the driver's; readiness checks the pinned executables, the daemon's health, turns usage
analytics off and Computer Use on, and checks what the server says and lists and that it answers; and the driver's
files agree on the pinned version, folder, helper, daemon and Aside's home, where the helper's runtime copy is placed
and granted."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import shlex
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

import agent_env
from agentenv_protocol import SESSION_META_KEY
from agentenv_protocol.computer import Platform, _macos

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "aside" / "driver.py"
FAKE = textwrap.dedent('''\
    import json, os, sys
    from pathlib import Path
    state = Path(os.environ["HOME"]) / "fake"
    if sys.argv[1:] == ["settings", "computer-use", "enable"]:
        with open(state / "enabled", "a") as out:
            out.write("enabled\\n")
        (state / "on").touch()
        print("Computer Use is on.")
        sys.exit(0)
    if sys.argv[1:] == ["settings", "analytics", "disable"]:
        print("Analytics are off.")
        sys.exit(0)
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    (state / f"env-{os.getpid()}.json").write_text(json.dumps(dict(os.environ)))
    manifest = json.loads((state / "tools.json").read_text())
    server = Server("aside", instructions=manifest["instructions"])
    bindings = {}

    @server.list_tools()
    async def tools():
        return [types.Tool(**t) for t in manifest["tools"]]

    @server.call_tool(validate_input=False)
    async def call(name, arguments):
        if not (state / "on").exists():
            raise RuntimeError("Computer Use is turned off. Turn it on with 'aside settings computer-use enable'.")
        code = arguments["code"]
        if code.startswith("set "):
            bindings["x"] = code[4:]
            text = "set"
        elif code == "get":
            text = f"{bindings.get('x')} in {os.getpid()}"
        elif code == "args":
            text = json.dumps(sys.argv[1:])
        elif code == "die":
            os._exit(1)
        elif "computer.listApps" in code:
            text = "3"
        else:
            text = "ran"
        return [types.TextContent(type="text", text=text)]

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
''')
MACOS = Platform(os="macos", version="26.5", arch="arm64", display="aqua")


class Granted:
    def missing(self) -> list[str]:
        return []


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """The driver's module, as it ships."""
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTENV_CONTROL_TOKEN", "a-deploys-secret")
    monkeypatch.setattr(_macos, "Quartz", Granted)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("agentenv_protocol.computer.driver.detect", lambda: MACOS)
    spec = importlib.util.spec_from_file_location("aside_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def module(pinned, monkeypatch, tmp_path):
    """The driver's module on a fake Aside whose CLI is the fake server, its daemon healthy and Computer Use on."""
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "tools.json").write_text(pinned.MANIFEST.read_text())
    (tmp_path / "fake" / "on").touch()
    prefix = tmp_path / "aside"
    cli = prefix / pinned.CLI
    cli.parent.mkdir(parents=True)
    cli.write_text(f"#!{sys.executable}\n{FAKE}")
    cli.chmod(0o755)
    digests = {}
    for path in pinned.EXECUTABLES:
        file = prefix / path
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(f"the pinned {file.name}")
        digests[path] = hashlib.sha256(file.read_bytes()).hexdigest()
    monkeypatch.setattr(pinned, "PREFIX", prefix)
    monkeypatch.setattr(pinned, "EXECUTABLES", digests)
    monkeypatch.setattr(pinned, "_health", lambda: {"ready": True})
    return pinned


def _driver(module, monkeypatch, **settings):
    monkeypatch.setenv("AGENTENV_COMPUTER_SETTINGS", json.dumps(settings))
    return module.AsideDriver()


@contextlib.asynccontextmanager
async def mcp_session(driver):
    async with create_connected_server_and_client_session(driver.mcp or driver.create_app()) as session:
        yield session


def _text(result) -> str:
    return " ".join(c.text for c in result.content if c.type == "text")


async def _gone(pid: int) -> bool:
    for _ in range(200):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        await asyncio.sleep(0.01)
    return False


@pytest.mark.asyncio
async def test_agents_get_asides_repl_as_pinned_with_its_instructions_marked_as_acting_on_apps(module, monkeypatch):
    driver = _driver(module, monkeypatch)
    async with mcp_session(driver) as session:
        listed = {t.name: t for t in (await session.list_tools()).tools}
        answered = await session.call_tool("repl", {"code": "1 + 1"})

    (pinned,) = module._manifest()["tools"]
    assert listed["repl"].inputSchema == pinned["inputSchema"] and listed["repl"].description == pinned["description"]
    hints = listed["repl"].annotations
    assert (hints.readOnlyHint, hints.destructiveHint, hints.openWorldHint) == (False, True, True)
    assert driver.mcp.instructions.startswith(module._manifest()["instructions"])
    assert not answered.isError and _text(answered) == "ran"


@pytest.mark.asyncio
async def test_allowed_apps_become_one_app_flag_each_and_none_for_every_app(module, monkeypatch):
    async with mcp_session(_driver(module, monkeypatch, allowed_apps=["Calculator", "com.apple.TextEdit"])) as session:
        named = json.loads(_text(await session.call_tool("repl", {"code": "args"})))
    async with mcp_session(_driver(module, monkeypatch)) as session:
        every = json.loads(_text(await session.call_tool("repl", {"code": "args"})))

    assert named == ["mcp", "--computer-use", "--app", "Calculator", "--app", "com.apple.TextEdit"]
    assert every == ["mcp", "--computer-use"]
    with pytest.raises(ValidationError):
        module.AsideDriver.Settings(allowed_apps=[])


@pytest.mark.asyncio
async def test_a_failing_cell_reaches_the_agent_as_aside_answers_it(module, monkeypatch, tmp_path):
    (tmp_path / "fake" / "on").unlink()
    async with mcp_session(_driver(module, monkeypatch)) as session:
        failed = await session.call_tool("repl", {"code": "1 + 1"})

    assert failed.isError and "Computer Use is turned off" in _text(failed)


@pytest.mark.asyncio
async def test_each_agent_session_keeps_a_repl_of_its_own_and_one_that_stops_starts_afresh_while_it_still_ends(
        module, monkeypatch):
    """A server that stops takes a while to end; the session's next call starts a new one rather than wait on it."""
    serve = module.AsideDriver._server

    @contextlib.asynccontextmanager
    async def slow_to_end(self):
        async with serve(self) as opened:
            try:
                yield opened
            finally:
                await asyncio.sleep(0.5)

    monkeypatch.setattr(module.AsideDriver, "_server", slow_to_end)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        for caller, value in (("a", "1"), ("b", "2")):
            await session.call_tool("repl", {"code": f"set {value}"}, meta={SESSION_META_KEY: caller})
        a = _text(await session.call_tool("repl", {"code": "get"}, meta={SESSION_META_KEY: "a"}))
        b = _text(await session.call_tool("repl", {"code": "get"}, meta={SESSION_META_KEY: "b"}))
        died = await session.call_tool("repl", {"code": "die"}, meta={SESSION_META_KEY: "a"})
        after = _text(await session.call_tool("repl", {"code": "get"}, meta={SESSION_META_KEY: "a"}))

    (a_value, _, a_pid), (b_value, _, b_pid) = a.partition(" in "), b.partition(" in ")
    assert (a_value, b_value) == ("1", "2") and a_pid != b_pid
    assert died.isError and "may have run" in _text(died)
    assert after.startswith("None in ") and after != f"None in {a_pid}"


@pytest.mark.asyncio
async def test_a_call_cancelled_while_it_waits_behind_another_never_reaches_aside(pinned):
    sent, release = [], asyncio.Event()

    class Session:
        async def call_tool(self, tool, arguments):
            sent.append(arguments["code"])
            if arguments["code"] == "slow":
                await release.wait()
            return arguments["code"]

    @contextlib.asynccontextmanager
    async def open_server():
        yield Session(), None

    repl = pinned._Repl(open_server)
    slow = asyncio.create_task(repl.call("repl", {"code": "slow"}))
    while not sent:
        await asyncio.sleep(0)
    abandoned = asyncio.create_task(repl.call("repl", {"code": "abandoned"}))
    await asyncio.sleep(0)
    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    release.set()
    answers = (await slow, await repl.call("repl", {"code": "after"}))
    await repl.close()

    assert answers == ("slow", "after") and sent == ["slow", "after"] and repl.busy == 0


@pytest.mark.asyncio
async def test_the_cli_gets_the_session_users_environment_and_none_of_the_drivers(module, monkeypatch, tmp_path):
    async with mcp_session(_driver(module, monkeypatch)) as session:
        await session.call_tool("repl", {"code": "1 + 1"})

    (env_file,) = (tmp_path / "fake").glob("env-*.json")
    env = json.loads(env_file.read_text())
    assert "AGENTENV_CONTROL_TOKEN" not in env and "AGENTENV_COMPUTER_SETTINGS" not in env
    assert env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin" and env["HOME"] == str(tmp_path)
    assert env["ASIDE_HOME"] == str(module.ASIDE_HOME) and env["ASIDE_CLI_UPDATE_CHECK"] == "0"


@pytest.mark.asyncio
async def test_readiness_checks_the_pinned_executables_the_daemon_turns_computer_use_on_and_checks_the_server(
        module, monkeypatch, tmp_path):
    driver = _driver(module, monkeypatch)
    manifest = module._manifest()
    (tmp_path / "fake" / "on").unlink()

    async def checks(served: dict | None = None):
        if served is not None:
            (tmp_path / "fake" / "tools.json").write_text(json.dumps(served))
        driver._pinned = None
        return await driver.ready_checks()

    ready = await driver.ready_checks()
    again = await driver.ready_checks()
    described = (await checks({"instructions": "other", "tools": [
        {**t, "description": "other"} for t in manifest["tools"]]}))[4]
    extra = (await checks({**manifest, "tools": [*manifest["tools"], {**manifest["tools"][0], "name": "exec"}]}))[4]
    monkeypatch.setattr(module, "_health", lambda: {"ready": False})
    unready = await checks(manifest)
    (module.PREFIX / module.HELPER / "Contents" / "MacOS" / "aside-computer-use").write_text("another build")
    (changed, *rest) = await checks()

    assert [(c.name, c.ok) for c in ready] == [("Aside Computer Use", True), ("Aside daemon", True),
                                               ("Analytics off", True), ("Computer Use on", True),
                                               ("Aside tools", True),
                                               ("Aside answers", True), ("macOS permissions", True)]
    assert (ready[2].detail, ready[3].detail) == ("Analytics are off.", "Computer Use is on.") and again == ready
    assert (tmp_path / "fake" / "enabled").read_text().count("enabled") == 3
    assert (described.ok, described.detail) == (False, "these differ from mcp.json: instructions, repl")
    assert (extra.ok, extra.detail) == (False, "these differ from mcp.json: exec")
    assert [(c.name, c.ok) for c in unready] == [("Aside Computer Use", True), ("Aside daemon", False),
                                                 ("macOS permissions", True)]
    assert not changed.ok and "Aside Computer Use.app/Contents/MacOS/aside-computer-use" in changed.detail
    assert [c.name for c in rest] == ["macOS permissions"]


def test_env_toml_install_sh_and_the_driver_agree_on_the_pinned_version_folder_helper_daemon_and_home(pinned):
    """A move to other Aside releases, or of their folder, helper, daemon or home, has to change them together; the
    helper's runtime copy, which install.sh places in the home, is granted as the helper is."""
    folder = DRIVER.parent
    macos = tomllib.loads((folder / "env.toml").read_text())["install"]["macos"]
    install = (folder / "install.sh").read_text()
    runtime = pinned.ASIDE_HOME / "runtime-computer-use" / pinned.HELPER

    daemon_download, cli_download = macos["downloads"]
    assert daemon_download["url"] == ("https://releases.aside.com/dev-updater/components/"
                                      f"{pinned.VERSION}/AsideDaemon/mac-arm64/AsideDaemon-{pinned.VERSION}.crx")
    assert cli_download["url"] == f"https://releases.aside.com/cli/AsideCLI-darwin-arm64-{pinned.CLI_VERSION}.zip"
    assert all(len(download["sha256"]) == 64 for download in macos["downloads"])
    assert f"daemon={daemon_download['to']}\n" in install and f"cli={cli_download['to']}\n" in install
    assert [grant["app"] for grant in macos["grants"] if "app" in grant] == [str(pinned.PREFIX / pinned.HELPER),
                                                                             str(runtime)]
    assert f"dest={pinned.PREFIX}\n" in install and f'helper="{pinned.HELPER}"\n' in install
    assert f'"$dest/{runtime.relative_to(pinned.PREFIX).parent}/$helper"' in install
    (daemon,) = [service for service in macos["services"] if service.get("direct")]
    assert shlex.split(daemon["command"]) == [str(pinned.PREFIX / pinned.DAEMON)]
    assert daemon["environment"] == {"ASIDE_HOME": str(pinned.ASIDE_HOME)}
    assert {str(pinned.DAEMON), f"{pinned.HELPER}/Contents/MacOS/aside-computer-use", str(pinned.CLI)} <= set(
        pinned.EXECUTABLES)
    assert all(len(digest) == 64 for digest in pinned.EXECUTABLES.values())


@pytest.mark.asyncio
async def test_past_the_most_repls_kept_the_least_recently_used_is_closed_and_the_others_keep_their_bindings(
        module, monkeypatch):
    monkeypatch.setattr(module, "MAX_REPLS", 2)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        async def repl(caller: str, code: str) -> str:
            return _text(await session.call_tool("repl", {"code": code}, meta={SESSION_META_KEY: caller}))

        for caller in "abc":
            await repl(caller, f"set {caller}")
        b, c, a = await repl("b", "get"), await repl("c", "get"), await repl("a", "get")

    assert a.startswith("None in ") and b.startswith("b in ") and c.startswith("c in ")


@pytest.mark.asyncio
async def test_a_repl_left_idle_is_closed_without_another_call_and_its_session_starts_afresh(module, monkeypatch):
    monkeypatch.setattr(module, "SWEEP_S", 0.05)
    monkeypatch.setattr(module, "IDLE_S", 0.2)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        await session.call_tool("repl", {"code": "set 1"}, meta={SESSION_META_KEY: "a"})
        before = _text(await session.call_tool("repl", {"code": "get"}, meta={SESSION_META_KEY: "a"}))
        closed = await _gone(int(before.rpartition(" in ")[2]))
        after = _text(await session.call_tool("repl", {"code": "get"}, meta={SESSION_META_KEY: "a"}))

    assert before.startswith("1 in ") and closed and after.startswith("None in ")


@pytest.mark.asyncio
async def test_calls_a_session_makes_together_while_another_repl_closes_share_one_repl(module, monkeypatch):
    monkeypatch.setattr(module, "MAX_REPLS", 1)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        async def repl(caller: str, code: str) -> str:
            return _text(await session.call_tool("repl", {"code": code}, meta={SESSION_META_KEY: caller}))

        await repl("a", "set a")
        _, got = await asyncio.gather(repl("b", "set b"), repl("b", "get"))
        after = await repl("b", "get")

    assert got.partition(" in ")[2] == after.partition(" in ")[2] and after.startswith("b in ")
