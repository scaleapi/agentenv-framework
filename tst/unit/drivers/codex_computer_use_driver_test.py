"""The codex-computer-use driver against a fake cua_repl: agents get its js and js_reset as pinned, with its
instructions, and not the tools it serves only the ChatGPT app, a choice of objects as one; each call carries Codex's
read-only sandbox on macOS unless the setting turns it off, and on Windows none unless it turns it on; Codex's approval requests are answered from allowed_apps by name or bundle identifier, from the
request's metadata or its message, every app approved when it names none, and nothing else; each agent session keeps a
REPL of its own across calls, one that stops starting afresh; cua_repl gets the app's environment and none of the
driver's, with Windows paths on Windows; readiness checks the pinned executables, what cua_repl says and lists and
that it answers; the MSIX unpacks with its names decoded; and the driver's files agree on the pinned versions."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import textwrap
import tomllib
import types
import zipfile
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

import agent_env
from agentenv_protocol import SESSION_META_KEY
from agentenv_protocol.computer import Platform, _macos

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "codex-computer-use" / "driver.py"
FAKE = textwrap.dedent('''\
    import json, os
    from pathlib import Path
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp.shared.message import ServerMessageMetadata
    state = Path(os.environ["HOME"]) / "fake"
    (state / f"env-{os.getpid()}.json").write_text(json.dumps(dict(os.environ)))
    manifest = json.loads((state / "tools.json").read_text())
    server = Server("cua_repl", instructions=manifest["instructions"])
    bindings = {}

    @server.list_tools()
    async def tools():
        return [types.Tool(**t) for t in manifest["tools"]]

    async def ask(message, tool=None, app=None, shown=None):
        context = server.request_context
        meta = types.RequestParams.Meta(**({} if tool is None else {
            "codex_approval_kind": "mcp_tool_call", "connector_id": "computer-use", "tool_name": tool,
            "tool_params": {"app": app} if app else {},
            "tool_params_display": [{"display_name": "App", "name": "app", "value": shown}] if shown else []}))
        result = await context.session.send_request(
            types.ServerRequest(types.ElicitRequest(params=types.ElicitRequestFormParams(
                message=message, requestedSchema={"type": "object", "properties": {}}, _meta=meta))),
            types.ElicitResult, metadata=ServerMessageMetadata(related_request_id=context.request_id))
        return result.action

    @server.call_tool(validate_input=False)
    async def call(name, arguments):
        context = server.request_context
        with open(state / "calls.jsonl", "a") as out:
            out.write(json.dumps({"pid": os.getpid(), "name": name, "arguments": arguments,
                                  "meta": context.meta.model_dump() if context.meta else {}}) + "\\n")
        code = arguments.get("code", "")
        if name == "js_reset":
            bindings.clear()
            text = "reset"
        elif code.startswith("app "):
            _, shown, bundle = code.split(" ")
            text = await ask(f'Allow Computer Use to use "{shown}"?', "get_app_state", bundle, shown)
        elif code.startswith("reworded "):
            _, shown, bundle = code.split(" ")
            text = await ask(f"Codex would like to control {shown}.", "get_app_state", bundle, shown)
        elif code.startswith("bare "):
            text = await ask(f'Allow Computer Use to use "{code[5:]}"?')
        elif code == "audio":
            text = await ask("Allow Computer Use to record computer audio?", "start_audio_recording")
        elif code.startswith("set "):
            bindings["x"] = code[4:]
            text = "set"
        elif code == "get":
            text = f"{bindings.get('x')} in {os.getpid()}"
        elif code == "die":
            os._exit(1)
        elif "cua.getState" in code:
            text = "ready" if not (state / "unready").exists() else "no apps"
        else:
            text = "ran"
        return [types.TextContent(type="text", text=text)]

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
''')
MACOS = Platform(os="macos", version="26.5", arch="arm64", display="aqua")
WINDOWS = Platform(os="windows", version="11", arch="amd64", display="dwm")


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
    spec = importlib.util.spec_from_file_location("codex_computer_use_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


@pytest.fixture
def module(pinned, monkeypatch, tmp_path):
    """The driver's module on a fake ChatGPT app whose Node is the fake cua_repl."""
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "tools.json").write_text(pinned.MANIFEST.read_text())
    helper = tmp_path / "Codex Computer Use.app"
    monkeypatch.setattr(pinned, "OS", pinned.OS._replace(app=tmp_path / "ChatGPT.app"))
    monkeypatch.setattr(pinned, "HELPER", helper)
    node = pinned._resources() / pinned.OS.node
    node.parent.mkdir(parents=True)
    node.write_text(f"#!{sys.executable}\n{FAKE}")
    node.chmod(0o755)
    digests = {}
    for path in pinned.OS.executables:
        root, _, rest = path.partition("/")
        file = helper / rest if root == helper.name else tmp_path / path
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(f"the pinned {file.name}")
        digests[path] = hashlib.sha256(file.read_bytes()).hexdigest()
    monkeypatch.setattr(pinned, "OS", pinned.OS._replace(executables=digests))
    return pinned


@pytest.fixture
def windows(monkeypatch, tmp_path):
    """The driver's module as it loads on Windows, with WindowsDriver under it."""
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("AGENTENV_CONTROL_TOKEN", "a-deploys-secret")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr("agentenv_protocol.computer.driver.detect", lambda: WINDOWS)
    monkeypatch.setitem(sys.modules, "mss", types.ModuleType("mss"))
    monkeypatch.delitem(sys.modules, "agentenv_protocol.computer._windows", raising=False)
    spec = importlib.util.spec_from_file_location("codex_computer_use_driver_windows", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    monkeypatch.setattr(sys, "platform", "linux")
    return loaded


def _driver(module, monkeypatch, **settings):
    monkeypatch.setenv("AGENTENV_COMPUTER_SETTINGS", json.dumps(settings))
    return module.CodexComputerUse()


@contextlib.asynccontextmanager
async def mcp_session(driver):
    async with create_connected_server_and_client_session(driver.mcp or driver.create_app()) as session:
        yield session


def _calls(home: Path) -> list[dict]:
    path = home / "fake" / "calls.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


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
async def test_agents_get_codexs_js_and_js_reset_as_pinned_with_its_instructions_and_not_the_apps_own_tools(
        module, monkeypatch):
    driver = _driver(module, monkeypatch)
    async with mcp_session(driver) as session:
        listed = {t.name: t for t in (await session.list_tools()).tools}

    pinned = {t["name"]: t for t in module._manifest()["tools"]}
    assert {"js", "js_reset"} <= set(listed) and not {"js_add_node_module_dir", "turn_ended"} & set(listed)
    assert listed["js"].inputSchema == pinned["js"]["inputSchema"]
    assert listed["js"].annotations.readOnlyHint is False and listed["js"].annotations.destructiveHint is True
    assert listed["js_reset"].annotations.readOnlyHint is True
    assert listed["js"].description == pinned["js"]["description"]
    assert driver.mcp.instructions.startswith(module._manifest()["instructions"])


@pytest.mark.asyncio
async def test_each_call_runs_in_codexs_read_only_sandbox_unless_the_setting_turns_it_off(
        module, monkeypatch, tmp_path):
    async with mcp_session(_driver(module, monkeypatch)) as session:
        await session.call_tool("js", {"code": "1 + 1"})
    async with mcp_session(_driver(module, monkeypatch, js_sandbox=False)) as session:
        await session.call_tool("js", {"code": "1 + 1"})

    sandboxed, open_ = (call["meta"]["codex/sandbox-state-meta"] for call in _calls(tmp_path))
    profile = sandboxed["permissionProfile"]
    entries = {json.dumps(e["path"], sort_keys=True): e["access"] for e in profile["file_system"]["entries"]}
    assert profile["type"] == "managed" and profile["network"] == "restricted"
    assert entries == {'{"type": "special", "value": {"kind": "root"}}': "read",
                       '{"type": "special", "value": {"kind": "tmpdir"}}': "write",
                       '{"type": "special", "value": {"kind": "slash_tmp"}}': "write",
                       json.dumps({"path": str(tmp_path / ".codex"), "type": "path"}, sort_keys=True): "write"}
    assert sandboxed["sandboxCwd"] == (tmp_path / ".codex" / "agentenv").as_uri()
    assert open_["permissionProfile"] == {"type": "disabled"}


@pytest.mark.asyncio
async def test_approvals_follow_allowed_apps_by_name_or_bundle_identifier_and_nothing_else_is_approved(
        module, monkeypatch, caplog):
    async with mcp_session(_driver(module, monkeypatch, allowed_apps=["calculator", "com.apple.TextEdit"])) as session:
        async def js(code: str) -> str:
            return _text(await session.call_tool("js", {"code": code}))

        named, by_id = await js("app Calculator com.apple.calculator"), await js("app TextEdit com.apple.TextEdit")
        other, audio = await js("app Finder com.apple.finder"), await js("audio")
        reworded, bare = await js("reworded Calculator com.apple.calc"), await js("bare Calculator")
    async with mcp_session(_driver(module, monkeypatch)) as session:
        any_app = _text(await session.call_tool("js", {"code": "app Finder com.apple.finder"}))
        no_audio = _text(await session.call_tool("js", {"code": "audio"}))

    assert [named, by_id, other, audio] == ["accept", "accept", "decline", "decline"]
    assert [reworded, bare] == ["accept", "accept"]
    assert [any_app, no_audio] == ["accept", "decline"]
    warned = [r.getMessage() for r in caplog.records if r.name == module.logger.name and r.levelname == "WARNING"]
    assert len(warned) == 2 and all("record computer audio" in message for message in warned)


@pytest.mark.asyncio
async def test_each_agent_session_keeps_a_repl_of_its_own_and_one_that_stops_starts_afresh_while_it_still_ends(
        module, monkeypatch, tmp_path):
    """A server that stops takes a while to end; the session's next call starts a new one rather than wait on it."""
    serve = module.CodexComputerUse._server

    @contextlib.asynccontextmanager
    async def slow_to_end(self):
        async with serve(self) as opened:
            try:
                yield opened
            finally:
                await asyncio.sleep(0.5)

    monkeypatch.setattr(module.CodexComputerUse, "_server", slow_to_end)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        for caller, value in (("a", "1"), ("b", "2")):
            await session.call_tool("js", {"code": f"set {value}"}, meta={SESSION_META_KEY: caller})
        a = _text(await session.call_tool("js", {"code": "get"}, meta={SESSION_META_KEY: "a"}))
        b = _text(await session.call_tool("js", {"code": "get"}, meta={SESSION_META_KEY: "b"}))
        died = await session.call_tool("js", {"code": "die"}, meta={SESSION_META_KEY: "a"})
        after = _text(await session.call_tool("js", {"code": "get"}, meta={SESSION_META_KEY: "a"}))

    (a_value, _, a_pid), (b_value, _, b_pid) = a.partition(" in "), b.partition(" in ")
    assert (a_value, b_value) == ("1", "2") and a_pid != b_pid
    assert died.isError and "may have run" in _text(died)
    assert after.startswith("None in ") and after != f"None in {a_pid}"


@pytest.mark.asyncio
async def test_cua_repl_gets_the_apps_environment_and_none_of_the_drivers(module, monkeypatch, tmp_path):
    async with mcp_session(_driver(module, monkeypatch)) as session:
        await session.call_tool("js", {"code": "1 + 1"})

    (env_file,) = (tmp_path / "fake").glob("env-*.json")
    env = json.loads(env_file.read_text())
    modules = str(module._resources() / module.OS.node_modules)
    assert "AGENTENV_CONTROL_TOKEN" not in env and env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert env["CUA_REPL_ENABLED_SURFACES"] == "computer" and env["SKY_CUA_SERVICE_PATH"] == str(module.HELPER)
    assert env["NODE_REPL_TRUSTED_SERVICES"] == '{"sky": "@oai/sky/service"}'
    assert env["NODE_REPL_TRUSTED_CODE_PATHS"] == f"{tmp_path / '.codex'}:{modules}"
    assert env["CODEX_HOME"] == str(tmp_path / ".codex")
    assert env["NODE_REPL_NODE_PATH"] == str(module._resources() / module.OS.node)


@pytest.mark.asyncio
async def test_readiness_checks_the_pinned_executables_what_cua_repl_says_and_lists_and_that_it_answers(
        module, monkeypatch, tmp_path):
    driver = _driver(module, monkeypatch)
    manifest = module._manifest()

    async def tools_check(served: dict):
        (tmp_path / "fake" / "tools.json").write_text(json.dumps(served))
        driver._pinned = None
        return (await driver.ready_checks())[1]

    ready = await driver.ready_checks()
    fewer = await tools_check({**manifest, "tools": manifest["tools"][:3]})
    described = await tools_check({"instructions": "other", "tools": [
        {**t, "description": "other"} if t["name"] == "js" else t for t in manifest["tools"]]})
    hinted = await tools_check({**manifest, "tools": [
        {**t, "annotations": {}} if t["name"] == "js_reset" else t for t in manifest["tools"]]})
    (module.HELPER / "Contents" / "MacOS" / "SkyComputerUseService").write_text("another build")
    driver._pinned = None
    (changed, *_) = await driver.ready_checks()

    assert [(c.name, c.ok) for c in ready] == [("Codex Computer Use", True), ("Codex tools", True),
                                               ("Codex answers", True), ("macOS permissions", True)]
    assert (fewer.ok, fewer.detail) == (False, "these differ from mcp.json: turn_ended")
    assert (described.ok, described.detail) == (False, "these differ from mcp.json: instructions, js")
    assert (hinted.ok, hinted.detail) == (False, "these differ from mcp.json: js_reset")
    assert not changed.ok and "Codex Computer Use.app/Contents/MacOS/SkyComputerUseService" in changed.detail


def test_env_toml_install_sh_and_the_driver_agree_on_the_pinned_version_and_the_helpers_path(pinned):
    """A move to another ChatGPT version, or the helper to another folder, has to change them together."""
    folder = DRIVER.parent
    macos = tomllib.loads((folder / "env.toml").read_text())["install"]["macos"]
    install = (folder / "install.sh").read_text()

    (download,) = macos["downloads"]
    assert download["url"].endswith(f"/ChatGPT-darwin-arm64-{pinned.VERSION}.zip")
    assert [grant["app"] for grant in macos["grants"] if "app" in grant] == [str(pinned.HELPER)]
    assert f'helper="{pinned.HELPER.name}"\n' in install and f"dest={pinned.HELPER.parent}\n" in install
    assert pinned.OS.app == folder / "ChatGPT.app" and all(
        path.startswith(("ChatGPT.app/", f"{pinned.HELPER.name}/")) for path in pinned.OS.executables)


@pytest.mark.asyncio
async def test_past_the_most_repls_kept_the_least_recently_used_is_closed_and_the_others_keep_their_bindings(
        module, monkeypatch):
    monkeypatch.setattr(module, "MAX_REPLS", 2)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        async def js(caller: str, code: str) -> str:
            return _text(await session.call_tool("js", {"code": code}, meta={SESSION_META_KEY: caller}))

        for caller in "abc":
            await js(caller, f"set {caller}")
        b, c, a = await js("b", "get"), await js("c", "get"), await js("a", "get")

    assert a.startswith("None in ") and b.startswith("b in ") and c.startswith("c in ")


@pytest.mark.asyncio
async def test_a_repl_left_idle_is_closed_without_another_call_and_its_session_starts_afresh(module, monkeypatch):
    monkeypatch.setattr(module, "SWEEP_S", 0.05)
    monkeypatch.setattr(module, "IDLE_S", 0.2)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        await session.call_tool("js", {"code": "set 1"}, meta={SESSION_META_KEY: "a"})
        before = _text(await session.call_tool("js", {"code": "get"}, meta={SESSION_META_KEY: "a"}))
        closed = await _gone(int(before.rpartition(" in ")[2]))
        after = _text(await session.call_tool("js", {"code": "get"}, meta={SESSION_META_KEY: "a"}))

    assert before.startswith("1 in ") and closed and after.startswith("None in ")


@pytest.mark.asyncio
async def test_calls_a_session_makes_together_while_another_repl_closes_share_one_repl(
        module, monkeypatch, tmp_path):
    monkeypatch.setattr(module, "MAX_REPLS", 1)
    async with mcp_session(_driver(module, monkeypatch)) as session:
        async def js(caller: str, code: str) -> str:
            return _text(await session.call_tool("js", {"code": code}, meta={SESSION_META_KEY: caller}))

        await js("a", "set a")
        await asyncio.gather(js("b", "set b"), js("b", "get"))
        after = await js("b", "get")

    pids = {call["pid"] for call in _calls(tmp_path) if call["arguments"]["code"] in ("set b", "get")}
    assert pids == {int(after.rpartition(" in ")[2])} and after.startswith("b in ")


def test_on_windows_cua_repl_gets_the_apps_environment_with_windows_paths_and_none_of_the_drivers(windows, tmp_path):
    env = windows._environment()
    resources = DRIVER.parent / "ChatGPT" / "resources"
    modules = resources / "cua_node" / "bin" / "node_modules"

    assert windows.OS_NAME == "windows" and windows.OSDriver.__name__ == "WindowsDriver"
    assert env["PATH"] == (r"C:\Windows\System32;C:\Windows;C:\Windows\System32\Wbem;"
                           r"C:\Windows\System32\WindowsPowerShell\v1.0")
    assert env["USERPROFILE"] == str(tmp_path) and env["SystemRoot"] == r"C:\Windows"
    assert "AGENTENV_CONTROL_TOKEN" not in env and "HOME" not in env
    assert env["NODE_REPL_NODE_PATH"] == str(resources / "cua_node" / "bin" / "node.exe")
    assert env["CUA_REPL_NODE_REPL_PATH"] == str(resources / "cua_node" / "bin" / "node_repl.exe")
    assert env["SKY_CUA_SERVICE_PATH"] == str(modules / "@oai" / "sky" / "bin" / "windows" / "codex-computer-use.exe")
    assert env["CODEX_CLI_PATH"] == str(resources / "codex.exe")
    assert env["NODE_REPL_TRUSTED_CODE_PATHS"].endswith(f";{modules}")
    assert env["NODE_REPL_INSTRUCTIONS_USE_CASE_COMPUTER_USE"].endswith("on Windows through Computer Use.")


def test_a_tools_choice_of_objects_reaches_agents_as_one_object_requiring_what_every_choice_requires(pinned):
    code, timeout, title = {"type": "string"}, {"type": "integer"}, {"type": "string", "description": "a title"}
    choices = {"description": "Run JavaScript.", "oneOf": [
        {"type": "object", "properties": {"code": code, "timeout_ms": timeout}, "required": ["code"]},
        {"type": "object", "properties": {"code": code, "title": title}, "required": ["code", "title"]}]}
    one = {"type": "object", "properties": {"code": code}, "required": ["code"]}

    assert pinned._one_object(choices) == {"description": "Run JavaScript.", "type": "object", "required": ["code"],
                                           "properties": {"code": code, "timeout_ms": timeout, "title": title}}
    assert pinned._one_object(one) is one


def test_the_msix_unpacks_into_the_apps_resources_with_their_names_decoded_and_without_its_electron_code(tmp_path):
    """An MSIX names its entries as URLs do, '@' as %40, which Windows decodes when it installs one."""
    msix, out = tmp_path / "ChatGPT.msix", tmp_path / "ChatGPT" / "resources"
    with zipfile.ZipFile(msix, "w") as package:
        package.writestr("app/resources/cua_node/bin/node_modules/%40oai/cua-repl/bin/cua-repl.mjs", "the launcher")
        package.writestr("app/resources/cua_node/bin/node_modules/%40img/a%2Bb%24.js", "a module")
        package.writestr("app/resources/codex.exe", "codex")
        package.writestr("app/resources/app.asar", "the app's own code")
        package.writestr("app/ChatGPT.exe", "the app")
        package.writestr("AppxManifest.xml", "the package's manifest")

    subprocess.run([sys.executable, str(DRIVER.parent / "unpack-msix.py"), str(msix), str(out)], check=True)

    assert sorted(path.relative_to(out).as_posix() for path in out.rglob("*") if path.is_file()) == [
        "codex.exe", "cua_node/bin/node_modules/@img/a+b$.js", "cua_node/bin/node_modules/@oai/cua-repl/bin/cua-repl.mjs"]
    assert (out / "cua_node/bin/node_modules/@oai/cua-repl/bin/cua-repl.mjs").read_text() == "the launcher"


def test_env_toml_the_windows_install_and_the_driver_agree_on_the_pinned_msix_and_where_it_unpacks(pinned):
    """OpenAI serves only its latest MSIX, so the pin names that link and a new build moves the sha256 and the version."""
    folder, win = DRIVER.parent, pinned.OSES["windows"]
    windows = tomllib.loads((folder / "env.toml").read_text())["install"]["windows"]
    install = (folder / "install-windows.ps1").read_text()
    unpacked = f"{win.app.name}/{win.resources}"

    (download,) = windows["downloads"]
    assert (download["url"], download["to"], download["arch"]) == (
        "https://persistent.oaistatic.com/codex-app-prod/ChatGPT-x64.msix", "ChatGPT.msix", "amd64")
    assert len(download["sha256"]) == 64 and windows["run"] == [{"script": "install-windows.ps1"}]
    assert ('"$env:PREFIX\\ChatGPT.msix" "$env:PREFIX\\' + unpacked.replace("/", "\\") + '"') in install
    assert win.app == folder / "ChatGPT" and all(path.startswith(f"{unpacked}/") for path in win.executables)
    assert {f"{unpacked}/{path}" for path in (win.node, win.node_repl, win.service)} <= set(win.executables)


def test_on_windows_js_runs_unsandboxed_unless_the_setting_turns_codexs_sandbox_on_and_reaches_agents_as_one_object(
        windows, monkeypatch):
    """Codex's sandbox on Windows refuses to start Codex Computer Use's own service (spawn EPERM)."""
    default, sandboxed = _driver(windows, monkeypatch), _driver(windows, monkeypatch, js_sandbox=True)
    tools = {t.tool.name: t.tool for t in default.forwarded_tools()}
    pinned = {t["name"]: t for t in windows._manifest()["tools"]}

    assert default._sandbox()["codex/sandbox-state-meta"]["permissionProfile"] == {"type": "disabled"}
    assert sandboxed._sandbox()["codex/sandbox-state-meta"]["permissionProfile"]["type"] == "managed"
    assert "oneOf" in pinned["js"]["inputSchema"] and "oneOf" not in tools["js"].inputSchema
    assert {"code", "completion_summary"} <= set(tools["js"].inputSchema["properties"])
    assert tools["js"].inputSchema["required"] == [] and set(tools) == {"js", "js_reset"}
    assert "listWindows" in windows.OS.probe


def test_a_windows_approval_for_an_app_run_from_a_renamed_python_is_matched_by_the_name_it_shows(pinned):
    """BenchLab and the bench's sentinel both run as copies of pythonw.exe, which the request names; only the name it
    shows tells them apart."""
    meta = {"codex_approval_kind": "mcp_tool_call", "connector_id": "computer-use",
            "tool_params": {"app": "pythonw.exe"},
            "tool_params_display": [{"display_name": "App", "name": "app", "value": "benchlab"}]}

    names = pinned._app_names("Allow Codex to use benchlab?", meta)
    assert names == {"pythonw.exe", "benchlab"}
    assert names & {"benchlab"} and not names & {"benchsentinel"}
