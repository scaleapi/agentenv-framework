"""The codex-computer-use driver against a fake cua_repl: agents get its js and js_reset as pinned, with its
instructions, and not the tools it serves only the ChatGPT app; each call carries Codex's read-only sandbox unless the
setting turns it off; Codex's approval requests are answered from allowed_apps by name or bundle identifier, every app
approved when it names none, and nothing else; each agent session keeps a REPL of its own across calls, one that stops
starting afresh; cua_repl gets the app's environment and none of the driver's; and readiness checks the pinned
executables, the tools cua_repl lists and that it answers."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import textwrap
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
        return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                for t in manifest["tools"]]

    async def ask(message, app):
        context = server.request_context
        meta = types.RequestParams.Meta(**({"tool_params": {"app": app}} if app else {}))
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
            _, display, bundle = code.split(" ")
            text = await ask(f'Allow Computer Use to use "{display}"?', bundle)
        elif code == "audio":
            text = await ask("Allow Computer Use to record computer audio?", None)
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


class Granted:
    def missing(self) -> list[str]:
        return []


@pytest.fixture
def module(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTENV_CONTROL_TOKEN", "a-deploys-secret")
    monkeypatch.setattr(_macos, "Quartz", Granted)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr("agentenv_protocol.computer.driver.detect", lambda: MACOS)
    spec = importlib.util.spec_from_file_location("codex_computer_use_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    (tmp_path / "fake").mkdir()
    (tmp_path / "fake" / "tools.json").write_text(loaded.MANIFEST.read_text())
    app, helper = tmp_path / "ChatGPT.app", tmp_path / "Codex Computer Use.app"
    node = app / "Contents" / "Resources" / "cua_node" / "bin" / "node"
    node.parent.mkdir(parents=True)
    node.write_text(f"#!{sys.executable}\n{FAKE}")
    node.chmod(0o755)
    pinned = {}
    for path in loaded.EXECUTABLES:
        root, _, rest = path.partition("/")
        file = (app if root == "ChatGPT.app" else helper) / rest
        if not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text(f"the pinned {file.name}")
        pinned[path] = hashlib.sha256(file.read_bytes()).hexdigest()
    monkeypatch.setattr(loaded, "APP", app)
    monkeypatch.setattr(loaded, "HELPER", helper)
    monkeypatch.setattr(loaded, "NODE", node)
    monkeypatch.setattr(loaded, "EXECUTABLES", pinned)
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
        module, monkeypatch):
    async with mcp_session(_driver(module, monkeypatch, allowed_apps=["calculator", "com.apple.TextEdit"])) as session:
        named = await session.call_tool("js", {"code": "app Calculator com.apple.calculator"})
        by_id = await session.call_tool("js", {"code": "app TextEdit com.apple.TextEdit"})
        other = await session.call_tool("js", {"code": "app Finder com.apple.finder"})
        audio = await session.call_tool("js", {"code": "audio"})
    async with mcp_session(_driver(module, monkeypatch)) as session:
        any_app = await session.call_tool("js", {"code": "app Finder com.apple.finder"})
        no_audio = await session.call_tool("js", {"code": "audio"})

    assert [_text(r) for r in (named, by_id, other, audio)] == ["accept", "accept", "decline", "decline"]
    assert [_text(r) for r in (any_app, no_audio)] == ["accept", "decline"]


@pytest.mark.asyncio
async def test_each_agent_session_keeps_a_repl_of_its_own_and_one_that_stops_starts_afresh(
        module, monkeypatch, tmp_path):
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
    modules = str(module.NODE_MODULES)
    assert "AGENTENV_CONTROL_TOKEN" not in env and env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert env["CUA_REPL_ENABLED_SURFACES"] == "computer" and env["SKY_CUA_SERVICE_PATH"] == str(module.HELPER)
    assert env["NODE_REPL_TRUSTED_SERVICES"] == '{"sky": "@oai/sky/service"}'
    assert env["NODE_REPL_TRUSTED_CODE_PATHS"] == f"{tmp_path / '.codex'}:{modules}"
    assert env["CODEX_HOME"] == str(tmp_path / ".codex") and env["NODE_REPL_NODE_PATH"] == str(module.NODE)


@pytest.mark.asyncio
async def test_readiness_checks_the_pinned_executables_the_tools_cua_repl_lists_and_that_it_answers(
        module, monkeypatch, tmp_path):
    driver = _driver(module, monkeypatch)
    ready = await driver.ready_checks()
    three = {**module._manifest(), "tools": module._manifest()["tools"][:3]}
    (tmp_path / "fake" / "tools.json").write_text(json.dumps(three))
    driver._pinned = None
    tools = (await driver.ready_checks())[1]
    (module.HELPER / "Contents" / "MacOS" / "SkyComputerUseService").write_text("another build")
    driver._pinned = None
    (changed, *_) = await driver.ready_checks()

    assert [(c.name, c.ok) for c in ready] == [("Codex Computer Use", True), ("Codex tools", True),
                                               ("Codex answers", True), ("macOS permissions", True)]
    assert not tools.ok and tools.detail == "these tools differ from mcp.json: turn_ended"
    assert not changed.ok and "Codex Computer Use.app/Contents/MacOS/SkyComputerUseService" in changed.detail


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
