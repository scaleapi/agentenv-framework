"""The agent-desktop driver against a fake agent-desktop: its instructions and tools are its server's as pinned, each call
reaches a server of its own, ``--headed`` when the driver's setting says, with none of the driver's environment, and
the screenshots it saves come back beside its answer; a server that fails answers an error; readiness checks the pinned executables and the tools the server lists; and its
skill is agent-desktop's own, with its references and its guide for macOS."""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest

import agent_env
from agentenv_protocol.computer import _macos

DRIVER = Path(agent_env.__file__).parent / "drivers" / "envs" / "agent-desktop" / "driver.py"
FAKE = textwrap.dedent('''\
    import json, sys
    from pathlib import Path
    state = Path.home() / "fake"
    with open(state / "argv.jsonl", "a") as out:
        out.write(json.dumps({"argv": sys.argv[1:], "env": sorted(__import__("os").environ)}) + "\\n")
    if sys.argv[1] == "skills":
        print(json.dumps(json.loads((state / "skills.json").read_text())[" ".join(sys.argv[2:])]))
        sys.exit(0)
    if (state / "die").exists():
        sys.exit(1)
    import anyio
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    manifest = json.loads((state / "tools.json").read_text())
    server = Server("agent-desktop", instructions=manifest["instructions"])

    @server.list_tools()
    async def tools():
        return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
                for t in manifest["tools"]]

    def screenshot(args):
        if not Path(args["output_path"]).is_dir():
            Path(args["output_path"]).write_bytes(b"\\x89PNG fake")
        return {"ok": True, "command": "screenshot", "data": {"format": "png", "path": args["output_path"]}}

    @server.call_tool(validate_input=False)
    async def call(name, arguments):
        if name == "desktop_screenshot":
            answer = screenshot(arguments)
        elif name == "desktop_batch":
            answer = {"ok": True, "command": "batch", "data": {"results": [
                screenshot(c["args"]) if c["command"] == "screenshot" else {"ok": True, "command": c["command"]}
                for c in json.loads(arguments["commands_json"])]}}
        else:
            answer = {"ok": True, "command": name, "data": arguments}
        return [types.TextContent(type="text", text=json.dumps(answer))]

    async def main():
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    anyio.run(main)
''')


class Granted:
    def missing(self) -> list[str]:
        return []


@pytest.fixture
def module(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENTENV_COMPUTER_SETTINGS", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENTENV_CONTROL_TOKEN", "a-deploys-secret")
    monkeypatch.setattr(_macos, "Quartz", Granted)
    spec = importlib.util.spec_from_file_location("agent_desktop_driver", DRIVER)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    (tmp_path / "fake").mkdir()
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "agent-desktop").write_text(f"#!{sys.executable}\n{FAKE}")
    (bin_ / "agent-desktop").chmod(0o755)
    (bin_ / "agent-desktop-macos-helper").write_text("helper")
    (tmp_path / "fake" / "tools.json").write_text(loaded.MANIFEST.read_text())
    monkeypatch.setattr(loaded, "BIN", bin_)
    monkeypatch.setattr(loaded, "EXECUTABLE", bin_ / "agent-desktop")
    return loaded


def _calls(home: Path) -> list[dict]:
    return [json.loads(line) for line in (home / "fake" / "argv.jsonl").read_text().splitlines()]


def test_its_instructions_and_tools_are_its_servers_at_the_pinned_version(module):
    driver = module.AgentDesktop()
    manifest = json.loads(module.MANIFEST.read_text())

    assert [t.tool.name for t in driver.forwarded_tools()] == [t["name"] for t in manifest["tools"]]
    assert len(manifest["tools"]) == 67 and "desktop_skills_get" in manifest["instructions"]
    assert driver.agent_desktop_instructions() == f"{manifest['instructions']}\n\n{module.FILES}"
    assert not driver.settings.headed


@pytest.mark.asyncio
async def test_a_call_reaches_a_server_of_its_own_headed_when_the_setting_says_with_none_of_the_drivers_environment(
        module, tmp_path):
    driver = module.AgentDesktop()
    headless = await driver._forward("desktop_snapshot", app="BenchLab", interactive_only=True)
    driver.settings = driver.Settings(headed=True)
    headed = await driver._forward("desktop_click", ref_id="@s1:e3")

    assert not headless.isError and json.loads(headless.content[0].text) == {
        "ok": True, "command": "desktop_snapshot", "data": {"app": "BenchLab", "interactive_only": True}}
    assert json.loads(headed.content[0].text)["data"] == {"ref_id": "@s1:e3"}
    (first, second) = _calls(tmp_path)
    assert first["argv"] == ["mcp"] and second["argv"] == ["mcp", "--headed"]
    assert "AGENTENV_CONTROL_TOKEN" not in first["env"] and "HOME" in first["env"]


@pytest.mark.asyncio
async def test_each_screenshot_a_call_saves_comes_back_beside_its_answer_up_to_the_cap_and_no_other_file_does(
        module, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    named, unreadable = tmp_path / "named.png", tmp_path / "folder.png"
    named.write_bytes(b"\x89PNG named")
    unreadable.mkdir()
    driver = module.AgentDesktop()

    saved = await driver._forward("desktop_screenshot", output_path=str(tmp_path / "a.png"))
    relative = await driver._forward("desktop_screenshot", output_path="rel.png")
    echoed = await driver._forward("desktop_find", name=str(named))
    unread = await driver._forward("desktop_screenshot", output_path=str(unreadable))
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", 2 * len(b"\x89PNG fake"))
    shots = [{"command": "screenshot", "args": {"output_path": str(tmp_path / f"b{n}.png")}} for n in range(3)]
    batch = await driver._forward("desktop_batch", commands_json=json.dumps(
        [*shots, {"command": "trace export", "args": {"out": str(tmp_path / "trace.html")}}, shots[0]]))

    text, image = saved.content
    assert json.loads(text.text)["data"]["path"] == str(tmp_path / "a.png")
    assert (image.type, image.mimeType, base64.b64decode(image.data)) == ("image", "image/png", b"\x89PNG fake")
    assert [block.type for block in relative.content] == ["text", "image"]
    assert [block.type for block in echoed.content] == ["text"]
    assert [block.type for block in unread.content] == ["text", "text"]
    assert unread.content[-1].text == f"Saved but not readable: {unreadable}"
    assert [block.type for block in batch.content] == ["text", "image", "image", "text"]
    assert batch.content[-1].text == f"Saved but not attached, past the most one answer carries: {tmp_path / 'b2.png'}"


@pytest.mark.asyncio
async def test_a_server_that_fails_before_it_answers_is_an_error_saying_the_command_may_have_run(module, tmp_path):
    (tmp_path / "fake" / "die").write_text("")

    result = await module.AgentDesktop()._forward("desktop_click", ref_id="@s1:e3")

    assert result.isError and "so desktop_click may have run" in result.content[0].text


@pytest.mark.asyncio
async def test_readiness_checks_the_pinned_executables_then_the_tools_its_server_lists(module, tmp_path, monkeypatch):
    driver = module.AgentDesktop()
    (wrong, *_) = await driver.ready_checks()
    monkeypatch.setattr(module, "EXECUTABLES", {name: hashlib.sha256((tmp_path / "bin" / name).read_bytes()).hexdigest()
                                                for name in module.EXECUTABLES})
    ready = await driver.ready_checks()
    manifest = json.loads(module.MANIFEST.read_text())
    manifest["tools"][0]["inputSchema"] = {"type": "object", "properties": {}}
    (tmp_path / "fake" / "tools.json").write_text(json.dumps(manifest))
    driver._pinned = None
    (_, changed, _) = await driver.ready_checks()

    assert not wrong.ok and "agent-desktop, agent-desktop-macos-helper" in wrong.detail
    assert [(check.name, check.ok) for check in ready] == [("agent-desktop", True), ("agent-desktop tools", True),
                                                           ("macOS permissions", True)]
    assert not changed.ok and changed.detail == "these tools differ from mcp.json: desktop_snapshot"


@pytest.mark.asyncio
async def test_its_skill_is_agent_desktops_own_with_its_references_and_its_guide_for_this_os(module, tmp_path):
    core = "---\nname: agent-desktop\ndescription: >\n  Operate desktop apps\n  through accessibility trees.\n---\n\n# Loop\n"
    answers = {
        "list": {"skills": [{"name": "agent-desktop", "references": ["references/workflows.md"]},
                            {"name": "agent-desktop-macos", "current": True, "references": ["references/tcc.md"]},
                            {"name": "agent-desktop-linux", "current": False, "references": ["references/x11.md"]}]},
        "get desktop": {"content": core},
        "get desktop references/workflows.md": {"content": "# Workflows"},
        "get platform": {"content": "# macOS"},
        "get platform references/tcc.md": {"content": "# TCC"},
    }
    (tmp_path / "fake" / "skills.json").write_text(json.dumps({key: {"ok": True, "data": data}
                                                               for key, data in answers.items()}))

    description, body, files = await module.AgentDesktop().bundled_skill()

    assert description == "Operate desktop apps through accessibility trees." and body == "# Loop\n"
    assert files == {"references/workflows.md": "# Workflows", "platform/SKILL.md": "# macOS",
                     "platform/references/tcc.md": "# TCC"}
