"""A computer deployed with a tool prefix serves every tool under it, and names its tools that way wherever it names
them: the card, tools/list, its skill, its extensions' descriptions and its errors. Without one, nothing changes."""
from __future__ import annotations

import re

import pytest

pytest.importorskip("mcp.server.fastmcp")
pytest.importorskip("PIL")

from computer_fake import LINUX, FakeDriver, http_client, mcp_session  # noqa: E402
from mcp.types import CallToolResult, TextContent  # noqa: E402

from agentenv_protocol import EnvironmentTool, ForwardedTool, server_instructions  # noqa: E402
from agentenv_protocol.computer import TOOL_PREFIX_ENV  # noqa: E402

PREFIX = "desktop_"
UNPREFIXED = re.compile(r"(?<![\w-])computer_")


async def _names(driver) -> tuple[set, set]:
    async with http_client(driver) as http, mcp_session(driver) as session:
        card = (await http.get("/.well-known/agent-env.json")).json()
        listed = {tool.name for tool in (await session.list_tools()).tools}
    return {tool["name"] for tool in card["capabilities"]["tools"]}, listed


@pytest.mark.asyncio
async def test_a_deploys_tool_prefix_names_every_tool_on_the_card_and_in_tools_list(monkeypatch):
    own_card, own_listed = await _names(FakeDriver())
    monkeypatch.setenv(TOOL_PREFIX_ENV, PREFIX)
    driver = FakeDriver()
    card, listed = await _names(driver)
    assert card == listed == {PREFIX + name for name in own_card} and own_card == own_listed
    async with mcp_session(driver) as session:
        shot = await session.call_tool(f"{PREFIX}computer_screenshot", {})
        unknown = await session.call_tool("computer_screenshot", {})
    assert not shot.isError and any(part.type == "image" for part in shot.content)
    assert unknown.isError


def test_the_declaration_a_driver_env_stores_names_the_tools_without_the_deploys_prefix(monkeypatch):
    monkeypatch.setenv(TOOL_PREFIX_ENV, PREFIX)
    names = {tool.name for tool in FakeDriver.declaration(LINUX).capabilities.tools}
    assert "computer_screenshot" in names and not any(name.startswith(PREFIX) for name in names)


@pytest.mark.asyncio
async def test_its_skill_its_extensions_and_its_notes_name_the_tools_as_it_serves_them(monkeypatch):
    monkeypatch.setenv(TOOL_PREFIX_ENV, PREFIX)
    monkeypatch.setattr(FakeDriver, "notes", "Take a computer_screenshot first; desktop_computer_click is already right.")
    async with http_client(FakeDriver()) as http:
        skill = (await http.get("/agentenv/ext/skill/get", params={"name": "desktop"})).json()
        card = (await http.get("/.well-known/agent-env.json")).json()
    text = skill["description"] + skill["skill_md"]
    assert f"`{PREFIX}computer_screenshot`" in text and f"`{PREFIX}computer_*`" in text and f"`{PREFIX}computer_click`" in text
    assert "Take a desktop_computer_screenshot first; desktop_computer_click is already right." in text
    assert not UNPREFIXED.search(text) and "desktop_desktop_" not in text and "serves its tools under" not in text
    descriptions = " ".join(e["description"] for e in card["capabilities"]["extensions"] if e.get("description"))
    assert f"The seven {PREFIX}computer_* tools" in descriptions and not UNPREFIXED.search(descriptions)
    tools = " ".join(t.get("description") or "" for t in card["capabilities"]["tools"])
    assert not UNPREFIXED.search(tools)


@pytest.mark.asyncio
async def test_an_error_names_the_tool_to_use_instead_as_it_is_served(monkeypatch):
    monkeypatch.setenv(TOOL_PREFIX_ENV, PREFIX)
    async with mcp_session(FakeDriver()) as session:
        refused = await session.call_tool(f"{PREFIX}computer_type", {"text": "a\x07b"})
    assert refused.isError and f"press keys with {PREFIX}computer_key" in refused.content[0].text


@pytest.mark.asyncio
async def test_tool_access_rules_name_the_tools_as_it_serves_them(monkeypatch):
    monkeypatch.setenv(TOOL_PREFIX_ENV, PREFIX)
    driver = FakeDriver()
    async with http_client(driver) as http, mcp_session(driver) as session:
        disabled = await http.post("/tools/disable", json={"role": "default", "tools": [f"{PREFIX}computer_click"]})
        listed = {tool.name for tool in (await session.list_tools()).tools}
        refused = await session.call_tool(f"{PREFIX}computer_click", {"x": 1, "y": 1})
    assert disabled.status_code == 200
    assert f"{PREFIX}computer_click" not in listed and f"{PREFIX}computer_screenshot" in listed
    assert refused.isError and "is disabled for role 'default'" in refused.content[0].text


@pytest.mark.asyncio
async def test_setting_the_default_space_rewrites_the_prefixed_point_tools(monkeypatch):
    monkeypatch.setenv(TOOL_PREFIX_ENV, PREFIX)
    driver = FakeDriver()
    async with http_client(driver) as http, mcp_session(driver) as session:
        await http.post("/agentenv/ext/computer-spaces/set", json={"default_space": "norm", "default_norm_scale": 1000})
        after = {t.name: t.description for t in (await session.list_tools()).tools}
    assert "norm space by default: 0 to 1000" in after[f"{PREFIX}computer_click"]


@pytest.mark.parametrize("prefix, problem", [
    ("desk top_", "holds characters a tool's name can't"),
    ("desktop/", "holds characters a tool's name can't"),
    ("décor_", "holds characters a tool's name can't"),
    ("accounting-desk1_", "is 17 characters, and a tool prefix is at most 16"),
])
def test_a_prefix_a_tool_name_cant_hold_or_that_is_too_long_is_refused(monkeypatch, prefix, problem):
    monkeypatch.setenv(TOOL_PREFIX_ENV, prefix)
    with pytest.raises(ValueError, match=problem):
        FakeDriver()


@pytest.mark.asyncio
async def test_a_prefix_of_sixteen_characters_is_taken(monkeypatch):
    monkeypatch.setenv(TOOL_PREFIX_ENV, "accounting-desk_")
    card, _listed = await _names(FakeDriver())
    assert "accounting-desk_computer_pointer_position" in card


class Forwarding(FakeDriver):
    """A driver forwarding another program's batch tool, whose steps name that program's own actions."""

    def forwarded_tools(self):
        schema = {"type": "object", "properties": {"tool": {"type": "string", "enum": ["click", "double_click"]}}}
        return [ForwardedTool(EnvironmentTool(name="run_steps", description="Run click or double_click in one call.",
                                              inputSchema=schema), self.steps)]

    async def steps(self, tool):
        return CallToolResult(content=[TextContent(type="text", text=f"ran {tool}")])

    @server_instructions
    def instructions(self):
        return "Act through run_steps, and take a computer_screenshot first."


@pytest.mark.asyncio
async def test_a_forwarded_tool_keeps_its_own_text_and_the_skill_and_instructions_lead_with_the_prefix_rule(monkeypatch):
    monkeypatch.setenv(TOOL_PREFIX_ENV, PREFIX)
    driver = Forwarding()
    instructions = driver.create_app().instructions
    async with http_client(driver) as http, mcp_session(driver) as session:
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        ran = await session.call_tool(f"{PREFIX}run_steps", {"tool": "click"})
        skill = (await http.get("/agentenv/ext/skill/get", params={"name": "desktop"})).json()
    forwarded = tools[f"{PREFIX}run_steps"]
    assert forwarded.description == "Run click or double_click in one call."
    assert forwarded.inputSchema["properties"]["tool"]["enum"] == ["click", "double_click"]
    assert not ran.isError and ran.content[0].text == "ran click"
    rule = (f"This computer serves its tools under `{PREFIX}`: a tool this text names without `{PREFIX}` is served with it, "
            f"so `run_steps` is `{PREFIX}run_steps`. Names inside a tool's arguments stay as written.")
    assert skill["skill_md"].partition("---\n\n")[2].startswith(rule)
    assert instructions == f"{rule}\n\nAct through run_steps, and take a {PREFIX}computer_screenshot first."


def test_without_a_prefix_a_forwarding_drivers_instructions_are_its_own(monkeypatch):
    monkeypatch.delenv(TOOL_PREFIX_ENV, raising=False)
    assert Forwarding().create_app().instructions == "Act through run_steps, and take a computer_screenshot first."
