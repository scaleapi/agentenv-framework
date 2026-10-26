"""@tool annotations, structured output and _meta on both FastMCPs, and served_tools choosing what a handler serves."""
from __future__ import annotations

from typing import Annotated

import anyio
import pytest
from pydantic import BaseModel

pytest.importorskip("mcp.server.fastmcp")
fastmcp = pytest.importorskip("fastmcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402
from mcp.types import CallToolResult, ImageContent, TextContent  # noqa: E402

from agentenv_protocol import (  # noqa: E402
    HIDDEN_BY_DEFAULT_META_KEY,
    AgentEnvEnvironment,
    AgentEnvFastMCPApplication,
    EnvironmentCard,
    EnvironmentTool,
    environment_card,
    tool,
)


class _Frame(BaseModel):
    frame_id: str
    width: int


@environment_card(name="tools")
class _Env(AgentEnvEnvironment):
    @tool(annotations={"readOnlyHint": True, "title": "Look"}, meta={"agentenv.io/group": "minimum"})
    def look(self) -> _Frame:
        return _Frame(frame_id="f1", width=1280)

    @tool(structured_output=False)
    def plain(self) -> dict:
        return {"a": 1}

    @tool(hidden_by_default=True, meta={"agentenv.io/group": "admin"})
    def secret(self) -> str:
        return "s"

    @tool()
    def shot(self) -> Annotated[CallToolResult, _Frame]:
        return CallToolResult(
            content=[TextContent(type="text", text="Captured f2"), ImageContent(type="image", data="aGk=", mimeType="image/png")],
            structuredContent={"frame_id": "f2", "width": 640},
        )


async def _mcp_tools(env: AgentEnvEnvironment) -> dict:
    with anyio.fail_after(30):
        async with create_connected_server_and_client_session(env.create_app()) as session:
            return {t.name: t for t in (await session.list_tools()).tools}


@pytest.mark.asyncio
async def test_annotations_meta_and_output_schemas_are_listed_on_mcps_fastmcp():
    tools = await _mcp_tools(_Env())
    look = tools["look"]
    assert look.annotations.readOnlyHint is True and look.annotations.title == "Look"
    assert look.meta == {"agentenv.io/group": "minimum"}
    assert look.outputSchema["properties"]["frame_id"] == {"title": "Frame Id", "type": "string"}
    assert tools["plain"].outputSchema is None
    assert tools["secret"].meta == {"agentenv.io/group": "admin", HIDDEN_BY_DEFAULT_META_KEY: True}
    assert tools["shot"].outputSchema["required"] == ["frame_id", "width"]


@pytest.mark.asyncio
async def test_structured_results_reach_the_caller_with_their_content_blocks():
    with anyio.fail_after(30):
        async with create_connected_server_and_client_session(_Env().create_app()) as session:
            look = await session.call_tool("look", {})
            shot = await session.call_tool("shot", {})
    assert look.structuredContent == {"frame_id": "f1", "width": 1280}
    assert [c.type for c in shot.content] == ["text", "image"] and shot.structuredContent == {"frame_id": "f2", "width": 640}


@pytest.mark.asyncio
async def test_annotations_meta_and_output_schemas_are_listed_on_fastmcp():
    app = fastmcp.FastMCP("tools")
    _Env().mount(app)
    with anyio.fail_after(30):
        async with fastmcp.Client(app) as client:
            tools = {t.name: t for t in await client.list_tools()}
            look = await client.call_tool("look", {})
            shot = await client.call_tool("shot", {})
    assert tools["look"].annotations.readOnlyHint is True
    assert tools["look"].meta["agentenv.io/group"] == "minimum"
    assert tools["look"].outputSchema["properties"]["frame_id"]["type"] == "string"
    assert tools["plain"].outputSchema is None
    assert tools["secret"].meta[HIDDEN_BY_DEFAULT_META_KEY] is True
    assert tools["shot"].outputSchema["required"] == ["frame_id", "width"]
    assert look.structured_content == {"frame_id": "f1", "width": 1280}
    assert [c.type for c in shot.content] == ["text", "image"] and shot.structured_content == {"frame_id": "f2", "width": 640}


@pytest.mark.asyncio
async def test_a_bare_call_tool_result_on_fastmcp_lists_no_output_schema():
    @environment_card(name="bare")
    class _Bare(AgentEnvEnvironment):
        @tool()
        async def bare(self) -> CallToolResult:
            return CallToolResult(content=[TextContent(type="text", text="done")])

    app = fastmcp.FastMCP("bare")
    _Bare().mount(app)
    with anyio.fail_after(30):
        async with fastmcp.Client(app) as client:
            listed = (await client.list_tools())[0]
            result = await client.call_tool("bare", {})
    assert listed.outputSchema is None and result.content[0].text == "done"


class _Chooser(_Env):
    def served_tools(self, tools: list[EnvironmentTool]) -> list[EnvironmentTool]:
        return [t.model_copy(update={"description": "Take a look."}) if t.name == "look" else t
                for t in tools if t.name != "secret"]


@pytest.mark.asyncio
async def test_served_tools_chooses_what_is_listed_and_advertised():
    env = _Chooser()
    tools = await _mcp_tools(env)
    assert set(tools) == {"look", "plain", "shot"} and tools["look"].description == "Take a look."
    advertised = {t.name: t for t in AgentEnvFastMCPApplication(EnvironmentCard(name="t"), _Chooser()).environment_card.capabilities.tools}
    assert set(advertised) == {"look", "plain", "shot"} and advertised["look"].description == "Take a look."


@pytest.mark.parametrize("chosen, error", [
    (lambda tools: [*tools, EnvironmentTool(name="ghost")], r"no @tool method: \['ghost'\]"),
    (lambda tools: [*tools, tools[0]], "a tool twice"),
])
def test_served_tools_naming_an_unmarked_tool_or_one_twice_is_refused(chosen, error):
    class _Bad(_Env):
        def served_tools(self, tools):
            return chosen(tools)

    with pytest.raises(ValueError, match=error):
        AgentEnvFastMCPApplication(EnvironmentCard(name="t"), _Bad())


@pytest.mark.asyncio
async def test_structured_output_false_lists_no_schema_for_a_call_tool_result_on_fastmcp():
    @environment_card(name="opted-out")
    class _OptedOut(AgentEnvEnvironment):
        @tool(structured_output=False)
        def shot(self) -> Annotated[CallToolResult, _Frame]:
            return CallToolResult(content=[TextContent(type="text", text="ok")], structuredContent={"frame_id": "f1", "width": 1})

    app = fastmcp.FastMCP("opted-out")
    _OptedOut().mount(app)
    with anyio.fail_after(30):
        async with fastmcp.Client(app) as client:
            listed = (await client.list_tools())[0]
            result = await client.call_tool("shot", {})
    assert listed.outputSchema is None and result.content[0].text == "ok"
