"""@tool annotations, structured output and _meta on both FastMCPs, served_tools choosing what a handler serves,\nupdate_tool changing a served description, and forwarded tools served with their own schemas."""
from __future__ import annotations

import base64
import re
from typing import Annotated

import anyio
import httpx
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
    ForwardedTool,
    create_fastmcp_app,
    environment_card,
    resource,
    server_instructions,
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


async def _card_tools(app) -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as http:
        card = (await http.get("/.well-known/agent-env.json")).json()
    return {t["name"]: t for t in card["capabilities"]["tools"]}


@pytest.mark.asyncio
async def test_update_tool_rewrites_a_description_on_the_card_and_in_tools_list_on_mcps_fastmcp():
    env = _Env()
    app = env.create_app()
    await env.update_tool("look", description="Look again.")
    with anyio.fail_after(30):
        async with create_connected_server_and_client_session(app) as session:
            tools = {t.name: t for t in (await session.list_tools()).tools}
            look = await session.call_tool("look", {})
    card = await _card_tools(app.streamable_http_app())
    assert tools["look"].description == card["look"]["description"] == "Look again."
    assert set(card) == set(tools) == {"look", "plain", "secret", "shot"}
    assert {t.name: t.description for t in env.mounted_card().capabilities.tools}["look"] == "Look again."
    assert look.structuredContent == {"frame_id": "f1", "width": 1280}


@pytest.mark.asyncio
async def test_update_tool_rewrites_a_description_on_the_card_and_in_tools_list_on_fastmcp():
    env = _Env()
    app = fastmcp.FastMCP("tools")
    env.mount(app)
    await env.update_tool("look", description="Look again.")
    with anyio.fail_after(30):
        async with fastmcp.Client(app) as client:
            tools = {t.name: t for t in await client.list_tools()}
            look = await client.call_tool("look", {})
    card = await _card_tools(app.http_app())
    assert tools["look"].description == card["look"]["description"] == "Look again."
    assert tools["look"].annotations.readOnlyHint is True and tools["look"].meta["agentenv.io/group"] == "minimum"
    assert look.structured_content == {"frame_id": "f1", "width": 1280}


@pytest.mark.asyncio
async def test_update_tool_needs_a_mounted_env_and_a_served_tool():
    with pytest.raises(RuntimeError, match="_Env is not mounted"):
        await _Env().update_tool("look", description="x")
    with pytest.raises(RuntimeError, match="_Env is not mounted"):
        _Env().mounted_card()
    env = _Chooser()
    env.create_app()
    with pytest.raises(ValueError, match="no served tool is named 'secret'"):
        await env.update_tool("secret", description="x")


_CLICK = {"type": "object", "additionalProperties": False, "properties": {
    "element_token": {"type": "string"}, "x": {"type": "integer"}, "count": {"type": "integer", "minimum": 1}}}
_FOUND = {"type": "object", "required": ["rows"], "properties": {"rows": {"type": "array"}}}


@environment_card(name="forwarding")
class _Forwarding(AgentEnvEnvironment):
    def __init__(self, calls: list) -> None:
        self.calls = calls

    @tool()
    def look(self) -> str:
        return "looked"

    def forwarded_tools(self) -> list[ForwardedTool]:
        async def click(**arguments) -> CallToolResult:
            self.calls.append(arguments)
            return CallToolResult(content=[TextContent(type="text", text="clicked"),
                                           ImageContent(type="image", data="aGk=", mimeType="image/png")],
                                  structuredContent={"hit": "button OK"})

        async def find(**arguments) -> CallToolResult:
            return CallToolResult(isError=True, content=[TextContent(type="text", text="the daemon isn't answering")])

        return [ForwardedTool(EnvironmentTool(name="click", description="Click an element.", inputSchema=_CLICK), click,
                              {"readOnlyHint": False, "destructiveHint": False}),
                ForwardedTool(EnvironmentTool(name="find", inputSchema={"type": "object"}), find, None, _FOUND)]

    @server_instructions
    def guide_first(self) -> str:
        return "Read the guide first."


@pytest.mark.asyncio
async def test_a_forwarded_tool_is_listed_with_its_own_schema_and_called_with_its_arguments_as_they_came():
    calls = []
    env = _Forwarding(calls)
    with anyio.fail_after(30):
        async with create_connected_server_and_client_session(env.create_app()) as session:
            listed = {t.name: t for t in (await session.list_tools()).tools}
            result = await session.call_tool("click", {"element_token": "s1:4", "count": 2})
            found = await session.call_tool("find", {})

    assert listed["click"].inputSchema == _CLICK and listed["click"].description == "Click an element."
    assert listed["click"].annotations.readOnlyHint is False and listed["click"].outputSchema is None
    assert listed["find"].outputSchema == _FOUND and found.isError and found.content[0].text == "the daemon isn't answering"
    assert calls == [{"element_token": "s1:4", "count": 2}]
    assert [c.type for c in result.content] == ["text", "image"] and result.structuredContent == {"hit": "button OK"}
    assert {t.name: t.inputSchema for t in env.mounted_card().capabilities.tools}["click"] == _CLICK


_OTHER_FILES = [
    {"uri": "skill://other/SKILL.md", "name": "other", "description": "The skill.", "mimeType": "text/markdown"},
    {"uri": "skill://other/LINUX.md", "name": "LINUX.md", "mimeType": "text/markdown"},
]


@environment_card(name="library")
class _Library(AgentEnvEnvironment):
    @resource("docs://library/guide", mime_type="text/markdown")
    def guide(self) -> str:
        """How to use it."""
        return "# Guide"

    @resource("skill://other/{file}", listed=lambda env: _OTHER_FILES)
    async def other_file(self, file: str) -> str:
        return f"# {file}"

    @resource("orders://{order_id}", mime_type="application/octet-stream")
    def order(self, order_id: str) -> bytes:
        return order_id.encode()


@pytest.mark.asyncio
async def test_fixed_listed_and_templated_resources_are_served_by_their_methods():
    with anyio.fail_after(30):
        async with create_connected_server_and_client_session(_Library().create_app()) as session:
            listed = (await session.list_resources()).resources
            templates = (await session.list_resource_templates()).resourceTemplates
            guide = await session.read_resource("docs://library/guide")
            linux = await session.read_resource("skill://other/LINUX.md")
            order = await session.read_resource("orders://42")

    assert [(str(r.uri), r.name, r.description, r.mimeType) for r in listed] == [
        ("docs://library/guide", "guide", "How to use it.", "text/markdown"),
        ("skill://other/SKILL.md", "other", "The skill.", "text/markdown"),
        ("skill://other/LINUX.md", "LINUX.md", None, "text/markdown")]
    assert [(t.uriTemplate, t.name) for t in templates] == [("orders://{order_id}", "order")]
    assert guide.contents[0].text == "# Guide" and linux.contents[0].text == "# LINUX.md"
    assert base64.b64decode(order.contents[0].blob) == b"42"


def test_a_resource_served_twice_listed_off_its_template_or_taking_other_parts_is_refused():
    class Twice(_Library):
        @resource("docs://library/guide")
        def again(self) -> str:
            return "again"

    class Elsewhere(_Library):
        @resource("skill://more/{file}", listed=lambda env: [{"uri": "skill://elsewhere/A.md", "name": "A.md"}])
        def more(self, file: str) -> str:
            return file

    def nothing(self) -> str:
        return ""

    with pytest.raises(ValueError, match=re.escape("resources served twice: ['docs://library/guide']")):
        Twice().create_app()
    with pytest.raises(ValueError, match=re.escape("'skill://elsewhere/A.md' doesn't match 'skill://more/{file}'")):
        Elsewhere().create_app()
    with pytest.raises(ValueError, match=re.escape("needs a method taking ['order_id'] besides self, not []")):
        resource("orders://{order_id}")(nothing)
    with pytest.raises(ValueError, match="lists resources only from a template URI"):
        resource("docs://library/all", listed=lambda env: [])(nothing)
    with pytest.raises(RuntimeError, match="the resource 'docs://library/guide' needs mcp's FastMCP"):
        _Library().mount(fastmcp.FastMCP("library"))


def test_server_instructions_are_one_plain_method_and_reach_an_app_built_from_the_handler():
    class Twice(_Forwarding):
        @server_instructions
        def again(self) -> str:
            return "Or this."

    with pytest.raises(ValueError, match="one @server_instructions method, not \\['again', 'guide_first'\\]"):
        Twice([]).create_app()
    with pytest.raises(TypeError, match="a plain method, not the coroutine 'later'"):
        @server_instructions
        async def later(self) -> str:
            return "Later."
    assert create_fastmcp_app(_Forwarding([]), card=EnvironmentCard(name="forwarding")).instructions == "Read the guide first."


def test_a_forwarded_tool_named_like_a_tool_is_refused():
    class Clashing(_Forwarding):
        @tool()
        def click(self) -> str:
            return "mine"

    with pytest.raises(ValueError, match="forwarded tools named like another tool: \\['click'\\]"):
        Clashing([]).create_app()


def test_a_forwarded_tool_needs_mcps_fastmcp():
    with pytest.raises(RuntimeError, match="the forwarded tool 'click' needs mcp's FastMCP"):
        _Forwarding([]).mount(fastmcp.FastMCP("forwarding"))
