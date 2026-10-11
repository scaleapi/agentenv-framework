"""Deterministic review agent: add one item through the configured MCP server."""

import json

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from agentenv_protocol.a2a_agent import (
    MCP_CONFIG_V1, TRAJECTORY_V1, AgentConfig, AgentEnvAgent,
    AgentIdentity, TaskRequest, TaskResult, TextPart, Usage, a2a_agent,
)


@a2a_agent(
    identity=AgentIdentity(name="vercel-review-agent", version="1.0.0",
                           description="Adds the requested item through MCP and reports stored state."),
    config=AgentConfig,
    extensions=(MCP_CONFIG_V1, TRAJECTORY_V1),
)
class ReviewAgent(AgentEnvAgent):
    async def run(self, request: TaskRequest[AgentConfig]) -> TaskResult:
        item = "".join(part.text for part in request.parts if isinstance(part, TextPart)).strip()
        if len(request.mcp_servers) != 1 or not item:
            raise ValueError("The review task requires one MCP server and an item name")
        registration = next(iter(request.mcp_servers.values()))
        async with httpx.AsyncClient(headers=registration.get("headers") or {}, timeout=60) as client:
            async with streamable_http_client(registration["url"], http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = [tool.name for tool in tools.tools]
                    assert "items_add_item" in names and "list_items" in names, names
                    added = await session.call_tool("items_add_item", {"item": item})
                    assert not added.isError, added
                    listed = await session.call_tool("list_items", {})
                    assert not listed.isError, listed
                    text = "\n".join(c.text for c in listed.content if c.type == "text")
        reply = "Stored items: " + text
        return (TaskResult.builder().succeeded().parts([TextPart(text=reply)])
                .usage(Usage(tool_call_count=2, input_tokens=0, output_tokens=0))
                .native_trajectory(format="vercel-review/v1", payload=[
                    {"tool": "items_add_item", "arguments": {"item": item}},
                    {"tool": "list_items", "response": json.loads(text)},
                ]).build())


if __name__ == "__main__":
    ReviewAgent().serve()
