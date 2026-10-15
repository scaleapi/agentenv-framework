"""Calls the echo tool through the env's MCP endpoint and checks it sends the text back."""
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

SENT = "hello from agent-env"


async def verify(mcp_url):
    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("echo", {"text": SENT})
    text = "".join(getattr(part, "text", "") for part in result.content)
    return [{"id": "echoes", "result": not result.isError and text == SENT, "description": f"echo returned {text!r}"}]
