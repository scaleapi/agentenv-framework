"""Calls the search_emails tool through the env's MCP endpoint and checks it finds the inbox's one invoice."""
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def verify(mcp_url):
    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("search_emails", {"query": "invoice"})
    text = "".join(getattr(part, "text", "") for part in result.content)
    found = [email["id"] for email in json.loads(text)] if not result.isError else []
    return [{"id": "finds-the-invoice", "result": found == ["m-1"], "description": f"search_emails found {found}"}]
