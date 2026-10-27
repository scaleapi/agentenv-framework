"""Calls the search_emails tool through the env's MCP endpoint and checks it finds the inbox's one invoice."""
import json

from agentenv_protocol.mcp_compat import client_session, field


async def verify(mcp_url):
    async with client_session(mcp_url) as session:
        result = await session.call_tool("search_emails", {"query": "invoice"})
    text = "".join(getattr(part, "text", "") for part in result.content)
    found = [email["id"] for email in json.loads(text)] if not field(result, "isError") else []
    return [{"id": "finds-the-invoice", "result": found == ["m-1"], "description": f"search_emails found {found}"}]
