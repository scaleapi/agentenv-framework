"""Calls the echo tool through the env's MCP endpoint and checks it sends the text back."""
from agentenv_protocol.mcp_compat import client_session, field

SENT = "hello from agent-env"


async def verify(mcp_url):
    async with client_session(mcp_url) as session:
        result = await session.call_tool("echo", {"text": SENT})
    text = "".join(getattr(part, "text", "") for part in result.content)
    return [{"id": "echoes", "result": not field(result, "isError") and text == SENT,
             "description": f"echo returned {text!r}"}]
