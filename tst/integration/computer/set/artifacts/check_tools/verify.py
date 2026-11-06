"""What the agents' role sees on the computer: AgentEnv's computer_* tools and the driver's own, each result naming them."""
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def verify(mcp_url):
    async with streamable_http_client(mcp_url) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            names = sorted(tool.name for tool in (await session.list_tools()).tools)
    computer = [name for name in names if name.startswith("computer_")]
    own = [name for name in names if not name.startswith("computer_")]
    return [{"id": "computer", "result": True, "description": " ".join(computer)},
            {"id": "own", "result": True, "description": " ".join(own)}]
