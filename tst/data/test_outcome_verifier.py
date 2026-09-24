"""Minimal outcome verifier for integration tests.

Checks that email and slack MCP servers have loaded data.
"""

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

MAX_RETRIES = 10
RETRY_DELAY = 5.0


async def _call_tool(mcp_url: str, tool_name: str, arguments: dict) -> dict | list | str:
    """Call an MCP tool with a fresh connection and retries."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with streamable_http_client(mcp_url) as (r, w, _):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    text = result.content[0].text
                    return json.loads(text)
        except Exception:
            if attempt == MAX_RETRIES:
                raise
            await asyncio.sleep(RETRY_DELAY)


def _make(id_: str, description: str, result, depends_on: list[str]) -> dict:
    return {
        "id": id_,
        "description": description,
        "result": result,
        "depends_on": depends_on,
    }


async def verify(mcp_url: str) -> list[dict]:
    """Verify that email and slack data have been loaded."""
    criteria = []

    # Check 1: Email data loaded
    cid1 = "email_data_loaded"
    try:
        data = await _call_tool(mcp_url, "list_emails", {"folder_name": "INBOX"})
        total = data.get("total_emails", 0) if isinstance(data, dict) else 0
        criteria.append(_make(cid1, "Email database has at least one email", total > 0, []))
    except Exception:
        criteria.append(_make(cid1, "Email database has at least one email", False, []))

    # Check 2: Slack channels exist
    cid2 = "slack_channels_exist"
    try:
        data = await _call_tool(mcp_url, "channels_list", {"channel_types": "public_channel"})
        channels = data.get("channels", []) if isinstance(data, dict) else []
        channel_names = [c.get("name", "") for c in channels] if isinstance(channels, list) else []
        criteria.append(_make(cid2, "Slack has a 'general' channel", "general" in channel_names, []))
    except Exception:
        criteria.append(_make(cid2, "Slack has a 'general' channel", False, []))

    return criteria
