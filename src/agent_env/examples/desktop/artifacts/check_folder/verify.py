"""Whether Documents/Reports is a folder in the session user's home, asked through the driver's computer-exec."""
import uuid

from agentenv_protocol import client as protocol

EXEC = "urn:agentenv:computer-exec/v1"


async def verify(mcp_url):
    base = mcp_url.rsplit("/mcp", 1)[0]
    card = await protocol.get_card(base)
    out = await protocol.invoke_extension(base, card, EXEC, {"argv": ["test", "-d", "Documents/Reports"],
                                                             "request_id": uuid.uuid4().hex}, method="run")
    return [{"id": "folder", "result": out["exit_code"] == 0,
             "description": "Documents/Reports is a folder" if out["exit_code"] == 0 else "no folder Documents/Reports"}]
