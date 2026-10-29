"""Whether the figures moved from Downloads into a Reports folder in Documents, asked through the driver's
computer-exec."""
import uuid

from agentenv_protocol import client as protocol

EXEC = "urn:agentenv:computer-exec/v1"
MOVED = "test -f Documents/Reports/figures.csv && test ! -e Downloads/figures.csv"


async def verify(mcp_url, env_url):
    card = await protocol.get_card(env_url)
    out = await protocol.invoke_extension(env_url, card, EXEC, {"argv": ["sh", "-c", MOVED],
                                                                "request_id": uuid.uuid4().hex}, method="run")
    moved = out["exit_code"] == 0
    return [{"id": "folder", "result": moved, "description": "the figures are in Documents/Reports" if moved else
             "the figures aren't in Documents/Reports, or are still in Downloads"}]
