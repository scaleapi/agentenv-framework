"""Whether the agent did the desk task through the computer: the Fixture Desk's own record of what reached it, read
through the driver's computer-exec."""
import json
import uuid

from agentenv_protocol import client as protocol

EXEC = "urn:agentenv:computer-exec/v1"


async def verify(mcp_url):
    base = mcp_url.rsplit("/mcp", 1)[0]
    card = await protocol.get_card(base)
    out = await protocol.invoke_extension(base, card, EXEC, {"argv": ["cat", "fixture/state.json"],
                                                             "request_id": uuid.uuid4().hex}, method="run")
    state = json.loads(out["stdout"])
    return [
        {"id": "name", "result": state["name"] == "Ada Lovelace", "description": f"name {state['name']!r}"},
        {"id": "subscribed", "result": state["subscribed"] is True, "description": f"subscribed {state['subscribed']}"},
        {"id": "item", "result": state["selected"] == "Item 37", "description": f"selected {state['selected']!r}"},
        {"id": "saved", "result": state["saved_count"] >= 1, "description": f"saved {state['saved_count']} times"},
        {"id": "cell", "result": state["cells"].get("3,4", 0) >= 1, "description": f"cells {state['cells']}"},
        {"id": "typed-as-keys", "result": len(state["keys"]) >= len("Ada Lovelace"),
         "description": f"{len(state['keys'])} key events reached the field"},
    ]
