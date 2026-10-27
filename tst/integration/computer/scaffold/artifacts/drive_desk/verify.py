"""A scripted drive of the Fixture Desk while the desktop records: the clicks, typing and scroll a person would make,
each answered with a screenshot. A test finds each moment in the video by its send time, its screenshot's capture time
and thumbnail, and the clicked grid cell's box."""
import asyncio
import base64
import io
import json
from datetime import datetime, timezone

from PIL import Image

from agentenv_protocol.computer.conformance.client import Driver

THUMB = (64, 40)


def _thumb(result):
    data = next(c.data for c in result.content if c.type == "image")
    return Image.open(io.BytesIO(base64.b64decode(data))).convert("L").resize(THUMB).tobytes().hex()


async def verify(mcp_url):
    driver = await Driver.connect(mcp_url.rsplit("/mcp", 1)[0], 60)
    for _ in range(60):  # the Fixture Desk starts beside the driver and writes its layout once it is up
        out = await driver.run("cat", "fixture/layout.json")
        if out["exit_code"] == 0 and out["stdout"].strip():
            break
        await asyncio.sleep(0.5)
    layout = json.loads(out["stdout"])

    def centre(box):
        x, y, w, h = layout[box]
        return {"x": x + w // 2, "y": y + h // 2}

    gx, gy, cell = layout["grid"][0], layout["grid"][1], layout["cell"]
    clicked = [gx + 4 * cell, gy + 3 * cell, cell, cell]
    steps = [("computer_click", centre("name")), ("computer_type", {"text": "Ada Lovelace"}),
             ("computer_click", centre("subscribe")),
             ("computer_click", {"x": clicked[0] + cell // 2, "y": clicked[1] + cell // 2}),
             ("computer_scroll", {**centre("list"), "direction": "down", "amount": 10}),
             ("computer_click", centre("save"))]
    shots = []
    async with driver.session() as session:
        await session.call_tool("computer_screenshot", {})
        for tool, args in steps:
            sent_at = datetime.now(timezone.utc).isoformat()
            result = await session.call_tool(tool, {**args, "screenshot": True})
            if result.isError:
                return [{"id": "drove", "result": False, "description": f"{tool}: {result.content[0].text}"}]
            shots.append({"step": tool, "args": args, "sent_at": sent_at,
                          "captured_at": result.structuredContent["frame"]["captured_at"], "thumb": _thumb(result)})
            await asyncio.sleep(0.7)
    return [{"id": "drove", "result": True, "description": json.dumps({"shots": shots, "clicked_cell": clicked})}]
