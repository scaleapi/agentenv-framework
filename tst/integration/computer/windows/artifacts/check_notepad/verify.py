"""Whether Notepad holds the line the agent was asked to type, saved as hello.txt: a Notepad window titled for the
file with no unsaved changes, hello or hello.txt as Windows shows file extensions or hides them, whose text has the
line, asked through cua-driver's own tools."""
import json
import re

from agentenv_protocol.computer.conformance.client import Driver

LINE = "agent-env on Windows"
SAVED = re.compile(r"^hello(\.txt)?\b")


async def verify(mcp_url, env_url):
    driver = await Driver.connect(env_url, 60)
    async with driver.session() as session:
        listed = await session.call_tool("list_windows", {"app": "Notepad"})
        windows = (listed.structuredContent or {}).get("windows") or []
        saved = [w for w in windows if SAVED.match(w.get("title") or "")]
        text = ""
        if saved:
            state = await session.call_tool("get_window_state", {"pid": saved[0]["pid"], "query": LINE})
            text = json.dumps(state.structuredContent or {}) + " ".join(
                getattr(c, "text", "") for c in state.content)
    found = bool(saved) and LINE in text
    titles = [w.get("title") for w in windows]
    return [{"id": "notepad", "result": found, "description": "Notepad shows hello.txt saved, holding the line" if found
             else f"no saved Notepad window titled hello.txt holds the line; Notepad's windows: {titles}"}]
