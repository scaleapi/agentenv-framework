"""Stands in for ``node agent.mjs``: records the turn it's given to ``FAKE_TURN_RECORD`` and replays the stream
``FAKE_SCENARIO`` names (``ok``, ``max_steps``, ``crash``, ``hang``)."""

import json
import os
import sys
import time
from pathlib import Path

turn = json.loads(sys.stdin.read())
Path(os.environ["FAKE_TURN_RECORD"]).write_text(json.dumps({**turn, "pid": os.getpid()}))
scenario = os.environ.get("FAKE_SCENARIO", "ok")
if scenario == "hang":
    time.sleep(60)
if scenario == "crash":
    sys.stderr.write("Error: Browser Use Cloud POST failed (401).")
    sys.exit(1)
for event in ({"type": "agent_start"}, {"type": "message_update"}, {"type": "tool_execution_end"}, {"type": "agent_end"}):
    print(json.dumps({"type": "event", "event": event}))
print(json.dumps({"type": "result", "result": {
    "status": "completed" if scenario == "ok" else scenario,
    "text": "The top story." if scenario == "ok" else "",
    "steps": 3, "model": "agentenv/gpt-test",
    "usage": {"input": 100, "output": 20, "totalTokens": 120},
}}))
