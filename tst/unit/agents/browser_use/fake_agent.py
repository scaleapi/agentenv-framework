"""Stands in for ``agent.py``: records the turn it's given, and its environment, to ``FAKE_TURN_RECORD`` and answers
as the run ``FAKE_SCENARIO`` names (``ok``, ``max_steps``, ``crash``, ``hang``)."""

import json
import os
import sys
import time
from pathlib import Path

turn = json.loads(sys.stdin.read())
Path(os.environ["FAKE_TURN_RECORD"]).write_text(json.dumps({
    **turn, "pid": os.getpid(), "env": {name: os.environ.get(name) for name in ("ANONYMIZED_TELEMETRY", "BH_TELEMETRY")}}))
scenario = os.environ.get("FAKE_SCENARIO", "ok")
if scenario == "hang":
    time.sleep(60)
if scenario == "crash":
    sys.stderr.write("RuntimeError: Browser(cdp_url=...) could not connect to a browser at ws://computer.test: 401")
    sys.exit(1)
# A library's own stdout, before agent.py's line.
print("{ not json")
print(json.dumps({"level": "info", "msg": "connected"}))
print(json.dumps({"type": "result", "result": {
    "status": "completed" if scenario == "ok" else scenario,
    "text": "The top story." if scenario == "ok" else "",
    "success": True if scenario == "ok" else None, "error": None,
    "steps": 3, "model": turn["model"], "cost_usd": None,
    "usage": {"input": 100, "output": 20, "cache_read": 60, "cache_write": 10, "requests": 3},
    "messages": [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": turn["prompt"]}]},
        {"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "python", "args": {"code": "page_info()"}}]},
        {"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "finish", "args": {"result": "x"}}]},
    ],
}}))
