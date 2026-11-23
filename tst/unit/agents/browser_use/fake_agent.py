"""Stands in for ``agent.py``: records the turn it's given, and its environment, to ``FAKE_TURN_RECORD`` and answers
as the run ``FAKE_SCENARIO`` names (``ok``, ``max_steps``, ``crash``, ``hang``; ``cut``, ``mixed`` and ``refused`` for
requests that got no response; ``follow_up`` for a context's third turn)."""

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
TURN_MESSAGES = [
    {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": turn["prompt"]}]},
    {"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "python", "args": {"code": "page_info()"}}]},
    {"kind": "response", "parts": [{"part_kind": "tool-call", "tool_name": "finish", "args": {"result": "x"}}]},
]
USAGE = {"input": 100, "output": 20, "cache_read": 60, "cache_write": 10, "requests": 3}
messages, prior, usage, calls, cost = TURN_MESSAGES, 0, USAGE, {"sent": 3, "refused": 0}, None
if scenario == "cut":  # browser-use-cut-request: the run's 1 s budget cut its only request; the proxy still answered
    messages, usage, calls, cost = TURN_MESSAGES[:1], dict.fromkeys(USAGE, 0), {"sent": 1, "refused": 0}, 0.0
elif scenario == "mixed":
    usage, calls = {**USAGE, "requests": 2}, {"sent": 3, "refused": 0}
elif scenario == "refused":
    usage, calls = {**USAGE, "requests": 2}, {"sent": 3, "refused": 1}
elif scenario == "follow_up":  # two earlier turns of three messages each, then this turn's
    messages, prior = TURN_MESSAGES * 3, 6
print(json.dumps({"type": "result", "result": {
    "status": "completed" if scenario in ("ok", "mixed", "refused", "follow_up") else
              "timeout" if scenario == "cut" else scenario,
    "text": "" if scenario in ("max_steps", "cut") else "The top story.",
    "success": None if scenario in ("max_steps", "cut") else True, "error": None,
    "steps": 0 if scenario == "cut" else 3, "model": turn["model"], "cost_usd": cost,
    "usage": usage, "messages": messages, "prior": prior, "calls": calls,
}}))
