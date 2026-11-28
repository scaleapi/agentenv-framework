"""Stands in for ``agent.py``: records the turn it's given, and its environment, to ``FAKE_TURN_RECORD`` and answers
as the run ``FAKE_SCENARIO`` names (``ok``, ``max_steps``, ``crash``, ``hang``; ``cut``, ``mixed`` and ``refused`` for
requests that got no response; ``follow_up`` for a context's third turn). A turn that builds an atif/1 copy is a second
turn answered as browser-use dumps its messages, with what agent.py keeps for the copy: a refused request, a
compaction's summary and a pruned screenshot."""

import base64
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
extras = {}
if turn["atif"]:
    JPEG = base64.b64encode(b"\xff\xd8\xff\xe0screenshot").decode()

    def at(ms: int) -> str:
        return f"2026-11-27T10:00:{ms // 1000:02d}.{ms % 1000:03d}000Z"

    def message(kind: str, run: str, ms: int, *parts: dict, **fields) -> dict:
        return {"parts": list(parts), "timestamp": at(ms), "kind": kind, "run_id": run, "conversation_id": "conv",
                "metadata": None, "state": "complete", **fields}

    def response(run: str, ms: int, part: dict, counts: tuple[int, int, int, int], cost: str) -> dict:
        usage = dict(zip(("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"), counts))
        return message("response", run, ms, part, usage={**usage, "cost": cost}, model_name="claude-haiku-4-5-20251001",
                       provider_name="anthropic", provider_response_id=f"msg_{ms}", finish_reason="tool_call")

    def call(call_id: str, name: str, args: dict) -> dict:
        return {"tool_name": name, "args": args, "tool_call_id": call_id, "part_kind": "tool-call"}

    def returned(call_id: str, name: str, content, ms: int) -> dict:
        return {"tool_name": name, "content": content, "tool_call_id": call_id, "outcome": "success",
                "part_kind": "tool-return", "timestamp": at(ms)}

    run = "run-2"
    messages = [  # the earlier turn's, which the run returns again, then this run's
        message("request", "run-1", 1, {"content": "earlier", "part_kind": "user-prompt"}),
        response("run-1", 2, {"content": "Earlier.", "part_kind": "text"}, (10, 1, 0, 0), "0.0001"),
        message("request", run, 10, {"content": turn["prompt"], "part_kind": "user-prompt"}),
        response(run, 20, call("toolu_1", "python", {"code": "screenshot()"}), (40, 8, 20, 10), "0.0004"),
        message("request", run, 30, returned("toolu_1", "python", ["Top story", "[older screenshot removed]"], 25)),
        response(run, 60, call("toolu_2", "finish", {"result": "The top story.", "success": True}), (45, 7, 40, 0),
                 "0.0003"),
        message("request", run, 70, returned("toolu_2", "finish", "Final result processed.", 65)),
    ]
    summary = response(run, 50, {"content": "The user asked for the top story.", "part_kind": "text"}, (15, 5, 0, 0),
                       "0.0001")
    prior, cost, calls = 2, 0.0008, {"sent": 4, "refused": 1}
    extras = {"atif": {
        "attempts": [{"started": at(15), "status": 200, "timestamp": at(18)},
                     {"started": at(40), "status": 200, "timestamp": at(48)},
                     {"started": at(52), "status": 503, "timestamp": at(53)},
                     {"started": at(54), "status": 200, "timestamp": at(58)}],
        "summaries": [{"compacted": True, "response": summary}],
        "images": {"toolu_1": [{"index": 1, "media_type": "image/jpeg", "data": JPEG}]}}}
print(json.dumps({"type": "result", "result": {
    "status": "completed" if scenario in ("ok", "mixed", "refused", "follow_up") else
              "timeout" if scenario == "cut" else scenario,
    "text": "" if scenario in ("max_steps", "cut") else "The top story.",
    "success": None if scenario in ("max_steps", "cut") else True, "error": None,
    "steps": 0 if scenario == "cut" else 3, "model": turn["model"], "cost_usd": cost,
    "usage": usage, "messages": messages, "prior": prior, "calls": calls, **extras,
}}))
