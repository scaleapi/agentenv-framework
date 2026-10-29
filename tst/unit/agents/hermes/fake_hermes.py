"""A scripted `hermes agentenv turn` for the bridge tests, run as the real one is: a process of its own. It writes the
events, usage report and stdout a turn would, as FAKE_HERMES_MODE says: ``ok`` answers, ``fail`` fails the model call,
``hang`` starts a tool and never finishes. Its arguments, environment and cwd go to FAKE_HERMES_RECORD."""
import json
import os
import subprocess
import sys
import time

args = sys.argv[1:]
option = {name: args[i + 1] for i, name in enumerate(args) if name.startswith("--") and i + 1 < len(args)}
mode = os.environ.get("FAKE_HERMES_MODE", "ok")
events = int(os.environ["AGENTENV_EVENTS_FD"])
session = option.get("--resume") or "20260101_000000_abcdef"
with open(os.environ["FAKE_HERMES_RECORD"], "w") as record:
    json.dump({"args": args, "cwd": os.getcwd(), "home": os.environ.get("HERMES_HOME"),
               "prompt": open(option["--prompt-file"]).read()}, record)


def emit(kind, **fields):
    os.write(events, (json.dumps({"type": kind, "timestamp": 0, "session_id": session, **fields}) + "\n").encode())


def report(**fields):
    with open(option["--usage-file"], "w") as usage:
        json.dump({"session_id": session, "input_tokens": 120, "output_tokens": 8, "total_tokens": 128,
                   "estimated_cost_usd": 0.002, "api_calls": 2, **fields}, usage)


emit("on_session_start", model=option.get("--model"))
emit("pre_tool_call", tool_call_id="call_1", tool_name="terminal", args={"command": "echo $((6 * 7))"})
if mode == "hang":
    tool = subprocess.Popen(["sleep", "600"])
    with open(os.environ["FAKE_HERMES_PIDS"], "w") as pids:
        pids.write(f"{os.getpid()} {tool.pid}")
    time.sleep(600)
emit("post_tool_call", tool_call_id="call_1", tool_name="terminal", result='{"output": "42"}', status="ok")
if mode == "fail":
    emit("api_request_error", status_code=401, error="HTTP 401: invalid key")
    emit("on_session_end", completed=False, failed=True, interrupted=False)
    report(completed=False, failed=True)
    sys.stderr.write("hermes: the model rejected the key\n")
    sys.exit(1)
answer = os.environ.get("FAKE_HERMES_ANSWER", "42")
emit("post_llm_call", assistant_response=answer)
emit("on_session_end", completed=True, failed=False, interrupted=False)
report(completed=True, failed=False)
print(answer)
