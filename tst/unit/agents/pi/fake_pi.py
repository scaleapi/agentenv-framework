"""Stands in for ``pi --mode json``: records its invocation and replays a scripted event stream.

``FAKE_PI_RECORD`` names the file the argv, cwd, stdin and pi config dir contents are written to;
``FAKE_PI_SCENARIO`` selects the stream (``ok``, ``error``, ``crash``, ``hang``). Like pi, it keeps the
conversation in ``<session-dir>/<stamp>_<id>.jsonl``, found by its header's id and cwd, and its tool call
writes ``FAKE_PI_WRITE`` (``name=text``) into the working directory when set.
"""

import json
import os
import sys
import time
from pathlib import Path


def emit(event):
    sys.stdout.write(json.dumps(event) + "\n")
    sys.stdout.flush()


def assistant(text, stop_reason="stop", **extra):
    return {
        "role": "assistant",
        "content": [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": text}],
        "stopReason": stop_reason,
        "usage": {
            "input": 100, "output": 20, "cacheRead": 5, "cacheWrite": 1, "totalTokens": 126,
            "cost": {"total": 0.25},
        },
        **extra,
    }


def session_file(directory, session_id):
    for path in directory.glob("*.jsonl"):
        header = json.loads(path.read_text().splitlines()[0])
        if header["id"] == session_id and header["cwd"] == os.getcwd():
            return path
    return None


argv = sys.argv[1:]
agent_dir = Path(os.environ["PI_CODING_AGENT_DIR"])
session_id = argv[argv.index("--session-id") + 1]
sessions = Path(argv[argv.index("--session-dir") + 1])
existing = session_file(sessions, session_id)
prompt = sys.stdin.read()
Path(os.environ["FAKE_PI_RECORD"]).write_text(json.dumps({
    "argv": argv,
    "cwd": os.getcwd(),
    "stdin": prompt,
    "models": json.loads((agent_dir / "models.json").read_text()),
    "mcp": json.loads((agent_dir / "mcp.json").read_text()),
    "env": {key: value for key, value in os.environ.items() if key.startswith(("AGENTENV_MCP_", "PI_"))},
    "attachments": {arg: Path(arg[1:]).read_bytes().decode() for arg in argv if arg.startswith("@")},
    "history": existing.read_text().splitlines()[1:] if existing else None,
}))
if existing is None:
    existing = sessions / f"2026-01-01T00-00-00-000Z_{session_id}.jsonl"
    existing.write_text(json.dumps({"type": "session", "version": 3, "id": session_id, "cwd": os.getcwd()}) + "\n")


def persist(message):
    with existing.open("a") as handle:
        handle.write(json.dumps({"type": "message", "message": message}) + "\n")


scenario = os.environ.get("FAKE_PI_SCENARIO", "ok")
emit({"type": "session", "version": 3, "id": session_id, "cwd": os.getcwd()})
emit({"type": "agent_start"})
if scenario == "crash":
    sys.stderr.write("boom: provider not configured\n")
    sys.exit(1)
if scenario == "hang":
    time.sleep(60)
user = {"role": "user", "content": prompt}
emit({"type": "message_end", "message": user})
persist(user)
emit({"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "x"}})
emit({"type": "message_end", "message": assistant("", stop_reason="toolUse")})
emit({"type": "tool_execution_start", "toolCallId": "t1", "toolName": "bash", "args": {"command": "ls"}})
if os.environ.get("FAKE_PI_WRITE"):
    name, text = os.environ["FAKE_PI_WRITE"].split("=", 1)
    Path(name).write_text(text)
emit({"type": "tool_execution_end", "toolCallId": "t1", "toolName": "bash", "result": {"content": []}, "isError": False})
tool_result = {"role": "toolResult", "toolCallId": "t1", "content": [{"type": "text", "text": "ok"}]}
persist(tool_result)
emit({"type": "message_end", "message": tool_result})
if scenario == "error":
    emit({"type": "message_end", "message": assistant("", stop_reason="error", errorMessage="429 rate limited")})
    emit({"type": "agent_settled", "aborted": False})
    sys.exit(1)
final = assistant("Done.")
emit({"type": "message_end", "message": final})
persist(final)
emit({"type": "agent_settled", "aborted": False})
