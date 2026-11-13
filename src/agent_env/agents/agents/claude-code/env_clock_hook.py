"""The CLI's PostToolUse and PostToolUseFailure hook while the task env's clock is armed: the env's time now, as context
ending the tool's result, read through the protocol's env clock (``agentenv_protocol.a2a_agent.env_clock``). Any
failure prints nothing, so a clock blip never fails a tool call; the model can still call get_time itself.

``hook_settings`` is the CLI's --settings that runs it, on the command line rather than in the user settings, so a
snapshot of the CLI's state never carries it to an agent with no clock or no such script. The module loads only the
standard library until the hook runs."""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import sys

# After every tool call, whether it succeeded or failed: a long command that fails takes as much of the clock.
HOOK_EVENTS = ("PostToolUse", "PostToolUseFailure")
# How long the CLI lets the hook run, a little longer than the protocol's read (READ_TIMEOUT_S).
HOOK_TIMEOUT_S = 10
_GET_TIME = "get_time"


def hook_settings() -> str:
    command = f"{shlex.quote(sys.executable)} {shlex.quote(os.path.abspath(__file__))}"
    entry = {"matcher": "", "hooks": [{"type": "command", "command": command, "timeout": HOOK_TIMEOUT_S}]}
    return json.dumps({"hooks": {event: [entry] for event in HOOK_EVENTS}})


def main(stdin: str, environ: dict[str, str]) -> str:
    """What the hook prints for the CLI's hook input ``stdin``: its JSON output, or nothing."""
    from agentenv_protocol.a2a_agent.env_clock import (
        ENV_CLOCK_SERVER_ENV, ClockServer, failure_reason, read_env_clock, turn_note,
    )

    configured = environ.get(ENV_CLOCK_SERVER_ENV)
    if not configured:
        return ""
    try:
        given = json.loads(stdin or "{}") or {}
    except ValueError:
        given = {}
    tool = given.get("tool_name") or ""
    # A get_time call's own result already is the reading.
    if tool == _GET_TIME or tool.endswith(f"__{_GET_TIME}"):
        return ""
    # The output names the event the CLI ran the hook for, which it checks.
    event = given.get("hook_event_name") if given.get("hook_event_name") in HOOK_EVENTS else HOOK_EVENTS[0]
    try:
        current = asyncio.run(read_env_clock(ClockServer.from_env(configured)))
    except Exception as e:
        print(f"env clock: {failure_reason(e)}", file=sys.stderr)
        return ""
    return json.dumps({"hookSpecificOutput": {"hookEventName": event, "additionalContext": turn_note(current)}})


if __name__ == "__main__":
    out = main(sys.stdin.read(), dict(os.environ))
    if out:
        print(out)
