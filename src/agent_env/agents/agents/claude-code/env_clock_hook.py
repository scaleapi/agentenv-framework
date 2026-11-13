"""The CLI's PostToolUse and PostToolUseFailure hook while the env's clock is armed: the env's time now, as context
ending the tool's result (``env_clock``). Any failure prints nothing, so a clock blip never fails a tool call; the model can still call
get_time itself."""

from __future__ import annotations

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_clock import (  # noqa: E402
    ENV_CLOCK_SERVER_ENV, ENV_CLOCK_TOOL_NAME, HOOK_EVENTS, ClockServer, failure_reason, read_env_clock, turn_note,
)


def main(stdin: str, environ: dict[str, str]) -> str:
    """What the hook prints for the CLI's hook input ``stdin``: its JSON output, or nothing."""
    configured = environ.get(ENV_CLOCK_SERVER_ENV)
    if not configured:
        return ""
    try:
        given = json.loads(stdin or "{}") or {}
    except ValueError:
        given = {}
    tool = given.get("tool_name") or ""
    # The output names the event the CLI ran the hook for, which it checks.
    event = given.get("hook_event_name") if given.get("hook_event_name") in HOOK_EVENTS else HOOK_EVENTS[0]
    # A get_time call's own result already is the reading.
    if tool == ENV_CLOCK_TOOL_NAME or tool.endswith(f"__{ENV_CLOCK_TOOL_NAME}"):
        return ""
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
