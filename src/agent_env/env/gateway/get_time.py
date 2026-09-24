"""Agent-facing clock read for urn:agentenv:clock/v1.

Registered only while the clock is armed, so an unarmed gateway is a zero-delta no-op. The wording
never says the clock is simulated — the agent is told its own host clock is not authoritative here,
which is the actionable part without disclosing that the environment is synthetic.
"""
from __future__ import annotations

from typing import Any, Callable

from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata
from mcp.types import ToolAnnotations

from .clock import Clock, ClockError, _iso

GET_TIME_TOOL_NAME = "get_time"
GET_TIME_TOOL_DESCRIPTION = (
    "Returns the current date and time for this environment. The clock on the machine you are "
    "running on is not synchronized with this environment, so do not use it and do not rely on "
    "your own assumption about today's date. Call this tool and derive from its value every "
    "absolute date, time or timestamp you pass to another tool, and every relative phrase you "
    'resolve ("today", "this week", "overdue", "the last 30 days"). A date taken from any other '
    "source can silently match no records on a read, or file a record at the wrong time on a "
    "write. Time advances while you work, so re-read this before each write rather than reusing "
    "an earlier value. Inside a raw SQL or free-form query string, prefer the literal date this "
    "tool returns over CURRENT_DATE or NOW(), so the query is reproducible."
)

_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

# readOnlyHint is load-bearing: without it, TriggerEngine._is_readonly makes every clock read
# re-evaluate every armed state trigger. Shared with the descriptor so the two cannot drift.
_ANNOTATIONS = ToolAnnotations(readOnlyHint=True)


def read_time(clock: Clock) -> dict[str, str]:
    """Second precision via the clock's own `_iso`, so get_time, /clock/time and the trajectory's
    virtual_time stamps read byte-identical."""
    now = clock.now()
    if now is None:
        raise ClockError("time service unavailable")
    return {"current_time": _iso(now.replace(microsecond=0))}


def build_get_time_tool(clock: Clock, make_arg_model: Callable[[str, dict], Any]) -> FastMCPTool:
    """Built by hand: ToolManager.add_tool only warns on a collision and would silently shadow this."""

    async def get_time() -> dict[str, str]:
        return read_time(clock)

    return FastMCPTool(
        fn=get_time,
        name=GET_TIME_TOOL_NAME,
        description=GET_TIME_TOOL_DESCRIPTION,
        parameters=_SCHEMA,
        fn_metadata=FuncMetadata(arg_model=make_arg_model(GET_TIME_TOOL_NAME, _SCHEMA)),
        is_async=True,
        annotations=_ANNOTATIONS,
    )
