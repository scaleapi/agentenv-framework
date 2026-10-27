"""What wakes an agent between its turns (urn:agentenv:wakeups/v1).

Two sources, both on the gateway's virtual clock:
- checks the agent schedules for itself with the schedule_check / list_scheduled_checks / cancel_scheduled_check
  tools, which exist only once a task turns them on (`configure`);
- notifications a task's triggers send it with the `notify_agent` action, through a channel the task declares.

Nothing here wakes anyone: a harness (a post-turn hook) reads `state` to learn when the next one is due, moves the
clock there, and `take`s what has come due to put it in front of the agent.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any, Callable

from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from .clock import Clock, ClockError, _iso, _parse_duration, _parse_rfc3339

SCHEDULE_CHECK_TOOL_NAME = "schedule_check"
LIST_SCHEDULED_CHECKS_TOOL_NAME = "list_scheduled_checks"
CANCEL_SCHEDULED_CHECK_TOOL_NAME = "cancel_scheduled_check"
WAKEUP_TOOL_NAMES = (SCHEDULE_CHECK_TOOL_NAME, LIST_SCHEDULED_CHECKS_TOOL_NAME, CANCEL_SCHEDULED_CHECK_TOOL_NAME)

DEFAULT_MAX_ACTIVE = 10
DEFAULT_MIN_INTERVAL = "PT1H"
_MAX_NOTE_CHARS = 2000
_MAX_CHANNEL_CHARS = 60

SCHEDULE_CHECK_DESCRIPTION = (
    "Schedule a check-in for yourself. You only run when you are messaged, so to look at something again later "
    "(a reply you are waiting for, a listing to watch, a deadline), schedule a check: when it comes due you are "
    "woken with your note, so write the note so it tells you what to do. Give `at` (a time, or a duration from now "
    "such as PT6H) and/or `every` (a duration such as P1D, to repeat), with an optional `until`. Times are in this "
    "environment's clock (see get_time); a time without a UTC offset is refused."
)
_SCHEDULE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "note": {"type": "string", "description": "What to do when the check comes due."},
        "at": {"type": "string", "description": "When: an RFC 3339 time with its offset (2027-02-02T18:00:00-08:00), "
                                                "or an ISO 8601 duration from now (PT6H)."},
        "every": {"type": "string", "description": "Repeat this often: an ISO 8601 duration (PT6H, P1D)."},
        "until": {"type": "string", "description": "Stop repeating after this time (RFC 3339) or duration from now."},
    },
    "required": ["note"],
    "additionalProperties": False,
}
_LIST_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
_CANCEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"check_id": {"type": "string"}},
    "required": ["check_id"],
    "additionalProperties": False,
}


class WakeupError(ValueError):
    """A bad configuration or tool argument (a 400 on the harness routes, a tool error for the agent)."""


def _when(text: Any, now: datetime, field: str) -> datetime:
    """An RFC 3339 instant, or an ISO 8601 duration counted from now."""
    if not isinstance(text, str) or not text:
        raise WakeupError(f"{field} must be an RFC 3339 time or an ISO 8601 duration")
    try:
        return now + _parse_duration(text) if text.startswith("P") else _parse_rfc3339(text)
    except ClockError as e:
        raise WakeupError(f"{field}: {e}")
    except OverflowError:
        raise WakeupError(f"{field} is too far away")


class Wakeups:
    """The gateway's record of scheduled checks and agent notifications; threadsafe. `clock` returns the gateway's
    current Clock, read at each use so a swapped clock (tests) is followed."""

    def __init__(self, clock: Callable[[], Clock]):
        self._clock_of = clock
        self._lock = threading.Lock()
        self._enabled = False
        self._max_active = DEFAULT_MAX_ACTIVE
        self._min_interval = _parse_duration(DEFAULT_MIN_INTERVAL)
        self._checks: dict[str, dict] = {}
        self._notifications: list[dict] = []
        self._seq = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def configure(self, body: Any) -> dict:
        """Turn scheduled checks on or off, with their limits. Notifications need no switch."""
        if not isinstance(body, dict):
            raise WakeupError("body must be a JSON object")
        enabled = body.get("scheduled_checks", True)
        if not isinstance(enabled, bool):
            raise WakeupError("scheduled_checks must be a boolean")
        max_active = body.get("max_active", DEFAULT_MAX_ACTIVE)
        if isinstance(max_active, bool) or not isinstance(max_active, int) or max_active < 1:
            raise WakeupError("max_active must be a positive integer")
        try:
            min_interval = _parse_duration(body.get("min_interval", DEFAULT_MIN_INTERVAL))
        except (ClockError, OverflowError) as e:
            raise WakeupError(f"min_interval: {e}")
        if min_interval <= timedelta(0):
            raise WakeupError("min_interval must be a positive duration")
        with self._lock:
            self._enabled, self._max_active, self._min_interval = enabled, max_active, min_interval
        return self.state()

    def clear(self) -> dict:
        with self._lock:
            self._enabled = False
            self._checks.clear()
            self._notifications.clear()
        return {"ok": True}

    @property
    def _clock(self) -> Clock:
        return self._clock_of()

    def _now(self) -> datetime:
        now = self._clock.now()
        if now is None:
            raise WakeupError("this environment's clock is not set, so checks can't be scheduled")
        return now

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    # ---- agent tools ----

    def schedule(self, note: Any, at: Any = None, every: Any = None, until: Any = None) -> dict:
        if not self._enabled:
            raise WakeupError("scheduled checks are not available here")
        if not isinstance(note, str) or not note.strip():
            raise WakeupError("note must say what to do when the check comes due")
        if len(note) > _MAX_NOTE_CHARS:
            raise WakeupError(f"note must be at most {_MAX_NOTE_CHARS} characters")
        now = self._now()
        step = None
        if every is not None:
            try:
                step = _parse_duration(every)
            except (ClockError, OverflowError) as e:
                raise WakeupError(f"every: {e}")
            if step < self._min_interval:
                raise WakeupError(f"every must be at least {self._min_interval} apart")
        if at is None and step is None:
            raise WakeupError("give at (when) or every (how often)")
        first = _when(at, now, "at") if at is not None else _when(every, now, "every")
        if step is not None:
            try:
                first + step  # a repeat that can't be counted forward would break every later take
            except OverflowError:
                raise WakeupError("every is too long")
        if first <= now:
            raise WakeupError(f"at {_iso(first)} has already passed; it is now {_iso(now)}")
        end = _when(until, now, "until") if until is not None else None
        if end is not None and step is None:
            raise WakeupError("until only applies with every")
        if end is not None and end < first:
            raise WakeupError("until is before the first check")
        with self._lock:
            if sum(c["status"] == "active" for c in self._checks.values()) >= self._max_active:
                raise WakeupError(f"you already have {self._max_active} scheduled checks; cancel one first")
            cid = self._next_id("check")
            self._checks[cid] = {"check_id": cid, "note": note, "next": first, "every": step, "until": end,
                                 "created_at": now, "status": "active", "fired": 0}
            return self._public(self._checks[cid])

    def list(self) -> dict:
        with self._lock:
            return {"checks": [self._public(c) for c in self._checks.values() if c["status"] == "active"]}

    def cancel(self, check_id: Any) -> dict:
        with self._lock:
            check = self._checks.get(check_id) if isinstance(check_id, str) else None
            if check is None or check["status"] != "active":
                raise WakeupError(f"no active scheduled check {check_id!r}")
            check["status"] = "cancelled"
            return {"cancelled": check_id}

    @staticmethod
    def _public(check: dict) -> dict:
        out = {"check_id": check["check_id"], "note": check["note"], "next": _iso(check["next"])}
        if check["every"]:
            out["every"] = str(check["every"])
        if check["until"]:
            out["until"] = _iso(check["until"])
        return out

    # ---- trigger action ----

    def notify(self, channel: str, text: str, trigger_id: str | None = None) -> dict:
        """Queue a notification for the agent at the current virtual time (wall time if the clock is unset)."""
        now = self._clock.now()
        with self._lock:
            item = {"id": self._next_id("note"), "channel": channel, "text": text, "trigger_id": trigger_id,
                    "at": _iso(now) if now else None, "status": "pending", "_at": now}
            self._notifications.append(item)
            return _public_note(item)

    # ---- harness ----

    def state(self) -> dict:
        with self._lock:
            active = [c for c in self._checks.values() if c["status"] == "active"]
            pending = [_public_note(n) for n in self._notifications if n["status"] == "pending"]
            nxt = min((c["next"] for c in active), default=None)
            return {"scheduled_checks": self._enabled, "max_active": self._max_active,
                    "min_interval": str(self._min_interval), "next_check_at": _iso(nxt) if nxt else None,
                    "checks": [{**self._public(c), "status": c["status"], "fired": c["fired"]}
                               for c in self._checks.values()],
                    "pending_notifications": pending}

    def take(self, body: Any = None) -> dict:
        """Everything due by `until` (default: now), oldest first; each is handed out once. A repeating check moves to
        its next time after `until` and says how many times it came due, so a long jump wakes the agent once."""
        body = body if isinstance(body, dict) else {}
        now = self._clock.now()
        until = _when(body["until"], now, "until") if body.get("until") is not None and now else now
        due: list[tuple[datetime | None, dict]] = []
        with self._lock:
            # work the whole batch out before marking anything handed out
            checks = [(c, *_after(c, until)) for c in self._checks.values()
                      if c["status"] == "active" and until is not None and c["next"] <= until]
            notes = [n for n in self._notifications
                     if n["status"] == "pending" and (until is None or n["_at"] is None or n["_at"] <= until)]
            for check, times, nxt in checks:
                due.append((check["next"], {"kind": "scheduled_check", "check_id": check["check_id"],
                                            "note": check["note"], "at": _iso(check["next"]), "times": times}))
                check["fired"] += 1
                if nxt is None:
                    check["status"] = "done"
                else:
                    check["next"] = nxt
            for note in notes:
                note["status"] = "delivered"
                due.append((note["_at"], {"kind": "notification", "id": note["id"], "channel": note["channel"],
                                          "text": note["text"], "at": note["at"], "trigger_id": note["trigger_id"]}))
        due.sort(key=lambda d: (d[0] is not None, d[0]))
        return {"items": [item for _, item in due], "until": _iso(until) if until else None}


def _after(check: dict, until: datetime) -> tuple[int, datetime | None]:
    """How many times a due check came due by `until`, and its next time after that (None: it is done). Counted,
    not stepped, so a long jump over a short `every` costs nothing."""
    step, end = check["every"], check["until"]
    if not step:
        return 1, None
    times = (until - check["next"]) // step + 1
    if end is not None:
        times = min(times, (end - check["next"]) // step + 1)
    try:
        nxt = check["next"] + step * times
    except OverflowError:
        return times, None
    return times, nxt if end is None or nxt <= end else None


def _public_note(note: dict) -> dict:
    return {k: v for k, v in note.items() if not k.startswith("_")}


def _tool_error(text: str) -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text=text)])


def build_wakeup_tools(wakeups: Wakeups, make_arg_model: Callable[[str, dict], Any]) -> list[FastMCPTool]:
    """The agent-facing tools, built by hand like get_time so a backing server's tool can't silently shadow them."""

    async def schedule_check(note: str, at: str | None = None, every: str | None = None,
                             until: str | None = None) -> dict | CallToolResult:
        try:
            return wakeups.schedule(note, at=at, every=every, until=until)
        except WakeupError as e:
            return _tool_error(str(e))

    async def list_scheduled_checks() -> dict:
        return wakeups.list()

    async def cancel_scheduled_check(check_id: str) -> dict | CallToolResult:
        try:
            return wakeups.cancel(check_id)
        except WakeupError as e:
            return _tool_error(str(e))

    specs = [
        (SCHEDULE_CHECK_TOOL_NAME, schedule_check, SCHEDULE_CHECK_DESCRIPTION, _SCHEDULE_SCHEMA, False),
        (LIST_SCHEDULED_CHECKS_TOOL_NAME, list_scheduled_checks, "Your scheduled checks that are still to come.",
         _LIST_SCHEMA, True),
        (CANCEL_SCHEDULED_CHECK_TOOL_NAME, cancel_scheduled_check, "Cancel one of your scheduled checks.",
         _CANCEL_SCHEMA, False),
    ]
    return [FastMCPTool(fn=fn, name=name, description=description, parameters=schema,
                        fn_metadata=FuncMetadata(arg_model=make_arg_model(name, schema)), is_async=True,
                        annotations=ToolAnnotations(readOnlyHint=read_only))
            for name, fn, description, schema, read_only in specs]


def validate_channel(channel: Any, where: str) -> None:
    """A notify_agent channel: the short source label the agent sees, e.g. 'Gmail' or 'Curbfinch alert'."""
    if not isinstance(channel, str) or not channel.strip():
        raise ValueError(f"{where}: channel must be a non-empty string, e.g. 'Gmail' or 'Curbfinch alert'")
    if len(channel) > _MAX_CHANNEL_CHARS:
        raise ValueError(f"{where}: channel must be at most {_MAX_CHANNEL_CHARS} characters")
