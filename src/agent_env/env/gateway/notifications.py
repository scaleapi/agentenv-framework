"""Notifications for an agent, due on the gateway's virtual clock (urn:agentenv:notifications/v1).

Two kinds, each for one recipient (the AgentEnv-Role it goes to):
- checks a role schedules for itself with the schedule_check / list_scheduled_checks / cancel_scheduled_check
  tools, which only the roles a task names in `configure` see;
- notifications a task's triggers send with the `notify` action, through a channel the task declares, to the
  solver's role unless the action names another.

Nothing here wakes anyone: prompt_agent `take`s what has come due for its agent's role after each turn and opens the
agent's next message with it, and a harness (a post-turn hook) may read `state` to learn when the next one is due and
move the clock there. Unrelated to A2A push notifications.

`take` hands out both as one shape, with where each came from as data, so a consumer never branches on the source:
    {"id", "to", "at", "times", "source": "scheduled_check" | "trigger", "source_id": the check's or trigger's id,
     "channel": the label the agent sees ("your scheduled check", or the trigger's channel), "parts": [one text part]}
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import Any, Callable

from mcp.server.fastmcp.tools import Tool as FastMCPTool
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata
from mcp.types import CallToolResult, TextContent, ToolAnnotations

from .caller import Caller
from .clock import Clock, ClockError, _iso, _parse_duration, _parse_rfc3339
from .constants import DEFAULT_ROLE, WILDCARD

SCHEDULE_CHECK_TOOL_NAME = "schedule_check"
LIST_SCHEDULED_CHECKS_TOOL_NAME = "list_scheduled_checks"
CANCEL_SCHEDULED_CHECK_TOOL_NAME = "cancel_scheduled_check"
CHECK_TOOL_NAMES = (SCHEDULE_CHECK_TOOL_NAME, LIST_SCHEDULED_CHECKS_TOOL_NAME, CANCEL_SCHEDULED_CHECK_TOOL_NAME)

CHECK_CHANNEL = "your scheduled check"  # the channel every check is delivered through; no trigger may use it
CHECK_SOURCE, TRIGGER_SOURCE = "scheduled_check", "trigger"

DEFAULT_MAX_ACTIVE = 10
DEFAULT_MIN_INTERVAL = "PT1H"
_MAX_NOTE_CHARS = 2000
_MAX_CHANNEL_CHARS = 60
_MAX_ROLE_CHARS = 100
_MAX_TAKE_ID_CHARS = 128
_KEPT_TAKES = 64  # take results kept for retries, oldest dropped first
_CONFIG_FIELDS = ("roles", "max_active", "min_interval")
_HISTORY = 50  # finished and cancelled checks state() still shows

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


class NotificationError(ValueError):
    """A bad configuration or tool argument (a 400 on the harness routes, a tool error for the agent)."""


def _when(text: Any, now: datetime, field: str) -> datetime:
    """An RFC 3339 instant, or an ISO 8601 duration counted from now."""
    if not isinstance(text, str) or not text:
        raise NotificationError(f"{field} must be an RFC 3339 time or an ISO 8601 duration")
    try:
        return now + _parse_duration(text) if text.startswith("P") else _parse_rfc3339(text)
    except ClockError as e:
        raise NotificationError(f"{field}: {e}")
    except OverflowError:
        raise NotificationError(f"{field} is too far away")


class Notifications:
    """The gateway's record of scheduled checks and trigger notifications; threadsafe. `clock` returns the gateway's
    current Clock, read at each use so a swapped clock (tests) is followed."""

    def __init__(self, clock: Callable[[], Clock]):
        self._clock_of = clock
        self._lock = threading.Lock()
        self._roles: tuple[str, ...] = ()  # who sees the check tools; none until a task names them
        self._max_active = DEFAULT_MAX_ACTIVE
        self._min_interval = _parse_duration(DEFAULT_MIN_INTERVAL)
        self._checks: dict[str, dict] = {}
        self._queue: list[dict] = []  # what triggers sent, pending until taken
        self._takes: dict[tuple[str, str], dict] = {}  # recent take results by (to, take_id), so a retry gets the same batch
        self._seq = 0

    @property
    def roles(self) -> tuple[str, ...]:
        return self._roles

    @property
    def enabled(self) -> bool:
        """Whether any role has the scheduled-check tools."""
        return bool(self._roles)

    def configure(self, body: Any) -> dict:
        """Name the roles that get the scheduled-check tools, with their limits; `roles: []` (the default) turns them
        off. Notifications need no switch."""
        if not isinstance(body, dict):
            raise NotificationError("body must be a JSON object")
        unknown = sorted(set(body) - set(_CONFIG_FIELDS))
        if unknown:
            raise NotificationError(f"config takes {list(_CONFIG_FIELDS)}, not {unknown}")
        roles = _roles(body.get("roles", []))
        max_active = body.get("max_active", DEFAULT_MAX_ACTIVE)
        if isinstance(max_active, bool) or not isinstance(max_active, int) or max_active < 1:
            raise NotificationError("max_active must be a positive integer")
        try:
            min_interval = _parse_duration(body.get("min_interval", DEFAULT_MIN_INTERVAL))
        except (ClockError, OverflowError) as e:
            raise NotificationError(f"min_interval: {e}")
        if min_interval <= timedelta(0):
            raise NotificationError("min_interval must be a positive duration")
        with self._lock:
            self._roles, self._max_active, self._min_interval = roles, max_active, min_interval
            # a role left out loses its tools, so the checks it can no longer cancel go too
            for check in self._checks.values():
                if check["status"] == "active" and check["to"] not in roles:
                    check["status"] = "cancelled"
            self._prune()
        return self.state()

    def clear(self) -> dict:
        with self._lock:
            self._roles = ()
            self._checks.clear()
            self._queue.clear()
            self._takes.clear()
        return {"ok": True}

    @property
    def _clock(self) -> Clock:
        return self._clock_of()

    def _now(self) -> datetime:
        now = self._clock.now()
        if now is None:
            raise NotificationError("this environment's clock is not set, so checks can't be scheduled")
        return now

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    # ---- agent tools ----

    def schedule(self, note: Any, at: Any = None, every: Any = None, until: Any = None, *,
                 to: str = DEFAULT_ROLE) -> dict:
        """A check for the role `to` that schedules it, refused for a role the config doesn't name."""
        if to not in self._roles:
            raise NotificationError("scheduled checks are not available here")
        if not isinstance(note, str) or not note.strip():
            raise NotificationError("note must say what to do when the check comes due")
        if len(note) > _MAX_NOTE_CHARS:
            raise NotificationError(f"note must be at most {_MAX_NOTE_CHARS} characters")
        now = self._now()
        step = None
        if every is not None:
            try:
                step = _parse_duration(every)
            except (ClockError, OverflowError) as e:
                raise NotificationError(f"every: {e}")
            if step < self._min_interval:
                raise NotificationError(f"every must be at least {_iso_duration(self._min_interval)} apart")
        if at is None and step is None:
            raise NotificationError("give at (when) or every (how often)")
        first = _when(at, now, "at") if at is not None else _when(every, now, "every")
        if step is not None:
            try:
                first + step  # a repeat that can't be counted forward would break every later take
            except OverflowError:
                raise NotificationError("every is too long")
        if first <= now:
            raise NotificationError(f"at {_iso(first)} has already passed; it is now {_iso(now)}")
        end = _when(until, now, "until") if until is not None else None
        if end is not None and step is None:
            raise NotificationError("until only applies with every")
        if end is not None and end < first:
            raise NotificationError("until is before the first check")
        with self._lock:
            if sum(c["status"] == "active" and c["to"] == to for c in self._checks.values()) >= self._max_active:
                raise NotificationError(f"you already have {self._max_active} scheduled checks; cancel one first")
            cid = self._next_id("check")
            self._checks[cid] = {"check_id": cid, "to": to, "note": note, "next": first, "every": step, "until": end,
                                 "created_at": now, "status": "active", "fired": 0}
            # only what the gateway decided: the agent wrote the note, and the trajectory already logs it
            return {k: v for k, v in self._public(self._checks[cid]).items() if k != "note"}

    def list(self, *, to: str = DEFAULT_ROLE) -> dict:
        """The role's own checks still to come."""
        with self._lock:
            return {"checks": [self._public(c) for c in self._checks.values()
                               if c["status"] == "active" and c["to"] == to]}

    def cancel(self, check_id: Any, *, to: str = DEFAULT_ROLE) -> dict:
        """Cancel one of the role's own checks."""
        with self._lock:
            check = self._checks.get(check_id) if isinstance(check_id, str) else None
            if check is None or check["status"] != "active" or check["to"] != to:
                raise NotificationError(f"no active scheduled check {check_id!r}")
            check["status"] = "cancelled"
            self._prune()
            return {"cancelled": check_id}

    @staticmethod
    def _public(check: dict) -> dict:
        out = {"check_id": check["check_id"], "note": check["note"], "next": _iso(check["next"])}
        if check["every"]:
            out["every"] = _iso_duration(check["every"])
        if check["until"]:
            out["until"] = _iso(check["until"])
        return out

    # ---- trigger action ----

    def notify(self, channel: str, text: str, trigger_id: str | None = None, to: str = DEFAULT_ROLE) -> dict:
        """Queue a notification for the role `to` at the current virtual time (none while the clock is unset: due at
        once). Receiving one needs no tools, so `to` need not be among the config's roles."""
        now = self._clock.now()
        with self._lock:
            item = {"id": self._next_id("notification"), "to": to, "channel": channel, "text": text,
                    "trigger_id": trigger_id, "at": _iso(now) if now else None, "status": "pending", "_at": now}
            self._queue.append(item)
            return _public_note(item)

    # ---- harness ----

    def state(self, to: Any = None) -> dict:
        """The config, every check (with its recipient and status) and the pending notifications: those for the
        recipient `to`, or for every recipient when it is None. `next_due_at` is the earliest of their active checks'
        next time and pending notifications' time; one queued while the clock was unset is due now."""
        to = None if to is None else _recipient(to)
        now = self._clock.now()
        with self._lock:
            checks = [c for c in self._checks.values() if to is None or c["to"] == to]
            queued = [n for n in self._queue if n["status"] == "pending" and (to is None or n["to"] == to)]
            times = [c["next"] for c in checks if c["status"] == "active"] + [n["_at"] or now for n in queued]
            nxt = min((t for t in times if t is not None), default=None)
            return {"roles": list(self._roles), "max_active": self._max_active,
                    "min_interval": _iso_duration(self._min_interval), "next_due_at": _iso(nxt) if nxt else None,
                    "checks": [{"check_id": c["check_id"], "to": c["to"], **self._public(c), "status": c["status"],
                                "fired": c["fired"]} for c in checks],
                    "pending_notifications": [_public_note(n) for n in queued]}

    def _prune(self) -> None:
        """Drop delivered notifications and all but the last _HISTORY finished checks; the caller holds the lock."""
        finished = [cid for cid, c in self._checks.items() if c["status"] != "active"]
        for cid in finished[:max(0, len(finished) - _HISTORY)]:
            del self._checks[cid]
        self._queue = [n for n in self._queue if n["status"] == "pending"]

    def take(self, body: Any = None) -> dict:
        """Everything due for the recipient `to` (default: the solver's role) by `until` (default: now), oldest
        first, as items of the one shape the module docstring gives; each is handed out once. A repeating check moves
        to its next time after `until` and says how many times it came due (`times`), so a long jump wakes the agent
        once. A `take_id` makes it safe to retry: the same id for the same recipient gets the same batch back instead
        of an empty one."""
        body = body if isinstance(body, dict) else {}
        to = _recipient(body.get("to", DEFAULT_ROLE))
        take_id = body.get("take_id")
        if take_id is not None and (not isinstance(take_id, str) or not take_id or len(take_id) > _MAX_TAKE_ID_CHARS):
            raise NotificationError(f"take_id must be a non-empty string of at most {_MAX_TAKE_ID_CHARS} characters")
        if take_id is not None:
            with self._lock:
                seen = self._takes.get((to, take_id))
            if seen is not None:
                return seen
        now = self._clock.now()
        until = _when(body["until"], now, "until") if body.get("until") is not None and now else now
        due: list[tuple[datetime | None, dict]] = []
        with self._lock:
            # work the whole batch out before marking anything handed out
            checks = [(c, *_after(c, until)) for c in self._checks.values()
                      if c["status"] == "active" and c["to"] == to and until is not None and c["next"] <= until]
            notes = [n for n in self._queue if n["status"] == "pending" and n["to"] == to
                     and (until is None or n["_at"] is None or n["_at"] <= until)]
            for check, times, nxt in checks:
                due.append((check["next"], _item(check["check_id"], to, _iso(check["next"]), times, CHECK_SOURCE,
                                                 check["check_id"], CHECK_CHANNEL, check["note"])))
                check["fired"] += 1
                if nxt is None:
                    check["status"] = "done"
                else:
                    check["next"] = nxt
            for note in notes:
                note["status"] = "delivered"
                due.append((note["_at"], _item(note["id"], to, note["at"], 1, TRIGGER_SOURCE, note["trigger_id"],
                                               note["channel"], note["text"])))
            self._prune()
            due.sort(key=lambda d: (d[0] is not None, d[0]))
            result = {"to": to, "items": [item for _, item in due], "until": _iso(until) if until else None}
            if take_id is not None:
                self._takes[(to, take_id)] = result
                while len(self._takes) > _KEPT_TAKES:
                    del self._takes[next(iter(self._takes))]
        return result


def _item(item_id: str, to: str, at: str | None, times: int, source: str, source_id: str | None, channel: str,
          text: str) -> dict:
    """One thing `take` hands out, the same shape whatever its source."""
    return {"id": item_id, "to": to, "at": at, "times": times, "source": source, "source_id": source_id,
            "channel": channel, "parts": [{"kind": "text", "text": text}]}


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


def _iso_duration(td: timedelta) -> str:
    """A duration in the ISO 8601 form the tools take (PT3H, P1DT6H, PT0.5S), so what they echo can be sent back."""
    hours, rest = divmod(td.seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    secs = f"{seconds}.{td.microseconds:06d}".rstrip("0") if td.microseconds else (str(seconds) if seconds else "")
    clock = (f"{hours}H" if hours else "") + (f"{minutes}M" if minutes else "") + (f"{secs}S" if secs else "")
    return "P" + (f"{td.days}D" if td.days else "") + (f"T{clock}" if clock else "") if td.days or clock else "PT0S"


def _roles(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise NotificationError('roles must be a list of role names, e.g. ["default"]; [] turns scheduled checks off')
    roles = tuple(dict.fromkeys(_recipient(r, "roles") for r in value))
    if WILDCARD in roles:
        raise NotificationError(f"roles names each role; {WILDCARD!r} is not one")
    return roles


def _recipient(value: Any, field: str = "to") -> str:
    """A role name, as the AgentEnv-Role header gives it."""
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_ROLE_CHARS:
        raise NotificationError(f"{field} must name a role (a non-empty string of at most {_MAX_ROLE_CHARS} "
                                "characters), e.g. 'default'")
    return value.strip()


def _public_note(note: dict) -> dict:
    return {k: v for k, v in note.items() if not k.startswith("_")}


def _tool_error(text: str) -> CallToolResult:
    return CallToolResult(isError=True, content=[TextContent(type="text", text=text)])


def build_check_tools(notifications: Notifications, make_arg_model: Callable[[str, dict], Any]) -> list[FastMCPTool]:
    """The agent-facing tools, built by hand like get_time so a backing server's tool can't silently shadow them. Each
    acts for the calling role (the AgentEnv-Role the call came with): its checks are its own."""

    async def schedule_check(note: str, at: str | None = None, every: str | None = None,
                             until: str | None = None) -> dict | CallToolResult:
        try:
            return notifications.schedule(note, at=at, every=every, until=until, to=Caller.current().role)
        except NotificationError as e:
            return _tool_error(str(e))

    async def list_scheduled_checks() -> dict:
        return notifications.list(to=Caller.current().role)

    async def cancel_scheduled_check(check_id: str) -> dict | CallToolResult:
        try:
            return notifications.cancel(check_id, to=Caller.current().role)
        except NotificationError as e:
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


def validate_recipient(to: Any, where: str) -> None:
    """A notify action's `to`: the role the notification goes to."""
    try:
        _recipient(to)
    except NotificationError as e:
        raise ValueError(f"{where}: {e}")


def validate_channel(channel: Any, where: str) -> None:
    """A notify action's channel: the short source label the agent sees, e.g. 'Gmail' or 'Curbfinch alert'."""
    if not isinstance(channel, str) or not channel.strip():
        raise ValueError(f"{where}: channel must be a non-empty string, e.g. 'Gmail' or 'Curbfinch alert'")
    if len(channel) > _MAX_CHANNEL_CHARS:
        raise ValueError(f"{where}: channel must be at most {_MAX_CHANNEL_CHARS} characters")
    if channel.strip().casefold() == CHECK_CHANNEL.casefold():  # a trigger can't pass itself off as the agent's own
        raise ValueError(f"{where}: channel {CHECK_CHANNEL!r} is reserved for the agent's own scheduled checks; "
                         "name where the notification comes from, e.g. 'Gmail'")
