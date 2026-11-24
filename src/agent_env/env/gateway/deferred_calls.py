"""Calls a server's tool result declares (``AGENT_ENV_DEFERRED_CALLS_META_KEY``) for the gateway to make on that server
when its clock gets to their time, as the declaring call's role, each reported to the trigger engine as the call it
stands for, so what a server does later by itself (a scheduled message posted) provokes triggers as a live call would."""
from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from mcp.types import CallToolResult

from .caller import Caller
from .clock import ClockError, _iso, _parse_rfc3339
from .constants import AGENT_ENV_DEFERRED_CALLS_META_KEY, DEFERRED_CALL_DONE, DEFERRED_CALL_FAILED, DEFERRED_CALL_PENDING
from .triggers import _ECHO_DIGEST_CHARS, _VIRTUAL_RECHECK_S, _VIRTUAL_RESOLUTION_S, _result_text

logger = logging.getLogger(__name__)


class DeferredCalls:
    """The calls servers declared, by server and key. One made stays known, so its server declaring the key again is
    ignored; one that failed may be declared afresh."""

    RETRY_ATTEMPTS = 5  # a call a transport failure stopped is made again, so the tool a server defers is idempotent
    RETRY_BACKOFF_S = 1.0  # real seconds before the second attempt, doubled before each later one
    ERROR_TEXT_LIMIT = 300  # characters of a failed call's error kept in the log and the trajectory
    FINISHED_KEPT = 100  # the latest calls made or failed that state lists; every key made stays known for good

    def __init__(self, gateway):
        self._gw = gateway
        self._calls: dict[tuple[str, str], _Call] = {}  # the calls still to make
        self._made: set[tuple[str, str]] = set()
        self._finished: deque[tuple[tuple[str, str], _Call]] = deque(maxlen=self.FINISHED_KEPT)
        self._tasks: set[asyncio.Task] = set()
        self._closed = False

    def take(self, result: CallToolResult, server_url: str, role: str, report: bool = True) -> CallToolResult:
        """``result`` without its declaration, which no caller sees, each call it declares scheduled as ``role``'s
        unless the result is an error, and reported to the trigger engine once made unless ``report`` is False (the
        engine's own calls, which provoke no trigger, declare calls that provoke none either)."""
        meta = result.meta
        if not meta or AGENT_ENV_DEFERRED_CALLS_META_KEY not in meta:
            return result
        declared = meta[AGENT_ENV_DEFERRED_CALLS_META_KEY]
        rest = {k: v for k, v in meta.items() if k != AGENT_ENV_DEFERRED_CALLS_META_KEY}
        result = result.model_copy(update={"meta": rest or None})
        if result.isError:
            return result
        if not isinstance(declared, list):
            logger.warning(f"ignoring the deferred calls declared by {server_url}: {type(declared).__name__}, not a list")
            return result
        for entry in declared:
            self._schedule(entry, server_url, role, report)
        return result

    def state(self) -> list[dict]:
        """The latest calls finished, then each still to make, for GET /triggers/state. ``due`` marks one still to make
        whose time the gateway clock has reached: it, and the triggers it provokes, are under way, so a settle check
        waits for it. One not yet due keeps nothing waiting."""
        now = _now(self._gw._clock)
        return [{"key": key, "server": server, "at": _iso(call.at), "tool": call.tool, "role": call.role,
                 "status": call.status, "attempts": call.attempts,
                 "due": call.status == DEFERRED_CALL_PENDING and (call.at - now).total_seconds() < _VIRTUAL_RESOLUTION_S}
                for (server, key), call in [*self._finished, *self._calls.items()]]

    def in_state(self, state: dict) -> dict:
        """The trigger engine's ``state`` with the calls listed (GET /triggers/state), and the earliest still to make
        counted in its next_event_at, so a harness that jumps the clock to the next event stops at it."""
        upcoming = min((call.at for call in self._calls.values() if call.status == DEFERRED_CALL_PENDING), default=None)
        if upcoming is not None and state.get("next_event_at") is not None:
            upcoming = min(upcoming, _parse_rfc3339(state["next_event_at"]))
        return {**state, **({"next_event_at": _iso(upcoming)} if upcoming is not None else {}),
                "deferred_calls": self.state()}

    async def cancel(self) -> None:
        """Cancel the calls still pending, and schedule none declared from now on. The gateway's process lifespan calls
        it at shutdown before it cancels the scheduled trigger actions, so no call made then sets off a fire that
        outlives the child sessions."""
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _schedule(self, entry: Any, server_url: str, role: str, report: bool) -> None:
        if self._closed:
            return
        try:
            key, at, call, report_as = _parse(entry)
        except ValueError as e:
            logger.warning(f"ignoring a deferred call declared by {server_url}: {e}")
            return
        if (server_url, key) in self._calls or (server_url, key) in self._made:
            return
        record = _Call(at, call["tool"], call["arguments"], report_as, role, report)
        self._calls[(server_url, key)] = record
        task = asyncio.get_running_loop().create_task(self._make(key, record, server_url))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _make(self, key: str, record: _Call, server_url: str) -> None:
        """Wait for the call's time and make it, again after a transport failure, each attempt under the tool-call lock
        as a live call is, with a changelog window of its own read under that lock, so a write made between attempts is
        not the call's; report it to the trigger engine when it went through, then log it. An error result is final,
        and so is anything unexpected. A call finished leaves the calls to make for the latest finished, its key kept
        for good once made."""
        try:
            await self._until(record.at)
            result, error, windows = None, None, []
            while result is None and record.attempts < self.RETRY_ATTEMPTS:
                if record.attempts:
                    await asyncio.sleep(self.RETRY_BACKOFF_S * 2 ** (record.attempts - 1))
                record.attempts += 1
                async with self._gw._tool_call_guard():
                    before = self._gw._query_changelog_id()
                    try:
                        result = await self._gw._call_child_tool(server_url, record.tool, record.arguments,
                                                                 caller=Caller(role=record.role), report=record.report)
                    except Exception as e:
                        error = (str(e) or type(e).__name__)[:self.ERROR_TEXT_LIMIT]
                        logger.warning(f"deferred call {key!r} ({record.tool}) attempt {record.attempts} failed: {error}")
                    windows.append([before, self._gw._query_changelog_id()])
            ok = result is not None and not result.isError
            text = None if result is None else _result_text(result)
            if text is not None and not ok:
                error = text[:self.ERROR_TEXT_LIMIT]
            report_as = record.report_as
            reported = ok and record.report
            if reported:  # before the status, so no state read sees the call made and its trigger not yet at work
                self._gw._fire_triggers(record.role, report_as["tool"], report_as["arguments"], result)
            record.status = DEFERRED_CALL_DONE if ok else DEFERRED_CALL_FAILED
            await self._gw._log_event({
                "event_type": "deferred_call", "key": key, "at": _iso(record.at), "tool": record.tool,
                "report_as": report_as["tool"], "role": record.role, "ok": ok, "reported": reported,
                "attempts": record.attempts, **({} if text is None else {"result_sha256": _digest(text)}),
                "changelog_id_before": windows[0][0], "changelog_id_after": windows[-1][1], "changelog_windows": windows,
                **({} if ok else {"error": error})})
        except Exception:
            logger.exception(f"deferred call {key!r} ({record.tool}) failed")
            if record.status == DEFERRED_CALL_PENDING:
                record.status = DEFERRED_CALL_FAILED
        if record.status == DEFERRED_CALL_DONE:
            self._made.add((server_url, key))
        self._calls.pop((server_url, key), None)
        self._finished.append(((server_url, key), record))

    async def _until(self, at: datetime) -> None:
        """Wait until the gateway clock gets to ``at``: its virtual time while armed, held while frozen and read again
        whenever it changes (the trigger engine's wake), and wall UTC while unarmed."""
        clock, engine = self._gw._clock, self._gw._trigger_engine
        while True:
            left = (at - _now(clock)).total_seconds()
            if left < _VIRTUAL_RESOLUTION_S:
                return
            rate = clock.rate
            await engine._nap(min(left / rate, _VIRTUAL_RECHECK_S) if rate else _VIRTUAL_RECHECK_S)


def _parse(entry: Any) -> tuple[str, datetime, dict, dict]:
    """A declared call's key, time, call and report, or ValueError naming what in it breaks the contract."""
    if not isinstance(entry, dict) or not isinstance(entry.get("key"), str) or not entry["key"]:
        raise ValueError(f"an entry must be an object with a non-empty string key, got {type(entry).__name__}")
    key = entry["key"]
    try:
        at = _parse_rfc3339(entry.get("at"))
    except ClockError as e:
        raise ValueError(f"{key!r}: at: {e}") from None
    calls = []
    for name in ("call", "report_as"):
        call = entry.get(name)
        if not isinstance(call, dict) or not isinstance(call.get("tool"), str) or not call["tool"] \
                or not isinstance(call.get("arguments", {}), dict):
            raise ValueError(f"{key!r}: {name} must be an object with a tool name and its arguments")
        calls.append({"tool": call["tool"], "arguments": call.get("arguments", {})})
    return key, at, calls[0], calls[1]


def _digest(text: str) -> str:
    """A result's digest as the trigger engine's internal_tool_call events carry it."""
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:_ECHO_DIGEST_CHARS]


def _now(clock) -> datetime:
    """The gateway clock's time: virtual while armed, wall UTC while not."""
    now = clock.now()
    return datetime.now(timezone.utc) if now is None else now


@dataclass
class _Call:
    """One declared call: when, which tool with what, the call its triggers see, as whom, whether its result is
    reported, and how far it got."""

    at: datetime
    tool: str
    arguments: dict
    report_as: dict
    role: str
    report: bool
    status: str = DEFERRED_CALL_PENDING
    attempts: int = 0
