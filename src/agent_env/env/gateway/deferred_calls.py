"""Calls a server's tool result declares (``AGENT_ENV_DEFERRED_CALLS_META_KEY``) for the gateway to make on that server
when its clock gets to their time, as the declaring call's role, each reported to the trigger engine as the call it
stands for, so what a server does later by itself (a scheduled message posted) provokes triggers as a live call would."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from mcp.types import CallToolResult

from .caller import Caller
from .clock import ClockError, _iso, _parse_rfc3339
from .constants import AGENT_ENV_DEFERRED_CALLS_META_KEY
from .triggers import _VIRTUAL_RECHECK_S, _VIRTUAL_RESOLUTION_S, _result_text

logger = logging.getLogger(__name__)


class DeferredCalls:
    """The calls servers declared, by key. One is kept once made, so the key declared again is ignored; one that failed
    may be declared afresh."""

    RETRY_ATTEMPTS = 5  # a call a transport failure stopped is made again, so the tool a server defers is idempotent
    RETRY_BACKOFF_S = 1.0  # real seconds before the second attempt, doubled before each later one

    def __init__(self, gateway):
        self._gw = gateway
        self._calls: dict[str, dict] = {}
        self._tasks: set[asyncio.Task] = set()

    def take(self, result: CallToolResult, server_url: str, role: str) -> CallToolResult:
        """``result`` without its declaration, which no caller sees, each call it declares scheduled as ``role``'s
        unless the result is an error."""
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
            self._schedule(entry, server_url, role)
        return result

    def state(self) -> list[dict]:
        """Each call declared, for GET /triggers/state; nothing that waits for the triggers to settle reads it."""
        return [{"key": key, "at": _iso(call["at"]), "tool": call["tool"], "role": call["role"],
                 "status": call["status"], "attempts": call["attempts"]} for key, call in self._calls.items()]

    async def cancel(self) -> None:
        """Cancel the calls still pending and wait for them to exit; the gateway's process lifespan calls it at
        shutdown, before the child sessions close."""
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

    def _schedule(self, entry: Any, server_url: str, role: str) -> None:
        try:
            key, at, call, report_as = _parse(entry)
        except ValueError as e:
            logger.warning(f"ignoring a deferred call declared by {server_url}: {e}")
            return
        known = self._calls.get(key)
        if known is not None and known["status"] != "failed":
            return
        record = {"at": at, **call, "report_as": report_as, "role": role, "status": "pending", "attempts": 0}
        self._calls[key] = record
        task = asyncio.get_running_loop().create_task(self._make(key, record, server_url))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _make(self, key: str, record: dict, server_url: str) -> None:
        """Wait for the call's time and make it, again after a transport failure; log it, and report it to the trigger
        engine when it went through. An error result is final."""
        await self._until(record["at"])
        result, error = None, None
        while result is None and record["attempts"] < self.RETRY_ATTEMPTS:
            if record["attempts"]:
                await asyncio.sleep(self.RETRY_BACKOFF_S * 2 ** (record["attempts"] - 1))
            record["attempts"] += 1
            try:
                result = await self._gw._call_child_tool(server_url, record["tool"], record["arguments"],
                                                         caller=Caller(role=record["role"]))
            except Exception as e:
                error = (str(e) or type(e).__name__)[:300]
                logger.warning(f"deferred call {key!r} ({record['tool']}) attempt {record['attempts']} failed: {error}")
        ok = result is not None and not result.isError
        if result is not None and not ok:
            error = _result_text(result)[:300]
        record["status"] = "done" if ok else "failed"
        await self._gw._log_event({"event_type": "deferred_call", "key": key, "tool": record["tool"],
                                   "role": record["role"], "ok": ok, "attempts": record["attempts"],
                                   **({} if ok else {"error": error})})
        if ok:
            report_as = record["report_as"]
            self._gw._fire_triggers(record["role"], report_as["tool"], report_as["arguments"], result)

    async def _until(self, at: datetime) -> None:
        """Wait until the gateway clock gets to ``at``: its virtual time while armed, held while frozen and read again
        whenever it changes (the trigger engine's wake), and wall UTC while unarmed."""
        clock, engine = self._gw._clock, self._gw._trigger_engine
        while True:
            now = clock.now()
            left = (at - (datetime.now(timezone.utc) if now is None else now)).total_seconds()
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
