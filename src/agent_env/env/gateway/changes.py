"""Who made each change to an env's state (urn:agentenv:changes/v1).

Every write the gateway serves is recorded in its trajectory with the changelog ids it spanned: a tool call's
``tool_call_result`` carries ``changelog_id_before`` and ``changelog_id``, a trigger's tool action its
``internal_tool_call`` and a request that may write through the REST proxy its ``http_request``, logged once the
response has ended, each with ``changelog_id_before`` / ``changelog_id_after``. A changelog row whose id lies in ``(before, after]`` of such a
window was written while that call ran. :class:`ChangeAttribution` keeps those windows as the gateway logs them
and joins the changelog's rows onto them: one actor for a row one call wrote, several for a row written while
calls overlapped (``performance`` mode serves calls concurrently), none for a write the gateway did not serve.
"""
from __future__ import annotations

import heapq
from typing import Any


class ChangeAttribution:
    """The write windows of the trajectory events the gateway has logged since it started."""

    def __init__(self) -> None:
        self._calls: dict[str, dict] = {}  # tool_call event id -> its actor, until its result closes the window
        self._windows: list[tuple[int, int, int, dict]] = []  # (before, after, order, actor)

    def observe(self, event: dict) -> None:
        """Take one trajectory event as `_log_event` writes it."""
        kind = event.get("event_type")
        if kind == "tool_call":
            self._calls[event["event_id"]] = {"kind": "tool_call", "event_id": event["event_id"],
                                              "tool": (event.get("tool_call") or {}).get("function_name"),
                                              **_who(event)}
        elif kind == "tool_call_result":
            actor = self._calls.pop(event.get("tool_call_event_id"), None)
            if actor is not None:
                self._add(event.get("changelog_id_before"), event.get("changelog_id"), actor)
        elif kind == "internal_tool_call":
            self._add(event.get("changelog_id_before"), event.get("changelog_id_after"), {
                "kind": "trigger_action", "event_id": event["event_id"], "trigger_id": event.get("trigger_id"),
                "action_index": event.get("action_index"), "tool": event.get("tool"), "role": event.get("as")})
        elif kind == "http_request":
            self._add(event.get("changelog_id_before"), event.get("changelog_id_after"), {
                "kind": "request", "event_id": event["event_id"],
                "request": {k: event.get(k) for k in ("route", "service", "method", "path", "status")},
                **_who(event)})

    def _add(self, before: Any, after: Any, actor: dict) -> None:
        if isinstance(before, int) and isinstance(after, int) and after > before:
            self._windows.append((before, after, len(self._windows), actor))

    def actors(self, row_ids: list[int]) -> list[list[dict]]:
        """For each of ``row_ids`` (ascending), the actors whose window holds it, in the order the calls finished."""
        windows = sorted(self._windows)
        open_windows: list[tuple[int, int, dict]] = []  # a heap of (after, order, actor)
        out: list[list[dict]] = []
        i = 0
        for row_id in row_ids:
            while i < len(windows) and windows[i][0] < row_id:
                before, after, order, actor = windows[i]
                heapq.heappush(open_windows, (after, order, actor))
                i += 1
            while open_windows and open_windows[0][0] < row_id:
                heapq.heappop(open_windows)
            out.append([actor for _, _, actor in sorted(open_windows, key=lambda w: w[1])])
        return out


def _who(event: dict) -> dict:
    return {"role": event.get("role"), **({"caused_by": event["caused_by"]} if event.get("caused_by") else {})}
