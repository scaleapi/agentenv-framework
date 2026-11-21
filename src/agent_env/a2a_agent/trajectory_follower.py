"""Follow one A2A task's trajectory while it runs, through the trajectory extension's cursor read."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import httpx
from agentenv_protocol.a2a_agent import (
    TaskEventsTrajectoryRequest,
    TrajectoryEventsResponse,
    TrajectoryState,
    card_request_accepts,
    request_fields,
)

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import parse_response
from agent_env.a2a_agent.protocol import raise_for_extension_status

logger = logging.getLogger(__name__)

_CURSOR_FIELDS = frozenset(request_fields(TaskEventsTrajectoryRequest).required)
_OPEN_STATES = frozenset({TrajectoryState.PENDING, TrajectoryState.RUNNING})
_READ_TIMEOUT_SECONDS = 30
_MAX_BACKOFF_SECONDS = 60
# Once stopped, the task has ended and its log is sealed, so the next read is normally the last.
_STOPPED_INTERVAL_SECONDS = 0.2
# After the task has ended, a failing read gets a short, bounded drain: enough to ride out a blip
# without losing the trajectory's tail, short enough that a gone agent frees the step.
_STOPPED_RETRY_SECONDS = 1.0
_STOPPED_MAX_FAILURES = 3
_ERROR_AFTER_FAILURES = 10


@dataclass(frozen=True)
class TrajectoryBatch:
    """Events ``after`` to ``next`` of a task's trajectory, as one read returned them; ``last``
    when the task has ended and no event remains unread."""

    after: int
    next: int
    events: list[Any]
    format: str | None
    state: TrajectoryState
    last: bool


TrajectorySink = Callable[[TrajectoryBatch], Awaitable[None]]


def live_trajectory_endpoint(card: dict, a2a_url: str) -> str | None:
    """The URL of the agent's live trajectory read, or None when its card does not offer one."""
    extension = A2AAgent.find_extension(card, A2AAgent.EXT_TRAJECTORY)
    method, path = A2AAgent.operation(extension, "get")
    if not method or path is None:
        return None
    if not card_request_accepts(method.get("request") or {}, _CURSOR_FIELDS):
        return None
    return a2a_url + path


def _log_read_failure(task_id: str, failures: int, exc: Exception) -> None:
    logger.log(
        logging.ERROR if failures >= _ERROR_AFTER_FAILURES else logging.WARNING,
        "Live trajectory read failed (consec=%d, will retry): %r",
        failures,
        exc,
        extra={"event": "live_trajectory_read_failed", "consec": failures, "task_id": task_id},
    )


async def follow_trajectory(
    endpoint: str,
    task_id: str,
    sink: TrajectorySink,
    *,
    poll_interval_seconds: float,
    stop: asyncio.Event,
    client: httpx.AsyncClient | None = None,
    headers: Mapping[str, str] | None = None,
    format: str | None = None,
) -> None:
    """Read task ``task_id``'s trajectory from its first event, handing ``sink`` each page that has
    events and the last one, until the task has ended and every event is read. Each read names
    ``format`` when given: only to a card that offers formats.

    Between reads it waits ``poll_interval_seconds``, or less once ``stop`` is set. Transport errors
    and 5xx answers are retried with backoff. Once ``stop`` is set they are retried a few more times,
    then end following; a task the agent no longer knows (404) ends following."""
    after = 0
    failures = 0
    named = {"format": format} if format is not None else {}
    async with nullcontext(client) if client is not None else httpx.AsyncClient() as http:
        while True:
            try:
                response = await http.post(
                    endpoint, json={"task_id": task_id, "after": after, **named}, headers=headers,
                    timeout=_READ_TIMEOUT_SECONDS,
                )
                if response.status_code == 404:
                    logger.warning("Agent no longer knows task %s; stopped following its trajectory", task_id)
                    return
                raise_for_extension_status(response, operation="trajectory get", include_body=True)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code < 500:
                    raise
                failures += 1
                _log_read_failure(task_id, failures, exc)
            except httpx.TransportError as exc:
                failures += 1
                _log_read_failure(task_id, failures, exc)
            else:
                failures = 0
                page = parse_response(TrajectoryEventsResponse, response.json(), operation="trajectory get")
                last = page.state not in _OPEN_STATES and not page.has_more
                if page.events or last:
                    await sink(
                        TrajectoryBatch(
                            after=after,
                            next=page.next,
                            events=page.events,
                            format=page.format,
                            state=page.state,
                            last=last,
                        )
                    )
                    after = page.next
                if last:
                    return
                if page.has_more:
                    continue
            if stop.is_set():
                if failures >= _STOPPED_MAX_FAILURES:
                    logger.warning(
                        "Agent stopped answering after task %s ended; stopped following its trajectory",
                        task_id,
                        extra={"event": "live_trajectory_follow_abandoned", "consec": failures, "task_id": task_id},
                    )
                    return
                await asyncio.sleep(_STOPPED_RETRY_SECONDS * failures or _STOPPED_INTERVAL_SECONDS)
                continue
            # Not `wait_for`: on 3.11 it drops a cancel that lands as `stop` is set, the step's own
            # teardown, and the follower then outlives its step.
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(min(poll_interval_seconds + failures * 5, _MAX_BACKOFF_SECONDS)):
                    await stop.wait()
