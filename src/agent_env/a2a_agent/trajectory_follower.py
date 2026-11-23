"""Follow one A2A task's trajectory while it runs, through the trajectory extension's cursor read."""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import httpx
from agentenv_protocol.a2a_agent import (
    AtifEventsResponse,
    TaskEventsTrajectoryRequest,
    TrajectoryEventsResponse,
    TrajectoryState,
    card_request_accepts,
    request_fields,
)
from agentenv_protocol.atif import FORMAT as ATIF

from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.a2a_agent.object_transfer import parse_response
from agent_env.a2a_agent.protocol import extension_error_code, raise_for_extension_status

logger = logging.getLogger(__name__)

_CURSOR_FIELDS = frozenset(request_fields(TaskEventsTrajectoryRequest).required)
_OPEN_STATES = frozenset({TrajectoryState.PENDING, TrajectoryState.RUNNING})
_READ_TIMEOUT_SECONDS = 30
_MAX_BACKOFF_SECONDS = 60
# Once stopped, the task has ended and its log is sealed, so the next read is normally the last.
_STOPPED_INTERVAL_SECONDS = 0.2
# What an agent answers for an atif/1 read it won't serve: the run didn't prescribe it, its adapter failed, or it builds
# no atif/1.
_ATIF_REFUSALS = frozenset({"format_not_prescribed", "atif_unavailable", "format_not_offered"})


@dataclass(frozen=True)
class TrajectoryBatch:
    """Events ``after`` to ``next`` of a task's trajectory, as one read returned them; ``last``
    when the task has ended and no event remains unread. An atif/1 read's also carry the document's ``header``, its
    ``subagents`` as far as served and its versions; ``stopped`` names why an atif/1 follow ended before the task did
    (the agent's refusal), with no events."""

    after: int
    next: int
    events: list[Any]
    format: str | None
    state: TrajectoryState
    last: bool
    header: dict[str, Any] | None = None
    subagents: list[dict[str, Any]] = field(default_factory=list)
    schema_version: str | None = None
    profile_version: str | None = None
    stopped: str | None = None


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
    fallback_format: str | None = None,
) -> None:
    """Read task ``task_id``'s trajectory from its first event, handing ``sink`` each page that has
    events and the last one, until the task has ended and every event is read. Each read names
    ``format`` when given: only to a card that offers formats.

    An atif/1 follow the agent refuses before any of its pages was stored goes on in ``fallback_format`` from the
    task's first event; one refused after that hands ``sink`` a batch naming why it ``stopped`` and ends, so a turn's
    pages never mix formats.

    Between reads it waits ``poll_interval_seconds``, or less once ``stop`` is set. Transport errors
    and 5xx answers are retried with backoff until ``stop`` is set, and end following after it; a
    task the agent no longer knows (404) ends following."""
    after = 0
    failures = 0
    reading = format
    named = {"format": format} if format is not None else {}
    stored_atif = False
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
                refused = extension_error_code(exc.response) if reading == ATIF else None
                if refused in _ATIF_REFUSALS:
                    if not stored_atif and fallback_format is not None:
                        logger.warning("Agent refused task %s's atif/1 trajectory (%s); following it as %s",
                                       task_id, refused, fallback_format)
                        reading, named, after = fallback_format, {"format": fallback_format}, 0
                        continue
                    logger.warning("Agent refused task %s's atif/1 trajectory (%s); stopped following it",
                                   task_id, refused)
                    await sink(TrajectoryBatch(after=after, next=after, events=[], format=ATIF,
                                               state=TrajectoryState.RUNNING, last=False, stopped=refused))
                    return
                if exc.response.status_code < 500:
                    raise
                failures += 1
                logger.warning("Live trajectory read failed (consec=%d, will retry): %s", failures, exc)
            except httpx.TransportError as exc:
                failures += 1
                logger.warning("Live trajectory read failed (consec=%d, will retry): %r", failures, exc)
            else:
                failures = 0
                atif = reading == ATIF
                page = parse_response(AtifEventsResponse if atif else TrajectoryEventsResponse, response.json(),
                                      operation="trajectory get")
                last = page.state not in _OPEN_STATES and not page.has_more
                if page.events or last:
                    extra = ({"header": page.header, "subagents": page.subagents, "schema_version": page.schema_version,
                              "profile_version": page.profile_version} if atif else {})
                    await sink(
                        TrajectoryBatch(
                            after=after,
                            next=page.next,
                            events=page.events,
                            format=page.format,
                            state=page.state,
                            last=last,
                            **extra,
                        )
                    )
                    after = page.next
                    stored_atif = stored_atif or (atif and bool(page.events))
                if last:
                    return
                if page.has_more:
                    continue
            if stop.is_set():
                if failures:
                    # The turn has ended, so an agent that stops answering now is gone, not busy: retrying
                    # would poll a dead address until the step's drain gives up. The step marks the turn's end.
                    logger.warning("Agent stopped answering after task %s ended; stopped following its trajectory", task_id)
                    return
                await asyncio.sleep(_STOPPED_INTERVAL_SECONDS)
                continue
            # Not `wait_for`: on 3.11 it drops a cancel that lands as `stop` is set, the step's own
            # teardown, and the follower then outlives its step.
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(min(poll_interval_seconds + failures * 5, _MAX_BACKOFF_SECONDS)):
                    await stop.wait()
