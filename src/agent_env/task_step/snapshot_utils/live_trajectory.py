"""A turn's live trajectory, stored as it is read: ``<prefix><turn>/live/<after>-<next>.jsonl`` per
page, ``meta.json`` naming the events' format, and ``end.json`` once following stops: with the
agent's final state when the follower read it, else the state the step saw the turn end in, or
``canceled`` or ``failed`` when the step stopped first. A follower that stopped early leaves
``end.json``'s ``next`` short of the turn's events; the final trajectory holds them all.

A step's turns are linked in order: each sent turn's ``next.json`` is ``{"turn": "<id>"}``, naming the
turn sent after it, or ``{"turn": null}`` once the step has ended, so a reader of one turn goes on to
the next and knows when the step is over.

Written from the worker with its own credentials, like the final trajectory. Every object is
written once: one already there holds the same bytes, so a repeated write is a no-op."""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager

from agentenv_protocol.a2a_agent import TrajectoryState

from agent_env.a2a_agent.trajectory_follower import TrajectoryBatch, follow_trajectory
from agent_env.store.base import ObjectAlreadyExistsError
from agent_env.store.object_store import ObjectStore

logger = logging.getLogger(__name__)

_WRITE_ATTEMPTS = 3
# The task has ended when the turn's wait returns, so draining its last events is normally one read.
_DRAIN_SECONDS = 60
# A step marks or links its turns in passing: the write must not hold up the turn or a cancel.
_PASSING_WRITE_SECONDS = 5
# The A2A task states a turn's trajectory ends in as themselves; any other (failed, rejected) ends it failed.
_ENDED_AS = {"completed": TrajectoryState.COMPLETED, "canceled": TrajectoryState.CANCELED}


class LiveTrajectoryChunks:
    """The ``follow_trajectory`` sink that stores one turn's pages."""

    def __init__(self, store: ObjectStore, prefix_key: str, turn_id: str) -> None:
        self._store = store
        self._directory = f"{prefix_key}{turn_id}/live/"
        self._format_written = False
        self._stored = 0
        self._ended = False

    async def __call__(self, batch: TrajectoryBatch) -> None:
        if batch.events:
            body = b"".join(
                json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in batch.events
            )
            await self._put(f"{batch.after:08d}-{batch.next:08d}.jsonl", body, "application/x-ndjson")
            self._stored = batch.next
        if batch.format and not self._format_written:
            await self._put("meta.json", json.dumps({"format": batch.format}).encode(), "application/json")
            self._format_written = True
        if batch.last:
            await self.end(batch.state)

    async def end(self, state: TrajectoryState) -> None:
        """Mark the turn ended in ``state`` after the events stored so far; the first mark stays."""
        if self._ended:
            return
        end = {"state": state.value, "next": self._stored}
        await self._put("end.json", json.dumps(end).encode(), "application/json")
        self._ended = True

    async def _put(self, name: str, body: bytes, content_type: str) -> None:
        await _put(self._store, self._directory + name, body, content_type)


class LiveTurn:
    """What a turn's body tells its follower: the task to ``follow`` once the message is sent, and
    the state it ``ended`` in once the wait for it returns."""

    def __init__(
        self,
        endpoint: str,
        store: ObjectStore,
        prefix_key: str,
        turn_id: str,
        poll_interval_seconds: float,
        previous_turn_id: str | None,
    ) -> None:
        self._endpoint = endpoint
        self._store = store
        self._prefix_key = prefix_key
        self._turn_id = turn_id
        self._poll_interval_seconds = poll_interval_seconds
        self._previous_turn_id = previous_turn_id
        self.stop = asyncio.Event()
        self.followers: list[asyncio.Task] = []
        self.sinks: list[LiveTrajectoryChunks] = []
        self.links: list[asyncio.Task] = []
        self.ended_as: TrajectoryState | None = None

    def follow(self, task_id: str) -> None:
        if self._previous_turn_id is not None:
            linking = _link(self._store, self._prefix_key, self._previous_turn_id, self._turn_id)
            self.links.append(
                asyncio.create_task(_in_passing(linking, self._previous_turn_id, f"linked to turn {self._turn_id}"))
            )
        sink = LiveTrajectoryChunks(self._store, self._prefix_key, self._turn_id)
        self.sinks.append(sink)
        self.followers.append(
            asyncio.create_task(
                follow_trajectory(
                    self._endpoint, task_id, sink, poll_interval_seconds=self._poll_interval_seconds, stop=self.stop
                )
            )
        )

    def ended(self, task_state: str) -> None:
        """Report the A2A state the turn's task ended in."""
        self.ended_as = _ENDED_AS.get(task_state, TrajectoryState.FAILED)


@asynccontextmanager
async def following_live_trajectory(
    endpoint: str,
    store: ObjectStore,
    prefix_key: str,
    turn_id: str,
    *,
    poll_interval_seconds: float,
    previous_turn_id: str | None = None,
) -> AsyncIterator[LiveTurn]:
    """Yield the turn whose task is followed once its message is sent; then ``previous_turn_id``'s
    ``next.json`` names it.

    When the body returns, the follower drains what remains, for at most ``_DRAIN_SECONDS``; when it
    raises, the follower is stopped at once, since the agent may be gone. Either way, a turn whose
    end the follower did not read is marked ended in the state the body reported, else ``canceled``
    for a cancelled step or ``failed``, so readers do not wait on it. A follower's own failure is
    logged and never fails the turn."""
    turn = LiveTurn(endpoint, store, prefix_key, turn_id, poll_interval_seconds, previous_turn_id)
    stopped_as = TrajectoryState.FAILED

    try:
        yield turn
        turn.stop.set()
        if turn.followers:
            _, draining = await asyncio.wait(turn.followers, timeout=_DRAIN_SECONDS)
            if draining:
                logger.warning(
                    "Live trajectory of turn %s did not finish draining in %ds (continuing)",
                    turn_id,
                    _DRAIN_SECONDS,
                )
    except asyncio.CancelledError:
        stopped_as = TrajectoryState.CANCELED
        raise
    finally:
        turn.stop.set()
        for follower in turn.followers:
            follower.cancel()
        if turn.followers:
            await asyncio.wait(turn.followers)
        for follower in turn.followers:
            if not follower.cancelled() and follower.exception() is not None:
                logger.warning(
                    "Live trajectory of turn %s stopped (continuing): %r", turn_id, follower.exception()
                )
        if turn.links:
            await asyncio.wait(turn.links)
        if turn.sinks:
            state = turn.ended_as or stopped_as
            await _in_passing(turn.sinks[-1].end(state), turn_id, f"marked {state.value}")


class LiveTurns:
    """A step's turns, followed one after another: a turn sent after another is named in that one's
    ``next.json``, and ``end`` marks the last turn sent as the step's last."""

    def __init__(self, endpoint: str, store: ObjectStore, prefix_key: str, *, poll_interval_seconds: float) -> None:
        self._endpoint = endpoint
        self._store = store
        self._prefix_key = prefix_key
        self._poll_interval_seconds = poll_interval_seconds
        self._last_sent: str | None = None

    @asynccontextmanager
    async def following(self, turn_id: str) -> AsyncIterator[LiveTurn]:
        """``following_live_trajectory`` for the step's next turn."""
        async with following_live_trajectory(
            self._endpoint,
            self._store,
            self._prefix_key,
            turn_id,
            poll_interval_seconds=self._poll_interval_seconds,
            previous_turn_id=self._last_sent,
        ) as turn:
            try:
                yield turn
            finally:
                if turn.sinks:
                    self._last_sent = turn_id

    async def end(self) -> None:
        """Mark the last turn sent as the step's last, once the step has ended."""
        if self._last_sent is not None:
            ending = _link(self._store, self._prefix_key, self._last_sent, None)
            await _in_passing(ending, self._last_sent, "marked last")


async def _link(store: ObjectStore, prefix_key: str, turn_id: str, next_turn_id: str | None) -> None:
    """Write ``turn_id``'s ``next.json``: the turn sent after it, or None after the step's last."""
    body = json.dumps({"turn": next_turn_id}).encode()
    await _put(store, f"{prefix_key}{turn_id}/live/next.json", body, "application/json")


async def _put(store: ObjectStore, key: str, body: bytes, content_type: str) -> None:
    for attempt in range(1, _WRITE_ATTEMPTS + 1):
        try:
            await asyncio.to_thread(store.put, key, body, content_type=content_type)
            return
        except ObjectAlreadyExistsError:
            return
        except Exception as exc:
            if attempt == _WRITE_ATTEMPTS:
                raise
            logger.warning("Live trajectory write of %s failed (attempt %d, will retry): %s", key, attempt, exc)
            await asyncio.sleep(attempt)


async def _in_passing(write: Awaitable[None], turn_id: str, what: str) -> None:
    """Run ``write`` for at most ``_PASSING_WRITE_SECONDS``, logging rather than raising its failure."""
    writing = asyncio.ensure_future(write)
    done, _ = await asyncio.wait({writing}, timeout=_PASSING_WRITE_SECONDS)
    if not done:
        writing.cancel()
        logger.warning("Live trajectory of turn %s was not %s in time (continuing)", turn_id, what)
    elif writing.exception() is not None:
        logger.warning("Live trajectory of turn %s was not %s (continuing): %r", turn_id, what, writing.exception())
