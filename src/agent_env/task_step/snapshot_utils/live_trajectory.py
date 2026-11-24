"""A turn's live trajectory, stored as it is read: ``<prefix><turn>/live/<after>-<next>.jsonl`` per
page, ``meta.json`` naming the events' format, and ``end.json`` once following stops: with the
agent's final state when the follower read it, else the state the step saw the turn end in, or
``canceled`` or ``failed`` when the step stopped first. A follower that stopped early leaves
``end.json``'s ``next`` short of the turn's events; the final trajectory holds them all.

An atif/1 turn's pages hold its closed root steps, one per line, named by the step ids they run between; the document's
header (its root without steps) is ``<after>-<next>.<n>.header.json`` beside the page where it changed, ``n`` counting
the header's versions, so the greatest is the latest; each sub-agent's new steps and header go the same way under
``sub/<trajectory id>/``; ``meta.json`` also names the schema and profile versions, the document's session and
trajectory ids and ``media_base``, the folder its ``media/`` paths resolve in; ``end.json`` says why a follow the agent
refused partway ``stopped``. The latest header with every page's steps is a prefix of the turn's stored copy.

A step's turns are linked in order: each sent turn's ``next.json`` is ``{"turn": "<id>"}``, naming the
turn sent after it, or ``{"turn": null}`` once the step has ended, so a reader of one turn goes on to
the next and knows when the step is over.

Written from the worker with its own credentials, like the final trajectory. Every object is
written once: one already there holds the same bytes, so a repeated write is a no-op."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from agentenv_protocol.a2a_agent import TrajectoryState
from agentenv_protocol.atif import FORMAT as ATIF

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
# A sub-agent's trajectory id names its folder, so only a plain file name does.
_SUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
LiveMedia = Callable[[Mapping[str, Any], Any], Awaitable[None]]


@dataclass(frozen=True)
class LiveAtif:
    """A turn followed as atif/1, its native format the fallback for a turn the agent refuses atif/1 before its first
    page: ``media`` pulls the media a page's new steps name (given them as a document, and the session) into
    ``media_base``, the folder their ``media/`` paths resolve in. With ``cursors``, its reads name what the follower
    holds, for an agent whose card takes that (``live_reads_take_cursors``)."""

    media: LiveMedia
    media_base: str
    cursors: bool = False


class LiveTrajectoryChunks:
    """The ``follow_trajectory`` sink that stores one turn's pages."""

    def __init__(self, store: ObjectStore, prefix_key: str, turn_id: str, *, atif: Optional[LiveAtif] = None,
                 stop: Optional[asyncio.Event] = None) -> None:
        self._store = store
        self._directory = f"{prefix_key}{turn_id}/live/"
        self._format_written = False
        self._stored = 0
        self._storing: asyncio.Future | None = None
        self._ended = False
        self._atif, self._stop = atif, stop
        self._headers: dict[str, tuple[str, int]] = {}  # each trajectory's folder: its header's digest, versions
        self._sub_steps: dict[str, int] = {}  # each sub-agent's folder: the last step id stored
        self._session: Any = None
        self._stopped: Optional[str] = None
        self._pulls: list[asyncio.Task] = []

    async def __call__(self, batch: TrajectoryBatch) -> None:
        if batch.stopped:
            self._stopped = batch.stopped
            return
        atif = batch.format == ATIF
        if batch.events or (atif and (batch.header is not None or batch.subagents)):
            # Shielded, so a cancelled follower's write still counts its events: ``end`` waits for it.
            self._storing = asyncio.ensure_future(self._store_atif(batch) if atif else self._store_events(batch))
            await asyncio.shield(self._storing)
        if batch.format and not self._format_written:
            meta: dict[str, Any] = {"format": batch.format}
            if atif:
                header = batch.header or {}
                meta.update(schema_version=batch.schema_version, profile_version=batch.profile_version,
                            session_id=header.get("session_id"), trajectory_id=header.get("trajectory_id"),
                            media_base=self._atif.media_base if self._atif else None)
            await self._put("meta.json", json.dumps(meta).encode(), "application/json")
            self._format_written = True
        if batch.last:
            await self.end(batch.state)

    async def _store_events(self, batch: TrajectoryBatch) -> None:
        body = b"".join(json.dumps(event, separators=(",", ":")).encode() + b"\n" for event in batch.events)
        await self._put(f"{batch.after:08d}-{batch.next:08d}.jsonl", body, "application/x-ndjson")
        self._stored = batch.next

    async def _store_atif(self, batch: TrajectoryBatch) -> None:
        """An atif/1 page: its root steps, the header when it changed, each sub-agent's new steps and header; then,
        while the turn runs, a pull of the media these new steps name."""
        name = f"{batch.after:08d}-{batch.next:08d}"
        if batch.events:
            await self._put(f"{name}.jsonl", _lines(batch.events), "application/x-ndjson")
        if batch.header is not None:
            self._session = batch.header.get("session_id", self._session)
            await self._header("", name, batch.header)
        new_subs = [sub for found in [await self._store_sub(s, "sub/") for s in batch.subagents] for sub in found]
        self._stored = batch.next
        if self._atif is not None and not (self._stop is not None and self._stop.is_set()):
            steps = {"steps": list(batch.events), "subagent_trajectories": new_subs}
            if batch.events or new_subs:
                self._pulls.append(asyncio.ensure_future(self._atif.media(steps, self._session)))

    async def _store_sub(self, sub: Mapping[str, Any], path: str) -> list[dict[str, Any]]:
        """A sub-agent's steps not stored yet and its header if it changed, under ``path``; and its embedded sub-agents
        the same way, under its folder. Returns each trajectory with only its new steps, for the media they name."""
        trajectory_id = sub.get("trajectory_id")
        if not isinstance(trajectory_id, str) or not _SUB_ID.fullmatch(trajectory_id):
            return []
        where = f"{path}{trajectory_id}/"
        last = self._sub_steps.get(where, 0)
        steps = [step for step in sub.get("steps") or [] if isinstance(step, Mapping)
                 and isinstance(step.get("step_id"), int) and step["step_id"] > last]
        name = f"{last:08d}-{(steps[-1]['step_id'] if steps else last):08d}"
        if steps:
            await self._put(f"{where}{name}.jsonl", _lines(steps), "application/x-ndjson")
            self._sub_steps[where] = steps[-1]["step_id"]
        await self._header(where, name, {k: v for k, v in sub.items() if k not in ("steps", "subagent_trajectories")})
        found = [{"steps": steps}] if steps else []
        for child in sub.get("subagent_trajectories") or []:
            found += await self._store_sub(child, f"{where}sub/")
        return found

    async def _header(self, where: str, name: str, header: Mapping[str, Any]) -> None:
        body = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
        digest = hashlib.sha256(body).hexdigest()
        seen, versions = self._headers.get(where, (None, 0))
        if digest != seen:
            await self._put(f"{where}{name}.{versions + 1:04d}.header.json", body, "application/json")
            self._headers[where] = (digest, versions + 1)

    async def cancel_pulls(self) -> None:
        """Cancel the media pulls this turn's pages started: once the turn ends, its copy's own pull takes what's left."""
        for pull in self._pulls:
            pull.cancel()
        if self._pulls:
            await asyncio.wait(self._pulls)

    async def end(self, state: TrajectoryState) -> None:
        """Mark the turn ended in ``state`` after the events stored so far; the first mark stays."""
        if self._ended:
            return
        if self._storing is not None:
            await asyncio.wait({self._storing})
        end = {"state": state.value, "next": self._stored, **({"stopped": self._stopped} if self._stopped else {})}
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
        *,
        poll_interval_seconds: float,
        previous_turn_id: str | None,
        headers: Mapping[str, str],
        format: str | None = None,
        atif: Optional[LiveAtif] = None,
    ) -> None:
        self._endpoint = endpoint
        self._headers = headers
        self._format = format
        self._atif = atif
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
        sink = LiveTrajectoryChunks(self._store, self._prefix_key, self._turn_id, atif=self._atif, stop=self.stop)
        self.sinks.append(sink)
        self.followers.append(
            asyncio.create_task(
                follow_trajectory(
                    self._endpoint, task_id, sink, poll_interval_seconds=self._poll_interval_seconds, stop=self.stop,
                    headers=self._headers, format=ATIF if self._atif else self._format,
                    fallback_format=self._format if self._atif else None,
                    cursors=self._atif is not None and self._atif.cursors,
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
    headers: Mapping[str, str],
    previous_turn_id: str | None = None,
    format: str | None = None,
    atif: Optional[LiveAtif] = None,
) -> AsyncIterator[LiveTurn]:
    """Yield the turn whose task is followed once its message is sent; then ``previous_turn_id``'s
    ``next.json`` names it.

    When the body returns, the follower drains what remains, for at most ``_DRAIN_SECONDS``; when it
    raises, the follower is stopped at once, since the agent may be gone. Either way, a turn whose
    end the follower did not read is marked ended in the state the body reported, else ``canceled``
    for a cancelled step or ``failed``, so readers do not wait on it. A follower's own failure is
    logged and never fails the turn. The media pulls a turn's atif/1 pages started are cancelled once it ends: the
    turn's copy pulls what's left, off the turn's path."""
    turn = LiveTurn(endpoint, store, prefix_key, turn_id, poll_interval_seconds=poll_interval_seconds,
                    previous_turn_id=previous_turn_id, headers=headers, format=format, atif=atif)
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
        for sink in turn.sinks:
            await sink.cancel_pulls()
        if turn.links:
            await asyncio.wait(turn.links)
        if turn.sinks:
            state = turn.ended_as or stopped_as
            await _in_passing(turn.sinks[-1].end(state), turn_id, f"marked {state.value}")


class LiveTurns:
    """A step's turns, followed one after another: a turn sent after another is named in that one's
    ``next.json``, and ``end`` marks the last turn sent as the step's last. Each is read in ``format``, or as atif/1
    with ``atif``."""

    def __init__(self, endpoint: str, store: ObjectStore, prefix_key: str, *, poll_interval_seconds: float,
                 headers: Mapping[str, str], format: str | None = None, atif: Optional[LiveAtif] = None) -> None:
        self._endpoint = endpoint
        self._headers = headers
        self._format = format
        self._atif = atif
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
            headers=self._headers,
            previous_turn_id=self._last_sent,
            format=self._format,
            atif=self._atif,
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


def _lines(items: list[Any]) -> bytes:
    return b"".join(json.dumps(item, separators=(",", ":")).encode() + b"\n" for item in items)


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
