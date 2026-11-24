"""The SDK's own state in an agent's snapshots: each context's atif/1 turn count. The SDK uploads it after the agent's
save handler and reads it before its load handler, through the request's ``state`` grant, and the card names it, so a
caller grants the object only to an SDK that takes it. While such a snapshot is in flight no task of its context
starts, so the count is the conversation's the agent captured; a running task goes on, as a capture mid-task expects."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ..errors import AgentEnvError
from ..transfers import download, upload
from ._trajectory_log import TaskTrajectories
from .extensions import SNAPSHOT_V1, ObjectSnapshotLoadRequest, ObjectSnapshotSaveRequest

#: The snapshot extension's card param naming the fields of the state an SDK keeps in each snapshot: a caller grants
#: the ``state`` object only to an agent whose card has it, since an SDK from before it refuses the object.
SNAPSHOT_STATE_PARAM = "state"


class SnapshotState(BaseModel):
    """What the SDK keeps in a snapshot beside the agent's own objects: the context's turn count. A newer SDK may add
    fields, which an older one reading the snapshot leaves alone."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    turns: int = Field(ge=0)


class SnapshotStates:
    """The SDK's state in the snapshots of an agent that serves atif/1."""

    def __init__(self, trajectories: TaskTrajectories) -> None:
        self._trajectories = trajectories
        self._in_flight = _InFlightSnapshots()

    @staticmethod
    def card_extension(item: dict[str, Any]) -> dict[str, Any]:
        """An extension's card entry; the snapshot extension's names the state the SDK keeps."""
        if item.get("uri") != SNAPSHOT_V1.uri:
            return item
        return {**item, "params": {**item.get("params", {}), SNAPSHOT_STATE_PARAM: list(SnapshotState.model_fields)}}

    async def save(self, request: Any, invoke: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        """The agent's save, then the context's count uploaded through the ``state`` grant and named in the answer. A
        request that grants no state is the agent's alone."""
        if not isinstance(request, ObjectSnapshotSaveRequest) or request.objects.state is None:
            return await invoke()
        async with self._in_flight.holding(request.context_id):
            answer = await invoke()
            state = SnapshotState(turns=self._trajectories.turns(request.context_id))
        uploaded = await upload(request.objects.state, state.model_dump_json().encode())
        return {**answer, "objects": {**answer["objects"], "state": uploaded.model_dump(mode="json", exclude_none=True)}}

    async def load(self, request: Any, invoke: Callable[[], Awaitable[dict[str, Any]]]) -> dict[str, Any]:
        """The state the snapshot carries, read before the agent's load, then the context the agent restored into
        seeded with its count. A request that grants no state is the agent's alone."""
        if not isinstance(request, ObjectSnapshotLoadRequest) or request.objects.state is None:
            return await invoke()
        state = await _read(request.objects.state)
        async with self._in_flight.holding(request.target_context_id):
            answer = await invoke()
            self._trajectories.restored(str(answer["context_id"]), state.turns)
        return answer

    async def task_may_start(self, context_id: str) -> None:
        """Wait until no snapshot of the context is in flight."""
        await self._in_flight.wait(context_id)


async def _read(grant: Any) -> SnapshotState:
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "state.json"
        await download(grant, path)
        try:
            return SnapshotState.model_validate_json(path.read_bytes())
        except ValidationError as exc:
            raise AgentEnvError(
                "snapshot_state_invalid",
                f"the snapshot's state object isn't the SDK's snapshot state: {exc.errors()[0]['msg']}",
                status=400,
            ) from exc


@dataclass(slots=True)
class _InFlight:
    count: int = 0
    turn: asyncio.Lock = field(default_factory=asyncio.Lock)
    done: asyncio.Event = field(default_factory=asyncio.Event)


class _InFlightSnapshots:
    """Each context's snapshots in flight, taken one at a time, so a load can't change the count a save is about to
    read, and a task of the context waits to start until none is."""

    def __init__(self) -> None:
        self._contexts: dict[str, _InFlight] = {}

    @asynccontextmanager
    async def holding(self, context_id: str | None) -> AsyncIterator[None]:
        if context_id is None:
            yield
            return
        entry = self._contexts.setdefault(context_id, _InFlight())
        entry.count += 1
        try:
            async with entry.turn:
                yield
        finally:
            entry.count -= 1
            if not entry.count:
                del self._contexts[context_id]
                entry.done.set()

    async def wait(self, context_id: str) -> None:
        while (entry := self._contexts.get(context_id)) is not None:
            await entry.done.wait()
