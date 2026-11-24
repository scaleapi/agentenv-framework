"""A running step's usage, written to its run's ledger while the step runs, every ``INTERVAL_S`` in which it changed,
so a budget sees a long step's spend before the step ends and a worker that dies mid-step keeps it
(``store.record_running_usage``, which marks the entries ``running``). The step's completion writes its final entries
over these: the ledger keeps an entry by attempt, step and role."""

from __future__ import annotations

import asyncio
import contextlib
import copy
from collections.abc import AsyncIterator

from agent_env.task import store
from agent_env.task.usage.ledger import entries_of
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.usage import STEP_USAGE_KEY

INTERVAL_S = 15.0


@contextlib.asynccontextmanager
async def running_usage(instance_id: str | None, step_id: str, context: TaskStepContext) -> AsyncIterator[None]:
    """Writes ``step_id``'s entries in ``context.metadata.step_usage`` to ``instance_id``'s ledger while the block runs,
    a write that failed tried again at the next interval; leaving the block waits for a write in flight. Without an
    instance, it writes nothing."""
    if not instance_id:
        yield
        return
    stopped = asyncio.Event()
    writer = asyncio.create_task(_write(instance_id, step_id, context, stopped))
    try:
        yield
    finally:
        stopped.set()
        await writer


async def _write(instance_id: str, step_id: str, context: TaskStepContext, stopped: asyncio.Event) -> None:
    written = None
    while True:
        try:
            await asyncio.wait_for(stopped.wait(), INTERVAL_S)
            return
        except TimeoutError:
            pass
        entries = copy.deepcopy((context.metadata.get(STEP_USAGE_KEY) or {}).get(step_id))
        if entries and entries != written and await store.record_running_usage(
                instance_id, entries_of({step_id: entries})):
            written = entries
