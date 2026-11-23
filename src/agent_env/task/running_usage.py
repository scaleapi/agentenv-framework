"""A running step's usage, written to its run's ledger while the step runs, every ``INTERVAL_S`` in which it changed,
so a budget sees a long step's spend before the step ends and a worker that dies mid-step keeps it. Each entry is
marked ``running``, which makes the run's total a lower bound, as the turn in flight isn't in it. The step's completion
writes its final entries over these: the ledger keeps an entry by attempt, step and role."""

from __future__ import annotations

import asyncio
import copy

from agent_env.task import store
from agent_env.task.usage_ledger import entries_of
from agent_env.task_step.context import TaskStepContext
from agent_env.task_step.usage import STEP_USAGE_KEY

INTERVAL_S = 15.0


class RunningUsage:
    """Writes ``step_id``'s entries in ``context.metadata.step_usage`` to ``instance_id``'s ledger from ``start`` until
    ``stop``, which waits for a write in flight; one a cancel leaves running can't take counts back (``ahead``). A
    write that failed is tried again at the next interval."""

    def __init__(self, instance_id: str, step_id: str, context: TaskStepContext) -> None:
        self._instance_id, self._step_id, self._context = instance_id, step_id, context
        self._stopped = asyncio.Event()
        self._task: asyncio.Task | None = None

    def start(self) -> RunningUsage:
        self._task = asyncio.create_task(self._run())
        return self

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            await self._task

    async def _run(self) -> None:
        written = None
        while True:
            try:
                await asyncio.wait_for(self._stopped.wait(), INTERVAL_S)
                return
            except TimeoutError:
                pass
            entries = copy.deepcopy((self._context.metadata.get(STEP_USAGE_KEY) or {}).get(self._step_id))
            if entries and entries != written and await store.record_usage(
                    self._instance_id, [{**entry, "running": True} for entry in entries_of({self._step_id: entries})]):
                written = entries
