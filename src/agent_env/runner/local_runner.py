"""In-process runner: runs ``Task.run()`` in the hub's own event loop, bounded by a worker
semaphore. ``submit()`` fires the run in the background and returns immediately; there is no
persisted queue or lease, so the run's asyncio task lives only in this process.

Consequence: a run does not survive its process. Each run records the process that owns it, which marks it seen
while it runs; ``start()``, and a started runner from then on, fails a run left non-terminal by an owner that is
gone (it can't still be executing), and ``stop()`` cancels the ones in flight. A durable, lease-based runner that resurrects a run across a restart — and supports
multiple worker processes — is a follow-up; configure an external durable ``[runner]`` where that matters.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from agent_env.runner import store as run_store
from agent_env.runner.runner import KEEP_KEY, RunHandle, RunRecord, Runner, RunStatus
from agent_env.task.store import new_instance_id
from agent_env.task.teardown import TERMINATE_TIMEOUT_SECONDS, teardown_run

logger = logging.getLogger(__name__)


class LocalRunner(Runner):
    """In-process runner. The DocumentStore holds run records for status/listing only."""

    type = "local"
    # How long a run's teardown may take, from when it starts: each sandbox gets TERMINATE_TIMEOUT_SECONDS,
    # and the folder of a local one is removed after that, outside it.
    TEARDOWN_WAIT_SECONDS = 2 * TERMINATE_TIMEOUT_SECONDS
    # How long stop() gives a cancelled run to unwind its steps and reach its teardown.
    STOP_WAIT_SECONDS = 60
    # After those waits, how long stop() gives a run it cancels again before shutting down without it.
    ABANDON_WAIT_SECONDS = 1
    # How often a runner marks its runs seen, and a started one reconciles orphans.
    OWNER_SEEN_SECONDS = 30
    # How long after it last marked a run seen an owner on another host still counts as alive.
    OWNER_LEASE_SECONDS = 120

    def __init__(self, workers: int = 2) -> None:
        if workers < 1:
            raise ValueError(f"workers must be >= 1, got {workers}")
        self.workers = int(workers)
        self._sem = asyncio.Semaphore(self.workers)
        self._inflight: dict[str, asyncio.Task] = {}
        self._tearing_down: set[str] = set()  # runs past their task, removing what it deployed
        self._contexts: dict = {}  # each run's context, so stop() can still tear down a run it gives up on
        self._abandoned: set[str] = set()  # runs stop() gave up on and tore down itself
        self._stopping = False
        self._reconciles = False
        self._watch: Optional[asyncio.Task] = None

    # --- lifecycle ---------------------------------------------------------

    async def start(self) -> None:
        """Reconcile orphans now and every ``OWNER_SEEN_SECONDS`` after: a run left QUEUED/RUNNING by an owner that is
        gone has no live task anywhere, so fail it rather than leave it hanging. One a live process owns, such as a
        CLI following its runs, is left to it. There is no queue to resume."""
        run_store.ensure_indexes()
        self._reconciles = True
        self._reconcile()
        self._watching()

    def _watching(self) -> None:
        if self._watch is None:
            self._watch = asyncio.create_task(self._watch_runs(), name="agent-env-local-runner-watch")

    async def _watch_runs(self) -> None:
        while True:
            await asyncio.sleep(self.OWNER_SEEN_SECONDS)
            try:
                for run_id in list(self._inflight):
                    run_store.see(run_id)
                if self._reconciles:
                    self._reconcile()
            except Exception:
                logger.warning("Couldn't mark runs seen or reconcile orphans; trying again in %ss",
                               self.OWNER_SEEN_SECONDS, exc_info=True)

    def _reconcile(self) -> None:
        orphaned = 0
        for record in run_store.active_runs(self.type):
            if _owner_alive(record.owner):
                continue
            run_store.mark_terminal(record.run_id, RunStatus.FAILED, error="interrupted by a hub restart")
            orphaned += 1
        if orphaned:
            logger.info("Failed %d run(s) left non-terminal by a previous process", orphaned)

    async def stop(self) -> None:
        """Cancel the runs still working and wait for every run, letting a teardown under way finish: each bounds
        its own teardown by TEARDOWN_WAIT_SECONDS. A run still going after a run's whole allowance, a step that
        ignored its cancel or a teardown that never ended, is cancelled again and left behind."""
        self._stopping = True
        if self._watch is not None:
            self._watch.cancel()
            self._watch = None
        for run_id, task in list(self._inflight.items()):
            if run_id not in self._tearing_down:
                task.cancel()
        if self._inflight:
            _, stuck = await asyncio.wait(
                list(self._inflight.values()), timeout=self.STOP_WAIT_SECONDS + self.TEARDOWN_WAIT_SECONDS)
            if stuck:
                for task in stuck:
                    task.cancel()
                await asyncio.wait(stuck, timeout=self.ABANDON_WAIT_SECONDS)
                # A run still going never reached its own teardown: remove what it has recorded so far.
                left = [run_id for run_id, task in self._inflight.items()
                        if not task.done() and run_id not in self._tearing_down and run_id in self._contexts]
                if left:
                    logger.warning("Shutting down without %d run(s) that didn't stop; tearing down what they recorded",
                                   len(left))
                    self._abandoned.update(left)  # their own teardown waits while this one runs
                    try:
                        await self._tear_down_abandoned(left)
                    finally:  # a run that ends later tears down whatever this one didn't
                        self._abandoned.difference_update(left)
        self._inflight.clear()

    async def _tear_down_abandoned(self, run_ids: list[str]) -> None:
        try:
            async with asyncio.timeout(self.TEARDOWN_WAIT_SECONDS):
                results = await asyncio.gather(
                    *(self._tear_down(run_id, self._contexts[run_id]) for run_id in run_ids), return_exceptions=True)
        except TimeoutError:
            logger.warning("Gave up tearing down %s after %ss; what they deployed may still be up",
                           ", ".join(run_ids), self.TEARDOWN_WAIT_SECONDS)
            return
        for run_id, result in zip(run_ids, results):
            if isinstance(result, BaseException):
                logger.warning("Run %s: tearing it down failed (%s: %s); what it deployed may still be up",
                               run_id, type(result).__name__, result)

    # --- Runner API --------------------------------------------------------

    async def submit(
        self,
        task_id: str,
        task_version: Optional[int] = None,
        *,
        agent_model: Optional[str] = None,
        agent_artifact_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        instance_id: Optional[str] = None,
    ) -> RunHandle:
        run_id = f"local-{uuid.uuid4().hex}"
        instance_id = instance_id or new_instance_id(task_id)        # minted here so callers can link immediately
        record = RunRecord(
            run_id=run_id, runner=self.type, task_id=task_id, task_version=task_version,
            instance_id=instance_id, status=RunStatus.QUEUED,
            owner={"host": socket.gethostname(), "pid": os.getpid(), "seen": datetime.now(timezone.utc).isoformat()},
            overrides={
                "agent_model": agent_model,
                "agent_artifact_id": agent_artifact_id,
                "metadata": metadata or {},
            },
        )
        run_store.insert_run(record)
        self._inflight[run_id] = asyncio.create_task(self._run(record), name=f"agent-env-run-{run_id}")
        self._watching()
        logger.info("Submitted run %s for task %s v%s", run_id, task_id, task_version)
        return RunHandle(run_id=run_id, instance_id=instance_id)

    async def status(self, run_id: str) -> Optional[RunRecord]:
        record = run_store.get_run(run_id)
        if record is not None and record.runner != self.type:
            # Submitted under a different [runner]; report rather than pretend to own it.
            logger.warning("Run %s was submitted by runner %r, not %r", run_id, record.runner, self.type)
        return record

    async def cancel(self, run_id: str) -> bool:
        """Cancel a run this runner is running; False for any other, whose record it leaves to the process running it."""
        record = run_store.get_run(run_id)
        task = self._inflight.get(run_id)
        if record is None or record.status.is_terminal or task is None:
            return False
        # Signal only. `_run` is the sole writer of the terminal state, so a cancel that
        # races the run's completion can't mislabel a finished run (see `_run`). This
        # preempts Task.run() at its current await; it unwinds its in-flight step cleanly.
        task.cancel()
        return True

    # --- execution ---------------------------------------------------------

    async def _run(self, record: RunRecord) -> None:
        from agent_env.task import Task
        from agent_env.task.store import find_task_instance

        run_task: Optional[asyncio.Task] = None
        context = None
        try:
            async with self._sem:                     # bounded concurrency; the wait here IS the queue
                if run_store.get_run(record.run_id).status == RunStatus.CANCELED:
                    return                            # canceled while it waited for a slot
                run_store.set_running(record.run_id)
                logger.info("Run %s starting (task=%s v%s)", record.run_id, record.task_id, record.task_version)
                stored = find_task_instance(record.instance_id)
                version = record.task_version if record.task_version is not None or stored is None else stored.task_version
                if stored is not None and (stored.task_id, stored.task_version) != (record.task_id, version):
                    raise ValueError(f"Instance {record.instance_id} is a run of {stored.task_id} v{stored.task_version}, "
                                     f"not of {record.task_id} v{version}")
                task = Task.get(record.task_id, version)
                if task is None:
                    raise LookupError(f"Task {record.task_id} v{version} not found")

                context = self._seed_context(record, stored)
                self._contexts[record.run_id] = context
                # start_step rides in metadata but Task.run takes it as a keyword;
                # without lifting it out here every resume re-ran from zero.
                run_task = asyncio.ensure_future(task.run(
                    agent_model=record.overrides.get("agent_model"),
                    agent_artifact_id=record.overrides.get("agent_artifact_id"),
                    context=context,
                    instance_id=record.instance_id,
                    start_step=int(context.metadata.get("start_step") or 0),
                ))
                await run_task
                self._finish(record.run_id, context)
        except asyncio.CancelledError:
            # cancel()/stop() cancelled us. `_run` is the single writer of the terminal state,
            # decided here synchronously so no cancel can interpose: if Task.run() finished
            # before the cancel landed, honor its outcome (its side effects are done, so
            # CANCELED would misreport it); otherwise preempt the in-flight run and record
            # CANCELED — or, on shutdown, leave it non-terminal for start() to reconcile.
            if run_task is not None and run_task.done() and not run_task.cancelled():
                self._finish(record.run_id, context, exc=run_task.exception())
            else:
                if run_task is not None:
                    run_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await run_task
                if self._stopping:
                    logger.info("Run %s interrupted by shutdown", record.run_id)
                else:
                    run_store.mark_terminal(record.run_id, RunStatus.CANCELED)
                    logger.info("Run %s canceled", record.run_id)
        except Exception as e:
            logger.exception("Run %s failed", record.run_id)
            run_store.mark_terminal(record.run_id, RunStatus.FAILED, error=f"{type(e).__name__}: {e}")
        finally:
            try:
                kept = (context is not None and context.metadata.get(KEEP_KEY) and run_task is not None
                        and run_task.done() and not run_task.cancelled())
                if context is not None and record.run_id not in self._abandoned and not kept:
                    self._tearing_down.add(record.run_id)
                    try:
                        async with asyncio.timeout(self.TEARDOWN_WAIT_SECONDS):
                            await self._tear_down(record.run_id, context)
                    except TimeoutError:
                        logger.warning("Run %s: stopped waiting for its teardown after %ss; what it hadn't "
                                       "removed is still up", record.run_id, self.TEARDOWN_WAIT_SECONDS)
            finally:
                self._tearing_down.discard(record.run_id)
                self._abandoned.discard(record.run_id)
                self._contexts.pop(record.run_id, None)
                self._inflight.pop(record.run_id, None)

    @staticmethod
    async def _tear_down(run_id: str, context) -> None:
        """Remove what the run deployed, however it ended: nothing resumes from a local run's sandboxes."""
        report = await teardown_run(context)
        for sandbox, why in report.failed:
            logger.warning("Run %s: couldn't tear down %s: %s", run_id, sandbox.sandbox_id, why)
        for sandbox in report.left:
            logger.warning("Run %s: %s is still up", run_id, sandbox.sandbox_id)

    def _finish(self, run_id: str, context, *, exc: Optional[BaseException] = None) -> None:
        """Persist the terminal state of a finished Task.run(): a raised step is FAILED,
        otherwise an un-retried failed step on the context is authoritative (a step can fail
        without raising when fail_task_on_error is False). Failures the scheduler recovered via
        retry carry ``retried`` and don't count, so a run that recovers is COMPLETED."""
        if exc is not None:
            run_store.mark_terminal(run_id, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}")
            logger.warning("Run %s failed: %s", run_id, exc)
        elif failed := [
            f for f in (context.metadata or {}).get("failed_steps") or []
            if not (isinstance(f, dict) and f.get("retried"))   # a malformed entry counts as a failure
        ]:
            run_store.mark_terminal(run_id, RunStatus.FAILED, error=str(failed))
            logger.warning("Run %s finished with failed steps: %s", run_id, failed)
        else:
            run_store.mark_terminal(run_id, RunStatus.COMPLETED)
            logger.info("Run %s completed", run_id)

    def _seed_context(self, record: RunRecord, stored):
        """Build the context ``Task.run()`` mutates, from the ``stored`` instance it continues if any, carrying the run
        identity in metadata.

        ``workflow_id`` holds the run id — the hub, instance documents and UI all key off it —
        so nothing downstream needs a special case for the local runner.
        """
        from agent_env.task_step import TaskStepContext

        context = TaskStepContext.from_dict(stored.context) if stored and stored.context else TaskStepContext()
        context.metadata.update(record.overrides.get("metadata") or {})
        context.metadata["workflow_id"] = record.run_id
        context.metadata["instance_id"] = record.instance_id
        context.metadata["runner"] = self.type
        return context


def _owner_alive(owner: Optional[dict]) -> bool:
    """Whether the process that submitted a run may still be running it: on this host, whether its pid is alive; on
    another, whether it marked the run seen within ``OWNER_LEASE_SECONDS``. A run naming no owner predates owners."""
    if not owner:
        return False
    if owner["host"] == socket.gethostname():
        try:
            os.kill(owner["pid"], 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    seen = owner.get("seen")
    lease = timedelta(seconds=LocalRunner.OWNER_LEASE_SECONDS)
    return seen is not None and datetime.now(timezone.utc) - datetime.fromisoformat(seen) < lease
