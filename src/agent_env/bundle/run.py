"""Run a bundle from its folder: plan it, write what changed, and run each selected task once.

The run covers the tasks selected on their own and every task a selected eval names, each run once, at most
four at a time, at the version materializing wrote or reused. A run that fails is kept on its ``TaskRun``
and the others go on. Each run's sandboxes are torn down as it ends, unless ``keep`` holds them up. Ctrl-C or
SIGTERM cancels the runs: each one that started is marked cancelled and torn down, and a second one stops the
teardown. A bundle's evals run only the bundle's own tasks for now, so one naming a store task is refused
before anything is written.
"""

from __future__ import annotations

import asyncio
import logging
import shlex
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

from agent_env.providers import build_sandbox_provider
from agent_env.store.routing import namespace_routing, run_scope
from agent_env.task import Task, record_task_cancelled
from agent_env.task.interrupts import Interrupts
from agent_env.task.teardown import TeardownReport, teardown_run
from agent_env.task_step.context import TaskStepContext

from ._fs import relative
from .materialize import Materialization, Materialized, materialize
from .parse import BundleEntry, BundleError, BundleKind, parse_bundle
from .plan import Plan, Write, plan_bundle
from .resolve import BuiltImage, Reference, resolve_bundle

logger = logging.getLogger(__name__)

MAX_CONCURRENT_RUNS = 4


class Outcome(Enum):
    PASSED = "passed"  # it has a score, and every score is at least 1
    BELOW_ONE = "scored below 1"
    UNSCORED = "unscored"
    FAILED = "failed"  # the run raised, or a step failed without stopping it
    CANCELLED = "cancelled"  # Ctrl-C or SIGTERM stopped it, or stopped it starting


@dataclass(frozen=True)
class TaskRun:
    """One run of a bundle task, with the context it left, kept when it raised."""

    entry: BundleEntry
    task: Task
    context: TaskStepContext
    error: Exception | None
    duration: float  # seconds
    cancelled: bool = False
    started: bool = True  # False when it was cancelled before it started
    torn_down: TeardownReport | None = None  # None while its sandboxes are kept up

    @property
    def instance_id(self) -> str | None:
        """None when the instance couldn't be recorded."""
        return self.context.instance_id

    @property
    def failed_steps(self) -> list[dict]:
        """The step failures a retry didn't recover from."""
        return [failure for failure in self.context.metadata.get("failed_steps") or [] if not failure.get("retried")]

    @property
    def scores(self) -> dict[str, float]:
        """Every score the run recorded, by the id of the step that recorded it, or by its verification's key
        when that isn't a step's verifier id."""
        steps = {step.verifier_id: step.id for step in self.task.steps if getattr(step, "verifier_id", None)}
        scores = {}
        for key, verification in (self.context.metadata.get("verifications") or {}).items():
            score = verification.get("score")
            if isinstance(score, (int, float)):
                scores[steps.get(key, key)] = float(score)
        return scores

    @property
    def outcome(self) -> Outcome:
        if self.cancelled:
            return Outcome.CANCELLED
        if self.error is not None or self.failed_steps:
            return Outcome.FAILED
        if not self.scores:
            return Outcome.UNSCORED
        return Outcome.PASSED if all(score >= 1 for score in self.scores.values()) else Outcome.BELOW_ONE


@dataclass(frozen=True)
class EvalRun:
    entry: BundleEntry
    version: int
    runs: tuple[TaskRun, ...]  # in the eval's order


@dataclass(frozen=True)
class BundleRun:
    materialization: Materialization
    runs: tuple[TaskRun, ...]  # each task once, in the plan's order
    evals: tuple[EvalRun, ...]
    skipped: tuple[BundleEntry, ...]  # the tasks no eval names, which running every eval leaves out

    @property
    def failed(self) -> bool:
        return any(run.outcome is Outcome.FAILED for run in self.runs)

    def path(self, entry: BundleEntry) -> str:
        """``entry``'s path in the bundle, as the summary names it (``tasks/hello.json``)."""
        return relative(self.materialization.plan.bundle.bundle.root, entry.path)

    def teardown(self) -> BundleRun:
        """Tear down the runs whose sandboxes were kept up, as ``run_bundle(keep=True)`` leaves them, and
        return this result with each one's report. Ctrl-C or SIGTERM stops it early and raises
        ``RunInterrupted`` with what it reached; the rest stays up. It runs its own event loop, like
        ``run_bundle``."""
        kept = [run for run in self.runs if run.torn_down is None]
        if not kept:
            return self
        with Interrupts() as interrupts:
            reports = interrupts.run(interrupts.stopping(teardown_run(run.context) for run in kept))
            result = self._with_reports(kept, reports)
            if interrupts.count:
                raise RunInterrupted(result, interrupts.signum)
        return result

    def _with_reports(self, runs: list[TaskRun], reports: list[TeardownReport]) -> BundleRun:
        done = {id(run): replace(run, torn_down=report) for run, report in zip(runs, reports)}

        def swapped(runs: tuple[TaskRun, ...]) -> tuple[TaskRun, ...]:
            return tuple(done.get(id(run), run) for run in runs)

        return replace(self, runs=swapped(self.runs),
                       evals=tuple(replace(eval_run, runs=swapped(eval_run.runs)) for eval_run in self.evals))


class RunInterrupted(KeyboardInterrupt):
    """Ctrl-C or SIGTERM stopped ``run_bundle``. ``result`` holds every run, those it cancelled included, each
    torn down unless a second signal stopped that."""

    def __init__(self, result: BundleRun, signum: int):
        super().__init__(signum)
        self.result = result
        self.signum = signum


def run_bundle(
    root: Path | str,
    *,
    tasks: Sequence[str] = (),
    evals: Sequence[str] = (),
    model: str | None = None,
    sandbox: str | None = None,
    on_progress: Callable[[str], None] | None = None,
    id_root: str | None = None,
    keep: bool = False,
) -> BundleRun:
    """Write what the tasks and evals selected in ``root`` need, as ``plan_bundle`` selects them, then run
    each of their tasks once. ``model`` replaces the agent's model but not the judge's; ``sandbox`` is the
    provider that envs, agents, sandboxes and the judge deploy on. ``on_progress`` is given a line of text
    at each write and step. ``id_root`` roots the ids of a bundle an installed package provides
    (``InstalledBundle.id_root``); a folder's ids otherwise come from its path.

    Each run's sandboxes are torn down as it ends, so the contexts returned record sandboxes that are gone.
    ``keep`` leaves them up instead, for the caller to use and then remove with ``BundleRun.teardown()``.
    Ctrl-C or SIGTERM cancels the runs and tears down every one, kept or not, then raises ``RunInterrupted``
    with the result.

    It runs its own event loop, and turns namespace routing on for the whole process while it runs, so it
    suits the CLI and scripts, not a service. From async code, call it in a thread; there it can't take over
    the process's signal handlers, so a Ctrl-C doesn't reach its runs, though each is still torn down as it
    ends.

    Before anything is written, raises RuntimeError when an event loop is already running, ValueError when
    ``sandbox`` names no provider, and BundleError when the bundle can't be planned. A task that fails its
    preflight raises BundleError once the entities it reads are written, before any task is."""
    _refuse_a_running_loop()
    if sandbox:
        build_sandbox_provider(sandbox)
    say = _progress(on_progress)
    with namespace_routing():
        plan = plan_bundle(resolve_bundle(parse_bundle(Path(root), id_root=id_root)), tasks=tasks, evals=evals)
        refuse_store_tasks(plan)
        materialization = materialize(
            plan,
            on_wait=lambda: say("waiting for another agent-env run to finish writing this bundle's ids"),
            on_build=lambda write: say(f"{_label(plan, write)}: building with docker, which can take minutes"),
            on_write=lambda done: say(_written(plan, done)),
        )
        wanted = {entry.entry.id for entry in plan.tasks}
        wanted.update(ref.id for entry in plan.evals for ref in entry.references)
        entries = [write.source.entry for write in plan.writes if write.kind is BundleKind.TASK and write.id in wanted]
        to_run = [(entry, Task.get(entry.id, materialization.version_of("task", entry.id))) for entry in entries]
        with Interrupts() as interrupts:
            runs = interrupts.run(_run_all(plan, to_run, model, sandbox, keep, say, interrupts))
            _mark_cancelled(runs, interrupts.reason)
            result = _bundle_run(plan, materialization, runs, wanted, every=not tasks and not evals)
            if not interrupts.count:
                return result
            if interrupts.count == 1:
                try:
                    result = result.teardown()
                except RunInterrupted as stopped:
                    result = stopped.result
            raise RunInterrupted(result, interrupts.signum)


def _mark_cancelled(runs: tuple[TaskRun, ...], reason: str) -> None:
    """Mark each run a signal stopped mid-way as cancelled. It runs once the loop has closed, which waits for every
    store write the runs' unwinding started, so the mark lands last: a failure write a second signal cut short
    can't overwrite it."""
    for run in runs:
        if run.cancelled and run.instance_id:
            with run_scope(run.task.id):
                record_task_cancelled(run.instance_id, reason, Task._utc_now_str())


def _bundle_run(plan: Plan, materialization: Materialization, runs: tuple[TaskRun, ...], wanted: set[str],
                every: bool) -> BundleRun:
    by_id = {run.task.id: run for run in runs}
    eval_runs = tuple(
        EvalRun(entry.entry, materialization.version_of("eval", entry.entry.id),
                tuple(by_id[ref.id] for ref in entry.references))
        for entry in plan.evals
    )
    skipped = ()
    if every:
        skipped = tuple(entry.entry for entry in plan.bundle.entries
                        if entry.entry.kind is BundleKind.TASK and entry.entry.id not in wanted)
    return BundleRun(materialization, runs, eval_runs, skipped)


def refuse_store_tasks(plan: Plan) -> None:
    """Refuse a selected eval that names a store task: a bundle's evals run only the bundle's own tasks."""
    problems = [
        f"{_path(plan, entry.entry)}: {ref.where.partition('.')[0]}: {ref.id!r} is a store task, and a bundle's "
        f"evals run only the bundle's own tasks for now; run it on its own with {_task_run_command(ref)}"
        for entry in plan.evals for ref in entry.references if ref.local is None
    ]
    if problems:
        raise BundleError(problems)


async def _run_all(
    plan: Plan, to_run: list[tuple[BundleEntry, Task]], model: str | None, sandbox: str | None, keep: bool,
    say: Callable[[str], None], interrupts: Interrupts,
) -> tuple[TaskRun, ...]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_RUNS)
    run_group_id = uuid.uuid4().hex
    running: set[asyncio.Task] = set()  # the runs holding a slot

    async def settle(context: TaskStepContext, cancelled: bool) -> TeardownReport | None:
        """Tear a run down unless it's kept. A second signal stops that."""
        try:
            if keep and not cancelled:
                return None
            if interrupts.count > 1:
                return TeardownReport.skipped(context)
            return await teardown_run(context)
        except asyncio.CancelledError:
            return TeardownReport.skipped(context)

    async def run_one(entry: BundleEntry, task: Task) -> TaskRun:
        tag = f"[{_path(plan, entry)}]"
        metadata = {"run_group_id": run_group_id, "task_id": task.id}
        if sandbox:
            metadata["user_overrides"] = {"env_sandbox": sandbox, "agent_sandbox": sandbox, "sandbox": sandbox}
        context = TaskStepContext(metadata=metadata)
        try:
            await semaphore.acquire()
        except asyncio.CancelledError:
            return TaskRun(entry, task, context, None, 0.0, cancelled=True, started=False, torn_down=TeardownReport())
        me = asyncio.current_task()
        running.add(me)
        began = time.monotonic()
        error, cancelled = None, False
        try:
            try:
                await task.run(
                    on_step_start=lambda i, total, step, _: say(f"{tag} step {i + 1}/{total} {step.id} ({step.type})"),
                    on_step_complete=lambda i, total, step, _, seconds: say(
                        f"{tag} step {i + 1}/{total} {step.id} done in {seconds:.1f}s"),
                    agent_model=model,
                    context=context,
                )
            except Exception as e:
                error = e
            except asyncio.CancelledError:
                if me.cancelling() == 0:  # a step's own, not a signal's
                    await teardown_run(context)
                    raise
                cancelled = True
            duration = time.monotonic() - began
            report = await settle(context, cancelled)
        finally:
            running.discard(me)
            semaphore.release()
        run = TaskRun(entry, task, context, error, duration, cancelled=cancelled, torn_down=report)
        say(f"{tag} {run.outcome.value} in {run.duration:.1f}s")
        for sandbox_record, why in (report.failed if report else ()):
            say(f"{tag} couldn't tear down {sandbox_record.sandbox_id}: {why}")
        return run

    def on_signal(count: int) -> None:
        if count == 1:
            say(f"Cancelling: tearing down {len(running)} run{'s' if len(running) != 1 else ''} "
                "(Ctrl-C again to stop now)")
        else:
            say("Stopping the teardown now")

    runs = await interrupts.gather((run_one(entry, task) for entry, task in to_run), on_signal)
    if any(run.cancelled() for run in runs):
        kept = [run.result().context for run in runs if not run.cancelled() and run.result().torn_down is None]
        await asyncio.gather(*(teardown_run(context) for context in kept))
        raise asyncio.CancelledError
    return tuple(run.result() for run in runs)


def _refuse_a_running_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError("run_bundle runs its own event loop, so it can't be called from a running one; call it in a "
                       "thread instead, e.g. await asyncio.to_thread(run_bundle, root)")


def _progress(on_progress: Callable[[str], None] | None) -> Callable[[str], None]:
    """``on_progress``, made safe to call from a step callback, where raising would fail the task. Once it
    raises, as printing to a closed pipe does, it isn't called again."""
    broken = on_progress is None

    def say(line: str) -> None:
        nonlocal broken
        if broken:
            return
        try:
            on_progress(line)
        except Exception as e:
            broken = True
            logger.warning("progress stopped: on_progress raised %r", e)

    return say


def _task_run_command(ref: Reference) -> str:
    version = "" if ref.version is None else f" --version {ref.version}"
    return f"agent-env task run --id {shlex.quote(ref.id)}{version}"


def _written(plan: Plan, done: Materialized) -> str:
    what = f"{_label(plan, done.write)}: v{done.version}"
    return f"{what}, unchanged" if done.reused else f"{what} ({'; '.join(done.reasons)})"


def _label(plan: Plan, write: Write) -> str:
    """How a write is named: its entry's folder, and for an image built from it, which image."""
    path = _path(plan, write.source.entry)
    return f"{path} ({write.source.dockerfile} image)" if isinstance(write.source, BuiltImage) else path


def _path(plan: Plan, entry: BundleEntry) -> str:
    return relative(plan.bundle.bundle.root, entry.path)
