"""Hand runs to the configured ``[runner]`` and follow them to the end. Progress comes from the run index and the
task instance, the records every runner keeps."""

from __future__ import annotations

import asyncio
import functools
from dataclasses import dataclass, field

import click

from agent_env.config import get_runner
from agent_env.runner.runner import RunHandle, RunRecord, RunStatus
from agent_env.task import Task
from agent_env.task.interrupts import Interrupts, tearing_down
from agent_env.task.step_journal import replay_context
from agent_env.task.store import TaskInstance, TaskStepStatus, find_task_instance, get_task_instance_store
from agent_env.task.task import substitute_seed_in_steps
from agent_env.task_step.context import TaskStepContext

POLL_SECONDS = 2.0


@dataclass
class SubmittedRun:
    tag: str
    task: Task
    handle: RunHandle
    record: RunRecord | None = None
    instance: TaskInstance | None = None
    steps: set[tuple[str, str]] = field(default_factory=set)
    definitions: list | None = None

    @property
    def done(self) -> bool:
        return self.record is not None and self.record.status.is_terminal

    @property
    def completed(self) -> bool:
        return self.record is not None and self.record.status is RunStatus.COMPLETED

    @property
    def outcome(self) -> str:
        return f"{self.record.status}" + (f": {self.record.error}" if self.record.error else "")


def follow(interrupts: Interrupts, submissions, runs: list[SubmittedRun], *, step_lines=None, **submit_kwargs) -> None:
    """Submit each ``(tag, task, metadata, instance_id)`` into ``runs`` and follow the runs to the end, echoing each step
    that ends (with ``step_lines(step, context)`` when given, else a verifier's score). A first Ctrl-C cancels the
    runs; a second stops waiting for them. ``runs`` holds what was submitted even when following raises."""
    [outcome] = interrupts.run(interrupts.gather([_submit_and_follow(submissions, runs, step_lines, **submit_kwargs)],
                                                 _on_signal))
    if not outcome.cancelled() and outcome.exception() is not None:
        raise outcome.exception()


async def _submit_and_follow(submissions, runs, step_lines, *, limit=None, **submit_kwargs) -> None:
    """``stop()`` is always awaited, spared from a first Ctrl-C so a ``LocalRunner`` finishes tearing its runs down,
    short of a second. Never ``start()``: that is the serving process's."""
    runner = get_runner()
    submit = functools.partial(runner.submit, **submit_kwargs)
    pending = list(submissions)
    try:
        try:
            await _follow(runner, submit, pending, runs, limit, step_lines)
        except asyncio.CancelledError:
            pending.clear()
            await asyncio.gather(*(runner.cancel(run.handle.run_id) for run in runs if not run.done))
            await _follow(runner, submit, pending, runs, limit, step_lines)
            raise
    finally:
        if asyncio.current_task().cancelling() < 2:
            tearing_down()
            await runner.stop()


async def _follow(runner, submit, pending, runs: list[SubmittedRun], limit, step_lines) -> None:
    while True:
        await _poll(runner, runs, step_lines)
        while pending and (limit is None or sum(not run.done for run in runs) < limit):
            tag, task, metadata, instance_id = pending.pop(0)
            handle = await submit(task.id, task.version, metadata=metadata, instance_id=instance_id)
            _echo(tag, f"Submitted run {handle.run_id} (instance {handle.instance_id})", "blue")
            runs.append(SubmittedRun(tag, task, handle))
        if not pending and all(run.done for run in runs):
            return
        await asyncio.sleep(POLL_SECONDS)


def _on_signal(count: int) -> None:
    message = "Cancelling the runs (Ctrl-C again to stop waiting for them)" if count == 1 else (
        "Stopped waiting; what the runs deployed may still be up")
    click.echo(click.style(message, fg="yellow"))


async def _poll(runner, runs: list[SubmittedRun], step_lines) -> None:
    """Reads every run before raising the first read that failed, so a run that ended is seen as ended."""
    errors = []
    for run in runs:
        if not run.done:
            try:
                await _poll_run(runner, run, step_lines)
            except Exception as e:
                errors.append(e)
    if errors:
        raise errors[0]


async def _poll_run(runner, run: SubmittedRun, step_lines) -> None:
    record = await runner.status(run.handle.run_id)
    run.instance = find_task_instance(run.handle.instance_id) or run.instance
    for step in run.instance.completed_steps if run.instance else []:
        if (step.step_id, step.status) not in run.steps:
            run.steps.add((step.step_id, step.status))
            _echo_step(run, step.step_id, step.status == TaskStepStatus.SUCCESS, step_lines)
    if record is not None and (run.record is None or record.status != run.record.status):
        run.record = record
        _echo(run.tag, f"Run {run.outcome}",
              "green" if record.status is RunStatus.COMPLETED else "red" if record.status.is_terminal else "yellow")


def _echo_step(run: SubmittedRun, step_id: str, passed: bool, step_lines) -> None:
    if run.definitions is None:
        run.definitions = substitute_seed_in_steps(
            run.task.steps, ((run.instance.context or {}).get("metadata") or {}).get("seed"))
    i, step = next((i, step) for i, step in enumerate(run.definitions) if step.id == step_id)
    _echo(run.tag, f"{'Completed' if passed else 'Failed'} step [{i + 1}/{len(run.definitions)}]: {step.type} "
                   f"(id={step.id})", "green" if passed else "red")
    vid = getattr(step, "verifier_id", None)
    if not passed or not (step_lines or vid):
        return
    context = _step_context(run.instance, step_id)
    if step_lines:
        for line in step_lines(step, context):
            _echo(run.tag, line, "green")
    elif (score := context.metadata.get("verifications", {}).get(vid, {}).get("score")) is not None:
        _echo(run.tag, f"score (verifier_id={vid}): {score}", "green")


def _step_context(instance: TaskInstance, step_id: str) -> TaskStepContext:
    """What ``step_id`` wrote, replayed from its journal entry; the instance's whole context when it has none."""
    entries = get_task_instance_store().journal_entries_sync(instance.instance_id)
    entry = next((e for e in entries if e["step_id"] == step_id), None)
    return TaskStepContext.from_dict(replay_context([entry], ()) if entry else instance.context or {})


def _echo(tag: str, message: str, color: str) -> None:
    click.echo(click.style(f"{tag} {message}" if tag else message, fg=color))
