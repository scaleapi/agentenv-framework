"""``--submit``: hand runs to the configured ``[runner]`` and follow them to the end instead of running them in this
process. Progress comes from the run index and the task instance, the records every runner keeps."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import click

from agent_env.config import get_runner
from agent_env.runner.runner import RunHandle, RunRecord, RunStatus
from agent_env.task.interrupts import tearing_down
from agent_env.task.store import TaskInstance, TaskStepStatus, find_task_instance

POLL_SECONDS = 2.0

_STATUS_COLORS = {RunStatus.COMPLETED: "green", RunStatus.FAILED: "red", RunStatus.CANCELED: "red"}


@dataclass
class SubmittedRun:
    tag: str
    task_id: str
    handle: RunHandle
    record: RunRecord | None = None
    steps: set[tuple[str, str]] = field(default_factory=set)

    @property
    def done(self) -> bool:
        return self.record is not None and self.record.status.is_terminal

    @property
    def completed(self) -> bool:
        return self.record is not None and self.record.status is RunStatus.COMPLETED

    def instance(self) -> TaskInstance | None:
        return find_task_instance(self.handle.instance_id)


async def submit_and_follow(submissions, runs: list[SubmittedRun], *, agent_model=None, agent_artifact_id=None) -> None:
    """Submit each ``(tag, task, metadata)``, appending it to ``runs``, and follow every run until it is terminal.

    A cancel, the first Ctrl-C, asks the runner to cancel what is still going and follows it to the end; a second
    stops waiting. However it ends, short of that second one, ``stop()`` is awaited, spared from a first Ctrl-C: it
    lets a ``LocalRunner`` finish tearing its runs down. Never ``start()``: that is the serving process's, and
    ``LocalRunner``'s fails every local run another process has in flight."""
    runner = get_runner()
    stopping_now = False
    try:
        try:
            for tag, task, metadata in submissions:
                handle = await runner.submit(task.id, task.version, agent_model=agent_model,
                                             agent_artifact_id=agent_artifact_id, metadata=metadata)
                _echo(tag, f"Submitted run {handle.run_id} (instance {handle.instance_id})", "blue")
                runs.append(SubmittedRun(tag, task.id, handle))
            await _follow(runner, runs)
        except asyncio.CancelledError:
            try:
                await _cancel_and_follow(runner, runs)
            except asyncio.CancelledError:
                stopping_now = True
                raise
            raise
    finally:
        if not stopping_now:
            tearing_down()
            await runner.stop()


async def _cancel_and_follow(runner, runs: list[SubmittedRun]) -> None:
    for run in runs:
        if not run.done:
            await runner.cancel(run.handle.run_id)
    await _follow(runner, runs)


def on_signal(count: int) -> None:
    message = "Cancelling the runs (Ctrl-C again to stop waiting for them)" if count == 1 else (
        "Stopped waiting; what the runs deployed may still be up")
    click.echo(click.style(message, fg="yellow"))


async def _follow(runner, runs: list[SubmittedRun]) -> None:
    while True:
        for run in runs:
            if run.done:
                continue
            record = await runner.status(run.handle.run_id)
            instance = run.instance()
            for step in instance.completed_steps if instance else []:
                if (step.step_id, step.status) not in run.steps:
                    run.steps.add((step.step_id, step.status))
                    _echo(run.tag, f"Step {step.step_id}: {step.status}",
                          "green" if step.status == TaskStepStatus.SUCCESS else "red")
            if record is not None and (run.record is None or record.status != run.record.status):
                _echo(run.tag, f"Run {record.status}" + (f": {record.error}" if record.error else ""),
                      _STATUS_COLORS.get(record.status, "yellow"))
            run.record = record or run.record
        if all(run.done for run in runs):
            return
        await asyncio.sleep(POLL_SECONDS)


def _echo(tag: str, message: str, color: str) -> None:
    click.echo(click.style(f"{tag} {message}" if tag else message, fg=color))
