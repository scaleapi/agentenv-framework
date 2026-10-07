"""Run all tasks in an eval concurrently, tearing down each run's sandboxes as it ends."""

import asyncio
import contextlib
import copy
import json
import os
from uuid import uuid4

import click

from agent_env.cli.banner import print_banner
from agent_env.cli.identity import get_agent_env_client_id
from agent_env.cli.teardown_output import echo_teardown
from agent_env.store.ids import fs_safe
from agent_env.task.interrupts import Interrupts
from agent_env.task.teardown import teardown_run
from agent_env.task_step.context import TaskStepContext


def _make_callbacks(tag: str):
    """Return (on_step_start, on_step_complete) callbacks with a prefix tag."""
    last_score = None

    def on_step_start(i, total, step, context):
        click.echo(click.style(
            f"{tag} Running step [{i+1}/{total}]: {step.type} (id={step.id})...",
            fg="yellow",
        ))

    def on_step_complete(i, total, step, context, duration):
        nonlocal last_score
        click.echo(click.style(
            f"{tag} Completed step [{i+1}/{total}]: {step.type} (id={step.id}) [{duration:.1f}s]",
            fg="green",
        ))

        score = context.metadata.get("score")
        if score is not None and score != last_score:
            click.echo(click.style(f"{tag} score: {score}", fg="green"))
            last_score = score

    return on_step_start, on_step_complete


def _write_context(context, task_id, output_dir, prefix=""):
    """Write context JSON to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{fs_safe(task_id)}_{uuid4().hex[:8]}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(context.to_safe_dict(), f, indent=2)
    click.echo(click.style(f"{prefix}Output written to: {output_path}", fg="blue"))


async def _run_single(task, tag, output_dir, agent_model=None, agent_artifact_id=None, base_metadata=None):
    """Execute a single task run with logging callbacks, then tear down what it deployed, even when it
    raised or was cancelled."""
    on_start, on_complete = _make_callbacks(tag)
    context = TaskStepContext(metadata=copy.deepcopy(base_metadata or {}))
    try:
        await task.run(
            on_step_start=on_start,
            on_step_complete=on_complete,
            agent_model=agent_model,
            agent_artifact_id=agent_artifact_id,
            context=context,
        )
    finally:
        echo_teardown(tag, await teardown_run(context))
    if output_dir:
        _write_context(context, task.id, output_dir, prefix=f"{tag} ")
    return task.id, tag, context


@click.command()
@click.option("--id", "eval_id", required=True, help="Eval id")
@click.option("--version", "eval_version", default=None, type=int, help="Eval version (defaults to latest)")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False), help="Directory to write per-task context JSON output")
@click.option("--k", "k", default=1, type=int, help="Number of parallel runs per task")
@click.option("--max-concurrency", default=None, type=int, help="Maximum number of concurrent task runs")
@click.option("--agent-model", default=None, type=str, help="Override the model used by prompt_agent steps")
@click.option("--agent-artifact-id", default=None, type=str, help="Override the agent image artifact for deploy_agent steps")
def run(eval_id: str, eval_version: int | None, output_dir: str | None, k: int, max_concurrency: int | None, agent_model: str | None, agent_artifact_id: str | None):
    """Run all tasks in an eval concurrently."""
    if k < 1:
        raise click.BadParameter("must be at least 1", param_hint="'--k'")
    if max_concurrency is not None and max_concurrency < 1:
        raise click.BadParameter("must be at least 1", param_hint="'--max-concurrency'")

    from agent_env.eval import Eval
    from agent_env.task import Task

    click.echo(f"Fetching eval: id={eval_id} version={eval_version or 'latest'}...")
    eval_obj = Eval.get(eval_id, version=eval_version)
    click.echo(f"Found eval: id={eval_obj.id} version={eval_obj.version} tasks={len(eval_obj.tasks)}")

    tasks = []
    for et in eval_obj.tasks:
        task = Task.get(et.task_id, version=et.task_version)
        version_note = f" (resolved from latest)" if et.task_version is None else ""
        click.echo(f"  Loaded task: {task.id} v{task.version} ({len(task.steps)} steps){version_note}")
        tasks.append(task)

    print_banner()

    total_runs = len(tasks) * k
    concurrency_note = f" (max concurrency: {max_concurrency})" if max_concurrency else ""
    click.echo(click.style(
        f"Running {len(tasks)} tasks x {k} run(s) = {total_runs} total runs{concurrency_note}...",
        fg="blue",
    ))
    click.echo()

    client_id = get_agent_env_client_id()
    base_metadata = {"agent_env_hub": {"caller": client_id}} if client_id else None

    async def _run_safe(task, tag, slot):
        """Wrap _run_single so failures, and a cancel while it waits for its slot, carry task identity."""
        try:
            async with slot:
                return await _run_single(
                    task,
                    tag,
                    output_dir,
                    agent_model=agent_model,
                    agent_artifact_id=agent_artifact_id,
                    base_metadata=base_metadata,
                )
        except BaseException as exc:
            return (task.id, tag, exc)

    def _on_signal(count):
        message = "Cancelling the runs and tearing them down (Ctrl-C again to stop now)" if count == 1 else (
            "Stopping the teardown now")
        click.echo(click.style(message, fg="yellow"))

    async def _run_all(interrupts):
        slot = asyncio.Semaphore(max_concurrency) if max_concurrency else contextlib.nullcontext()
        coros = []
        for task in tasks:
            for run_idx in range(1, k + 1):
                if k > 1:
                    tag = f"[{task.id} run {run_idx}]"
                else:
                    tag = f"[{task.id}]"
                coros.append(_run_safe(task, tag, slot))
        return [run.result() for run in await interrupts.gather(coros, _on_signal)]

    with Interrupts() as interrupts:
        results = interrupts.run(_run_all(interrupts))

    click.echo()

    # Summarize results
    failures = []
    successes = []
    for result in results:
        task_id, tag, payload = result
        if isinstance(payload, BaseException):
            failures.append((task_id, tag, payload))
        else:
            score = payload.metadata.get("score")
            successes.append((task_id, tag, score))

    if successes:
        click.echo(click.style("Results:", fg="blue"))
        for task_id, tag, score in successes:
            score_str = f" score={score}" if score is not None else ""
            click.echo(click.style(f"  {tag} PASSED{score_str}", fg="green"))

    if failures:
        for task_id, tag, exc in failures:
            outcome = "CANCELLED" if isinstance(exc, asyncio.CancelledError) else f"FAILED: {exc}"
            click.echo(click.style(f"  {tag} {outcome}", fg="red"))
        click.echo(click.style(
            f"\n{len(successes)}/{total_runs} runs completed, {len(failures)}/{total_runs} failed.",
            fg="red",
        ))
    else:
        click.echo(click.style(f"\nAll {total_runs} runs completed!", fg="blue"))
    if interrupts.count:
        raise SystemExit(128 + interrupts.signum)
    if failures:
        raise SystemExit(1)
