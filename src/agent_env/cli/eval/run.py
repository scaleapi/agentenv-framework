"""Run all tasks in an eval concurrently."""

import asyncio
import copy
import json
import os
from uuid import uuid4

import click

from agent_env.cli.banner import print_banner
from agent_env.cli.identity import get_agent_env_client_id
from agent_env.store.ids import fs_safe
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
    """Execute a single task run with logging callbacks."""
    on_start, on_complete = _make_callbacks(tag)
    initial_context = (
        TaskStepContext(metadata=copy.deepcopy(base_metadata))
        if base_metadata
        else None
    )
    context = await task.run(
        on_step_start=on_start,
        on_step_complete=on_complete,
        agent_model=agent_model,
        agent_artifact_id=agent_artifact_id,
        context=initial_context,
    )
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

    async def _run_safe(task, tag, output_dir, agent_model=None, agent_artifact_id=None):
        """Wrap _run_single so failures carry task identity."""
        try:
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

    async def _run_all():
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None
        coros = []
        for task in tasks:
            for run_idx in range(1, k + 1):
                if k > 1:
                    tag = f"[{task.id} run {run_idx}]"
                else:
                    tag = f"[{task.id}]"
                if semaphore:
                    async def _limited(t=task, tg=tag, od=output_dir, am=agent_model, aa=agent_artifact_id):
                        async with semaphore:
                            return await _run_safe(t, tg, od, agent_model=am, agent_artifact_id=aa)
                    coros.append(_limited())
                else:
                    coros.append(_run_safe(task, tag, output_dir, agent_model=agent_model, agent_artifact_id=agent_artifact_id))
        return await asyncio.gather(*coros)

    results = asyncio.run(_run_all())

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
            click.echo(click.style(f"  {tag} FAILED: {exc}", fg="red"))
        click.echo(click.style(
            f"\n{len(successes)}/{total_runs} runs completed, {len(failures)}/{total_runs} failed.",
            fg="red",
        ))
        raise SystemExit(1)
    else:
        click.echo(click.style(f"\nAll {total_runs} runs completed!", fg="blue"))
