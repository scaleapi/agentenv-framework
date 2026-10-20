"""Run all tasks in an eval through the configured ``[runner]``, following each run to its end."""

import copy
import json
import os
from uuid import uuid4

import click

from agent_env.cli import submitted_runs
from agent_env.cli.banner import print_banner
from agent_env.cli.identity import get_agent_env_client_id
from agent_env.store.ids import fs_safe
from agent_env.task.interrupts import Interrupts
from agent_env.task_step.context import TaskStepContext


def _write_context(context, task_id, output_dir, prefix=""):
    """Write context JSON to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{fs_safe(task_id)}_{uuid4().hex[:8]}.json"
    output_path = os.path.join(output_dir, filename)
    with open(output_path, "w") as f:
        json.dump(context.to_safe_dict(), f, indent=2)
    click.echo(click.style(f"{prefix}Output written to: {output_path}", fg="blue"))


@click.command()
@click.option("--id", "eval_id", required=True, help="Eval id")
@click.option("--version", "eval_version", default=None, type=int, help="Eval version (defaults to latest)")
@click.option("--output-dir", default=None, type=click.Path(file_okay=False), help="Directory to write per-task context JSON output")
@click.option("--k", "k", default=1, type=int, help="Number of parallel runs per task")
@click.option("--max-concurrency", default=None, type=int, help="Maximum number of concurrent task runs")
@click.option("--agent-model", default=None, type=str, help="Override the model used by prompt_agent steps")
@click.option("--user-model", default=None, type=str, help="Override the model the user simulator runs on")
@click.option("--judge-model", default=None, type=str, help="Override the model rubrics_verifier judges grade with")
@click.option("--agent-artifact-id", default=None, type=str, help="Override the agent image artifact for deploy_agent steps")
def run(eval_id: str, eval_version: int | None, output_dir: str | None, k: int, max_concurrency: int | None, agent_model: str | None, user_model: str | None, judge_model: str | None, agent_artifact_id: str | None):
    """Run all tasks in an eval through the configured [runner], concurrently."""
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
        version_note = " (resolved from latest)" if et.task_version is None else ""
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
    base_metadata = {"agent_env_hub": {"caller": client_id}} if client_id else {}
    if overrides := {key: value for key, value in (("user_model", user_model), ("judge_model", judge_model)) if value}:
        base_metadata["user_overrides"] = overrides

    submissions = [(f"[{task.id} run {i}]" if k > 1 else f"[{task.id}]", task, copy.deepcopy(base_metadata), None)
                   for task in tasks for i in range(1, k + 1)]
    runs: list[submitted_runs.SubmittedRun] = []
    with Interrupts() as interrupts:
        submitted_runs.follow(interrupts, submissions, runs, agent_model=agent_model,
                              agent_artifact_id=agent_artifact_id, limit=max_concurrency)

    click.echo()

    # Summarize results
    failures = []
    successes = []
    for run in runs:
        context = run.instance.context if run.instance else None
        if output_dir and context is not None:
            _write_context(TaskStepContext.from_dict(context), run.task.id, output_dir, prefix=f"{run.tag} ")
        if run.completed:
            successes.append((run.tag, (context or {}).get("metadata", {}).get("score")))
        else:
            failures.append((run.tag, run.outcome if run.done else "CANCELLED"))

    if successes:
        click.echo(click.style("Results:", fg="blue"))
        for tag, score in successes:
            score_str = f" score={score}" if score is not None else ""
            click.echo(click.style(f"  {tag} PASSED{score_str}", fg="green"))

    if len(successes) < total_runs:
        for tag, outcome in failures:
            click.echo(click.style(f"  {tag} {outcome}", fg="red"))
        click.echo(click.style(
            f"\n{len(successes)}/{total_runs} runs completed, {total_runs - len(successes)}/{total_runs} failed.",
            fg="red",
        ))
    else:
        click.echo(click.style(f"\nAll {total_runs} runs completed!", fg="blue"))
    if interrupts.count:
        raise SystemExit(128 + interrupts.signum)
    if len(successes) < total_runs:
        raise SystemExit(1)
