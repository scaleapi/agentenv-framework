"""Run all tasks in an eval through the configured ``[runner]``, following each run to its end."""

import copy
import json
import os
from uuid import uuid4

import click

from agent_env.cli import submitted_runs
from agent_env.cli.utils import judge_samples_option, min_judge_agreement_option
from agent_env.cli.banner import print_banner
from agent_env.cli.identity import get_agent_env_client_id
from agent_env.cli.usage_output import cost_line, sources_line
from agent_env.store.ids import fs_safe
from agent_env.task.interrupts import Interrupts
from agent_env.task.usage.summary import summarize
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
@judge_samples_option
@min_judge_agreement_option
@click.option("--agent-artifact-id", default=None, type=str, help="Override the agent image artifact for deploy_agent steps")
@click.option("--skip-lint", is_flag=True,
              help="Run tasks agent-env task lint finds errors in, saying them, as eval run did before it linted")
def run(eval_id: str, eval_version: int | None, output_dir: str | None, k: int, max_concurrency: int | None, agent_model: str | None, user_model: str | None, judge_model: str | None, judge_samples: int | None, min_judge_agreement: float | None, agent_artifact_id: str | None, skip_lint: bool):
    """Run all tasks in an eval through the configured [runner], concurrently."""
    if k < 1:
        raise click.BadParameter("must be at least 1", param_hint="'--k'")
    if max_concurrency is not None and max_concurrency < 1:
        raise click.BadParameter("must be at least 1", param_hint="'--max-concurrency'")

    from agent_env.eval import Eval
    from agent_env.task import Task
    from agent_env.task.lint import UNPINNED, lint, refuses, said, unpinned_summary

    click.echo(f"Fetching eval: id={eval_id} version={eval_version or 'latest'}...")
    eval_obj = Eval.get(eval_id, version=eval_version)
    click.echo(f"Found eval: id={eval_obj.id} version={eval_obj.version} tasks={len(eval_obj.tasks)}")

    tasks = []
    for et in eval_obj.tasks:
        task = Task.get(et.task_id, version=et.task_version)
        version_note = " (resolved from latest)" if et.task_version is None else ""
        click.echo(f"  Loaded task: {task.id} v{task.version} ({len(task.steps)} steps){version_note}")
        tasks.append(task)

    findings = [(f"{task.id} v{task.version}", finding) for task in tasks for finding in lint(task)]
    refused = [] if skip_lint else [(task, f) for task, f in findings if refuses(f)]
    if refused:
        click.echo(f"Task lint found {len(refused)} error(s), so no run started (--skip-lint runs anyway):", err=True)
        for task, finding in refused:
            click.echo(f"  - {task}: {finding.text}", err=True)
        raise SystemExit(1)
    for line in [*said((task, f) for task, f in findings if f.rule != UNPINNED), unpinned_summary(findings)]:
        if line is not None:
            click.echo(f"  {line}")

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
    overrides = {key: value for key, value in (("user_model", user_model), ("judge_model", judge_model),
                                               ("judge_samples", judge_samples)) if value}
    if min_judge_agreement is not None:
        overrides["min_judge_agreement"] = min_judge_agreement
    if overrides:
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
    entries = []
    for run in runs:
        context = run.instance.context if run.instance else None
        if output_dir and context is not None:
            _write_context(TaskStepContext.from_dict(context), run.task.id, output_dir, prefix=f"{run.tag} ")
        ledger = run.instance.usage if run.instance else []
        summary = summarize(ledger)
        entries += ledger
        if run.completed:
            successes.append((run.tag, (context or {}).get("metadata", {}).get("score"), summary))
        else:
            failures.append((run.tag, run.outcome if run.done else "CANCELLED", summary))

    if successes:
        click.echo(click.style("Results:", fg="blue"))
        for tag, score, summary in successes:
            score_str = f" score={score}" if score is not None else ""
            click.echo(click.style(f"  {tag} PASSED{score_str}", fg="green"))
            if cost := cost_line(summary):
                click.echo(f"    {cost}")

    if len(successes) < total_runs:
        for tag, outcome, summary in failures:
            click.echo(click.style(f"  {tag} {outcome}", fg="red"))
            if cost := cost_line(summary):
                click.echo(f"    {cost}")
        click.echo(click.style(
            f"\n{len(successes)}/{total_runs} runs completed, {total_runs - len(successes)}/{total_runs} failed.",
            fg="red",
        ))
    else:
        click.echo(click.style(f"\nAll {total_runs} runs completed!", fg="blue"))
    total = summarize(entries)
    if cost := cost_line(total, f"Cost over {len(runs)} runs", attempts=False):
        click.echo(cost)
    if sources := sources_line(total):
        click.echo(sources)
    if interrupts.count:
        raise SystemExit(128 + interrupts.signum)
    if len(successes) < total_runs:
        raise SystemExit(1)
