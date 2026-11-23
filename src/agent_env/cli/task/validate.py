"""Check a task's step config without running it."""

import click


@click.command()
@click.option("--id", "task_id", required=True, help="Task id")
@click.option("--version", "task_version", default=None, type=int, help="Task version (defaults to latest)")
def validate(task_id: str, task_version: int | None):
    """Validate a stored task's step config without deploying anything.

    Reports what task lint finds, steps whose script artifact does not resolve, or
    resolves to the wrong type, and deploy_env steps that couldn't deploy their env.
    Exits non-zero when there are problems (lint's warnings aren't), so it can gate CI.
    """
    from agent_env.task import Task
    from agent_env.task.lint import Severity, lint
    from agent_env.task_step.task_step import TaskStep

    task = Task.get(task_id, task_version)
    findings = lint(task, config=False)
    for finding in findings:
        if finding.severity is Severity.WARNING:
            click.echo(f"{task.id} v{task.version}: {finding}", err=True)
    problems = [finding.text for finding in findings if finding.severity is Severity.ERROR] + task.preflight()
    # Most step types have no preflight, so report coverage rather than implying a full check.
    checked = sum(1 for s in task.steps if type(s).preflight is not TaskStep.preflight)
    if problems:
        click.echo(f"{task.id} v{task.version}: {len(problems)} problem(s)", err=True)
        for p in problems:
            click.echo(f"  - {p}", err=True)
        raise SystemExit(1)
    click.echo(
        f"{task.id} v{task.version}: {checked} of {len(task.steps)} step(s) have "
        f"preflight checks; no problems found"
    )
