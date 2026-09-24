"""Get a task definition and output its JSON."""

import json

import click


@click.command()
@click.option("--id", "task_id", required=True, help="Task id")
@click.option("--version", "task_version", default=None, type=int, help="Task version (defaults to latest)")
def get(task_id: str, task_version: int | None):
    """Get a task and output its step definitions as JSON."""
    from agent_env.task import Task

    task = Task.get(task_id, task_version)
    steps = task.to_dict()["steps"]
    click.echo(json.dumps(steps, indent=2))
