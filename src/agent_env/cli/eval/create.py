"""Create an eval from a JSON file listing tasks."""

import json
from pathlib import Path

import click


@click.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--id", "eval_id", required=True, help="Eval id")
def create(filepath: Path, eval_id: str):
    """Create an eval from a JSON file of task references.

    \b
    The JSON file should contain a list of task reference dicts, e.g.:
    [
        {"task_id": "my-task-1", "task_version": 1},
        {"task_id": "my-task-2"}
    ]

    task_version is optional; omit it to use the latest version.
    """
    from agent_env.eval import Eval, EvalTask
    from agent_env.task import Task

    click.echo(f"Reading tasks from {filepath}...")
    with open(filepath) as f:
        raw_tasks = json.load(f)

    if not isinstance(raw_tasks, list):
        click.echo("Error: JSON file must contain a list of task reference dicts", err=True)
        raise SystemExit(1)

    eval_tasks: list[EvalTask] = []
    for i, entry in enumerate(raw_tasks):
        task_id = entry.get("task_id")
        if not task_id:
            click.echo(f"Error: entry {i} missing task_id: {entry}", err=True)
            raise SystemExit(1)
        task_version = entry.get("task_version")

        task = Task.get(task_id, version=task_version)
        click.echo(f"  Validated task {i + 1}/{len(raw_tasks)}: {task.id} v{task.version} ({len(task.steps)} steps)")
        eval_tasks.append(EvalTask(task_id=task_id, task_version=task_version))

    click.echo(f"Creating eval '{eval_id}' with {len(eval_tasks)} tasks...")
    eval_obj = Eval.put(id=eval_id, tasks=eval_tasks)
    click.echo(f"Created eval: id={eval_obj.id} version={eval_obj.version} tasks={len(eval_obj.tasks)}")
