"""Add tasks to an existing eval, creating a new version."""

import json
from pathlib import Path

import click


@click.command("add-tasks")
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--id", "eval_id", required=True, help="Eval id")
@click.option("--version", "eval_version", default=None, type=int, help="Eval version (defaults to latest)")
def add_tasks(filepath: Path, eval_id: str, eval_version: int | None):
    """Add tasks from a JSON file to an existing eval.

    Creates a new eval version with the merged task list.
    Duplicate task_ids are replaced with the new entry.

    \b
    The JSON file should contain a list of task reference dicts, e.g.:
    [
        {"task_id": "new-task", "task_version": 2},
        {"task_id": "another-task"}
    ]

    task_version is optional; omit it to use the latest version.
    """
    from agent_env.eval import Eval, EvalTask
    from agent_env.task import Task

    click.echo(f"Fetching eval: id={eval_id} version={eval_version or 'latest'}...")
    existing_eval = Eval.get(eval_id, version=eval_version)
    click.echo(f"Found eval: id={existing_eval.id} version={existing_eval.version} tasks={len(existing_eval.tasks)}")

    click.echo(f"Reading new tasks from {filepath}...")
    with open(filepath) as f:
        raw_tasks = json.load(f)

    if not isinstance(raw_tasks, list):
        click.echo("Error: JSON file must contain a list of task reference dicts", err=True)
        raise SystemExit(1)

    new_tasks: list[EvalTask] = []
    for i, entry in enumerate(raw_tasks):
        task_id = entry.get("task_id")
        if not task_id:
            click.echo(f"Error: entry {i} missing task_id: {entry}", err=True)
            raise SystemExit(1)
        task_version = entry.get("task_version")

        task = Task.get(task_id, version=task_version)
        click.echo(f"  Validated task {i + 1}/{len(raw_tasks)}: {task.id} v{task.version} ({len(task.steps)} steps)")
        new_tasks.append(EvalTask(task_id=task_id, task_version=task_version))

    # Merge: existing tasks keyed by task_id, overwritten by new entries
    merged: dict[str, EvalTask] = {t.task_id: t for t in existing_eval.tasks}
    for t in new_tasks:
        merged[t.task_id] = t
    merged_tasks = list(merged.values())

    click.echo(f"Creating new eval version with {len(merged_tasks)} tasks ({len(new_tasks)} added/updated)...")
    eval_obj = Eval.put(id=eval_id, tasks=merged_tasks)
    click.echo(f"Created eval: id={eval_obj.id} version={eval_obj.version} tasks={len(eval_obj.tasks)}")
