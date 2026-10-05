"""Create a task from a JSON definition file."""

import json
from pathlib import Path

import click

from agent_env.plugins import _registration


@click.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--id", "task_id", required=True, help="Task id")
@click.option(
    "--skip-validation",
    is_flag=True,
    help="Save even if preflight finds problems (they are printed either way).",
)
def create(filepath: Path, task_id: str, skip_validation: bool):
    """Create a task from a JSON file containing step definitions.

    \b
    The JSON file should contain a list of step dicts, e.g.:
    [
        {"id": "step-1", "type": "deploy_env", "env_id": "my-env"},
        {"id": "step-2", "type": "prompt_agent", "prompt": "Hello"}
    ]
    """
    from agent_env.task import Task

    click.echo(f"Reading steps from {filepath}...")
    with open(filepath) as f:
        steps = json.load(f)

    if not isinstance(steps, list):
        click.echo("Error: JSON file must contain a list of step dicts", err=True)
        raise SystemExit(1)

    built = _build_steps(steps)
    _report_preflight(
        Task(id=task_id, version=None, steps=[s for _, s in built]),
        skip_validation,
    )

    task_steps = []
    for i, (kwargs, step) in enumerate(built):
        saved = type(step).put(**kwargs)
        click.echo(f"  Step {i + 1}/{len(built)}: {step.type} id={saved.id} version={saved.version}")
        task_steps.append(saved)

    click.echo(f"Creating task '{task_id}' with {len(task_steps)} steps...")
    task = Task.put(id=task_id, steps=task_steps)
    click.echo(f"Created task: id={task.id} version={task.version} steps={len(task.steps)}")


def _build_steps(steps: list[dict]) -> list[tuple[dict, object]]:
    """Instantiate every step as ``(kwargs, step)`` without writing anything.

    `step_cls.put` writes unconditionally, so a create that preflight rejects must
    not have left step documents behind — one extra version per retry.
    """
    from agent_env.task_step.registry import get_task_step_registry

    registry = get_task_step_registry()
    built = []
    for step_dict in steps:
        step_type = step_dict["type"]
        step_cls = registry.get(step_type)
        if step_cls is None:
            click.echo(
                f"Error: Unknown task step type: {step_type}{_registration.failure_note(_registration.TASK_STEPS, step_type)}",
                err=True,
            )
            raise SystemExit(1)
        kwargs = {k: v for k, v in step_dict.items() if k != "type"}
        kwargs.setdefault("version", None)
        # retry_config lives on the TaskStep base but most step __init__s don't
        # accept it; construct without it and hydrate (as Task.from_dict does),
        # so retry_config on any step type doesn't crash create. put() below
        # handles the kwargs form the same way.
        from agent_env.task_step.task_step import attach_retry_config

        ctor_kwargs = {k: v for k, v in kwargs.items() if k != "retry_config"}
        step = attach_retry_config(step_cls(**ctor_kwargs), step_dict)
        built.append((kwargs, step))
    return built


def _report_preflight(task, skip_validation: bool) -> None:
    problems = task.preflight()
    if not problems:
        return
    click.echo(f"Preflight found {len(problems)} problem(s):", err=True)
    for p in problems:
        click.echo(f"  - {p}", err=True)
    if not skip_validation:
        click.echo("Nothing was saved. Re-run with --skip-validation to save anyway.", err=True)
        raise SystemExit(1)
    click.echo("Saving anyway (--skip-validation).", err=True)
