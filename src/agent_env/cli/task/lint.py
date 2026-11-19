"""Check a task for what would make it run or grade wrong, before any agent runs."""

import json
from pathlib import Path

import click


@click.command()
@click.argument("path", required=False, type=click.Path(exists=True, path_type=Path))
@click.option("--id", "task_id", default=None, help="A stored task's id, in place of PATH")
@click.option("--version", "task_version", default=None, type=int, help="Its version (defaults to latest)")
def lint(path: Path | None, task_id: str | None, task_version: int | None):
    """Lint a task: a bundle folder's tasks, one task file, or a stored task (--id).

    \b
    Errors are configs that can't run or grade correctly: an unknown field, a step's
    own config problems, a headline that isn't a verifier, an aggregate source no
    verifier writes. They also refuse agent-env run, bundle register, task create and
    eval run. Warnings block nothing. Exits non-zero when there are errors. A task
    file on its own is checked without a store; a bundle folder is planned as a run
    plans it and --id reads the stored task, so both read the configured stores.
    """
    if (path is None) == (task_id is None):
        raise click.UsageError("give a bundle folder or task file, or --id, but not both")
    if task_id is not None:
        _report(*_stored(task_id, task_version))
    elif path.is_dir():
        _report(*_bundle(path))
    elif path.parent.name == "tasks" and path.suffix == ".json":
        _report(*_bundle(path.parent.parent, path.stem))
    else:
        _report(*_file(path))


def _stored(task_id: str, task_version: int | None) -> tuple[list[str], list[str]]:
    from agent_env.task import Task
    from agent_env.task.lint import Severity, lint as lint_task

    task = Task.get(task_id, task_version)
    findings = lint_task(task)
    where = f"{task.id} v{task.version}"
    return ([f"{where}: {f.text}" for f in findings if f.severity is Severity.ERROR],
            [f"{where}: {f}" for f in findings if f.severity is Severity.WARNING])


def _bundle(root: Path, task: str | None = None) -> tuple[list[str], list[str]]:
    """A bundle's tasks as ``agent-env run`` plans them, so its references resolve as a run resolves them."""
    from agent_env.bundle import BundleError, BundleKind, parse_bundle
    from agent_env.bundle.installed import installed_bundles
    from agent_env.bundle.plan import plan_bundle
    from agent_env.bundle.resolve import resolve_bundle
    from agent_env.store.routing import namespace_routing

    with namespace_routing():
        try:
            bundle = parse_bundle(root)
            names = [task] if task else [entry.name for entry in bundle.entries if entry.kind is BundleKind.TASK]
            plan = plan_bundle(resolve_bundle(bundle), tasks=names, installed=installed_bundles)
        except BundleError as e:
            return list(e.problems), []
    return [], list(plan.lint_warnings)


def _file(path: Path) -> tuple[list[str], list[str]]:
    """A task file on its own: its references name store ids, as ``task create`` reads them."""
    from agent_env.bundle.parse import task_file_parts
    from agent_env.bundle.resolve import build_step
    from agent_env.plugins import _registration
    from agent_env.task import Task
    from agent_env.task.lint import Severity, lint as lint_task, unknown_fields
    from agent_env.task_step.registry import get_task_step_registry

    try:
        steps, headlines = task_file_parts(json.loads(path.read_text()))
    except (ValueError, json.JSONDecodeError) as e:
        return [f"{path}: {e}"], []
    if unknown := list(unknown_fields(steps)):
        return [f"{path}: {f.text}" for f in unknown], []
    registry, built, problems = get_task_step_registry(), [], []
    for step in steps:
        step_type = step.get("type")
        if not isinstance(step_type, str) or registry.get(step_type) is None:
            note = _registration.failure_note(_registration.TASK_STEPS, step_type) if isinstance(step_type, str) else ""
            problems.append(f"step {step.get('id')!r}: {step_type!r} is not a known step type{note}")
            continue
        try:
            built.append(build_step(step))
        except Exception as e:
            problems.append(f"step {step.get('id')!r}: {step_type} can't read it ({type(e).__name__}: {e})")
    if problems:
        return [f"{path}: {problem}" for problem in problems], []
    try:
        task = Task(id=path.stem, version=None, steps=built, headline_verifiers=headlines)
    except ValueError as e:
        return [f"{path}: {e}"], []
    findings = lint_task(task, steps)
    return ([f"{path}: {f.text}" for f in findings if f.severity is Severity.ERROR],
            [f"{path}: {f}" for f in findings if f.severity is Severity.WARNING])


def _report(errors: list[str], warnings: list[str]) -> None:
    for line in warnings:
        click.echo(line)
    if errors:
        click.echo(f"{len(errors)} error{'s' * (len(errors) != 1)}:", err=True)
        for line in errors:
            click.echo(f"  - {line}", err=True)
        raise SystemExit(1)
    count = f"{len(warnings)} warning{'s' * (len(warnings) != 1)}" if warnings else "no problems"
    click.echo(f"task lint: {count}")
