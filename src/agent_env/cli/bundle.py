"""``agent-env bundle``: start a bundle folder from what a store holds."""

from __future__ import annotations

from pathlib import Path

import click

from agent_env.bundle.importer import BundleImport, import_eval, import_task
from agent_env.cli.utils import interruptible
from agent_env.entity.scopes import FieldScope


@click.group()
def bundle():
    """Create bundle folders, which ``agent-env run`` runs."""


@bundle.command("init")
@click.argument("folder", type=click.Path(file_okay=False, path_type=Path))
@click.option("--from-task", "task_id", default=None, help="Import this stored task, by id.")
@click.option("--from-eval", "eval_id", default=None, help="Import this stored eval and every task it names, by id.")
@click.option("--version", type=int, default=None, help="The version to import (defaults to the latest).")
@click.option("--include-scope", "include_scopes", multiple=True,
              type=click.Choice([scope.value for scope in FieldScope if scope is not FieldScope.TASK]),
              help="Keep the fields of this scope as stored, and leave in the store what the folder can't hold, for a "
                   "folder run against the same store. Repeatable.")
def init(folder: Path, task_id: str | None, eval_id: str | None, version: int | None,
         include_scopes: tuple[str, ...]):
    """Write a stored task, or an eval and its tasks, as a bundle at FOLDER, which must be new or empty.

    Each task is written as the store holds it, and every env, agent and artifact it uses is written into a folder of
    its own, so the bundle runs with no store: each image as the part of its stored build context its Dockerfile
    reads, each data artifact as its files. Fields scoped to who ran it (the agent, the sandbox provider, where outputs
    go, service endpoints, billing tags) are left out, so a run takes its own, unless --include-scope names their scope.
    What a folder can't hold fails the import, unless --include-scope user leaves it in the store, pinned. Values that
    look like secrets are left out. What the task needs beyond core is listed. Nothing is written to a store, and an
    import that fails writes nothing."""
    if (task_id is None) == (eval_id is None):
        raise click.UsageError("give one of --from-task or --from-eval")
    scopes = [FieldScope(scope) for scope in include_scopes]
    with interruptible():
        imported = (import_task(task_id, folder, version=version, include_scopes=scopes, progress=click.echo)
                    if task_id is not None
                    else import_eval(eval_id, folder, version=version, include_scopes=scopes, progress=click.echo))
    _report(imported, folder)


def _report(imported: BundleImport, folder: Path) -> None:
    what = (f"eval {imported.eval.id} v{imported.eval.version} and its tasks" if imported.eval
            else f"task {imported.tasks[0].id} v{imported.tasks[0].version}")
    click.echo(f"Imported {what} into {folder}:")
    if imported.pinned:
        click.echo(f"  It still needs the store for {len(imported.pinned)} reference(s): see Still needs the store.")
    for path in sorted([*imported.files, *{f"{entity.folder}/" for entity in imported.written}]):
        click.echo(f"  {path}")
    for title, lines in imported.report():
        click.echo(f"{title}:")
        for line in lines:
            click.echo(f"  {line}")
    click.echo(f"Run it with: agent-env run {folder}")
