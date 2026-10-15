"""``agent-env bundle``: start a bundle folder from what a store holds."""

from __future__ import annotations

from pathlib import Path

import click

from agent_env.bundle.importer import BundleImport, import_eval, import_task
from agent_env.cli.utils import interruptible


@click.group()
def bundle():
    """Create bundle folders, which ``agent-env run`` runs."""


@bundle.command("init")
@click.argument("folder", type=click.Path(file_okay=False, path_type=Path))
@click.option("--from-task", "task_id", default=None, help="Import this stored task, by id.")
@click.option("--from-eval", "eval_id", default=None, help="Import this stored eval and every task it names, by id.")
@click.option("--version", type=int, default=None, help="The version to import (defaults to the latest).")
def init(folder: Path, task_id: str | None, eval_id: str | None, version: int | None):
    """Write a stored task, or an eval and its tasks, as a bundle at FOLDER, which must be new or empty.

    Each task is written as the store holds it, and every store reference it makes is pinned at the version the
    store holds now, so the folder runs what the store ran; what a step writes stays unpinned, since the run writes
    it again. Values that look like secrets are left out. Nothing is written to a store, and an import that fails
    writes nothing."""
    if (task_id is None) == (eval_id is None):
        raise click.UsageError("give one of --from-task or --from-eval")
    with interruptible():
        imported = (import_task(task_id, folder, version=version) if task_id is not None
                    else import_eval(eval_id, folder, version=version))
    _report(imported, folder)


def _report(imported: BundleImport, folder: Path) -> None:
    what = (f"eval {imported.eval.id} v{imported.eval.version} and its tasks" if imported.eval
            else f"task {imported.tasks[0].id} v{imported.tasks[0].version}")
    click.echo(f"Imported {what} into {folder}:")
    for path in imported.files:
        click.echo(f"  {path}")
    for title, lines in imported.report():
        click.echo(f"{title}:")
        for line in lines:
            click.echo(f"  {line}")
    click.echo(f"Run it with: agent-env run {folder}")
