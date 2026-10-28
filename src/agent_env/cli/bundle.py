"""``agent-env bundle``: start a bundle folder from what a store holds, and register one into the stores."""

from __future__ import annotations

from pathlib import Path

import click

from agent_env.bundle.importer import BundleImport, import_eval, import_task
from agent_env.bundle.parse import BundleKind
from agent_env.bundle.register import describe_target, register_bundle
from agent_env.cli.utils import interruptible
from agent_env.entity.scopes import FieldScope


@click.group()
def bundle():
    """Create bundle folders, which ``agent-env run`` runs, and register them into the configured stores."""


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


@bundle.command("register")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--task", "tasks", multiple=True,
              help="Register this task, by name, with what it uses. Repeatable. With no --task or --eval, every task "
                   "and eval in the bundle.")
@click.option("--eval", "evals", multiple=True, help="Register this eval, by name, and the tasks it names. Repeatable.")
@click.option("--prefix", default=None,
              help="What each entry's registry id opens with; by default the folder's name and a dash.")
@click.option("--dry-run", is_flag=True, help="Check everything and say what would be written; write nothing.")
@click.option("--yes", is_flag=True, help="Write to a store other than this machine's own without asking.")
def register(folder: Path, tasks: tuple[str, ...], evals: tuple[str, ...], prefix: str | None,
             dry_run: bool, yes: bool):
    """Write the bundle at FOLDER into the configured stores under registry ids, so whatever runs the stores' tasks
    runs what the folder runs.

    Each env, agent, artifact, skill, task and eval takes the id PREFIX plus its name, lowercased; a task's outputs
    take the task's id and their names. A shipped agent a task names by its alias is registered under the alias when
    the store lacks one. Each task and eval names the version of everything it references, so a registered version
    never changes meaning; a step that leaves its agent unset is left so, for whatever runs it to give its own. Each
    image the bundle builds is built here for linux/amd64 with Docker and pushed to the configured image store, with
    its build context beside it.

    It checks everything first and says what it would write: an @local id left in what it would write or, for a store
    other than this machine's, a credential written out or a URL only this machine reaches refuses the whole
    registration. An id the store holds already takes a new version, whoever wrote it. A store other than this
    machine's is written only after you confirm, or with --yes. Registering again writes only what changed since."""
    options = {"tasks": tasks, "evals": evals, "prefix": prefix}
    click.echo(f"Registering {folder} into:")
    for line in describe_target():
        click.echo(f"  {line}")
    click.echo("Checking:")
    with interruptible():
        planned = register_bundle(folder, dry_run=True, on_progress=click.echo, **options)
    for line in planned.floating:
        click.echo(f"  floating, since its field takes no version: {line}")
    if dry_run:
        click.echo("Nothing was written: --dry-run.")
        return
    if all(done.reused for done in planned.materialization.writes):
        click.echo("Nothing to write: every version is registered already.")
        return
    if planned.shared and not yes:
        click.confirm("Write these to the stores above?", abort=True)
    click.echo("Writing:")
    with interruptible():
        registered = register_bundle(folder, on_progress=click.echo, **options)
    tasks_written = [done for done in registered.materialization.writes if done.write.kind is BundleKind.TASK]
    if tasks_written:
        first = tasks_written[0]
        click.echo(f"Run a registered task with: agent-env task run --id {first.write.id} --version {first.version}")
