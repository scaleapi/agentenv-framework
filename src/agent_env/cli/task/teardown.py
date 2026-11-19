"""Tear down what a task run left up."""

import asyncio
from dataclasses import replace

import click

from agent_env.cli.teardown_output import echo_teardown, gone
from agent_env.store.base import NotFoundError
from agent_env.store.routing import run_scope
from agent_env.task.interrupts import Interrupts
from agent_env.task.store import get_task_instance_store
from agent_env.task.teardown import TORN_DOWN_KEY, TeardownReport, record_teardown, recorded_still_up, teardown_run
from agent_env.task_step.context import TaskStepContext


@click.command("teardown")
@click.option("--instance", "instance_id", required=True, help="The task instance whose sandboxes to tear down.")
@click.option("--force", is_flag=True,
              help="Tear down an instance recorded running too, as a run killed outright leaves it.")
def teardown(instance_id: str, force: bool):
    """Tear down the sandboxes a task run left up, as its own teardown does: the recordings it left open end, each
    env still up closes, and each sandbox is terminated by id, a local one's folder removed. For a run whose teardown
    was stopped or failed, which `agent-env run` names. Ctrl-C stops it; it exits 1 while any is still up, and
    running it again goes on from there."""
    try:
        instance = get_task_instance_store().get(instance_id)
    except NotFoundError:
        raise click.ClickException(f"no task instance {instance_id!r}") from None
    if instance.status == "running" and not force:
        raise click.ClickException(f"instance {instance_id} is recorded running, so its run may still be using its "
                                   "sandboxes; pass --force once it isn't, as for a run killed outright")
    context = TaskStepContext.from_dict(instance.context or {})
    context.metadata[TORN_DOWN_KEY] = [*context.metadata.get(TORN_DOWN_KEY, []),
                                       *(sandbox.sandbox_id for sandbox in recorded_still_up(context) if gone(sandbox))]
    left = recorded_still_up(context)
    if not left:
        click.echo(f"Nothing instance {instance_id} recorded is still up.")
        return
    click.echo(f"Tearing down {len(left)} sandbox{'es' if len(left) != 1 else ''} of instance {instance_id} "
               "(Ctrl-C to stop)")
    with Interrupts() as stopper, run_scope(instance_id):
        try:
            [report] = stopper.run(stopper.stopping([teardown_run(context)]))
        except asyncio.CancelledError:
            report = TeardownReport(left=recorded_still_up(context))
        record_teardown(instance_id, context, report, instance.attempt_epoch)
    echo_teardown("", replace(report, terminated=tuple(sandbox for sandbox in left if sandbox not in report.still_up)))
    if report.still_up:
        raise SystemExit(1)
