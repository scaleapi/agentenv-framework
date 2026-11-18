"""Tear down what a task run left up."""

import click

from agent_env.cli.teardown_output import echo_teardown
from agent_env.store.base import NotFoundError
from agent_env.task.interrupts import Interrupts
from agent_env.task.store import get_task_instance_store, record_teardown
from agent_env.task.teardown import TeardownReport, teardown_run
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
    left = TeardownReport.skipped(context).left
    if not left:
        click.echo(f"Nothing instance {instance_id} recorded is still up.")
        return
    click.echo(f"Tearing down {len(left)} sandbox{'es' if len(left) != 1 else ''} of instance {instance_id} "
               "(Ctrl-C to stop)")
    with Interrupts() as stopper:
        [report] = stopper.run(stopper.stopping([teardown_run(context)]))
    record_teardown(instance_id, context)
    echo_teardown("", report)
    if report.still_up:
        raise SystemExit(1)
