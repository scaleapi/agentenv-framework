"""How a command reports what tearing a run down did."""

import click

from agent_env.providers.sandbox_providers.local_docker.provider import LocalSandbox
from agent_env.task.teardown import RecordedSandbox, TeardownReport, kind


def echo_teardown(tag: str, report: TeardownReport) -> None:
    prefix = f"{tag} " if tag else ""
    if report.terminated:
        n = len(report.terminated)
        click.echo(click.style(f"{prefix}Tore down {n} sandbox{'es' if n != 1 else ''}", fg="blue"))
    for sandbox, why in report.failed:
        click.echo(click.style(f"{prefix}Couldn't tear down {sandbox.sandbox_id}: {why}", fg="red"))
    for sandbox in report.left:
        click.echo(click.style(f"{prefix}Still up: {sandbox.sandbox_id} ({kind(sandbox)})", fg="red"))


def gone(sandbox: RecordedSandbox) -> bool:
    """Whether ``sandbox`` is a local one whose work folder is gone, which its teardown removes once the rest is down:
    gone itself, so neither listed as still up nor terminated again."""
    return sandbox.sandbox_type == "local" and LocalSandbox.find_work_dir(sandbox.sandbox_id) is None


def echo_finish_teardown(instance_ids: list[str]) -> None:
    """The command that finishes the teardown of each instance whose run left sandboxes up."""
    if instance_ids:
        click.echo("Tear down what's left with:")
        for instance_id in instance_ids:
            click.echo(f"  agent-env task teardown --instance {instance_id}")
