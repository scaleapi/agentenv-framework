"""How a command reports what tearing a run down did."""

import click

from agent_env.task.teardown import TeardownReport, kind


def echo_teardown(tag: str, report: TeardownReport) -> None:
    prefix = f"{tag} " if tag else ""
    if report.terminated:
        n = len(report.terminated)
        click.echo(click.style(f"{prefix}Tore down {n} sandbox{'es' if n != 1 else ''}", fg="blue"))
    for sandbox, why in report.failed:
        click.echo(click.style(f"{prefix}Couldn't tear down {sandbox.sandbox_id}: {why}", fg="red"))
    for sandbox in report.left:
        click.echo(click.style(f"{prefix}Still up: {sandbox.sandbox_id} ({kind(sandbox)})", fg="red"))
