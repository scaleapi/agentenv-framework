"""Snapshot a deployed MultiEnv's servicedb as a pre-loaded Docker image."""

import asyncio
import logging

import click

from agent_env.env.snapshot_store import EnvSnapshot

logger = logging.getLogger(__name__)


async def run_snapshot(instance_id: str) -> EnvSnapshot:
    """Core snapshot logic. Returns the EnvSnapshot.

    Can be called directly from tests or wrapped by the CLI command.
    """
    return await EnvSnapshot.create(instance_id)


@click.command()
@click.option("--instance-id", required=True, help="Deployed env instance ID")
def snapshot(instance_id: str):
    """Snapshot a deployed MultiEnv's servicedb as a pre-loaded Docker image."""
    try:
        result = asyncio.run(run_snapshot(instance_id))
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)
    click.echo(click.style("Snapshot complete!", fg="green"))
    click.echo(f"  instance_id: {result.instance_id}")
    click.echo(f"  env_id: {result.env_id}")
    click.echo(f"  environment_universe_id: {result.environment_universe_id}")
    click.echo(f"  is_clean: {result.is_clean}")
    click.echo(f"  db_image_artifact: {result.db_image_artifact_id} v{result.db_image_artifact_version}")
