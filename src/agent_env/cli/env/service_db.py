"""CLI commands for ServiceDBEnv."""

import click

from agent_env.cli.utils import build_platform_option
from agent_env.config import get_config
from agent_env.env.bootstrap import (  # noqa: F401
    DB_MCP_DOCKERFILE,
    DB_MCP_IMAGE_NAME,
    DB_WEB_DOCKERFILE,
    DB_WEB_IMAGE_NAME,
    SERVICE_DB_DOCKERFILE,
    SERVICE_DB_IMAGE_NAME,
    put_service_db_env,
)


@click.group(name="service-db")
def service_db():
    """ServiceDB environment commands."""
    pass


@service_db.command()
@click.option("--id", "env_id", default=None, help="Env id (default: config default_service_db_env_id)")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@build_platform_option
def put(env_id: str, metadata_pairs: tuple[str, ...], build_platform: str):
    """Build and upload a ServiceDB environment."""
    metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            raise click.Abort()
        key, value = pair.split("=", 1)
        metadata[key] = value
    put_service_db_env(env_id or get_config().default_service_db_env_id, platform=build_platform, metadata=metadata,
                       say=click.echo)
