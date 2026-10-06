import sys

import click

from agent_env.cli.utils import build_platform_option
from agent_env.env.bootstrap import GATEWAY_CONTEXT, GATEWAY_DOCKERFILE, GATEWAY_IMAGE_TAG, put_gateway_env  # noqa: F401


@click.group()
def gateway():
    """Gateway environment commands."""
    pass


@gateway.command()
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@build_platform_option
def put(env_id: str, metadata_pairs: tuple[str, ...], build_platform: str):
    """Build and upload a gateway environment."""
    metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        metadata[key] = value
    put_gateway_env(env_id, platform=build_platform, metadata=metadata, say=click.echo)
