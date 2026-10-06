import sys

import click

from agent_env.cli.utils import build_platform_option
from agent_env.env.bootstrap import WEBSITE_BROWSER_CONTEXT, WEBSITE_BROWSER_DOCKERFILE, put_website_browser_env  # noqa: F401


@click.group(name="website-browser")
def website_browser():
    """Website browser environment commands."""
    pass


@website_browser.command()
@click.option("--id", "env_id", default=None, help="Env id (default: config default_website_browser_env_id)")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@build_platform_option
def put(env_id: str, metadata_pairs: tuple[str, ...], build_platform: str):
    """Build and upload a website browser environment."""
    from agent_env.config import get_config

    metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        metadata[key] = value
    put_website_browser_env(env_id or get_config().default_website_browser_env_id, platform=build_platform,
                            metadata=metadata, say=click.echo)
