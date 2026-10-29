import sys

import click

from agent_env.cli.utils import build_platform_option
from agent_env.env.bootstrap import PLAYWRIGHT_MCP_CONTEXT, PLAYWRIGHT_MCP_DOCKERFILE, put_playwright_mcp_env  # noqa: F401
from agent_env.env.envs.playwright_mcp import PLAYWRIGHT_MCP_ENVIRONMENT_NAME


@click.group(name="playwright-mcp")
def playwright_mcp():
    """Playwright MCP website browser commands."""
    pass


@playwright_mcp.command()
@click.option("--id", "env_id", default=PLAYWRIGHT_MCP_ENVIRONMENT_NAME, show_default=True, help="Env id")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@build_platform_option
def put(env_id: str, metadata_pairs: tuple[str, ...], build_platform: str):
    """Build and upload the Playwright MCP website browser environment."""
    metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        metadata[key] = value
    put_playwright_mcp_env(env_id, platform=build_platform, metadata=metadata, say=click.echo)
