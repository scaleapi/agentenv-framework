import subprocess
import sys
from pathlib import Path

import click

from agent_env.artifact import DockerImageArtifact
from agent_env.cli.utils import build_platform_option, detect_env_metadata, docker_build_platform_args
from agent_env.env import MCPServerEnv
from agent_env.env.envs.website_browser import (
    PLAYWRIGHT_MCP_VERSION,
    WEBSITE_BROWSER_IMAGE_TAG,
    WEBSITE_BROWSER_ENVIRONMENT_NAME,
    WEBSITE_BROWSER_SERVICE_VERSION,
)

_PACKAGE_ROOT = Path(__file__).parent.parent.parent
WEBSITE_BROWSER_DOCKERFILE = _PACKAGE_ROOT / "env" / "envs" / "website_browser" / "Dockerfile"
WEBSITE_BROWSER_CONTEXT = _PACKAGE_ROOT / "env" / "envs" / "website_browser"


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

    if not env_id:
        env_id = get_config().default_website_browser_env_id

    click.echo("Building website browser Docker image...")
    result = subprocess.run(
        ["docker", "build", *docker_build_platform_args(build_platform),
         "--build-arg", f"PLAYWRIGHT_MCP_VERSION={PLAYWRIGHT_MCP_VERSION}",
         "-f", str(WEBSITE_BROWSER_DOCKERFILE), "-t", WEBSITE_BROWSER_IMAGE_TAG, str(WEBSITE_BROWSER_CONTEXT)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Docker build failed: {result.stderr}", err=True)
        sys.exit(1)

    click.echo("Creating DockerImageArtifact...")
    artifact = DockerImageArtifact.put(
        id=f"website-browser-{env_id}",
        description="Website browser MCP server",
        image_name=WEBSITE_BROWSER_IMAGE_TAG,
    )
    click.echo(f"Created artifact: id={artifact.id} version={artifact.version}")

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        user_metadata[key] = value
    metadata = detect_env_metadata(WEBSITE_BROWSER_DOCKERFILE, WEBSITE_BROWSER_CONTEXT)
    metadata.update(user_metadata)

    click.echo("Creating MCPServerEnv...")
    env = MCPServerEnv.put(
        id=env_id,
        docker_image_artifact=artifact,
        environment_name=WEBSITE_BROWSER_ENVIRONMENT_NAME,
        service_version=WEBSITE_BROWSER_SERVICE_VERSION,
        metadata=metadata if metadata else None,
    )
    click.echo(f"Created MCPServerEnv: id={env.id} version={env.version} environment_name={env.environment_name} service_version={env.service_version}")
