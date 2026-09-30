import sys
from pathlib import Path

import click

from agent_env.artifact import DockerImageArtifact
from agent_env.cli.utils import build_platform_option, detect_env_metadata
from agent_env.env import GatewayEnv
from agent_env.utils.docker_build import build_image

_PACKAGE_ROOT = Path(__file__).parent.parent.parent
GATEWAY_DOCKERFILE = _PACKAGE_ROOT / "env" / "gateway" / "Dockerfile"
GATEWAY_CONTEXT = _PACKAGE_ROOT / "env"
GATEWAY_IMAGE_TAG = "env-gateway"


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

    click.echo(f"Building gateway Docker image...")
    build_image(GATEWAY_DOCKERFILE, GATEWAY_CONTEXT, GATEWAY_IMAGE_TAG, platform=build_platform)

    click.echo(f"Creating DockerImageArtifact...")
    artifact = DockerImageArtifact.put(
        id=f"gateway-{env_id}",
        description="Created from agent-env CLI",
        image_name=GATEWAY_IMAGE_TAG,
    )
    click.echo(f"Created artifact: id={artifact.id} version={artifact.version}")

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        user_metadata[key] = value
    metadata = detect_env_metadata(GATEWAY_DOCKERFILE, GATEWAY_CONTEXT)
    metadata.update(user_metadata)

    click.echo(f"Creating GatewayEnv...")
    env = GatewayEnv.put(
        id=env_id,
        docker_image_artifact=artifact,
        metadata=metadata if metadata else None,
    )
    click.echo(f"Created GatewayEnv: id={env.id} version={env.version}")
