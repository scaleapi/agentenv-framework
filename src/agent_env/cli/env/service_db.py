"""CLI commands for ServiceDBEnv."""

import subprocess
from pathlib import Path

import click

from agent_env.artifact import DockerImageArtifact
from agent_env.cli.utils import build_platform_option, detect_env_metadata, docker_build_platform_args
from agent_env.env.envs.service_db import ServiceDBEnv
from agent_env.config import get_config

# Path to ServiceDB Dockerfile
SERVICE_DB_DOCKERFILE = Path(__file__).parent.parent.parent / "env" / "envs" / "service_db" / "Dockerfile"
SERVICE_DB_IMAGE_NAME = "agent-env-service-db"
DB_WEB_DOCKERFILE = Path(__file__).parent.parent.parent / "env" / "envs" / "service_db" / "Dockerfile.db-web"
DB_WEB_IMAGE_NAME = "agent-env-db-web"
DB_MCP_DOCKERFILE = Path(__file__).parent.parent.parent / "env" / "envs" / "service_db" / "Dockerfile.db-mcp"
DB_MCP_IMAGE_NAME = "agent-env-db-mcp"


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
    if not env_id:
        env_id = get_config().default_service_db_env_id

    click.echo(f"Building ServiceDB image from {SERVICE_DB_DOCKERFILE}...")

    # Build Docker image
    result = subprocess.run(
        ["docker", "build", *docker_build_platform_args(build_platform),
         "-f", str(SERVICE_DB_DOCKERFILE),
         "-t", SERVICE_DB_IMAGE_NAME,
         str(SERVICE_DB_DOCKERFILE.parent)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"Docker build failed: {result.stderr}", err=True)
        raise click.Abort()
    click.echo("Docker build successful")

    # Create DB DockerImageArtifact
    click.echo("Creating DB DockerImageArtifact...")
    db_artifact = DockerImageArtifact.put(
        id=f"service-db-{env_id}",
        description=f"ServiceDB PostgreSQL image for {env_id}",
        image_name=SERVICE_DB_IMAGE_NAME,
    )
    click.echo(f"Created DB DockerImageArtifact: id={db_artifact.id} version={db_artifact.version}")

    # Build db-web image (build instead of pull to avoid docker save manifest issues on Apple Silicon)
    click.echo(f"Building db-web image from {DB_WEB_DOCKERFILE}...")
    result = subprocess.run(
        ["docker", "build", *docker_build_platform_args(build_platform),
         "-f", str(DB_WEB_DOCKERFILE),
         "-t", DB_WEB_IMAGE_NAME,
         str(DB_WEB_DOCKERFILE.parent)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"db-web build failed: {result.stderr}", err=True)
        raise click.Abort()
    click.echo("db-web build successful")

    # Create db-web DockerImageArtifact
    click.echo("Creating db-web DockerImageArtifact...")
    db_web_artifact = DockerImageArtifact.put(
        id=f"db-web-{env_id}",
        description="db-web lightweight web UI for database inspection",
        image_name=DB_WEB_IMAGE_NAME,
    )
    click.echo(f"Created db-web DockerImageArtifact: id={db_web_artifact.id} version={db_web_artifact.version}")

    # Build db-mcp image (PostgreSQL MCP server for direct DB access)
    click.echo(f"Building db-mcp image from {DB_MCP_DOCKERFILE}...")
    result = subprocess.run(
        ["docker", "build", *docker_build_platform_args(build_platform),
         "-f", str(DB_MCP_DOCKERFILE),
         "-t", DB_MCP_IMAGE_NAME,
         str(DB_MCP_DOCKERFILE.parent)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        click.echo(f"db-mcp build failed: {result.stderr}", err=True)
        raise click.Abort()
    click.echo("db-mcp build successful")

    # Create db-mcp DockerImageArtifact
    click.echo("Creating db-mcp DockerImageArtifact...")
    db_mcp_artifact = DockerImageArtifact.put(
        id=f"db-mcp-{env_id}",
        description="db-mcp PostgreSQL MCP server for direct DB access",
        image_name=DB_MCP_IMAGE_NAME,
    )
    click.echo(f"Created db-mcp DockerImageArtifact: id={db_mcp_artifact.id} version={db_mcp_artifact.version}")

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            raise click.Abort()
        key, value = pair.split("=", 1)
        user_metadata[key] = value
    metadata = detect_env_metadata(SERVICE_DB_DOCKERFILE, SERVICE_DB_DOCKERFILE.parent)
    metadata.update(user_metadata)

    # Create ServiceDBEnv
    click.echo("Creating ServiceDBEnv...")
    env = ServiceDBEnv.put(
        id=env_id,
        db_docker_image_artifact=db_artifact,
        db_web_docker_image_artifact=db_web_artifact,
        db_mcp_docker_image_artifact=db_mcp_artifact,
        metadata=metadata if metadata else None,
    )
    click.echo(f"Created ServiceDBEnv: id={env.id} version={env.version}")
