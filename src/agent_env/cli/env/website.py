import asyncio
import os
import sys
from pathlib import Path

import click

from agent_env.artifact import DockerImageArtifact, EnvironmentArtifact
from agent_env.cli.utils import (
    deployed_env_from_instance,
    build_platform_option,
    detect_env_metadata,
    env_provider_type_option,
    environment_name_options,
    refuse_unwritable_ids,
    resolve_environment_name,
)
from agent_env.utils.card_naming import card_name_from_github, card_name_from_source
from agent_env.utils.docker_build import DEFAULT_BUILD_PLATFORM, build_image
from agent_env.env import Env
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers import get_env_sandbox_provider
from agent_env.store.ids import derive_id, image_repository



@click.group(name="website")
def website():
    """Website environment commands."""
    pass


@website.command()
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--backend-dockerfile", default=None, type=click.Path(exists=True), help="Path to backend Dockerfile")
@click.option("--backend-docker-context", default=None, type=click.Path(exists=True), help="Backend Docker build context (defaults to Dockerfile's directory)")
@click.option("--backend-dockerfile-github-url", default=None, help="GitHub URL to backend Dockerfile")
@click.option("--backend-docker-context-github-url", default=None, help="GitHub URL to backend build context directory")
@click.option("--frontend-dockerfile", default=None, type=click.Path(exists=True), help="Path to frontend Dockerfile")
@click.option("--frontend-docker-context", default=None, type=click.Path(exists=True), help="Frontend Docker build context (defaults to Dockerfile's directory)")
@click.option("--frontend-dockerfile-github-url", default=None, help="GitHub URL to frontend Dockerfile")
@click.option("--frontend-docker-context-github-url", default=None, help="GitHub URL to frontend build context directory")
@environment_name_options
@env_provider_type_option("What deploys the env: 'gateway' (a gateway in front of the website), or the type of an installed "
                          "agent_env.env_providers plugin", "website")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@click.option("--skip-validation", is_flag=True, default=False, help="Skip environment validation after registration")
@build_platform_option
def put(
    env_id: str,
    backend_dockerfile: str | None,
    backend_docker_context: str | None,
    backend_dockerfile_github_url: str | None,
    backend_docker_context_github_url: str | None,
    frontend_dockerfile: str | None,
    frontend_docker_context: str | None,
    frontend_dockerfile_github_url: str | None,
    frontend_docker_context_github_url: str | None,
    environment_name: str | None,
    env_provider_type: str,
    metadata_pairs: tuple[str, ...],
    skip_validation: bool,
    build_platform: str,
):
    """Build and upload a website environment (frontend + backend)."""
    environment_name = resolve_environment_name(environment_name, allow_missing=True)

    # Validate mutual exclusion
    if backend_dockerfile and backend_dockerfile_github_url:
        click.echo("Error: --backend-dockerfile and --backend-dockerfile-github-url are mutually exclusive", err=True)
        sys.exit(1)
    if not backend_dockerfile and not backend_dockerfile_github_url:
        click.echo("Error: either --backend-dockerfile or --backend-dockerfile-github-url is required", err=True)
        sys.exit(1)
    if frontend_dockerfile and frontend_dockerfile_github_url:
        click.echo("Error: --frontend-dockerfile and --frontend-dockerfile-github-url are mutually exclusive", err=True)
        sys.exit(1)
    if not frontend_dockerfile and not frontend_dockerfile_github_url:
        click.echo("Error: either --frontend-dockerfile or --frontend-dockerfile-github-url is required", err=True)
        sys.exit(1)
    use_github = bool(backend_dockerfile_github_url)
    if use_github != bool(frontend_dockerfile_github_url):
        click.echo("Error: backend and frontend must both use local Dockerfiles or both use GitHub URLs", err=True)
        sys.exit(1)

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        user_metadata[key] = value

    if use_github:
        if environment_name is None:
            environment_name = card_name_from_github(backend_dockerfile_github_url, backend_docker_context_github_url, github_token=os.environ.get("GITHUB_TOKEN"))
            if not environment_name:
                click.echo("Error: could not read @environment_card(name=...) from the backend GitHub source; pass --environment-name.", err=True)
                sys.exit(1)
            click.echo(f"Derived environment_name={environment_name!r} from the environment card.")
        if build_platform != DEFAULT_BUILD_PLATFORM:
            click.echo(f"Warning: --platform {build_platform!r} is ignored for --*-dockerfile-github-url builds", err=True)
        backend_status = [""]
        frontend_status = [""]
        lines_printed = [0]

        def _redraw():
            if lines_printed[0] > 0:
                click.echo(f"\033[{lines_printed[0]}A", nl=False)
            click.echo(f"\033[2K  {click.style('Backend:', fg='cyan')}  {backend_status[0]}")
            click.echo(f"\033[2K  {click.style('Frontend:', fg='magenta')} {frontend_status[0]}")
            lines_printed[0] = 2

        def _backend_progress(step: str, message: str, percent: int) -> None:
            backend_status[0] = f"[{percent:3d}%] {message}"
            _redraw()

        def _frontend_progress(step: str, message: str, percent: int) -> None:
            frontend_status[0] = f"[{percent:3d}%] {message}"
            _redraw()

        env = asyncio.run(WebsiteEnv.put_from_github(
            id=env_id,
            backend_dockerfile_github_url=backend_dockerfile_github_url,
            backend_docker_context_github_url=backend_docker_context_github_url,
            frontend_dockerfile_github_url=frontend_dockerfile_github_url,
            frontend_docker_context_github_url=frontend_docker_context_github_url,
            environment_name=environment_name,
            metadata=user_metadata if user_metadata else None,
            on_backend_progress=_backend_progress,
            on_frontend_progress=_frontend_progress,
            github_token=os.environ.get("GITHUB_TOKEN"),
            env_provider_type=env_provider_type,
        ))
        click.echo(f"Created WebsiteEnv: id={env.id} version={env.version} environment_name={env.environment_name}")
        if not skip_validation:
            click.echo("\nValidating environment (use --skip-validation to skip)...")
            instance_id = asyncio.run(env.validate(on_progress=click.echo))
            click.echo(f"Validation task: {instance_id}")
        return

    # Local Docker build path
    backend_dockerfile_path = Path(backend_dockerfile)
    backend_ctx = Path(backend_docker_context) if backend_docker_context else backend_dockerfile_path.parent
    backend_id, frontend_id = derive_id(env_id, "backend_image"), derive_id(env_id, "frontend_image")
    refuse_unwritable_ids(env_id, backend_id, frontend_id)
    backend_tag = image_repository(backend_id)
    if environment_name is None:
        environment_name = card_name_from_source(str(backend_dockerfile_path), str(backend_ctx))
        if not environment_name:
            click.echo("Error: no @environment_card(name=...) found in the backend source; pass --environment-name.", err=True)
            sys.exit(1)
        click.echo(f"Derived environment_name={environment_name!r} from the environment card.")

    click.echo(f"Building website backend Docker image...")
    build_image(backend_dockerfile_path, backend_ctx, backend_tag, platform=build_platform)

    click.echo(f"Creating backend DockerImageArtifact...")
    backend_artifact = DockerImageArtifact.put(
        id=backend_id,
        description="Website backend image created from agent-env CLI",
        image_name=backend_tag,
    )
    click.echo(f"Created backend artifact: id={backend_artifact.id} version={backend_artifact.version}")

    frontend_dockerfile_path = Path(frontend_dockerfile)
    frontend_ctx = Path(frontend_docker_context) if frontend_docker_context else frontend_dockerfile_path.parent
    frontend_tag = image_repository(frontend_id)

    click.echo(f"Building website frontend Docker image...")
    build_image(frontend_dockerfile_path, frontend_ctx, frontend_tag, platform=build_platform)

    click.echo(f"Creating frontend DockerImageArtifact...")
    frontend_artifact = DockerImageArtifact.put(
        id=frontend_id,
        description="Website frontend image created from agent-env CLI",
        image_name=frontend_tag,
    )
    click.echo(f"Created frontend artifact: id={frontend_artifact.id} version={frontend_artifact.version}")

    metadata = detect_env_metadata(backend_dockerfile_path, backend_ctx)
    metadata.update(user_metadata)

    click.echo(f"Creating WebsiteEnv...")
    env = WebsiteEnv.put(
        id=env_id,
        backend_docker_image_artifact=backend_artifact,
        frontend_docker_image_artifact=frontend_artifact,
        environment_name=environment_name,
        metadata=metadata if metadata else None,
        env_provider_type=env_provider_type,
    )
    click.echo(f"Created WebsiteEnv: id={env.id} version={env.version} environment_name={env.environment_name} env_provider_type={env.env_provider_type}")
    if not skip_validation:
        click.echo("\nValidating environment (use --skip-validation to skip)...")
        instance_id = asyncio.run(env.validate(on_progress=click.echo))
        click.echo(f"Validation task: {instance_id}")


@website.command(name="load-environment-artifact")
@click.option("--id", "env_id", default=None, help="WebsiteEnv id (optional; cross-checked against the instance)")
@click.option("--environment-artifact-id", "environment_artifact_id",
              required=True, help="EnvironmentArtifact id")
@click.option("--instance-id", "instance_id", required=True,
              help="Deployed env instance id, as printed by `env deploy`")
def load_environment_artifact(env_id: str | None, environment_artifact_id: str,
                              instance_id: str):
    """Load an environment artifact into a deployed website environment."""

    env = deployed_env_from_instance(env_id, instance_id)
    click.echo(f"Found env: id={env.id} version={env.version} environment_name={env.environment_name}")

    click.echo(f"Fetching environment artifact: id={environment_artifact_id}...")
    environment_artifact = EnvironmentArtifact.get(environment_artifact_id)
    click.echo(f"Found environment artifact: id={environment_artifact.id} version={environment_artifact.version} environment_name={environment_artifact.environment_name}")

    click.echo("Loading environment artifact...")
    asyncio.run(env.load_environment_artifact(environment_artifact))
    click.echo("Loaded environment artifact into env")
