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
from agent_env.utils.docker.build import DEFAULT_BUILD_PLATFORM, build_image
from agent_env.env.envs.website import WebsiteEnv
from agent_env.providers.env_providers.constants import AGENT_ENV_WEBSITE_HEALTH_PATH
from agent_env.store.ids import derive_id, image_repository



@click.group(name="website")
def website():
    """Website environment commands."""
    pass


@website.command()
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--server-dockerfile", "--backend-dockerfile", "backend_dockerfile", default=None, type=click.Path(exists=True),
              help="Path to the Dockerfile of the server image, which serves the site's pages and API")
@click.option("--server-docker-context", "--backend-docker-context", "backend_docker_context", default=None,
              type=click.Path(exists=True), help="The server image's build context (defaults to its Dockerfile's directory)")
@click.option("--server-dockerfile-github-url", "--backend-dockerfile-github-url", "backend_dockerfile_github_url",
              default=None, help="GitHub URL to the server image's Dockerfile")
@click.option("--server-docker-context-github-url", "--backend-docker-context-github-url",
              "backend_docker_context_github_url", default=None, help="GitHub URL to the server image's build context directory")
@click.option("--server-image-ref", "server_image_ref", default=None,
              help="A server image already in a registry, e.g. ghcr.io/org/site:v1: registered as it is, with its tag "
                   "pinned to the digest the registry serves now (instead of a Dockerfile)")
@click.option("--frontend-dockerfile", default=None, type=click.Path(exists=True),
              help="Path to a separate frontend Dockerfile (deprecated: build the pages into the server image)")
@click.option("--frontend-docker-context", default=None, type=click.Path(exists=True), help="Frontend Docker build context (defaults to Dockerfile's directory)")
@click.option("--frontend-dockerfile-github-url", default=None, help="GitHub URL to a separate frontend Dockerfile (deprecated)")
@click.option("--frontend-docker-context-github-url", default=None, help="GitHub URL to frontend build context directory")
@click.option("--health-path", default=AGENT_ENV_WEBSITE_HEALTH_PATH, show_default=True,
              help="The path that answers 200 once the site is usable")
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
    server_image_ref: str | None,
    frontend_dockerfile: str | None,
    frontend_docker_context: str | None,
    frontend_dockerfile_github_url: str | None,
    frontend_docker_context_github_url: str | None,
    health_path: str,
    environment_name: str | None,
    env_provider_type: str,
    metadata_pairs: tuple[str, ...],
    skip_validation: bool,
    build_platform: str,
):
    """Build and upload a website environment: its server image, which serves the site's pages and API, from a
    Dockerfile, GitHub or a registry, and any deprecated separate frontend image."""
    environment_name = resolve_environment_name(environment_name, allow_missing=True)

    if sum(map(bool, (backend_dockerfile, backend_dockerfile_github_url, server_image_ref))) != 1:
        click.echo("Error: exactly one of --server-dockerfile, --server-dockerfile-github-url and --server-image-ref is "
                   "required", err=True)
        sys.exit(1)
    if frontend_dockerfile and frontend_dockerfile_github_url:
        click.echo("Error: --frontend-dockerfile and --frontend-dockerfile-github-url are mutually exclusive", err=True)
        sys.exit(1)
    use_github = bool(backend_dockerfile_github_url)
    if (frontend_dockerfile and not backend_dockerfile) or (frontend_dockerfile_github_url and not use_github):
        click.echo("Error: a frontend image is built the way the server image is, both from local Dockerfiles or both "
                   "from GitHub URLs", err=True)
        sys.exit(1)
    if server_image_ref and not environment_name:
        click.echo("Error: --server-image-ref needs --environment-name, which a registry image's card can't give here",
                   err=True)
        sys.exit(1)
    if frontend_dockerfile or frontend_dockerfile_github_url:
        click.echo("Warning: a separate frontend image is deprecated; build the site's pages into its server image",
                   err=True)

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        user_metadata[key] = value
    site = {"health_path": health_path}

    if use_github:
        if environment_name is None:
            environment_name = card_name_from_github(backend_dockerfile_github_url, backend_docker_context_github_url, github_token=os.environ.get("GITHUB_TOKEN"))
            if not environment_name:
                click.echo("Error: could not read @environment_card(name=...) from the server image's GitHub source; pass --environment-name.", err=True)
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
            click.echo(f"\033[2K  {click.style('Server:', fg='cyan')}   {backend_status[0]}")
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
            frontend_dockerfile_github_url=frontend_dockerfile_github_url or "",
            frontend_docker_context_github_url=frontend_docker_context_github_url,
            environment_name=environment_name,
            metadata=user_metadata if user_metadata else None,
            on_backend_progress=_backend_progress,
            on_frontend_progress=_frontend_progress,
            github_token=os.environ.get("GITHUB_TOKEN"),
            env_provider_type=env_provider_type,
            **site,
        ))
        click.echo(f"Created WebsiteEnv: id={env.id} version={env.version} environment_name={env.environment_name}")
        if not skip_validation:
            click.echo("\nValidating environment (use --skip-validation to skip)...")
            instance_id = asyncio.run(env.validate(on_progress=click.echo))
            click.echo(f"Validation task: {instance_id}")
        return

    backend_id, frontend_id = derive_id(env_id, "backend_image"), derive_id(env_id, "frontend_image")
    refuse_unwritable_ids(env_id, backend_id, *([frontend_id] if frontend_dockerfile else []))
    metadata: dict[str, str] = {}
    if server_image_ref:
        if build_platform != DEFAULT_BUILD_PLATFORM:
            click.echo(f"Warning: --platform {build_platform!r} is ignored for --server-image-ref, which builds nothing", err=True)
        backend_artifact = DockerImageArtifact.put_ref(backend_id, description=f"Website server image for {env_id}",
                                                       image_name=server_image_ref)
        click.echo(f"Registered server image: id={backend_artifact.id} version={backend_artifact.version} image={backend_artifact.image_name}")
    else:
        backend_dockerfile_path = Path(backend_dockerfile)
        backend_ctx = Path(backend_docker_context) if backend_docker_context else backend_dockerfile_path.parent
        if environment_name is None:
            environment_name = card_name_from_source(str(backend_dockerfile_path), str(backend_ctx))
            if not environment_name:
                click.echo("Error: no @environment_card(name=...) found in the server image's source; pass --environment-name.", err=True)
                sys.exit(1)
            click.echo(f"Derived environment_name={environment_name!r} from the environment card.")
        backend_artifact = _build(backend_id, backend_dockerfile_path, backend_ctx, "server", build_platform)
        metadata = detect_env_metadata(backend_dockerfile_path, backend_ctx)

    frontend_artifact = None
    if frontend_dockerfile:
        frontend_dockerfile_path = Path(frontend_dockerfile)
        frontend_ctx = Path(frontend_docker_context) if frontend_docker_context else frontend_dockerfile_path.parent
        frontend_artifact = _build(frontend_id, frontend_dockerfile_path, frontend_ctx, "frontend", build_platform)
    metadata.update(user_metadata)

    click.echo("Creating WebsiteEnv...")
    try:
        env = WebsiteEnv.put(
            id=env_id,
            server_docker_image_artifact=backend_artifact,
            frontend_docker_image_artifact=frontend_artifact,
            environment_name=environment_name,
            metadata=metadata if metadata else None,
            env_provider_type=env_provider_type,
            **site,
        )
    except ValueError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Created WebsiteEnv: id={env.id} version={env.version} environment_name={env.environment_name} "
               f"env_provider_type={env.env_provider_type}")
    if not skip_validation:
        click.echo("\nValidating environment (use --skip-validation to skip)...")
        instance_id = asyncio.run(env.validate(on_progress=click.echo))
        click.echo(f"Validation task: {instance_id}")


def _build(image_id: str, dockerfile: Path, context: Path, role: str, platform: str) -> DockerImageArtifact:
    """Build ``dockerfile`` into the image ``image_id`` and store it with its build context, as an a2a agent's image
    is stored."""
    tag = image_repository(image_id)
    click.echo(f"Building website {role} Docker image...")
    build_image(dockerfile, context, tag, platform=platform)
    artifact = DockerImageArtifact.put(id=image_id, description=f"Website {role} image created from agent-env CLI",
                                       image_name=tag, build_context_path=str(context), dockerfile_path=str(dockerfile),
                                       platform=platform)
    click.echo(f"Created {role} artifact: id={artifact.id} version={artifact.version}")
    return artifact


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
