import sys
from pathlib import Path

import click

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import DockerImageArtifact
from agent_env.cli.utils import build_platform_option, detect_env_metadata, skips_local_validation
from agent_env.utils.docker_build import build_image


@click.command()
@click.option("--id", "agent_id", required=True, help="A2A agent id")
@click.option("--dockerfile", required=True, type=click.Path(exists=True), help="Path to Dockerfile")
@click.option("--context", "context_path", default=None, type=click.Path(exists=True), help="Docker build context (defaults to Dockerfile's directory)")
@click.option("--env-var", "env_var_pairs", multiple=True, help="Default env var KEY=VALUE (repeatable)")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@click.option("--min-disk-size-gb", type=float, default=None, help="Minimum disk size in GB for agent deployment")
@click.option("--default-model", type=str, default=None, help="Default model for agent prompts (e.g., claude-sonnet-4-6)")
@click.option("--skip-validation", is_flag=True, default=False, help="Skip A2A agent validation after registration")
@click.option("--litellm-api-key", type=str, default=None, help="LiteLLM API key for validation prompts")
@build_platform_option
def put(agent_id: str, dockerfile: str, context_path: str | None, env_var_pairs: tuple[str, ...], metadata_pairs: tuple[str, ...], min_disk_size_gb: float | None, default_model: str | None, skip_validation: bool, litellm_api_key: str | None, build_platform: str):
    """Build a Docker image and register an A2A agent."""

    default_env_vars = {}
    for pair in env_var_pairs:
        if "=" not in pair:
            click.echo(f"Invalid env-var format '{pair}', expected KEY=VALUE", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        default_env_vars[key] = value

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            click.echo(f"Invalid metadata format '{pair}', expected key=value", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        user_metadata[key] = value

    dockerfile_path = Path(dockerfile)
    context = Path(context_path) if context_path else dockerfile_path.parent
    image_tag = f"a2a-agent-{agent_id}"

    click.echo(f"Building Docker image...")
    build_image(dockerfile_path, context, image_tag, platform=build_platform)

    click.echo(f"Creating DockerImageArtifact...")
    artifact = DockerImageArtifact.put(
        id=f"a2a-agent-{agent_id}",
        description=f"A2A agent image for {agent_id}",
        image_name=image_tag,
        build_context_path=str(context),
        dockerfile_path=str(dockerfile_path),
    )
    click.echo(f"Created artifact: id={artifact.id} version={artifact.version}")

    metadata = detect_env_metadata(dockerfile_path, context)
    metadata.update(user_metadata)
    if min_disk_size_gb is not None:
        metadata["min_disk_size_gb"] = min_disk_size_gb
    if default_model is not None:
        metadata["default_model"] = default_model

    click.echo(f"Registering A2A agent...")
    agent = A2AAgent.put(
        id=agent_id,
        docker_image_artifact=artifact,
        default_env_vars=default_env_vars if default_env_vars else None,
        metadata=metadata if metadata else None,
    )
    click.echo(f"Created A2A agent: id={agent.id} version={agent.version} image={artifact.id}:{artifact.version}")

    if not skip_validation and not skips_local_validation(agent_id, "agent"):
        from agent_env.cli.a2a_agent.validate import run_validation
        click.echo()
        run_validation(agent, litellm_api_key=litellm_api_key)
