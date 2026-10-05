import asyncio

import click

from agent_env.artifact import EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.cli.utils import detect_base_metadata, env_provider_type_option
from agent_env.env import Env, MultiEnv


def parse_env_ref(ref: str) -> tuple[str, int | None]:
    """Parse 'id' or 'id:version' format."""
    if ":" in ref:
        id_part, version_part = ref.split(":", 1)
        return id_part, int(version_part)
    return ref, None


@click.group(name="multi")
def multi():
    """MultiEnv commands."""
    pass


@multi.command()
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--mcp-server", "mcp_servers", multiple=True, help="MCPServerEnv id[:version]")
@click.option("--website", "websites", multiple=True, help="WebsiteEnv id[:version]")
@click.option("--metadata", "metadata_pairs", multiple=True, help="Metadata key=value pair (repeatable)")
@click.option("--name", default=None, help="Name agents see this env's MCP server under, e.g. crm -> mcp__crm__<tool> (default: env + 4 random digits per deploy)")
@env_provider_type_option("What deploys the env: 'gateway' (a gateway in front of its servers and websites, whatever their own "
                          "env_provider_type), or the type of an installed agent_env.env_providers plugin, which deploys them all", "multi")
@click.option("--validate", "run_validation", is_flag=True, default=False, help="Validate the environment after registering it")
def put(env_id: str, mcp_servers: tuple[str, ...], websites: tuple[str, ...], metadata_pairs: tuple[str, ...], name: str | None, env_provider_type: str,
        run_validation: bool):
    """Create a MultiEnv from MCPServerEnv and/or WebsiteEnv ids."""

    if not mcp_servers and not websites:
        click.echo("Error: At least one of --mcp-server or --website must be provided.", err=True)
        raise SystemExit(1)

    mcp_server_envs = []
    for ref in mcp_servers:
        server_id, server_version = parse_env_ref(ref)
        click.echo(f"Fetching MCPServerEnv: id={server_id} version={server_version or 'latest'}...")
        env = Env.get(server_id, version=server_version)
        if env.type != "mcp_server":
            click.echo(f"Error: {server_id} is type '{env.type}', expected 'mcp_server'", err=True)
            raise SystemExit(1)
        mcp_server_envs.append(env)
        click.echo(f"  Found: id={env.id} version={env.version}")

    website_envs = []
    for ref in websites:
        website_id, website_version = parse_env_ref(ref)
        click.echo(f"Fetching WebsiteEnv: id={website_id} version={website_version or 'latest'}...")
        env = Env.get(website_id, version=website_version)
        if env.type != "website":
            click.echo(f"Error: {website_id} is type '{env.type}', expected 'website'", err=True)
            raise SystemExit(1)
        website_envs.append(env)
        click.echo(f"  Found: id={env.id} version={env.version}")

    user_metadata = {}
    for pair in metadata_pairs:
        if "=" not in pair:
            raise SystemExit(f"Invalid metadata format '{pair}', expected key=value")
        key, value = pair.split("=", 1)
        user_metadata[key] = value
    metadata = detect_base_metadata()
    metadata.update(user_metadata)

    refusal = MultiEnv(id=env_id, version=None, mcp_server_envs=mcp_server_envs, website_envs=website_envs, name=name,
                       env_provider_type=env_provider_type).deploy_refusal()
    if refusal:
        click.echo(f"Error: {refusal}", err=True)
        raise SystemExit(1)
    click.echo(f"Creating MultiEnv...")
    multi_env = MultiEnv.put(id=env_id, mcp_server_envs=mcp_server_envs, website_envs=website_envs, metadata=metadata if metadata else None, name=name,
                             env_provider_type=env_provider_type)
    click.echo(f"Created MultiEnv: id={multi_env.id} version={multi_env.version} env_provider_type={multi_env.env_provider_type}")
    if run_validation:
        click.echo("\nValidating environment...")
        instance_id = asyncio.run(multi_env.validate(on_progress=click.echo))
        click.echo(f"Validation task: {instance_id}")


@multi.command(name="load-environment-universe-artifact")
@click.option("--env-instance-id", "env_instance_id", required=True, help="Deployed env instance ID")
@click.option("--environment-universe-artifact-id", "environment_universe_artifact_id",
              required=True, help="EnvironmentUniverseArtifact id")
@click.option("--snapshot-after-load/--no-snapshot-after-load", "snapshot_after_load", default=None,
              help="After a full re-ingest, bake a clean snapshot so later loads of this "
                   "universe restore from an image instead of re-ingesting every service. "
                   "Defaults to AGENT_ENV_SNAPSHOT_AFTER_LOAD (off).")
def load_environment_universe_artifact(env_instance_id: str, environment_universe_artifact_id: str,
                                       snapshot_after_load: bool | None):
    """Load an environment universe artifact into a deployed env."""
    from agent_env.store.base import NotFoundError

    from agent_env.env.store import get_env_instance_store
    try:
        deployed_env = get_env_instance_store().get(env_instance_id)
    except NotFoundError:
        click.echo(f"Error: Instance '{env_instance_id}' not found", err=True)
        raise SystemExit(1)

    click.echo(f"Instance: {env_instance_id} (env={deployed_env.env_id} v{deployed_env.env_version})")
    if getattr(deployed_env, "sandbox_type", None):
        click.echo(f"Sandbox backend: {deployed_env.sandbox_type}")
    env = Env.get(deployed_env.env_id, deployed_env.env_version)
    env = asyncio.run(type(env).from_deployed_env(deployed_env))

    if not hasattr(env, "load_environment_universe_artifact"):
        click.echo(f"Error: Env type '{env.type}' does not support load_environment_universe_artifact", err=True)
        raise SystemExit(1)

    click.echo(f"Fetching environment universe artifact: id={environment_universe_artifact_id}...")
    universe_artifact = EnvironmentUniverseArtifact.get(environment_universe_artifact_id)
    click.echo(f"Found environment universe artifact: id={universe_artifact.id} version={universe_artifact.version}")

    click.echo("Loading environment universe artifact...")
    from agent_env.env.envs.multi_env import MultiEnv

    if isinstance(env, MultiEnv):
        result = asyncio.run(env.load_environment_universe_artifact(
            universe_artifact, snapshot_after_load=snapshot_after_load,
        ))
    else:
        if snapshot_after_load:
            click.echo(f"Note: env type '{env.type}' has no servicedb to snapshot; "
                       "--snapshot-after-load ignored")
        result = asyncio.run(env.load_environment_universe_artifact(universe_artifact))
    click.echo("Successfully loaded all environment artifacts into env")

    # Which path ran decides whether this load could time out at all, so say it out
    # loud. Measured on a 13-service / ~8.7GB universe: restore ~4min, re-ingest
    # ~3.5min on 8 vCPU but a FAILURE on the 1-vCPU default. Restore is not
    # reliably faster -- it is bounded, which is the part that matters.
    if getattr(result, "restored_from_snapshot", False):
        click.echo(f"Restored from snapshot image: {result.snapshot_db_image_artifact_id}")
    else:
        click.echo("Re-ingested every service over HTTP (no clean snapshot matched); each "
                   "service was racing its own 600s timeout")
        if result.snapshot_baked:
            click.echo("Baked a clean snapshot; the next load of this universe can restore "
                       "from it instead of re-ingesting")
        elif result.snapshot_baked is False:
            click.echo(f"Snapshot bake did not produce a reusable snapshot: {result.snapshot_bake_error}")

    if result.metadata_filepaths:
        click.echo("Metadata files downloaded:")
        for key, path in result.metadata_filepaths.items():
            click.echo(f"  {key} -> {path}")


@multi.command(name="load-environment-artifact")
@click.option("--env-instance-id", "env_instance_id", required=True, help="Deployed env instance ID")
@click.option("--environment-artifact-id", "environment_artifact_id",
              required=True, help="EnvironmentArtifact id")
def load_environment_artifact(env_instance_id: str, environment_artifact_id: str):
    """Load an environment artifact into a deployed env."""
    from agent_env.store.base import NotFoundError

    from agent_env.env.store import get_env_instance_store
    try:
        deployed_env = get_env_instance_store().get(env_instance_id)
    except NotFoundError:
        click.echo(f"Error: Instance '{env_instance_id}' not found", err=True)
        raise SystemExit(1)

    click.echo(f"Instance: {env_instance_id} (env={deployed_env.env_id} v{deployed_env.env_version})")
    env = Env.get(deployed_env.env_id, deployed_env.env_version)
    env = asyncio.run(type(env).from_deployed_env(deployed_env))

    if not hasattr(env, "load_environment_artifact"):
        click.echo(f"Error: Env type '{env.type}' does not support load_environment_artifact", err=True)
        raise SystemExit(1)

    click.echo(f"Fetching environment artifact: id={environment_artifact_id}...")
    environment_artifact = EnvironmentArtifact.get(environment_artifact_id)
    click.echo(f"Found environment artifact: id={environment_artifact.id} version={environment_artifact.version}")

    click.echo("Loading environment artifact...")
    asyncio.run(env.load_environment_artifact(environment_artifact))
    click.echo("Loaded environment artifact into env")


@multi.command()
@click.option("--id", "env_id", required=True, help="MultiEnv id")
@click.option("--version", "env_version", type=int, default=None, help="Env version (default: latest)")
def validate(env_id: str, env_version: int | None):
    """Validate a MultiEnv's environment card and persist it to env metadata."""
    from agent_env.store.base import NotFoundError
    from agent_env.task_step.task_steps.env_card_validator import VALIDATED_ENVIRONMENT_CARD_KEY

    try:
        env = Env.get(env_id, env_version)
    except NotFoundError:
        click.echo(f"Error: Env '{env_id}' not found", err=True)
        raise SystemExit(1)

    if not isinstance(env, MultiEnv):
        click.echo(f"Error: Env '{env_id}' is type '{env.type}', expected 'multi'", err=True)
        raise SystemExit(1)

    click.echo(f"Validating env: id={env.id} version={env.version}")
    instance_id = asyncio.run(env.validate(on_progress=click.echo))
    click.echo(f"Task instance: {instance_id}")

    env = Env.get(env_id, env.version)
    card_val = env.metadata.get(VALIDATED_ENVIRONMENT_CARD_KEY)
    click.secho("\nEnvironment card:", bold=True)
    if not card_val:
        click.echo("  (not validated)")
    elif card_val.get("accessible"):
        missing = [f for f, v in card_val.get("required_fields", {}).items() if not v.get("present")]
        click.secho(f"  ACCESSIBLE: {card_val.get('children_count', 0)} child env(s); extensions={card_val.get('extensions', [])}; tools={card_val.get('tools', [])}", fg="green")
        if missing:
            click.secho(f"  missing required fields: {missing}", fg="yellow")
    else:
        click.secho(f"  NOT ACCESSIBLE: {card_val.get('error', 'no card')}", fg="magenta")


@multi.command(name="validate-universe-compatibility")
@click.option("--env-id", required=True, help="MultiEnv id")
@click.option("--env-version", type=int, default=None, help="Env version (default: latest)")
@click.option("--universe-artifact-id", required=True, help="EnvironmentUniverseArtifact id")
def validate_universe_compatibility(env_id: str, env_version: int | None, universe_artifact_id: str):
    """Validate universe artifact compatibility with a MultiEnv via load/export roundtrip."""
    from agent_env.store.base import NotFoundError

    try:
        env = Env.get(env_id, env_version)
    except NotFoundError:
        click.echo(f"Error: Env '{env_id}' not found", err=True)
        raise SystemExit(1)

    if not isinstance(env, MultiEnv):
        click.echo(f"Error: Env '{env_id}' is type '{env.type}', expected 'multi'", err=True)
        raise SystemExit(1)

    click.echo(f"Validating env={env.id} v{env.version} universe={universe_artifact_id}")
    instance_id = asyncio.run(env.validate_universe_compatibility(universe_artifact_id=universe_artifact_id, on_progress=click.echo))
    click.echo(f"Task instance: {instance_id}")

    from agent_env.env.env_artifact_store import get_env_artifact_store
    from agent_env.env.env_artifact_store import EnvArtifactType, get_env_artifact_store
    results = get_env_artifact_store().get_by_env(env_id, env_version=env.version, type=EnvArtifactType.UNIVERSE_COMPATIBILITY)
    doc = next((r for r in results if r["artifact_id"] == universe_artifact_id), None)
    compat = doc["data"] if doc else {}

    overall = compat.get("compatible")
    click.echo(click.style(f"\nUniverse compatibility: {'COMPATIBLE' if overall else 'INCOMPATIBLE'}", fg="green" if overall else "magenta"))
    for svc_name, svc_result in sorted(compat.get("services", {}).items()):
        svc_compat = svc_result.get("compatible")
        click.echo(click.style(f"\n  {svc_name}: {'COMPATIBLE' if svc_compat else 'INCOMPATIBLE'}", fg="green" if svc_compat else "magenta"))
        for issue in svc_result.get("issues", []):
            critical = issue.get("critical")
            color = "red" if critical else "yellow"
            marker = "CRITICAL" if critical else "warning"
            click.echo(click.style(f"    [{issue['phase']}] [{marker}] {issue['entity']}.{issue['field']}: {issue['type']} - {issue['detail']}", fg=color))

    if compat.get("exported_universe_artifact_id"):
        click.echo(f"\nExported universe artifact: {compat['exported_universe_artifact_id']}")


@multi.command(name="compatible-universes")
@click.option("--env-id", required=True, help="MultiEnv id")
@click.option("--env-version", type=int, default=None, help="Env version (default: all versions)")
def compatible_universes(env_id: str, env_version: int | None):
    """List universe compatibility results for an env."""
    from agent_env.env.env_artifact_store import get_env_artifact_store

    from agent_env.env.env_artifact_store import EnvArtifactType
    results = get_env_artifact_store().get_by_env(env_id, env_version=env_version, type=EnvArtifactType.UNIVERSE_COMPATIBILITY)
    if not results:
        click.echo(f"No compatibility results found for env {env_id}")
        return
    for r in results:
        compat = r["data"].get("compatible", False)
        status = click.style("COMPATIBLE", fg="green") if compat else click.style("INCOMPATIBLE", fg="magenta")
        click.echo(f"  universe={r['artifact_id']} v{r['artifact_version']}: {status}")
