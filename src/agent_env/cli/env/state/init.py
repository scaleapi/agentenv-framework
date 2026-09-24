"""Pre-initialize a durable remote state store for an env, decoupled from a deploy.

Runs the same ``acquire`` + ``prepare`` a deploy would, but out-of-band and with a long TTL, so a
long-running store can be stood up ahead of time and referenced later by its ``instance_id``
"""

import asyncio

import click

from agent_env.env import Env, MultiEnv

# A durable store can't be shorter-lived than a normal run, so the floor is the default run TTL
# (agent-env's current default, mirrored from cli/env/deploy.py).
MIN_TTL_SECONDS = 10800  # 3 hours (= default run TTL)
DEFAULT_TTL_SECONDS = 2592000  # 30 days
MAX_TTL_SECONDS = 31536000  # 365 days


def _resolve_environment_names(env: Env) -> list[str]:
    """One schema per DB-client environment. A MultiEnv contributes its MCP + website environments; a
    standalone servicedb-backed env contributes its single one.

    Must match the list the gateway passes to ``prepare`` at deploy (``GatewayProvider._all_environments``,
    the authoritative one) — including the ``website_browser`` server the gateway auto-adds whenever an
    env has websites. A name missing here yields a base without that schema, and the deploy then builds
    it a silently-empty overlay."""
    if isinstance(env, MultiEnv):
        names = [e.environment_name for e in env.mcp_server_envs] + [
            e.environment_name for e in env.website_envs
        ]
        if env.website_envs:
            from agent_env.config import get_config

            names.append(Env.get(get_config().default_website_browser_env_id).environment_name)
    elif hasattr(env, "environment_name"):
        names = [env.environment_name]
    else:
        return []
    return list(dict.fromkeys(names))

# TODO: right now this takes an `--id` param because the schemas to initialize the EnvStateInstance with is
# derived from the env. Refactor this to use environment_cards, and consolidate _resolve_environment_names with the 
# same logic in GatewayProvider._all_environments, so that the code can't drift
@click.command(name="init-env-state")
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--env-state-type", "env_state_type", default=None,
              help="Env state type to initialize (required). Only external types — those that "
                   "allocate a store upstream of the gateway — can be pre-initialized out-of-band. "
                   "Which are registered depends on the installed config.toml.")
@click.option("--ttl-seconds", type=click.IntRange(min=MIN_TTL_SECONDS, max=MAX_TTL_SECONDS),
              default=DEFAULT_TTL_SECONDS,
              help=f"Store lifetime in seconds (min {MIN_TTL_SECONDS}, max {MAX_TTL_SECONDS}, "
                   f"default {DEFAULT_TTL_SECONDS} = 30 days)")
def init_env_state(env_id: str, env_state_type: str | None, ttl_seconds: int):
    """Pre-initialize a durable remote state store for an env and print its instance id."""
    from agent_env.providers.state import (
        acquire_state_for_deploy,
        build_state_provider,
    )

    if not env_state_type:
        raise click.BadParameter("an env state type must be supplied", param_hint="'--env-state-type'")
    state_type = env_state_type

    try:
        build_state_provider(state_type)  # resolve before any Mongo/AWS work, so a typo fails fast
    except ValueError as e:
        raise click.BadParameter(str(e), param_hint="'--env-state-type'") from e

    click.echo(f"Fetching env: id={env_id}...")
    env = Env.get(env_id)
    click.echo(f"Found env: id={env.id} version={env.version} type={env.type}")

    environment_names = _resolve_environment_names(env)
    if not environment_names:
        raise click.BadParameter(
            f"env type '{env.type}' has no environments to back with a DB state store",
            param_hint="'--id'",
        )
    click.echo(f"Environments (one schema each): {', '.join(environment_names)}")

    click.echo(f"Initializing {env_state_type} store (ttl={ttl_seconds}s)...")

    async def _init() -> "EnvStateInstance | None":
        inst = await acquire_state_for_deploy(
            env_state_type=state_type, ttl_seconds=ttl_seconds, name_hint=env_id,
        )
        if inst is not None:
            await build_state_provider(state_type).prepare(environment_names, instance=inst)
        return inst

    instance = asyncio.run(_init())
    if instance is None:
        raise click.BadParameter(
            f"env state type '{env_state_type}' cannot be pre-initialized out-of-band",
            param_hint="'--env-state-type'",
        )

    click.echo("Initialized!")
    click.echo(f"  host:      {instance.metadata.get('host')}")
    click.echo(f"  database:  {instance.metadata.get('dbname')}")
    click.echo(f"  schemas:   {', '.join(environment_names)}")
    click.echo(f"  created:   {instance.created_at_utc}")
    click.echo(f"  expires:   {instance.expires_at_utc}")
    click.echo("Env State Instance ID: " + click.style(instance.instance_id, fg="green"))
