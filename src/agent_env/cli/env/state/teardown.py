"""Tear down a durable env-state store initialized via ``env init-env-state``.

Idempotent: tearing down an instance whose store is already gone — or an unknown instance id — logs and no-ops
rather than erroring, so a future TTL reaper and a manual teardown can race without conflict.
"""

import asyncio

import click


@click.command(name="teardown-env-state")
@click.option("--instance-id", "instance_id", required=True, help="Env state instance id (esi-…)")
def teardown_env_state(instance_id: str):
    """Tear down an env-state store by instance id and retire its record."""
    from agent_env.providers.state import build_state_provider, get_env_state_instance_store
    from agent_env.store.base import NotFoundError

    try:
        instance = get_env_state_instance_store().get(instance_id)
    except NotFoundError:
        click.echo(f"No env state instance {instance_id} found; nothing to tear down.")
        return

    kind = instance.metadata.get("kind")
    if kind == "base":
        click.secho(
            f"⚠ {instance_id} is a persistent BASE (database "
            f"{instance.metadata.get('dbname')!r}). Tearing it down DROPS the shared base database "
            "and everything materialized in it. Any run overlay still pointing at this base "
            "(base_env_state_instance_id) will be orphaned — its data source disappears.",
            fg="yellow",
        )
    elif kind == "run_overlay":
        click.echo(
            f"{instance_id} is a persistent RUN_OVERLAY over base "
            f"{instance.metadata.get('base_env_state_instance_id')!r}; dropping only its overlay "
            "schemas + run role. The shared base is preserved (not orphaned)."
        )

    click.echo(f"Tearing down env state instance {instance_id} ({instance.state_type})...")
    provider = build_state_provider(instance.state_type)
    asyncio.run(provider.teardown(instance))
    click.echo("Done. " + click.style(f"{instance_id} torn down / retired", fg="green"))
