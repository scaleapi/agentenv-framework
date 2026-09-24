import click


@click.command("get-instance")
@click.option("--id", "instance_id", required=True, help="Instance ID")
def get_instance(instance_id: str):
    """Look up a deployed A2A agent instance."""
    from agent_env.a2a_agent.store import get_a2a_agent_instance_store
    from agent_env.store.base import NotFoundError

    try:
        deployed = get_a2a_agent_instance_store().get(instance_id)
    except NotFoundError:
        click.echo(f"Error: Instance '{instance_id}' not found", err=True)
        raise SystemExit(1)

    click.echo(f"Instance ID: {deployed.instance_id}")
    click.echo(f"Agent ID: {deployed.agent_id}")
    click.echo(f"Agent Version: {deployed.agent_version}")
    click.echo(f"A2A URL: {deployed.a2a_url}")
    click.echo(f"Sandbox ID: {deployed.sandbox_id}")
    click.echo(f"Agent Card: {deployed.agent_card.get('name', 'unknown')}")
    if deployed.created_at_utc:
        click.echo(f"Created At (UTC): {deployed.created_at_utc}")
    if deployed.expires_at_utc:
        click.echo(f"Expires At (UTC): {deployed.expires_at_utc}")
