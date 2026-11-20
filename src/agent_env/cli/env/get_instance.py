"""Get a deployed environment instance by ID."""

import click


@click.command("get-instance")
@click.option("--id", "instance_id", required=True, help="Instance ID")
def get_instance(instance_id: str):
    """Look up a deployed environment instance."""
    from agent_env.store.base import NotFoundError

    from agent_env.env.env import DeployedGatewayEnv, DeployedSandboxEnv
    from agent_env.env.store import get_env_instance_store

    try:
        deployed_env = get_env_instance_store().get(instance_id)
    except NotFoundError:
        click.echo(f"Error: Instance '{instance_id}' not found", err=True)
        raise SystemExit(1)

    click.echo(f"Instance ID: {deployed_env.instance_id}")
    click.echo(f"Env ID: {deployed_env.env_id}")
    click.echo(f"Env Version: {deployed_env.env_version}")
    # A record without a gateway has none of the gateway's URLs.
    gateway = deployed_env if isinstance(deployed_env, DeployedGatewayEnv) else None
    if gateway:
        click.echo(f"Env Base Url: {gateway.gateway_url}")
    click.echo(f"Env MCP Url: {deployed_env.mcp_url}")
    if gateway and gateway.db_web_gateway_url:
        click.echo(f"Env DB Web Url: {gateway.db_web_gateway_url}")
    if gateway and gateway.db_mcp_gateway_url:
        click.echo(f"Env DB MCP Url: {gateway.db_mcp_gateway_url}")
    for svc_name, url in ((gateway.website_frontend_urls if gateway else None) or {}).items():
        click.echo(f"Website Frontend ({svc_name}): {url}")
    if isinstance(deployed_env, DeployedSandboxEnv):
        click.echo(f"Sandbox ID: {deployed_env.sandbox_id}")
    if deployed_env.created_at_utc:
        click.echo(f"Created At (UTC): {deployed_env.created_at_utc}")
    if deployed_env.expires_at_utc:
        click.echo(f"Expires At (UTC): {deployed_env.expires_at_utc}")
