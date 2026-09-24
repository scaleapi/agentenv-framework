import asyncio
import sys

import click

from agent_env.a2a_agent import A2AAgent

MIN_TTL_SECONDS = 60
MAX_TTL_SECONDS = 1209600
DEFAULT_TTL_SECONDS = 7200


@click.command()
@click.option("--id", "agent_id", required=True, help="A2A agent id")
@click.option("--version", "agent_version", default=None, type=int, help="Agent version (defaults to latest)")
@click.option("--env-var", "env_var_pairs", multiple=True, help="Override env var KEY=VALUE (repeatable)")
@click.option("--ttl-seconds", type=click.IntRange(min=MIN_TTL_SECONDS, max=MAX_TTL_SECONDS), default=DEFAULT_TTL_SECONDS,
              help=f"VM lifetime in seconds (default {DEFAULT_TTL_SECONDS})")
@click.option("--sandbox", default=None,
              help="Sandbox backend(s): a built-in (modal, modal_vm, e2b, local) "
                   "or a name from [sandbox.providers] in .agentenv/config.toml; comma-separated for a "
                   "fallback chain. Defaults to [sandbox].agent_default (else local) when omitted.")
@click.option("--priority", type=int, default=0, show_default=True,
              help="Sandbox priority: 0=interactive, 1=non_interactive (honoured by backends with priority tiers)")
def deploy(agent_id: str, agent_version: int | None, env_var_pairs: tuple[str, ...], ttl_seconds: int, sandbox: str, priority: int):
    """Deploy an A2A agent on a sandbox VM."""
    from agent_env.providers import build_sandbox_provider, set_agent_sandbox_provider

    if sandbox:
        try:
            provider = build_sandbox_provider(sandbox)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="'--sandbox'")
        set_agent_sandbox_provider(provider)
    click.echo(f"Sandbox backend: {sandbox or 'config default'}")

    env_vars = {}
    for pair in env_var_pairs:
        if "=" not in pair:
            click.echo(f"Invalid env-var format '{pair}', expected KEY=VALUE", err=True)
            sys.exit(1)
        key, value = pair.split("=", 1)
        env_vars[key] = value

    click.echo(f"Fetching A2A agent: id={agent_id} version={agent_version or 'latest'}...")
    agent = A2AAgent.get(agent_id, agent_version)
    click.echo(f"Found: id={agent.id} version={agent.version} image={agent.docker_image_artifact.id}")

    click.echo(f"Deploying (ttl={ttl_seconds}s)...")
    deployed = asyncio.run(agent.deploy(ttl_seconds=ttl_seconds, env_vars=env_vars if env_vars else None, priority=priority))

    click.echo("Deployed!")
    click.echo("Instance ID: " + click.style(deployed.instance_id, fg="green"))
    click.echo("A2A URL: " + click.style(deployed.a2a_url, fg="green"))
    click.echo("Sandbox ID: " + click.style(deployed.sandbox_id, fg="yellow"))
    click.echo(f"Agent Card: {deployed.agent_card.get('name', 'unknown')}")
    if deployed.expires_at_utc:
        click.echo(f"Expires At (UTC): {deployed.expires_at_utc}")
