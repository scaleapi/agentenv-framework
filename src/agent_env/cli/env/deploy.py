import asyncio

import click

from agent_env.env import DeployedGatewayEnv, Env
from agent_env.env.gateway import GatewayMode
from agent_env.store.routing import run_scope

MIN_TTL_SECONDS = 60  # 1 minute
MAX_TTL_SECONDS = 1209600  # 2 weeks
DEFAULT_TTL_SECONDS = 10800  # 3 hours


@click.command()
@click.option("--id", "env_id", required=True, help="Env id")
@click.option("--version", "env_version", default=None, type=int, help="Env version (defaults to latest)")
@click.option("--ttl-seconds", type=click.IntRange(min=MIN_TTL_SECONDS, max=MAX_TTL_SECONDS), default=DEFAULT_TTL_SECONDS,
              help=f"VM lifetime in seconds (min {MIN_TTL_SECONDS}, max {MAX_TTL_SECONDS}, default {DEFAULT_TTL_SECONDS})")
@click.option("--gateway-mode", type=click.Choice([m.value for m in GatewayMode], case_sensitive=False),
              default=GatewayMode.PERFORMANCE.value, help="Gateway mode (performance or consistent)")
@click.option("--sandbox", default=None,
              help="Sandbox backend(s): a built-in (modal, modal_vm, e2b, sail, local) "
                   "or a name from [sandbox.providers] in .agentenv/config.toml; comma-separated for a "
                   "fallback chain. Defaults to [sandbox].default (else local) when omitted.")
@click.option("--service-db", "service_db_env_id", default=None,
              help="Override default_service_db_env_id (on Modal its images must be in the configured image store)")
@click.option("--gateway", "gateway_env_id", default=None,
              help="Override default_gateway_env_id")
@click.option("--disk-size-gb", type=float, default=10.0, show_default=True,
              help="Sandbox VM disk size in GB. Large universes (e.g. github/gdrive "
                   "root file trees) need more than the 10GB default.")
@click.option("--cpu", type=float, default=None,
              help="Sandbox VM vCPUs. Defaults to the provider value of 1.0, "
                   "which is one whole core after rounding. A universe loads its "
                   "components concurrently onto that one VM, and the loaders are "
                   "CPU-bound (JSON decode, row shaping, COPY), so a large "
                   "multi-component universe is competing with itself for a single "
                   "core.")
@click.option("--memory-mb", type=int, default=None,
              help="Sandbox VM memory in MB (default provider value, ~8192). "
                   "Stopgap for memory-heavy parallel loads; prefer sizing "
                   "--disk-size-gb since artifact staging is disk-backed.")
@click.option("--env-state-type", default=None,
              help="Env state type when creating a fresh store: 'local_postgres' (the built-in "
                   "default) or any backend registered under [state.providers] in config.toml.")
@click.option("--env-state-instance-id", default=None,
              help="Attach against an EXISTING env state instance instead of creating a fresh store "
                   "(its own type selects the provider).")
def deploy(env_id: str, env_version: int | None, ttl_seconds: int, gateway_mode: str, sandbox: str, service_db_env_id: str | None, gateway_env_id: str | None, disk_size_gb: float, cpu: float | None, memory_mb: int | None, env_state_type: str | None, env_state_instance_id: str | None):
    """Deploy an MCP server environment."""
    from agent_env.providers import build_sandbox_provider
    from agent_env.config import get_config

    if sandbox:
        try:
            build_sandbox_provider(sandbox)
        except ValueError as e:
            raise click.BadParameter(str(e), param_hint="'--sandbox'")
    click.echo(f"Sandbox backend: {sandbox or 'config default'}")

    if service_db_env_id:
        get_config().default_service_db_env_id = service_db_env_id
        click.echo(f"Override default_service_db_env_id: {service_db_env_id}")
    if gateway_env_id:
        get_config().default_gateway_env_id = gateway_env_id
        click.echo(f"Override default_gateway_env_id: {gateway_env_id}")

    click.echo(f"Fetching env: id={env_id} version={env_version or 'latest'}...")
    env = Env.get(env_id, version=env_version)
    click.echo(f"Found env: id={env.id} version={env.version} type={env.type}")

    mode = GatewayMode(gateway_mode)
    cpu_note = f", cpu={cpu}" if cpu else ""
    mem_note = f", memory={memory_mb}MB" if memory_mb else ""
    click.echo(f"Deploying (ttl={ttl_seconds}s, gateway_mode={mode.value}, disk={disk_size_gb}GB{cpu_note}{mem_note})...")
    deploy_kwargs = dict(ttl_seconds=ttl_seconds, gateway_mode=mode, sandbox_type=sandbox, disk_size_gb=disk_size_gb)
    if cpu is not None:
        deploy_kwargs["cpu"] = cpu
    if memory_mb is not None:
        deploy_kwargs["memory_mb"] = memory_mb
    if env_state_type is not None:
        deploy_kwargs["env_state_type"] = env_state_type
        click.echo(f"Env state type: {env_state_type}")
    if env_state_instance_id:
        deploy_kwargs["env_state_instance_id"] = env_state_instance_id
        click.echo(f"Env state instance id: {env_state_instance_id}")
    try:
        with run_scope(env.id):
            deployed_env = asyncio.run(env.deploy(**deploy_kwargs))
    except NotImplementedError as e:
        click.echo(f"Error: {e}", err=True)
        raise SystemExit(1)

    click.echo(f"Deployed!")
    if deployed_env.instance_id:
        click.echo("Instance ID: " + click.style(deployed_env.instance_id, fg="green"))
    click.echo("Env MCP Url: " + click.style(deployed_env.mcp_url, fg="blue"))
    # A record without a gateway has none of the gateway's URLs.
    if isinstance(deployed_env, DeployedGatewayEnv):
        click.echo("Env Gateway Url: " + click.style(deployed_env.gateway_url, fg="yellow"))
        if deployed_env.db_web_url:
            click.echo("Env DB Web Url: " + click.style(deployed_env.db_web_url, fg="yellow"))
        if deployed_env.db_mcp_url:
            click.echo("Env DB MCP Url: " + click.style(deployed_env.db_mcp_url, fg="yellow"))
        if deployed_env.vnc_url:
            click.echo("VNC Url: " + click.style(deployed_env.vnc_url, fg="yellow"))
        for svc_name, url in (deployed_env.website_frontend_urls or {}).items():
            click.echo(f"Website Frontend ({svc_name}): " + click.style(url, fg="yellow"))
    if deployed_env.expires_at_utc:
        click.echo(f"Expires At (UTC): {deployed_env.expires_at_utc}")
