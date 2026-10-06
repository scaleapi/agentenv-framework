"""``agent-env up`` — the local stack (stores, runner, explorer) in one command.

Everything is resolved from the discovered ``.agentenv/config.toml`` (or ``AGENT_ENV_CONFIG``);
`up` starts no more than the config asks for. The explorer and runner run as host processes, not
containers, so ``LocalSandbox`` shares the Docker daemon's path namespace.
"""

from __future__ import annotations

import importlib.util
import logging

import click

from agent_env.env.bootstrap import GATEWAY, SERVICE_DB, InfraError, default_env_id, ensure_default_envs


def _require_explorer_deps() -> None:
    """The explorer needs the optional ``[explorer]`` extra (fastapi, uvicorn). Check without importing
    them, so a missing extra is an actionable hint rather than a stack trace."""
    missing = [m for m in ("fastapi", "uvicorn") if importlib.util.find_spec(m) is None]
    if missing:
        raise click.ClickException(
            f"agent-env up serves the local explorer, which needs the optional 'explorer' extra "
            f"(missing: {', '.join(missing)}). Install it with:  pip install 'agentenv-framework[explorer]'"
        )


def _bootstrap_envs() -> None:
    """Build the two envs ``deploy_env`` resolves by id (service-db, gateway) when they're missing or another
    agent-env release built them, so `up` is cheap after the first run, which builds their images."""
    try:
        builds = ensure_default_envs((SERVICE_DB, GATEWAY), say=lambda line: click.echo(f"  {line}"))
    except InfraError as e:
        raise click.ClickException(f"{e}; or run `agent-env up --no-bootstrap`") from None
    for kind in (SERVICE_DB, GATEWAY):
        if kind not in {build.kind for build in builds}:
            click.echo(f"  {kind:<12} already registered ({default_env_id(kind)})")


@click.command()
@click.option("--no-bootstrap", is_flag=True, help="Skip building/registering the gateway and service-db envs.")
def up(no_bootstrap: bool) -> None:
    """Start the local agent-env stack (stores, runner, explorer) from .agentenv/config.toml."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    click.echo("agent-env up")
    _require_explorer_deps()

    from agent_env.config import configure, get_config, get_runner
    from agent_env.store.routing import configured_store

    configure()
    cfg = get_config()
    # After configure(), not before: the check resolves a document, and configure() would
    # discard that Config and resolve a second one.
    if cfg.config_path() is None:
        raise click.ClickException(
            "no .agentenv/config.toml found. Create one to select your backends "
            "(see .agentenv/config.example.toml in the agent-env repo), or point "
            "AGENT_ENV_CONFIG at an existing config file."
        )

    # The local image store (if the config resolves one) brings up its own registry:2 on
    # first push via ensure_repository — up manages no store infra of its own.
    click.echo("\n  resolved backends")
    click.echo(f"    document store   {type(configured_store(cfg.get_document_store())).__name__}")
    click.echo(f"    object store     {type(configured_store(cfg.get_object_store())).__name__}")
    click.echo(f"    image store      {type(configured_store(cfg.get_image_store())).__name__}")
    click.echo(f"    secret store     {type(cfg.get_secret_store()).__name__}")
    click.echo(f"    runner           {get_runner().type}")

    if not no_bootstrap:
        click.echo("\n  bootstrap")
        _bootstrap_envs()

    from agent_env.explorer.app import explorer_settings

    settings = explorer_settings()
    host, port = settings["host"], settings["port"]
    click.echo(f"\n  explorer  http://{host}:{port}")
    click.echo(f"  docs   http://{host}:{port}/docs        (UI)")
    click.echo(f"  api    http://{host}:{port}/api/docs    (Swagger)")
    click.echo("  ready — ctrl-c to stop\n")

    import uvicorn
    # Factory string so uvicorn builds the app exactly once, in the server process;
    # importing explorer_settings above no longer builds it (see explorer.app.__getattr__).
    uvicorn.run("agent_env.explorer.app:create_app", factory=True, host=host, port=port, log_level="info")
