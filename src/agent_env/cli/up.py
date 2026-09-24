"""``agent-env up`` — the local stack (stores, runner, explorer) in one command.

Everything is resolved from the discovered ``.agentenv/config.toml`` (or ``AGENT_ENV_CONFIG``);
`up` starts no more than the config asks for. The explorer and runner run as host processes, not
containers, so ``LocalSandbox`` shares the Docker daemon's path namespace.
"""

from __future__ import annotations

import importlib.util
import logging
import subprocess
import sys
from shutil import which

import click


def _docker(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def _require_explorer_deps() -> None:
    """The explorer needs the optional ``[explorer]`` extra (fastapi, uvicorn). Check without importing
    them, so a missing extra is an actionable hint rather than a stack trace."""
    missing = [m for m in ("fastapi", "uvicorn") if importlib.util.find_spec(m) is None]
    if missing:
        raise click.ClickException(
            f"agent-env up serves the local explorer, which needs the optional 'explorer' extra "
            f"(missing: {', '.join(missing)}). Install it with:  pip install 'agentenv-framework[explorer]'"
        )


def _require_docker() -> None:
    """Docker is required only where the host actually builds/runs containers (the bootstrap
    image builds; the local sandbox/registry later). Checked at point of use, not at ``up``
    startup, so a remote-backed or browse-only explorer needs no local Docker."""
    if which("docker") is None:
        raise click.ClickException(
            "docker not found on PATH — building the gateway / service-db envs needs a Docker "
            "daemon. Install and start Docker, or run `agent-env up --no-bootstrap`."
        )
    if _docker("info").returncode != 0:
        raise click.ClickException(
            "the Docker daemon is not reachable — start Docker, or run `agent-env up --no-bootstrap`."
        )


def _bootstrap_envs() -> None:
    """Register the two envs ``deploy_env`` resolves by id (service-db, gateway) if missing.
    Idempotent: existing envs are left as-is, so `up` is cheap after the first run (which builds
    their images)."""
    from agent_env.config import get_config
    from agent_env.env import Env
    from agent_env.store import NotFoundError

    cfg = get_config()
    wanted = {
        cfg.default_service_db_env_id: ("service-db", ["env", "service-db", "put", "--id"]),
        cfg.default_gateway_env_id: ("gateway", ["env", "gateway", "put", "--id"]),
    }
    for env_id, (label, argv) in wanted.items():
        try:
            if Env.get(env_id) is not None:
                click.echo(f"  {label:<12} already registered ({env_id})")
                continue
        except NotFoundError:
            pass
        click.echo(f"  {label:<12} building + registering ({env_id}) — first run only, this is slow")
        _require_docker()   # only when we actually build — an already-registered env needs no Docker
        if subprocess.run([sys.executable, "-m", "agent_env.cli", *argv, env_id]).returncode != 0:
            raise click.ClickException(f"bootstrap of {label} failed")


@click.command()
@click.option("--no-bootstrap", is_flag=True, help="Skip building/registering the gateway and service-db envs.")
def up(no_bootstrap: bool) -> None:
    """Start the local agent-env stack (stores, runner, explorer) from .agentenv/config.toml."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    click.echo("agent-env up")
    _require_explorer_deps()

    from agent_env.config import configure, get_config, get_runner

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
    click.echo(f"    document store   {type(cfg.get_document_store()).__name__}")
    click.echo(f"    object store     {type(cfg.get_object_store()).__name__}")
    click.echo(f"    image store      {type(cfg.get_image_store()).__name__}")
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
