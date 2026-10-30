"""Build the forms and arches of a computer env not built yet: those that failed, and those a put that stopped left
building."""

import click

from agent_env.env import Env
from agent_env.env.envs.computer.utils import composer
from agent_env.env.envs.computer.env import ComputerEnv
from agent_env.providers import build_sandbox_provider
from agent_env.store.base import NotFoundError


@click.command()
@click.option("--id", "env_id", required=True, help="The computer env's id")
@click.option("--version", type=int, default=None, help="Its version; the latest when left out")
@click.option("--sandbox", default=None, help="The sandbox provider that resolves, builds and probes, by default the "
              "env's; comma-separated for a chain, each arch going to the first link that runs it natively")
@click.option("--no-emulate", is_flag=True, help="Skip, rather than emulate, an arch no link runs natively")
def build(env_id: str, version: int | None, sandbox: str | None, no_emulate: bool):
    """Build a computer env's forms and arches not built yet, reusing any image of the same key, and print each one's
    status; exits 1 when one failed."""
    try:
        provider = build_sandbox_provider(sandbox) if sandbox else None
        env = Env.get(env_id, version)
    except (NotFoundError, ValueError) as e:
        raise click.ClickException(str(e)) from None
    if not isinstance(env, ComputerEnv):
        raise click.ClickException(f"{env_id} is a {env.type} env; only a computer env is built")
    if env.computer is None:
        raise click.ClickException(f"{env_id} names no computer and driver, so it has nothing to build alone")
    if (pairs := composer.pending(env, provider)) or not composer.current(env):
        composer.compose(env, pairs, on_progress=click.echo, sandbox_provider=provider, emulate=not no_emulate)
    failed = False
    for form, by_arch in env.metadata.get("composed", {}).items():
        for arch, result in by_arch.items():
            failed |= result["status"] == "failed"
            detail = result.get("reason") or result.get("image", "")
            click.echo(f"{form}/{arch}: {result['status']}" + (f" ({'reused, ' if result.get('reused') else ''}"
                                                              f"{detail})" if detail else ""))
    if failed:
        raise SystemExit(1)
