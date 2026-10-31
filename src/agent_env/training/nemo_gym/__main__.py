"""``python -m agent_env.training.nemo_gym SOURCE ROOT``: writes NeMo Gym's resources server for a bundle's tasks.
``python -m agent_env.training.nemo_gym serve FOLDER``: runs a written one on its own, for a trainer that takes a
resources server's URL, as verifiers' NeMo Gym taskset does."""

import click

from agent_env.training.nemo_gym.prepare import write_server

try:
    from agent_env.training.nemo_gym.server import serve as run_server
except ImportError as exc:
    run_server, UNSERVABLE = None, str(exc)


class _WriteByDefault(click.Group):
    """``serve`` and ``write`` by name; anything else is ``write``'s arguments, as the command took them before it had
    subcommands."""

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and args[0] not in self.commands and args[0] != "--help":
            args = ["write", *args]
        return super().parse_args(ctx, args)


@click.group(cls=_WriteByDefault)
def main() -> None:
    """Writes NeMo Gym's resources server for a bundle's tasks, or serves a written one on its own."""


@main.command()
@click.argument("source")
@click.argument("root", type=click.Path(file_okay=False))
@click.option("--name", help="The server's name; the bundle's env name by default.")
@click.option("--sandbox", help="Where the server deploys the bundle's envs, as agent-env run --sandbox takes it.")
@click.option("--reward", help="The step whose score is the reward, for tasks that record several.")
@click.option("--reward-components", is_flag=True,
              help="Answer verify with each of a task's scores as the reward's components, for NeMo RL's GDPO; the "
                   "reward is then their sum.")
@click.option("--max-concurrent-envs", type=int, default=8, show_default=True,
              help="The episodes the server runs at once, which should cover a step's rollouts: the rest wait their turn.")
def write(source: str, root: str, name: str | None, sandbox: str | None, reward: str | None, reward_components: bool,
          max_concurrent_envs: int) -> None:
    """Writes the resources server for SOURCE, a bundle or the URL of an OpenEnv server serving one, into the Gym root
    ROOT, with its config and a JSONL file of rows per split. Playing one task of each toolset to find its tools
    deploys its envs."""
    server = write_server(source, root, name=name, sandbox=sandbox, reward=reward, reward_components=reward_components,
                          max_concurrent_envs=max_concurrent_envs)
    click.echo(f"wrote {server}")


@main.command()
@click.argument("folder", type=click.Path(exists=True, file_okay=False))
@click.option("--host", default="127.0.0.1", show_default=True,
              help="The address to listen on. The server checks no token, so keep it off public interfaces.")
@click.option("--port", type=int, required=True, help="The port to listen on.")
@click.option("--masked-as-error", is_flag=True,
              help="Answer verify for a rollout the environment failed with HTTP 503 rather than mask_sample, for a "
                   "client that doesn't read mask_sample, as verifiers' NeMo Gym taskset doesn't: it then fails the "
                   "rollout, and prime-rl trains on none of it.")
@click.option("--idle-timeout", type=float,
              help="Seconds a session may go without a request before its episode is ended, the config's (600) by "
                   "default. A client that abandons sessions without closing them, as verifiers' NeMo Gym taskset "
                   "does with the rollouts prime-rl cancels, holds their episodes until then.")
def serve(folder: str, host: str, port: int, masked_as_error: bool, idle_timeout: float | None) -> None:
    """Runs the resources server written into FOLDER, a Gym root's resources_servers/<name>, on its own: without Gym's
    head server or the agent servers its config names. Point verifiers' NeMo Gym taskset at it with
    env.taskset.task.resources_url, and its env.taskset.dataset at the folder's data/<split>.jsonl."""
    if run_server is None:
        raise click.ClickException(UNSERVABLE)
    run_server(folder, host=host, port=port, masked_as_error=masked_as_error, idle_timeout=idle_timeout)


if __name__ == "__main__":
    main()
