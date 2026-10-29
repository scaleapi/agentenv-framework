"""``python -m agent_env.training.nemo_gym SOURCE ROOT``: writes NeMo Gym's resources server for a bundle's tasks."""

import click

from agent_env.training.nemo_gym.prepare import write_server


@click.command()
@click.argument("source")
@click.argument("root", type=click.Path(file_okay=False))
@click.option("--name", help="The server's name; the bundle's env name by default.")
@click.option("--sandbox", help="Where the server deploys the bundle's envs, as agent-env run --sandbox takes it.")
@click.option("--reward", help="The step whose score is the reward, for tasks that record several.")
@click.option("--max-concurrent-envs", type=int, default=8, show_default=True,
              help="The episodes the server runs at once, which must cover the rollouts Gym runs at once.")
def main(source: str, root: str, name: str | None, sandbox: str | None, reward: str | None,
         max_concurrent_envs: int) -> None:
    """Writes the resources server for SOURCE, a bundle or the URL of an OpenEnv server serving one, into the Gym root
    ROOT, with its config and a JSONL file of rows per split. Playing one task of each toolset to find its tools
    deploys its envs."""
    server = write_server(source, root, name=name, sandbox=sandbox, reward=reward,
                          max_concurrent_envs=max_concurrent_envs)
    click.echo(f"wrote {server}")


if __name__ == "__main__":
    main()
