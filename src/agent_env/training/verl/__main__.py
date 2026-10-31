"""``python -m agent_env.training.verl URL OUT``: writes a bundle's tasks as verl's RL dataset and agent loop config."""

import click

from agent_env.training.verl.prepare import write_dataset


@click.command()
@click.argument("url")
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--tool-timeout", type=float, default=60.0, show_default=True,
              help="One tool call's deadline in seconds; the server ends a call that misses it, and the episode.")
@click.option("--reset-timeout", type=float, default=600.0, show_default=True,
              help="How long an episode may take to deploy, and each turn to be ended, in seconds; also how long a "
                   "rollout waits for room on the server.")
@click.option("--attempts", type=int, default=3, show_default=True,
              help="How many times a rollout the env fails is played, each time from the start on a new episode.")
def main(url: str, out: str, tool_timeout: float, reset_timeout: float, attempts: int) -> None:
    """Writes the tasks of the OpenEnv server at URL as OUT/<split>.parquet, a row per task, and OUT/agent_loop.yaml,
    the agent loop that plays them on the server."""
    folder = write_dataset(url, out, tool_timeout=tool_timeout, reset_timeout=reset_timeout, attempts=attempts)
    click.echo(f"wrote {', '.join(sorted(path.name for path in folder.iterdir()))} to {folder}")


if __name__ == "__main__":
    main()
