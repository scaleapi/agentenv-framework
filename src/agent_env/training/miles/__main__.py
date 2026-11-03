"""``python -m agent_env.training.miles URL OUT``: writes a bundle's tasks as Miles' prompt data."""

import click

from agent_env.training.miles.prepare import write_rows


@click.command()
@click.argument("url")
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--tool-timeout", type=float, default=60.0, show_default=True,
              help="One tool call's deadline in seconds; the server ends a call that misses it, and the episode.")
@click.option("--reset-timeout", type=float, default=600.0, show_default=True,
              help="How long an episode may take to deploy, and each turn to be ended, in seconds; also how long a "
                   "rollout waits for room on the server.")
@click.option("--model-timeout", type=float, default=600.0, show_default=True,
              help="How long the agent function waits for each of the model's replies, in seconds; an episode whose "
                   "model misses it fails, and its group is left out.")
@click.option("--max-turns", type=int, default=4, show_default=True,
              help="The model turns a rollout may take; the episode is ended after the last.")
@click.option("--token-var", help="The environment variable that holds the token the server asks for, read now and in "
                                  "each of Miles' rollout processes.")
def main(url: str, out: str, tool_timeout: float, reset_timeout: float, model_timeout: float, max_turns: int,
         token_var: str | None) -> None:
    """Writes the tasks of the OpenEnv server at URL as OUT/<split>.jsonl, a row per task, for Miles' --prompt-data
    with --input-key messages and --metadata-key metadata."""
    folder = write_rows(url, out, tool_timeout=tool_timeout, reset_timeout=reset_timeout, model_timeout=model_timeout,
                        max_turns=max_turns, token_var=token_var)
    click.echo(f"wrote {', '.join(sorted(path.name for path in folder.iterdir()))} to {folder}")


if __name__ == "__main__":
    main()
