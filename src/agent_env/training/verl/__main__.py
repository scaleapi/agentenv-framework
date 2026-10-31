"""``python -m agent_env.training.verl URL OUT``: writes a bundle's tasks as verl's RL dataset and agent loop config."""

import json

import click

from agent_env.training.verl.prepare import write_dataset


@click.command()
@click.argument("url")
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--tool-timeout", type=float, default=60.0, show_default=True,
              help="One tool call's deadline in seconds; the server ends a call that misses it, and the episode.")
@click.option("--reset-timeout", type=float, default=600.0, show_default=True,
              help="How long an episode may take to deploy, and each turn to be ended, in seconds; also how long a "
                   "rollout waits for room on the server, and how long after it began it may be played again.")
@click.option("--attempts", type=int, default=3, show_default=True,
              help="How many times a rollout the env fails is played at most, each time from the start on a new "
                   "episode.")
@click.option("--token-var", help="The environment variable that holds the token the server asks for, read now and in "
                                  "each of verl's workers.")
@click.option("--tokenizer", help="The model's tokenizer, as transformers loads it, to count each prompt with its "
                                  "tools.")
@click.option("--max-prompt-length", type=int,
              help="Refuse rows whose prompt with its tools is longer, in tokens: verl's data.max_prompt_length.")
@click.option("--chat-template-kwargs", type=json.loads,
              help="The chat template's settings as JSON, verl's data.apply_chat_template_kwargs, to count prompts as "
                   'verl renders them, e.g. \'{"enable_thinking": false}\'.')
def main(url: str, out: str, tool_timeout: float, reset_timeout: float, attempts: int, token_var: str | None,
         tokenizer: str | None, max_prompt_length: int | None, chat_template_kwargs: dict | None) -> None:
    """Writes the tasks of the OpenEnv server at URL as OUT/<split>.parquet, a row per task, and OUT/agent_loop.yaml,
    the agent loop that plays them on the server."""
    written = write_dataset(url, out, tool_timeout=tool_timeout, reset_timeout=reset_timeout, attempts=attempts,
                            token_var=token_var, tokenizer=tokenizer, max_prompt_length=max_prompt_length,
                            chat_template_kwargs=chat_template_kwargs)
    click.echo(f"wrote {', '.join(sorted(path.name for path in written.folder.iterdir()))} to {written.folder}")
    if written.longest_prompt is not None:
        click.echo(f"the longest prompt is {written.longest_prompt} tokens with its tools")


if __name__ == "__main__":
    main()
