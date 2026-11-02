"""``python -m agent_env.training.skyrl URL OUT --tokenizer MODEL``: writes a bundle's tasks as SkyRL's RL dataset."""

import json

import click

from agent_env.training.skyrl.calls import FORMATS
from agent_env.training.skyrl.prepare import write_dataset


@click.command()
@click.argument("url")
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--tokenizer", required=True,
              help="The model's tokenizer, as transformers loads it, whose chat template renders each row's tools into "
                   "its system prompt and counts its prompt.")
@click.option("--tool-format", type=click.Choice(FORMATS), default="hermes", show_default=True,
              help="The tool-call format the model's chat template teaches: hermes for Qwen2.5 and Qwen3, qwen3_coder "
                   "for Qwen3.5 and Qwen3-Coder.")
@click.option("--tool-timeout", type=float, default=60.0, show_default=True,
              help="One tool call's deadline in seconds; the server ends a call that misses it, and the episode.")
@click.option("--reset-timeout", type=float, default=600.0, show_default=True,
              help="How long an episode may take to deploy, and each turn to be ended, in seconds; also how long a "
                   "rollout waits for room on the server, and how long after it began it may be played again.")
@click.option("--attempts", type=int, default=3, show_default=True,
              help="How many times a rollout the env fails is played at most, each time from the start on a new "
                   "episode.")
@click.option("--token-var", help="The environment variable that holds the token the server asks for, read now and by "
                                  "the trainer.")
@click.option("--max-prompt-length", type=int,
              help="Refuse rows whose prompt with its tools is longer, in tokens: SkyRL's trainer.max_prompt_length.")
@click.option("--chat-template-kwargs", type=json.loads,
              help="The chat template's settings as JSON, SkyRL's generator.chat_template_kwargs, to render prompts as "
                   'SkyRL does, e.g. \'{"enable_thinking": false}\'.')
def main(url: str, out: str, tokenizer: str, tool_format: str, tool_timeout: float, reset_timeout: float,
         attempts: int, token_var: str | None, max_prompt_length: int | None, chat_template_kwargs: dict | None) -> None:
    """Writes the tasks of the OpenEnv server at URL as OUT/<split>.parquet, a row per task, each with the tools its
    task offers in its system prompt and the server's settings for the env that plays it."""
    written = write_dataset(url, out, tokenizer=tokenizer, tool_format=tool_format, tool_timeout=tool_timeout,
                            reset_timeout=reset_timeout, attempts=attempts, token_var=token_var,
                            max_prompt_length=max_prompt_length, chat_template_kwargs=chat_template_kwargs)
    click.echo(f"wrote {', '.join(sorted(path.name for path in written.folder.iterdir()))} to {written.folder}")
    click.echo(f"the longest prompt is {written.longest_prompt} tokens with its tools")


if __name__ == "__main__":
    main()
