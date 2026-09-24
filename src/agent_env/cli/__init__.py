import logging

import click

from agent_env.plugins._cli import load_cli_plugins, load_cli_root_options
from agent_env.store.base import NotFoundError

from .a2a_agent import a2a_agent
from .artifact import artifact
from .config import config
from .env import env
from .eval import eval
from .plugin import plugin
from .task import task
from .up import up


# Errors that mean "not allowed" or "not there" rather than "agent-env is broken". Spelled out
# because there is no domain base class to catch: ConfigError subclasses ValueError (so does
# pydantic's ValidationError), while NotFoundError subclasses Exception. A TypeError or a
# retry-exhausted DuplicateKeyError is a defect, and a defect keeps its traceback.
_USER_FACING_ERRORS = (ValueError, NotFoundError)


class _UserErrorsAreNotCrashes(click.Group):
    """Render a rule the caller broke as one line, not a stack trace through click's frames.

    Click formats only ``ClickException`` specially, so an unknown store backend or a missing
    artifact otherwise arrives looking like agent-env crashed rather than like the answer it
    is. ``--verbose`` re-raises, so the traceback is deferred rather than lost.
    """

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except _USER_FACING_ERRORS as e:
            if ctx.params.get("verbose"):
                raise
            raise click.ClickException(str(e)) from e


@click.group(cls=_UserErrorsAreNotCrashes)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging (DEBUG level)")
def cli(verbose: bool):
    """Agent environment CLI."""
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(name)s %(message)s")


cli.add_command(a2a_agent)
cli.add_command(env)
cli.add_command(artifact)
cli.add_command(config)
cli.add_command(eval)
cli.add_command(plugin)
cli.add_command(task)
cli.add_command(up)

load_cli_plugins(cli)
load_cli_root_options(cli)
