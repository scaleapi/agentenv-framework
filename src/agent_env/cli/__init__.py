import logging

import click

from agent_env.plugins._cli import load_cli_plugins, load_cli_root_options
from agent_env.store.base import NotFoundError
from agent_env.store.routing import namespace_routing
from agent_env.utils.docker_build import DockerBuildError

from .a2a_agent import a2a_agent
from .artifact import artifact
from .config import config
from .env import env
from .eval import eval
from .plugin import plugin
from .run import run
from .task import task
from .up import up


# Errors that mean "not allowed", "not there" or "your build failed" rather than "agent-env is broken".
# Spelled out because there is no domain base class to catch: ConfigError subclasses ValueError (so does
# pydantic's ValidationError), while NotFoundError and DockerBuildError don't. A TypeError or a
# retry-exhausted DuplicateKeyError is a defect, and a defect keeps its traceback.
_USER_FACING_ERRORS = (ValueError, NotFoundError, DockerBuildError)


class _UserErrorsAreNotCrashes(click.Group):
    """Render a rule the caller broke as one line, not a stack trace through click's frames.

    Click formats only ``ClickException`` specially, so an unknown store backend or a missing
    artifact otherwise arrives looking like agent-env crashed rather than like the answer it
    is. The error's notes follow it, since they say what was being done when it happened.
    ``--verbose`` re-raises, so the traceback is deferred rather than lost.
    """

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except _USER_FACING_ERRORS as e:
            if ctx.params.get("verbose"):
                raise
            raise click.ClickException("\n".join([str(e), *getattr(e, "__notes__", ())])) from e


def _verbose_logging() -> None:
    """Log agent-env's own DEBUG messages, and only warnings from other libraries: their debug logs can carry
    credentials, as hpack's do, which log every HTTP/2 header Modal's client sends, its token among them."""
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(name)s %(message)s")
    for name in ("agent_env", "agentenv_protocol"):
        logging.getLogger(name).setLevel(logging.DEBUG)


@click.group(cls=_UserErrorsAreNotCrashes)
@click.version_option(package_name="agentenv-framework", prog_name="agent-env", message="%(prog)s %(version)s")
@click.option("--verbose", "-v", is_flag=True,
              help="Enable agent-env's DEBUG logging (other libraries log warnings only)")
@click.pass_context
def cli(ctx: click.Context, verbose: bool):
    """Agent environment CLI."""
    if verbose:
        _verbose_logging()
    # For this command only, so a CLI invoked in-process leaves the caller's stores as they were.
    ctx.with_resource(namespace_routing())


cli.add_command(a2a_agent)
cli.add_command(env)
cli.add_command(artifact)
cli.add_command(config)
cli.add_command(eval)
cli.add_command(plugin)
cli.add_command(run)
cli.add_command(task)
cli.add_command(up)

load_cli_plugins(cli)
load_cli_root_options(cli)
