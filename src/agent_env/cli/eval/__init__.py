import click

from .add_tasks import add_tasks
from .create import create
from .run import run


@click.group()
def eval():
    """Eval commands."""
    pass


eval.add_command(create)
eval.add_command(add_tasks)
eval.add_command(run)
