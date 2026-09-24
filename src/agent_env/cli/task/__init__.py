import click

from .create import create
from .get import get
from .get_instance import get_instance
from .run import run, run_batch
from .validate import validate


@click.group()
def task():
    """Task commands."""
    pass


task.add_command(create)
task.add_command(get)
task.add_command(get_instance)
task.add_command(run)
task.add_command(run_batch)
task.add_command(validate)
