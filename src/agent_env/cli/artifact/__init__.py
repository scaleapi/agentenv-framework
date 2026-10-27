import click

from .cli import cli
from .file_artifact_universe import file_artifact_universe
from .environment import environment
from .environment_universe import environment_universe
from .skill import skill


@click.group()
def artifact():
    """Artifact commands."""
    pass


artifact.add_command(cli)
artifact.add_command(file_artifact_universe)
artifact.add_command(environment)
artifact.add_command(environment_universe)
artifact.add_command(skill)
