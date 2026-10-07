from pathlib import Path

import click

from agent_env.artifact import FileArtifact, EnvironmentArtifact
from agent_env.cli.utils import environment_name_options, resolve_environment_name


@click.group(name="environment")
def environment():
    """EnvironmentArtifact commands."""
    pass


@environment.command()
@click.argument("filepath", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--id", "artifact_id", required=True, help="EnvironmentArtifact id")
@click.option("--description", required=True, help="Artifact description")
@environment_name_options
def put(artifact_id: str, description: str, environment_name: str | None, filepath: Path):
    """Create an EnvironmentArtifact (and underlying FileArtifact) from a local file."""
    environment_name = resolve_environment_name(environment_name)

    click.echo("Creating FileArtifact...")
    file_artifact = FileArtifact.put(
        id=EnvironmentArtifact.derived_file_id(artifact_id),
        description=description,
        file_path=str(filepath),
    )
    click.echo(
        f"Created FileArtifact: id={file_artifact.id} version={file_artifact.version} filename={file_artifact.filename}"
    )

    click.echo("Creating EnvironmentArtifact...")
    environment_artifact = EnvironmentArtifact.put(
        id=artifact_id,
        environment_name=environment_name,
        file_artifact=file_artifact,
    )
    click.echo(
        f"Created EnvironmentArtifact: id={environment_artifact.id} version={environment_artifact.version} "
        f"environment_name={environment_artifact.environment_name}"
    )
    click.echo(f"  pinned FileArtifact: {environment_artifact.file_artifact_ref}")
