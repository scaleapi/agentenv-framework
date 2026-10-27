import json
import tomllib
from pathlib import Path
from typing import Optional

import click

from agent_env.artifact import ComputerArtifact


@click.group(name="computer")
def computer():
    """ComputerArtifact commands."""


@computer.command("put")
@click.option("--id", "artifact_id", required=True, help="ComputerArtifact id")
@click.argument("folder", type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path))
@click.option("--no-inspect", is_flag=True,
              help="Write it without reading its images from their registries; each form is then uninspected")
def computer_put(artifact_id: str, folder: Path, no_inspect: bool):
    """Write the computer FOLDER's artifact.toml declares, with its form's image, after reading that image from its
    registry to check it's what the computer says."""
    try:
        result = ComputerArtifact.put_folder(artifact_id, folder, inspect=not no_inspect)
    except (ValueError, OSError, tomllib.TOMLDecodeError) as e:
        raise click.ClickException(str(e)) from None
    click.echo(f"Created ComputerArtifact: id={result.id} version={result.version}")
    for form, by_arch in result.metadata["validated_images"].items():
        for arch, found in by_arch.items():
            why = found.get("reason") or found.get("problem")
            click.echo(f"  {form} {arch}: {found['status']}" + (f" ({why})" if why else ""))


@computer.command("get")
@click.option("--id", "artifact_id", required=True, help="ComputerArtifact id")
@click.option("--version", type=int, default=None, help="The version (defaults to the latest)")
def computer_get(artifact_id: str, version: Optional[int]):
    """Print a computer artifact as JSON, with what put read from its images."""
    click.echo(json.dumps(ComputerArtifact.get(artifact_id, version).model_dump(mode="json"), indent=2))
