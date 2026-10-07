from pathlib import Path
from typing import Optional

import click

from agent_env.artifact import CliArtifact


@click.group(name="cli")
def cli():
    """CliArtifact commands."""
    pass


@cli.command("put")
@click.option("--id", "artifact_id", required=True, help="CliArtifact id")
@click.option(
    "--cli-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Local directory containing the CLI tree",
)
@click.option("--entrypoint", required=True, help="Relative path within --cli-dir to the executable file (e.g. 'bin/slack')")
@click.option("--command-name", required=True, help="Command name installed on PATH (e.g. 'slack' -> /usr/local/bin/slack)")
@click.option("--env-id", default=None, help="Optional source env id this CLI was generated from")
@click.option("--env-version", type=int, default=None, help="Optional source env version this CLI was generated from")
def cli_put(
    artifact_id: str,
    cli_dir: Path,
    entrypoint: str,
    command_name: str,
    env_id: Optional[str],
    env_version: Optional[int],
):
    """Upload a CLI directory as a CliArtifact."""

    file_count = sum(1 for p in cli_dir.rglob("*") if p.is_file())
    click.echo(f"Uploading CliArtifact bundle ({file_count} file(s)) from {cli_dir}...")
    result = CliArtifact.put(
        id=artifact_id,
        command_name=command_name,
        entrypoint=entrypoint,
        cli_dir=cli_dir,
        env_id=env_id,
        env_version=env_version,
    )
    click.echo(
        f"Created CliArtifact: id={result.id} version={result.version} "
        f"command_name={result.command_name} entrypoint={result.entrypoint} "
        f"cli_files_id={result.cli_files_id} cli_object_url={result.cli_object_url}"
    )


@cli.command("get")
@click.option("--id", "artifact_id", required=True, help="CliArtifact id")
@click.option("--version", "artifact_version", type=int, default=None, help="Version (default: latest)")
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Local directory to download the CLI into",
)
def cli_get(artifact_id: str, artifact_version: Optional[int], output_dir: Path):
    """Download a CliArtifact's files to a local directory."""

    click.echo(f"Fetching CliArtifact: id={artifact_id} version={artifact_version or 'latest'}...")
    cli_artifact = CliArtifact.get(artifact_id, artifact_version)
    click.echo(
        f"Found: id={cli_artifact.id} version={cli_artifact.version} "
        f"command_name={cli_artifact.command_name} entrypoint={cli_artifact.entrypoint}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    file_artifacts = cli_artifact.get_cli_files().get_file_artifacts()
    click.echo(f"Downloading {len(file_artifacts)} file(s)...")
    for rel_path, fa in file_artifacts.items():
        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fa.load())
        click.echo(f"  {rel_path} -> {dest}")

    click.echo(f"Done. Files written to {output_dir}")
    click.echo(f"Entrypoint: {output_dir / cli_artifact.entrypoint}")
