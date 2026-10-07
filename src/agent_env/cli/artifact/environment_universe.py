from pathlib import Path
from typing import Optional, Tuple

import click

from agent_env.artifact import FileArtifact, EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.cli.utils import parse_artifact_ref


# ---------------------------------------------------------------------------
# EnvironmentUniverseArtifact
#   agent-env artifact environment-universe put --id scenario_001 --environment-artifact slack-data --environment-artifact email-data
# ---------------------------------------------------------------------------

@click.group(name="environment-universe")
def environment_universe():
    """EnvironmentUniverseArtifact commands."""
    pass


@environment_universe.command("put")
@click.option("--id", "artifact_id", required=True, help="EnvironmentUniverseArtifact id")
@click.option(
    "--environment-artifact",
    "environment_artifacts",
    multiple=True,
    required=True,
    help="EnvironmentArtifact id[:version] (repeatable). Omit :version to pin to current latest.",
)
@click.option(
    "--metadata",
    "metadata_entries",
    multiple=True,
    help="Metadata entry as key=/path/to/file (repeatable)",
)
def environment_universe_put(
    artifact_id: str,
    environment_artifacts: Tuple[str, ...],
    metadata_entries: Tuple[str, ...],
):
    """Create an EnvironmentUniverseArtifact bundling existing EnvironmentArtifacts."""

    resolved = []
    for ref in environment_artifacts:
        sa_id, sa_version = parse_artifact_ref(ref)
        click.echo(f"Fetching EnvironmentArtifact: id={sa_id} version={sa_version or 'latest'}...")
        sa = EnvironmentArtifact.get(sa_id, version=sa_version)
        click.echo(f"  Found: id={sa.id} version={sa.version}")
        resolved.append(sa)

    metadata = None
    if metadata_entries:
        keys = [entry.split("=", 1)[0] for entry in metadata_entries if "=" in entry]
        if len(keys) != len(set(keys)):
            click.echo("Error: duplicate keys in --metadata entries", err=True)
            raise SystemExit(1)
        metadata = {}
        for entry in metadata_entries:
            if "=" not in entry:
                click.echo(f"Error: --metadata value must be key=/path/to/file, got '{entry}'", err=True)
                raise SystemExit(1)
            key, file_path = entry.split("=", 1)
            p = Path(file_path)
            if not p.is_file():
                click.echo(f"Error: metadata file not found: {file_path}", err=True)
                raise SystemExit(1)
            fa_id = EnvironmentUniverseArtifact.derived_metadata_id(artifact_id, key)
            click.echo(f"Creating metadata FileArtifact: key={key} file={file_path}...")
            fa = FileArtifact.put(
                id=fa_id,
                description=f"Metadata '{key}' for EnvironmentUniverseArtifact '{artifact_id}'",
                file_path=str(p),
            )
            metadata[key] = fa
            click.echo(f"  Created FileArtifact: id={fa.id} version={fa.version} filename={fa.filename}")

    click.echo("Creating EnvironmentUniverseArtifact...")
    result = EnvironmentUniverseArtifact.put(
        id=artifact_id,
        environment_artifacts=resolved,
        metadata=metadata,
    )
    refs_summary = [str(r) for r in result.environment_artifact_refs]
    md_summary = (
        {k: str(r) for k, r in result.metadata_refs.items()}
        if result.metadata_refs
        else None
    )
    click.echo(
        f"Created EnvironmentUniverseArtifact: id={result.id} version={result.version} "
        f"environment_artifacts={refs_summary} metadata={md_summary}"
    )


@environment_universe.command("get")
@click.option("--id", "artifact_id", required=True, help="EnvironmentUniverseArtifact id")
@click.option("--version", "artifact_version", type=int, default=None, help="Version (default: latest)")
@click.option("--output-dir", required=True, type=click.Path(), help="Local directory to download files into")
def environment_universe_get(
    artifact_id: str,
    artifact_version: Optional[int],
    output_dir: str,
):
    """Download an EnvironmentUniverseArtifact's files to a local directory."""

    click.echo(f"Fetching EnvironmentUniverseArtifact: id={artifact_id} version={artifact_version or 'latest'}...")
    universe = EnvironmentUniverseArtifact.get(artifact_id, artifact_version)
    if universe.environment_artifact_refs:
        children = [str(r) for r in universe.environment_artifact_refs]
    else:
        children = universe.legacy_environment_artifact_ids or []
    if universe.metadata_refs:
        md = {k: str(r) for k, r in universe.metadata_refs.items()}
    else:
        md = universe.legacy_metadata
    click.echo(
        f"Found: id={universe.id} version={universe.version} "
        f"environment_artifacts={children} metadata={md}"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    environment_artifacts = universe.get_environment_artifacts()
    click.echo(f"Downloading {len(environment_artifacts)} environment artifact(s)...")
    for sa in environment_artifacts:
        file_artifact = sa.get_file_artifact()
        data = file_artifact.load()
        sa_dir = out / sa.environment_name
        sa_dir.mkdir(parents=True, exist_ok=True)
        dest = sa_dir / file_artifact.filename
        dest.write_bytes(data)
        click.echo(f"  {sa.environment_name} -> {dest}")

    metadata_artifacts = universe.get_metadata()
    if metadata_artifacts:
        click.echo(f"Downloading {len(metadata_artifacts)} metadata file(s)...")
        for key, file_artifact in metadata_artifacts.items():
            data = file_artifact.load()
            metadata_dir = out / EnvironmentUniverseArtifact.metadata_name / key
            metadata_dir.mkdir(parents=True, exist_ok=True)
            dest = metadata_dir / file_artifact.filename
            dest.write_bytes(data)
            click.echo(f"  {key} -> {dest}")

    click.echo(f"Done. Files written to {out}")


@environment_universe.command("compatible-envs")
@click.option("--id", "artifact_id", required=True, help="EnvironmentUniverseArtifact id")
def compatible_envs(artifact_id: str):
    """List env compatibility results for a universe."""
    from agent_env.env.env_artifact_store import get_env_artifact_store

    from agent_env.env.env_artifact_store import EnvArtifactType
    results = get_env_artifact_store().get_by_artifact(artifact_id, type=EnvArtifactType.UNIVERSE_COMPATIBILITY)
    if not results:
        click.echo(f"No compatibility results found for universe {artifact_id}")
        return
    for r in results:
        compat = r["data"].get("compatible", False)
        status = click.style("COMPATIBLE", fg="green") if compat else click.style("INCOMPATIBLE", fg="magenta")
        click.echo(f"  env={r['env_id']} v{r['env_version']}: {status}")
