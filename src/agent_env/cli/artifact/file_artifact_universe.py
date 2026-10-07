"""CLI commands for FileArtifactUniverse — bundles FileArtifacts together."""

from pathlib import Path
from typing import Optional, Tuple

import click

from agent_env.artifact import FileArtifact, FileArtifactUniverse
from agent_env.cli.utils import deprecated_option, renamed_value
from agent_env.store.ids import fs_safe, is_local_id, key_segment


# ---------------------------------------------------------------------------
# FileArtifactUniverse
#   agent-env artifact file-artifact-universe put --id ds_001 \
#       --file-artifact raw_csv --file-artifact cleaned_csv
# ---------------------------------------------------------------------------


@click.group(name="file-artifact-universe")
def file_artifact_universe():
    """FileArtifactUniverse commands."""
    pass


@file_artifact_universe.command("put")
@click.option("--id", "universe_id", required=True, help="FileArtifactUniverse id")
@click.option(
    "--file-artifact",
    "file_artifact_ids",
    multiple=True,
    required=True,
    help="FileArtifact id to include (repeatable)",
)
def put(universe_id: str, file_artifact_ids: Tuple[str, ...]):
    """Create a FileArtifactUniverse bundling existing FileArtifacts."""

    click.echo(f"Resolving {len(file_artifact_ids)} FileArtifact(s)...")
    file_artifacts: dict[str, FileArtifact] = {}
    for fa_id in file_artifact_ids:
        fa = FileArtifact.get(fa_id)
        if fa.filename in file_artifacts:
            click.echo(
                f"Error: duplicate filename '{fa.filename}' — FileArtifacts "
                f"{file_artifacts[fa.filename].id} and {fa.id} both use it. "
                "Each FileArtifact in a universe must have a unique filename.",
                err=True,
            )
            raise SystemExit(1)
        file_artifacts[fa.filename] = fa
        click.echo(f"  {fa.filename} <- {fa.id} v{fa.version}")

    click.echo("Creating FileArtifactUniverse...")
    universe = FileArtifactUniverse.put(id=universe_id, file_artifacts=file_artifacts)
    click.echo(
        f"Created FileArtifactUniverse: id={universe.id} version={universe.version} "
        f"file_count={len(universe.file_artifact_ids)}"
    )


@file_artifact_universe.command("put-bundled")
@click.option("--id", "universe_id", required=True, help="FileArtifactUniverse id")
@click.option(
    "--file-dir",
    "file_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help="Local directory; every file under it is uploaded into the bundle (relative path preserved)",
)
@click.option(
    "--prefix-url",
    "prefix_url",
    default=None,
    help="Object-store prefix to upload the bundle under. Defaults to the version's own prefix in the configured "
    "object store (artifacts/file_artifact_universe/<id>/<version>/).",
)
@deprecated_option("--s3-url", "s3_url", "--prefix-url")
def put_bundled(universe_id: str, file_dir: Path, prefix_url: Optional[str], s3_url: Optional[str]):
    """Upload every file under --file-dir as a single bundled FileArtifactUniverse."""
    prefix_url = renamed_value("--prefix-url", prefix_url, "--s3-url", s3_url)
    files: dict[str, Path] = {}
    for p in sorted(file_dir.rglob("*")):
        if p.is_file():
            files[p.relative_to(file_dir).as_posix()] = p
    if not files:
        click.echo(f"Error: no files found under {file_dir}", err=True)
        raise SystemExit(1)

    click.echo(f"Uploading {len(files)} file(s)" + (f" under {prefix_url}..." if prefix_url else "..."))
    for rel in files:
        click.echo(f"  {rel}")
    universe = FileArtifactUniverse.put_bundled(id=universe_id, files=files, prefix_url=prefix_url)
    click.echo(
        f"Created FileArtifactUniverse: id={universe.id} version={universe.version} "
        f"file_count={len(universe.file_artifact_ids)} bundle_object_url={universe.bundle_object_url}"
    )


@file_artifact_universe.command("get")
@click.option("--id", "universe_id", required=True, help="FileArtifactUniverse id")
@click.option(
    "--version",
    "universe_version",
    type=int,
    default=None,
    help="Version (default: latest)",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Local directory to download files into",
)
def get(universe_id: str, universe_version: Optional[int], output_dir: str):
    """Download every file in a FileArtifactUniverse to a local directory."""

    click.echo(
        f"Fetching FileArtifactUniverse: id={universe_id} "
        f"version={universe_version or 'latest'}..."
    )
    universe = FileArtifactUniverse.get(universe_id, universe_version)
    click.echo(
        f"Found: id={universe.id} version={universe.version} "
        f"file_count={len(universe.file_artifact_ids)}"
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    file_artifacts = universe.get_file_artifacts()
    click.echo(f"Downloading {len(file_artifacts)} file(s)...")
    for filename, fa in file_artifacts.items():
        data = fa.load()
        # Preserve subdirectories if the filename contains '/'.
        dest = out / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        click.echo(f"  {filename} -> {dest} ({len(data)} bytes)")

    click.echo(f"Done. Files written to {out}")


@file_artifact_universe.command("list")
@click.option(
    "--batch-id",
    "batch_id",
    type=str,
    default=None,
    help=(
        "Only list universes produced by this task batch. Resolved by joining "
        "through task_instances — batch_id is a batch-runner concept and "
        "isn't stored on the universe itself."
    ),
)
@click.option(
    "--limit",
    type=int,
    default=50,
    help="Maximum number of results (default: 50)",
)
def list_universes(batch_id: Optional[str], limit: int):
    """List FileArtifactUniverses (most recent first)."""

    universes = _query_universes(batch_id=batch_id, limit=limit)

    if not universes:
        click.echo("No FileArtifactUniverses found.")
        return

    for u in universes:
        click.echo(f"  {u.id} v{u.version}  files={len(u.file_artifact_ids)}")


@file_artifact_universe.command("get-many")
@click.option(
    "--id",
    "universe_ids",
    multiple=True,
    help="FileArtifactUniverse id (repeatable). Mutually exclusive with --batch-id.",
)
@click.option(
    "--batch-id",
    "batch_id",
    type=str,
    default=None,
    help=(
        "Download every universe produced by this task batch. Resolved via "
        "task_instances (batch_id lives on the instance, not the universe)."
    ),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(),
    help="Local directory. Each universe's files go into a subdirectory named after its id.",
)
@click.option(
    "--concurrency",
    type=int,
    default=8,
    help="Max parallel file downloads (default: 8)",
)
def get_many(
    universe_ids: Tuple[str, ...],
    batch_id: Optional[str],
    output_dir: str,
    concurrency: int,
):
    """Mass-download many FileArtifactUniverses in parallel.

    Provide either --id (repeatable) or --batch-id. Files are pulled directly
    from S3 in parallel (no hop through the hub backend), so this scales to
    hundreds of universes.
    """
    import concurrent.futures

    if bool(universe_ids) == bool(batch_id):
        click.echo("Error: provide exactly one of --id (repeatable) or --batch-id.", err=True)
        raise SystemExit(1)

    if batch_id:
        click.echo(f"Resolving universes for batch_id={batch_id}...")
        universes = _query_universes(batch_id=batch_id, limit=10_000)
    else:
        click.echo(f"Resolving {len(universe_ids)} universe(s)...")
        universes = [FileArtifactUniverse.get(uid) for uid in universe_ids]

    if not universes:
        click.echo("No universes matched — nothing to download.")
        return

    click.echo(f"Preparing to download {len(universes)} universe(s) into {output_dir}")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Flatten into (universe_id, filename, FileArtifact) triples so we can
    # parallelize downloads across all files in all universes.
    jobs: list[tuple[str, str, FileArtifact]] = []
    for u in universes:
        for fname, fa_id in u.file_artifact_ids.items():
            jobs.append((u.id, fname, FileArtifact.get(fa_id)))
    click.echo(f"Total files to download: {len(jobs)}")

    def _download(job: tuple[str, str, FileArtifact]) -> tuple[str, str, int]:
        universe_id, filename, fa = job
        data = fa.load()
        dest = out / (fs_safe(universe_id) if is_local_id(universe_id) else key_segment(universe_id)) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return (universe_id, filename, len(data))

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        for (universe_id, filename, nbytes) in pool.map(_download, jobs):
            completed += 1
            click.echo(f"  [{completed}/{len(jobs)}] {universe_id}/{filename} ({nbytes} bytes)")

    click.echo(f"Done. {len(jobs)} file(s) across {len(universes)} universe(s) written to {out}")


def _query_universes(
    *,
    batch_id: Optional[str] = None,
    limit: int = 50,
) -> list[FileArtifactUniverse]:
    """Query file_artifact_universe artifacts, optionally filtered by batch_id.

    When batch_id is provided, resolution goes through the task instance
    store (see ``TaskInstanceStore.file_artifact_universe_ids_for_batch``),
    because batch_id is a batch-runner concept stored on the instance, not
    on the universe itself.
    """
    from agent_env.artifact.store import get_artifact_store
    from agent_env.task.store import get_task_instance_store

    if batch_id:
        ids = get_task_instance_store().file_artifact_universe_ids_for_batch(batch_id)
        if not ids:
            return []
        ids = ids[:limit]
    else:
        ids = None

    return get_artifact_store().latest_by_type("file_artifact_universe", ids=ids, limit=limit)
