import contextlib
from pathlib import Path
from typing import Optional

import click

from agent_env.artifact import SkillArtifact
from agent_env.artifact.artifacts.skill import _parse_skill_md, download_skill


# ---------------------------------------------------------------------------
# SkillArtifact
#   agent-env artifact skill put --id my-skill --skill-dir ./my-skill
#   agent-env artifact skill get --id my-skill --output-dir /tmp/my-skill
# ---------------------------------------------------------------------------

_SKILL_MD_FILENAME = "SKILL.md"
_BODY_PREVIEW_CHARS = 500

_BANNER = r"""
    _                    _     _____            ____  _    _ _ _
   / \   __ _  ___ _ __ | |_  | ____|_ ____   _/ ___|| | _(_) | |
  / _ \ / _` |/ _ \ '_ \| __| |  _| | '_ \ \ / /\___ \| |/ / | | |
 / ___ \ (_| |  __/ | | | |_  | |___| | | \ V /  ___) |   <| | | |
/_/   \_\__, |\___|_| |_|\__| |_____|_| |_|\_/  |____/|_|\_\_|_|_|
        |___/
"""


@click.group(name="skill")
def skill():
    """SkillArtifact commands."""
    pass


@skill.command("put")
@click.option("--id", "artifact_id", required=True, help="SkillArtifact id (must match frontmatter 'name')")
@click.option(
    "--skill-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Local skill directory containing SKILL.md at the root",
)
@click.option(
    "--s3-url",
    "s3_url",
    default=None,
    help="S3 prefix containing the skill (s3://bucket/prefix/) — alternative to --skill-dir",
)
def skill_put(artifact_id: str, skill_dir: Optional[Path], s3_url: Optional[str]):
    """Upload a skill as a SkillArtifact, from a local directory or an S3 prefix."""
    if (skill_dir is None) == (s3_url is None):
        click.echo("Error: specify exactly one of --skill-dir or --s3-url", err=True)
        raise SystemExit(1)

    source_cm = download_skill(s3_url) if s3_url else contextlib.nullcontext(skill_dir)
    with source_cm as source_dir:
        if s3_url:
            click.echo(f"Downloaded skill from {s3_url} to {source_dir}")

        file_count = sum(1 for p in source_dir.rglob("*") if p.is_file())
        click.echo(f"Uploading SkillArtifact bundle ({file_count} file(s))...")
        result = SkillArtifact.put(id=artifact_id, skill_dir=source_dir)
        click.echo(
            f"Created SkillArtifact: id={result.id} version={result.version} "
            f"spec_version={result.agent_skills_spec_version} "
            f"skill_files_id={result.skill_files_id} "
            f"skill_s3_url={result.skill_object_url}"
        )
        click.echo()
        click.secho(_BANNER, fg="magenta", bold=True)
        click.secho("Frontmatter:", fg="cyan", bold=True)
        _echo_field("name", result.skill_name)
        _echo_field("description", result.description)
        if result.license is not None:
            _echo_field("license", result.license)
        if result.compatibility is not None:
            _echo_field("compatibility", result.compatibility)
        if result.allowed_tools is not None:
            _echo_field("allowed-tools", result.allowed_tools)
        if result.skill_metadata:
            click.echo(f"  {click.style('metadata', fg='cyan')}:")
            for k, v in result.skill_metadata.items():
                click.echo(f"    {click.style(k, fg='cyan')}: {click.style(v, fg='green')}")

        _, body = _parse_skill_md((source_dir / _SKILL_MD_FILENAME).read_bytes())
        preview = body[:_BODY_PREVIEW_CHARS]
        truncated = len(body) > _BODY_PREVIEW_CHARS
        click.secho(f"Body (first {_BODY_PREVIEW_CHARS} chars):", fg="cyan", bold=True)
        click.echo(click.style(preview, fg="green") + ("…" if truncated else ""))


def _echo_field(key: str, value: str) -> None:
    click.echo(f"  {click.style(key, fg='cyan')}: {click.style(value, fg='green')}")


@skill.command("get")
@click.option("--id", "artifact_id", required=True, help="SkillArtifact id")
@click.option("--version", "artifact_version", type=int, default=None, help="Version (default: latest)")
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Local directory to download the skill into",
)
def skill_get(artifact_id: str, artifact_version: Optional[int], output_dir: Path):
    """Download a SkillArtifact's files to a local directory."""

    click.echo(f"Fetching SkillArtifact: id={artifact_id} version={artifact_version or 'latest'}...")
    skill_artifact = SkillArtifact.get(artifact_id, artifact_version)
    click.echo(
        f"Found: id={skill_artifact.id} version={skill_artifact.version} "
        f"spec_version={skill_artifact.agent_skills_spec_version} "
        f"name={skill_artifact.skill_name}"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    skill_files = skill_artifact.get_skill_files()
    file_artifacts = skill_files.get_file_artifacts()
    click.echo(f"Downloading {len(file_artifacts)} file(s)...")
    for rel_path, fa in file_artifacts.items():
        dest = output_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fa.load())
        click.echo(f"  {rel_path} -> {dest}")

    click.echo(f"Done. Files written to {output_dir}")
