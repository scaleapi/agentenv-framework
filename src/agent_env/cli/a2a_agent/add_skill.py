import asyncio
from pathlib import Path
from typing import Optional

import click

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.skill import fetch_skill_md, parse_skill_md
from agent_env.a2a_agent.store import get_a2a_agent_instance_store
from agent_env.store.base import NotFoundError
from agent_env.cli.utils import deprecated_option, renamed_value
from agent_env.task_step.task_steps.add_skills import Skill


# ---------------------------------------------------------------------------
#   agent-env a2a-agent add-skill --instance-id <id> --skill-artifact-id code-review
#   agent-env a2a-agent add-skill --instance-id <id> --skill-md-path ./SKILL.md
#   agent-env a2a-agent add-skill --instance-id <id> --skill-object-url s3://bucket/prefix/
# ---------------------------------------------------------------------------


@click.command("add-skill")
@click.option("--instance-id", "instance_id", required=True, help="Deployed A2A agent instance ID")
@click.option(
    "--skill-artifact-id",
    "skill_artifact_id",
    default=None,
    help="SkillArtifact id (loads the stored skill and registers it)",
)
@click.option(
    "--skill-artifact-version",
    "skill_artifact_version",
    type=int,
    default=None,
    help="SkillArtifact version (default: latest)",
)
@click.option(
    "--skill-md-path",
    "skill_md_path",
    type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=Path),
    default=None,
    help="Local SKILL.md file to register inline",
)
@click.option(
    "--skill-object-url",
    "skill_object_url",
    default=None,
    help="Object-store prefix containing a skill directory",
)
@deprecated_option("--skill-s3-url", "skill_s3_url", "--skill-object-url")
def add_skill(
    instance_id: str,
    skill_artifact_id: Optional[str],
    skill_artifact_version: Optional[int],
    skill_md_path: Optional[Path],
    skill_object_url: Optional[str],
    skill_s3_url: Optional[str],
):
    """Register a skill against a deployed A2A agent's /ext/skill-config endpoint."""
    skill_object_url = renamed_value("--skill-object-url", skill_object_url, "--skill-s3-url", skill_s3_url)
    sources = [bool(skill_artifact_id), bool(skill_md_path), bool(skill_object_url)]
    if sum(sources) != 1:
        click.echo(
            "Error: specify exactly one of --skill-artifact-id, --skill-md-path, --skill-object-url",
            err=True,
        )
        raise SystemExit(1)

    try:
        deployed = get_a2a_agent_instance_store().get(instance_id)
    except NotFoundError:
        click.echo(f"Error: instance '{instance_id}' not found", err=True)
        raise SystemExit(1)

    click.echo(f"Deployed agent: {deployed.agent_id} v{deployed.agent_version} @ {deployed.a2a_url}")

    if skill_artifact_id is not None:
        skill = Skill(skill_artifact_id=skill_artifact_id, skill_artifact_version=skill_artifact_version)
        click.echo(f"Registering SkillArtifact '{skill_artifact_id}' v{skill_artifact_version or 'latest'}...")
    elif skill_object_url is not None:
        frontmatter, _ = parse_skill_md(fetch_skill_md(skill_object_url))
        skill = Skill(name=frontmatter["name"], description=frontmatter["description"], object_url=skill_object_url)
        click.echo(f"Registering skill from {skill_object_url} (name={skill.name})...")
    else:
        frontmatter, body = parse_skill_md(skill_md_path.read_bytes())
        skill = Skill(
            name=frontmatter["name"],
            description=frontmatter["description"],
            body=body,
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            metadata=frontmatter.get("metadata"),
            allowed_tools=frontmatter.get("allowed-tools"),
        )
        click.echo(f"Registering inline skill from {skill_md_path} (name={skill.name})...")

    result = asyncio.run(A2AAgent.add_skill(deployed, skill))
    click.echo(f"Registered: {result}")
