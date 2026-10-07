"""Add skills to a deployed A2A agent."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from typing import ClassVar, Optional

from agent_env.task_step.context import TaskStepContext
from agent_env.entity_refs import EntityRef
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """Represents an Agent Skill following the Agent Skills specification.

    See https://agentskills.io/specification for the full spec.

    A skill can come from three mutually-exclusive sources:
    - **inline**: body + frontmatter fields, rendered into SKILL.md at send time
    - **s3_url**: object-store prefix containing a skill directory with its own SKILL.md,
      sent as a bundle of read grants
    - **skill_artifact_id**: a SkillArtifact in the store; name/description/skill_object_url
      are resolved from it at send time
    """
    name: Optional[str] = None
    description: Optional[str] = None
    body: Optional[str] = None
    license: Optional[str] = None
    compatibility: Optional[str] = None
    metadata: Optional[dict[str, str]] = None
    allowed_tools: Optional[str] = None
    s3_url: Optional[str] = None
    skill_artifact_id: Optional[str] = None
    skill_artifact_version: Optional[int] = None

    def __post_init__(self):
        modes = [bool(self.body), bool(self.s3_url), bool(self.skill_artifact_id)]
        if sum(modes) != 1:
            raise ValueError("Skill must have exactly one of: body (inline), s3_url, skill_artifact_id")
        if self.skill_artifact_id is not None:
            if any([self.body, self.s3_url, self.license, self.compatibility, self.metadata, self.allowed_tools]):
                raise ValueError(
                    "Skill with skill_artifact_id must not have body/s3_url/license/compatibility/metadata/allowed_tools"
                )
        else:
            if not self.name or not self.description:
                raise ValueError("Inline or s3_url skills require name and description")
            if self.s3_url and any([self.body, self.license, self.compatibility, self.metadata, self.allowed_tools]):
                raise ValueError(
                    "Skill with s3_url must not have body/license/compatibility/metadata/allowed_tools — "
                    "the S3 skill directory has its own SKILL.md"
                )

    def validate(self) -> None:
        from agent_env.artifact import SkillArtifact

        if self.skill_artifact_id is not None:
            return
        if self.s3_url is not None:
            SkillArtifact.validate(object_url=self.s3_url, expected_name=self.name)
            return
        SkillArtifact.validate(skill_md=self.to_skill_md().encode("utf-8"), expected_name=self.name)

    def to_skill_md(self) -> str:
        """Render the full SKILL.md content with YAML frontmatter + body."""
        lines = ["---", f"name: {self.name}", f"description: {self.description}"]
        if self.license:
            lines.append(f"license: {self.license}")
        if self.compatibility:
            lines.append(f"compatibility: {self.compatibility}")
        if self.metadata:
            lines.append("metadata:")
            for k, v in self.metadata.items():
                lines.append(f'  {k}: "{v}"')
        if self.allowed_tools:
            lines.append(f"allowed-tools: {self.allowed_tools}")
        lines.append("---")
        if self.body:
            lines.append("")
            lines.append(self.body)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d: dict = {}
        for field_name in (
            "name", "description", "body", "license", "compatibility",
            "metadata", "allowed_tools", "s3_url",
            "skill_artifact_id", "skill_artifact_version",
        ):
            value = getattr(self, field_name)
            if value is not None:
                d[field_name] = value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        return cls(
            name=data.get("name"), description=data.get("description"),
            body=data.get("body"), license=data.get("license"),
            compatibility=data.get("compatibility"), metadata=data.get("metadata"),
            allowed_tools=data.get("allowed_tools"), s3_url=data.get("s3_url"),
            skill_artifact_id=data.get("skill_artifact_id"),
            skill_artifact_version=data.get("skill_artifact_version"),
        )



def _sanitize_skill_name(s: str) -> str:
    """Coerce arbitrary text into the Agent Skills name regex
    (^(?!-)(?!.*--)[a-z0-9-]{1,64}(?<!-)$). Empty inputs raise."""
    s = s.lower()
    s = re.sub(r"[^a-z0-9-]+", "-", s)   # replace runs of disallowed chars with a hyphen
    s = re.sub(r"-+", "-", s)            # collapse consecutive hyphens
    s = s.strip("-")                     # drop leading/trailing hyphens
    if not s:
        raise ValueError(
            "Cannot derive a valid Agent Skills name from input: result was empty after sanitization"
        )
    return s[:60].rstrip("-")


def _build_skill_for_installed_cli(cli_artifact_id: str, entry: dict) -> Skill:
    command_name = entry["command_name"]
    install_path = entry["install_path"]
    skill_name = f"{_sanitize_skill_name(command_name)}-cli"
    return Skill(
        name=skill_name,
        description=(
            f"Use when working with the {command_name} environment. "
            f"Run with --help to discover available subcommands at runtime."
        ),
        body=(
            f"# {command_name} CLI\n\n"
            f"A command-line interface for the `{command_name}` MCP environment.\n\n"
            f"## Location\n\nThe binary is installed at `{install_path}`.\n\n"
            f"## Discovery\n\n"
            f"Run `{install_path} --help` to see all currently-available subcommands. "
            f"Run `{install_path} <subcommand> --help` for the flags of a specific subcommand.\n\n"
            f"## Output\n\nSubcommands print pretty-printed JSON on success. Errors go to stderr with non-zero exit code.\n"
        ),
    )


def _files_skill_name(universe_id: str) -> str:
    """``<slug>-<hash>-files``: a slug of the id's last segment and a hash of the whole id, since an id itself
    needn't be a valid skill name."""
    slug = re.sub(r"[^a-z0-9]+", "-", universe_id.rsplit("/", 1)[-1].lower()).strip("-")[:45].rstrip("-")
    digest = hashlib.sha256(universe_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}-files" if slug else f"{digest}-files"


def _build_skill_for_loaded_file_artifact_universe(universe_id: str, entry: dict) -> Skill:
    destination_path = entry["destination_path"]
    return Skill(
        name=_files_skill_name(universe_id),
        description=f"Files possibly relevant to the current task are available at {destination_path}.",
        body=f"Files possibly relevant to the current task are available at `{destination_path}`.\n",
    )


class AddSkillsTaskStep(TaskStep):
    type: ClassVar[str] = "add_skills"
    entity_refs = (
        EntityRef.artifact("skills[].skill_artifact_id", version_field="skill_artifact_version", artifact_type="skill"),
        EntityRef.artifact("cli_artifact_ids[]", artifact_type="cli"),
        EntityRef.artifact("file_artifact_universe_ids[]"),
    )

    def __init__(
        self,
        id: str,
        version: Optional[int],
        skills: Optional[list[Skill | dict]] = None,
        cli_artifact_ids: Optional[list[str]] = None,
        file_artifact_universe_ids: Optional[list[str]] = None,
        agent_name: Optional[str] = None,
        depends_on: Optional[list[TaskStepDependency]] = None,
        fail_task_on_error: bool = True,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.skills = [s if isinstance(s, Skill) else Skill.from_dict(s) for s in (skills or [])]
        self.cli_artifact_ids = list(cli_artifact_ids or [])
        self.file_artifact_universe_ids = list(file_artifact_universe_ids or [])
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["skills"] = [s.to_dict() for s in self.skills]
        base["cli_artifact_ids"] = self.cli_artifact_ids
        base["file_artifact_universe_ids"] = self.file_artifact_universe_ids
        base["agent_name"] = self.agent_name
        return base

    @classmethod
    def from_dict(cls, data: dict) -> AddSkillsTaskStep:
        return cls(
            **cls._base_from_dict(data),
            skills=data.get("skills"),
            cli_artifact_ids=data.get("cli_artifact_ids"),
            file_artifact_universe_ids=data.get("file_artifact_universe_ids"),
            agent_name=data.get("agent_name"),
        )

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.a2a_agent.store import get_a2a_agent_instance_store

        agent = next((a for a in context.deployed_agents if a.agent_name == self.agent_name), None)
        if agent is None:
            raise RuntimeError(f"Agent '{self.agent_name}' not found in context.deployed_agents")
        if agent.instance_id is None:
            raise RuntimeError(f"Agent '{self.agent_name}' is missing instance_id; cannot resolve deployment")

        skills_to_add = list(self.skills)
        agent_clis = (context.metadata.get("installed_clis") or {}).get(self.agent_name) or {}
        for cli_artifact_id in self.cli_artifact_ids:
            entry = agent_clis.get(cli_artifact_id)
            if entry is None:
                raise RuntimeError(
                    f"CliArtifact '{cli_artifact_id}' not in context.installed_clis['{self.agent_name}']; "
                    f"ensure LoadArtifactTaskStep ran first with agent_name='{self.agent_name}'"
                )
            skills_to_add.append(_build_skill_for_installed_cli(cli_artifact_id, entry))

        loaded_universes_by_id: dict[str, dict] = {
            e["id"]: e
            for e in (context.metadata.get("loaded_file_artifact_universes") or [])
            if e.get("agent_name") == self.agent_name
        }
        for fau_id in self.file_artifact_universe_ids:
            entry = loaded_universes_by_id.get(fau_id)
            if entry is None:
                raise RuntimeError(
                    f"FileArtifactUniverse '{fau_id}' was not loaded into agent '{self.agent_name}'; "
                    f"ensure a LoadArtifactTaskStep with agent_name='{self.agent_name}' for this FAU ran first"
                )
            skills_to_add.append(_build_skill_for_loaded_file_artifact_universe(fau_id, entry))

        deployed = get_a2a_agent_instance_store().get(agent.instance_id)
        for skill in skills_to_add:
            result = await A2AAgent.add_skill(deployed, skill)
            logger.info(f"Registered skill: {result}")

        return context
