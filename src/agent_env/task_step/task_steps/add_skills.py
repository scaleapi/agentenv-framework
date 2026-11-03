"""Add skills to a deployed A2A agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import ClassVar, Optional

from agentenv_protocol import SKILL_EXTENSION_URI
from agentenv_protocol.client import find_extension_method

from agent_env.config import get_config
from agent_env.task_step.context import DeployedAgent, TaskStepContext
from agent_env.entity.refs import EntityRef
from agent_env.entity.scopes import FieldScope
from agent_env.task_step.task_step import TaskStep, TaskStepDependency

logger = logging.getLogger(__name__)

#: ``context.metadata[ENV_SKILLS_KEY][agent_name][env_id][skill_name]``: the SKILL.md add_skills installed from an env, as
#: the env, or one of its child envs, wrote it for that deploy.
ENV_SKILLS_KEY = "env_skills"
#: Where an env's skill with files beside its SKILL.md is stored, under the artifact key prefix, by a digest of its
#: files, to be sent to the agent as a skill bundle.
_ENV_SKILL_PREFIX = "agentenv-env-skills"


@dataclass
class Skill:
    """Represents an Agent Skill following the Agent Skills specification.

    See https://agentskills.io/specification for the full spec.

    A skill can come from three mutually-exclusive sources, and names none when an import left its object_url out:
    - **inline**: body + frontmatter fields, rendered into SKILL.md at send time
    - **object_url**: object-store prefix containing a skill directory with its own SKILL.md,
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
    skill_artifact_id: Optional[str] = None
    skill_artifact_version: Optional[int] = None
    object_url: Optional[str] = None

    def __post_init__(self):
        modes = [bool(self.body), bool(self.object_url), bool(self.skill_artifact_id)]
        if sum(modes) > 1:
            raise ValueError("Skill must have at most one of: body (inline), object_url, skill_artifact_id")
        if self.skill_artifact_id is not None:
            if any([self.body, self.object_url, self.license, self.compatibility, self.metadata, self.allowed_tools]):
                raise ValueError(
                    "Skill with skill_artifact_id must not have body/object_url/license/compatibility/metadata/allowed_tools"
                )
        else:
            if not self.name or not self.description:
                raise ValueError("Inline or object_url skills require name and description")
            if self.object_url and any([self.body, self.license, self.compatibility, self.metadata, self.allowed_tools]):
                raise ValueError(
                    "Skill with object_url must not have body/license/compatibility/metadata/allowed_tools — "
                    "the skill directory has its own SKILL.md"
                )

    def validate(self) -> None:
        from agent_env.artifact import SkillArtifact

        if self.skill_artifact_id is not None:
            return
        if self.object_url is not None:
            SkillArtifact.validate(object_url=self.object_url, expected_name=self.name)
            return
        if not self.body:
            raise ValueError(self.no_source())
        SkillArtifact.validate(skill_md=self.to_skill_md().encode("utf-8"), expected_name=self.name)

    def names_a_source(self) -> bool:
        return bool(self.body or self.object_url or self.skill_artifact_id)

    def no_source(self) -> str:
        return (f"skill {self.name!r} names no source; set body, object_url or skill_artifact_id (object_url is user "
                "scope, so an import leaves it out)")

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
        """Both spellings of ``object_url``: a worker on an older version reads only ``s3_url``."""
        d: dict = {}
        for field_name in (
            "name", "description", "body", "license", "compatibility",
            "metadata", "allowed_tools", "s3_url", "object_url",
            "skill_artifact_id", "skill_artifact_version",
        ):
            value = self.object_url if field_name == "s3_url" else getattr(self, field_name)
            if value is not None:
                d[field_name] = value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> Skill:
        return cls(
            name=data.get("name"), description=data.get("description"),
            body=data.get("body"), license=data.get("license"),
            compatibility=data.get("compatibility"), metadata=data.get("metadata"),
            allowed_tools=data.get("allowed_tools"), object_url=data.get("s3_url") or data.get("object_url"),
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


def _env_skill_name(environment_name: str, env_id: str) -> str:
    """The env's environment name, as the agent's MCP alias for it is; for a name that isn't a skill name as it stands,
    a slug of it and a hash of the env's id, so two envs' names never come out the same."""
    if re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", environment_name) and len(environment_name) <= 64:
        return environment_name
    slug = re.sub(r"[^a-z0-9]+", "-", environment_name.lower()).strip("-")[:51].rstrip("-")
    digest = hashlib.sha256(env_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug}-{digest}" if slug else digest


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
        EntityRef.env("env_ids[]"),
    )
    field_scopes = {**TaskStep.field_scopes, "skills[].object_url": FieldScope.USER, "skills[].s3_url": FieldScope.USER}

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
        env_ids: Optional[list[str]] = None,
    ):
        super().__init__(id, version, depends_on=depends_on, fail_task_on_error=fail_task_on_error)
        self.skills = [s if isinstance(s, Skill) else Skill.from_dict(s) for s in (skills or [])]
        self.cli_artifact_ids = list(cli_artifact_ids or [])
        self.file_artifact_universe_ids = list(file_artifact_universe_ids or [])
        self.env_ids = list(env_ids or [])
        self.agent_name = agent_name or TaskStep.DEFAULT_AGENT_NAME

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["skills"] = [s.to_dict() for s in self.skills]
        base["cli_artifact_ids"] = self.cli_artifact_ids
        base["file_artifact_universe_ids"] = self.file_artifact_universe_ids
        if self.env_ids:  # left out otherwise, so a step without one serializes as before
            base["env_ids"] = self.env_ids
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
            env_ids=data.get("env_ids"),
        )

    def config_problems(self) -> list[str]:
        """Each skill naming no source."""
        return [f"add_skills '{self.id}': {skill.no_source()}" for skill in self.skills if not skill.names_a_source()]

    async def execute(self, context: TaskStepContext) -> TaskStepContext:
        from agent_env.a2a_agent import A2AAgent
        from agent_env.a2a_agent.store import get_a2a_agent_instance_store

        if problems := self.config_problems():
            raise ValueError("; ".join(problems))
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

        env_skills = {env_id: await _env_skills(context, agent, env_id) for env_id in self.env_ids}

        deployed = get_a2a_agent_instance_store().get(agent.instance_id)
        for skill in skills_to_add:
            result = await A2AAgent.add_skill(deployed, skill)
            logger.info(f"Registered skill: {result}")
        for env_id, skills in env_skills.items():
            for skill in skills:
                files = skill.get("files")
                sent = ({"object_url": await asyncio.to_thread(_stored_skill, skill["skill_md"], files)} if files else
                        {"skill_md": skill["skill_md"]})
                result = await A2AAgent.register_skill(deployed, name=skill["name"], description=skill["description"],
                                                       **sent)
                installed = context.metadata.setdefault(ENV_SKILLS_KEY, {}).setdefault(self.agent_name, {})
                installed.setdefault(env_id, {})[skill["name"]] = skill["skill_md"]
                logger.info(f"Registered the skill {skill['name']} of env {env_id}: {result}")

        return context


def _stored_skill(skill_md: str, files: dict[str, str]) -> str:
    """The object-store prefix holding an env skill's SKILL.md and its other files, one prefix per content; a file
    whose path leaves the skill's folder is refused."""
    if outside := [path for path in files if not path or PurePosixPath(path).is_absolute()
                   or ".." in PurePosixPath(path).parts]:
        raise ValueError(f"an env skill's files must stay inside its folder, not {outside}")
    config = get_config()
    store, contents = config.get_object_store(), {"SKILL.md": skill_md, **files}
    digest = hashlib.sha256(json.dumps(contents, sort_keys=True).encode()).hexdigest()
    prefix = f"{config.get_artifact_key_prefix()}{_ENV_SKILL_PREFIX}/{digest}"
    for path, text in contents.items():
        store.put(f"{prefix}/{path}", text.encode(), content_type="text/markdown", allow_overwrite=True)
    return store.object_url(prefix)


async def _env_skills(context: TaskStepContext, agent: DeployedAgent, env_id: str) -> list[dict]:
    """The skills the env ``env_id`` serves, each under its card's environment name: its own card's and each of its
    child envs' (a gateway's servers, a computer's apps) its name reaches; refused for an env this run didn't deploy or
    bind to ``agent``, or that serves none."""
    deployed = next((d for d in context.deployed_envs if d.env_id == env_id), None)
    if deployed is None:
        raise RuntimeError(f"env '{env_id}' isn't deployed in this run")
    if env_id not in agent.env_ids:
        raise RuntimeError(f"env '{env_id}' isn't bound to agent '{agent.agent_name}': name it in the env_ids of its "
                           "deploy_agent")
    card = deployed.environment_card or {}
    serving = [(None, _env_skill_name(card.get("name") or env_id, env_id))] if find_extension_method(
        card, SKILL_EXTENSION_URI, "get") is not None else []
    serving += [(child["name"], _env_skill_name(child["name"], f"{env_id}/{child['name']}"))
                for child in card.get("children_environments") or []
                if find_extension_method(child, SKILL_EXTENSION_URI, "get") is not None
                and deployed.get_child_env_card(child["name"]) is child]
    if not serving:
        raise RuntimeError(f"env '{env_id}' serves no skill: neither its card nor a child env's declares "
                           f"{SKILL_EXTENSION_URI}")
    return [await deployed.invoke(SKILL_EXTENSION_URI, "get", {"name": name}, environment_name=child)
            for child, name in serving]
