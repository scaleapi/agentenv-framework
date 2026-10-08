"""SkillArtifact — Agent Skills (agentskills.io) as first-class artifacts.

A SkillArtifact references one FileArtifactUniverse bundling every file in the skill
(SKILL.md + scripts/, references/, assets/, …) colocated under a single object-store prefix
(`artifacts/skill/<key_segment(id)>/<version>/`). That prefix IS the bundle — no duplication, and
deployed A2A agents get its files through read grants.
"""

from __future__ import annotations

import contextlib
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator, Literal, Optional

import yaml
from pydantic import ConfigDict, Field, model_serializer

from agent_env.artifact.artifact import Artifact, _write_twin
from agent_env.config import get_config
from agent_env.store.base import ObjectNotFoundError
from agent_env.store.ids import derive_id, key_segment

if TYPE_CHECKING:
    from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse


AGENT_SKILLS_SPEC_VERSION = "0.1"

_SKILL_NAME_RE = re.compile(r"^(?!-)(?!.*--)[a-z0-9-]{1,64}(?<!-)$")
_SKILL_MD_FILENAME = "SKILL.md"
_MAX_DESCRIPTION_LEN = 1024
_MAX_COMPATIBILITY_LEN = 500


class SkillArtifact(Artifact):
    model_config = ConfigDict(populate_by_name=True)

    type: Literal["skill"] = "skill"

    skill_files_id: str = Field(description="ID of the FileArtifactUniverse bundling every file in the skill")
    skill_object_url: str = Field(alias="skill_s3_url", description="Object-store prefix where the skill bundle lives")
    agent_skills_spec_version: str = Field(description="agentskills.io spec revision this skill was validated under")

    skill_name: str = Field(description="Skill name from SKILL.md frontmatter; equals the artifact id")
    description: str = Field(description="Skill description from SKILL.md frontmatter")
    license: Optional[str] = Field(default=None)
    compatibility: Optional[str] = Field(default=None)
    allowed_tools: Optional[str] = Field(default=None)
    skill_metadata: Optional[dict[str, str]] = Field(default=None)

    # No return annotation: pydantic builds the serialization schema from one, and a dict drops the fields.
    @model_serializer(mode="wrap")
    def _serialize(self, handler: Any):
        # Dual-write, from the attribute: `alias=` emits one spelling, which one depending on the caller's `by_alias`.
        data = handler(self)
        _write_twin(data, "skill_s3_url", "skill_object_url", self.skill_object_url)
        return data

    @classmethod
    def validate(
        cls,
        *,
        skill_md: Optional[bytes] = None,
        object_url: Optional[str] = None,
        expected_name: str,
    ) -> None:
        if (skill_md is None) == (object_url is None):
            raise ValueError("specify exactly one of skill_md or object_url")
        if object_url is not None:
            skill_md = fetch_skill_md(object_url)
        frontmatter, _body = parse_skill_md(skill_md)
        _validate_frontmatter(frontmatter, expected_name=expected_name)

    @classmethod
    def put(cls, id: str, *, skill_dir: Path) -> "SkillArtifact":
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
        from agent_env.artifact.store import get_artifact_store

        skill_md_path = skill_dir / _SKILL_MD_FILENAME
        if not skill_md_path.is_file():
            raise ValueError(f"{_SKILL_MD_FILENAME} not found at {skill_md_path}")

        skill_md_bytes = skill_md_path.read_bytes()
        cls.validate(skill_md=skill_md_bytes, expected_name=id)
        frontmatter, _body = parse_skill_md(skill_md_bytes)

        files: dict[str, Path] = {}
        for p in sorted(skill_dir.rglob("*")):
            if p.is_file():
                files[p.relative_to(skill_dir).as_posix()] = p

        store = get_artifact_store()
        version = store.next_version(id)
        key = f"{get_config().get_artifact_key_prefix()}artifacts/skill/{key_segment(id)}/{version}/"
        skill_object_url = get_config().get_object_store_for(id).object_url(key)

        universe = FileArtifactUniverse.put_bundled(
            id=derive_id(id, "files"),
            files=files,
            prefix_url=skill_object_url,
        )

        instance = cls(
            id=id,
            version=version,
            skill_files_id=universe.id,
            skill_object_url=skill_object_url,
            agent_skills_spec_version=AGENT_SKILLS_SPEC_VERSION,
            skill_name=frontmatter["name"],
            description=frontmatter["description"],
            license=frontmatter.get("license"),
            compatibility=frontmatter.get("compatibility"),
            allowed_tools=frontmatter.get("allowed-tools"),
            skill_metadata=frontmatter.get("metadata"),
        )
        return store.put_document(instance)

    def get_skill_files(self) -> "FileArtifactUniverse":
        from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
        return FileArtifactUniverse.get(self.skill_files_id)


def parse_skill_md(data: bytes) -> tuple[dict, str]:
    text = data.decode("utf-8")
    if not text.startswith("---"):
        raise ValueError("SKILL.md must begin with a YAML frontmatter block delimited by '---'")
    after_open = text[3:]
    if after_open.startswith("\n"):
        after_open = after_open[1:]
    end_match = re.search(r"^---\s*$", after_open, flags=re.MULTILINE)
    if end_match is None:
        raise ValueError("SKILL.md frontmatter is not closed with '---'")
    frontmatter_text = after_open[: end_match.start()]
    body = after_open[end_match.end():].lstrip("\n")
    parsed = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(parsed, dict):
        raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    return parsed, body


def _validate_frontmatter(fm: dict, expected_name: str) -> None:
    name = fm.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("SKILL.md frontmatter 'name' is required and must be a non-empty string")
    if not _SKILL_NAME_RE.match(name):
        raise ValueError(
            f"SKILL.md frontmatter 'name' {name!r} does not match the spec: 1-64 chars, lowercase "
            "alphanumerics and hyphens only, no leading/trailing or consecutive hyphens"
        )
    if name != expected_name:
        raise ValueError(
            f"SKILL.md frontmatter 'name' {name!r} must equal the SkillArtifact id {expected_name!r}"
        )

    description = fm.get("description")
    if not isinstance(description, str) or not description:
        raise ValueError("SKILL.md frontmatter 'description' is required and must be a non-empty string")
    if len(description) > _MAX_DESCRIPTION_LEN:
        raise ValueError(
            f"SKILL.md frontmatter 'description' is {len(description)} chars; max {_MAX_DESCRIPTION_LEN}"
        )

    compatibility = fm.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str) or not compatibility:
            raise ValueError("SKILL.md frontmatter 'compatibility' must be a non-empty string if present")
        if len(compatibility) > _MAX_COMPATIBILITY_LEN:
            raise ValueError(
                f"SKILL.md frontmatter 'compatibility' is {len(compatibility)} chars; max {_MAX_COMPATIBILITY_LEN}"
            )

    license_ = fm.get("license")
    if license_ is not None and (not isinstance(license_, str) or not license_):
        raise ValueError("SKILL.md frontmatter 'license' must be a non-empty string if present")

    allowed_tools = fm.get("allowed-tools")
    if allowed_tools is not None and (not isinstance(allowed_tools, str) or not allowed_tools):
        raise ValueError("SKILL.md frontmatter 'allowed-tools' must be a non-empty string if present")

    metadata = fm.get("metadata")
    if metadata is not None:
        if not isinstance(metadata, dict):
            raise ValueError("SKILL.md frontmatter 'metadata' must be a mapping if present")
        for k, v in metadata.items():
            if not isinstance(k, str) or not isinstance(v, str):
                raise ValueError("SKILL.md frontmatter 'metadata' must map strings to strings")


def fetch_skill_md(object_url: str) -> bytes:
    prefix = object_url if object_url.endswith("/") else object_url + "/"
    skill_md_url = prefix + _SKILL_MD_FILENAME
    try:
        return get_config().get_object_store_at(skill_md_url).get(skill_md_url)
    except ObjectNotFoundError as e:
        raise ValueError(f"{_SKILL_MD_FILENAME} not found at {object_url}") from e


@contextlib.contextmanager
def download_skill(object_url: str) -> Iterator[Path]:
    """Download an object-store prefix into a local tempdir; yields the dir, cleans up on exit."""
    store = get_config().get_object_store_at(object_url)
    prefix = object_url if object_url.endswith("/") else object_url + "/"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        count = 0
        for obj_url in store.list_at(prefix):
            rel = obj_url[len(prefix):]
            if not rel or obj_url.endswith("/"):
                continue
            store.download_to_file(obj_url, str(tmp_path / rel))
            count += 1
        if count == 0:
            raise ValueError(f"No objects found under {object_url}")
        yield tmp_path
