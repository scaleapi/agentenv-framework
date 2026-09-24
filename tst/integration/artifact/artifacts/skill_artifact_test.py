"""Integration tests for SkillArtifact.

Runs against the configured object and document stores, matching the
other artifact tests in this directory.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from agent_env.artifact import AGENT_SKILLS_SPEC_VERSION, Artifact, SkillArtifact
from agent_env.config import get_config


SAMPLE_SKILL_DIR = Path(__file__).resolve().parents[3] / "data" / "skills" / "sample_skill"


def _unique_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _write_skill_dir(artifact_id: str, frontmatter_name: str, tmp_path: Path) -> Path:
    skill_dir = tmp_path / artifact_id
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\n"
        f"name: {frontmatter_name}\n"
        f"description: Integration-test skill for {artifact_id}.\n"
        f"license: Apache-2.0\n"
        f"---\n\n"
        f"# {frontmatter_name}\n\nBody.\n"
    )
    (skill_dir / "scripts").mkdir()
    (skill_dir / "scripts" / "hello.py").write_text("print('hi')\n")
    return skill_dir


@pytest.mark.integration
class TestSkillArtifact:
    def test_put_valid_skill_roundtrips(self, tmp_path):
        artifact_id = _unique_id("skill-valid")
        skill_dir = _write_skill_dir(artifact_id, frontmatter_name=artifact_id, tmp_path=tmp_path)

        created = SkillArtifact.put(id=artifact_id, skill_dir=skill_dir)

        assert created.type == "skill"
        assert created.skill_name == artifact_id
        assert created.agent_skills_spec_version == AGENT_SKILLS_SPEC_VERSION
        assert created.license == "Apache-2.0"
        assert created.skill_object_url.startswith(get_config().get_object_store().object_url("artifacts/skill/"))
        assert created.skill_object_url.rstrip("/").endswith(f"/artifacts/skill/{artifact_id}/{created.version}")

        retrieved = Artifact.get(artifact_id)
        assert isinstance(retrieved, SkillArtifact)
        assert retrieved.skill_files_id == created.skill_files_id
        assert retrieved.skill_object_url == created.skill_object_url

        skill_files = retrieved.get_skill_files()
        fa_map = skill_files.get_file_artifacts()
        assert "SKILL.md" in fa_map
        assert "scripts/hello.py" in fa_map
        assert all(fa.object_url.startswith(created.skill_object_url) for fa in fa_map.values())

    def test_put_invalid_skill_rejected(self, tmp_path):
        artifact_id = _unique_id("skill-invalid")
        skill_dir = _write_skill_dir(artifact_id, frontmatter_name="something-else", tmp_path=tmp_path)

        with pytest.raises(ValueError, match="must equal the SkillArtifact id"):
            SkillArtifact.put(id=artifact_id, skill_dir=skill_dir)
