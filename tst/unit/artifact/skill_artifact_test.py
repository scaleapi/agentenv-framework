"""Writing a SkillArtifact: from a folder or a file listing, named by its id's last segment, retried safely."""

import pytest

from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.skill import SkillArtifact

SKILL_MD = "---\nname: pdf-extract\ndescription: Pull text out of PDFs.\n---\nBody\n"


@pytest.fixture
def skill_files(tmp_path):
    (tmp_path / "SKILL.md").write_text(SKILL_MD)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/extract.py").write_text("print('text')\n")
    return {"SKILL.md": tmp_path / "SKILL.md", "scripts/extract.py": tmp_path / "scripts/extract.py"}


@pytest.mark.parametrize("id", ["pdf-extract", "team/pdf-extract"])
def test_a_skill_is_named_by_its_ids_last_segment(local_stores, skill_files, id):
    skill = SkillArtifact.put(id, files=skill_files)

    assert (skill.id, skill.skill_name) == (id, "pdf-extract")
    assert sorted(skill.get_skill_files().get_file_artifacts()) == ["SKILL.md", "scripts/extract.py"]


def test_a_skill_whose_name_isnt_its_ids_last_segment_is_refused(local_stores, skill_files):
    with pytest.raises(ValueError, match="'pdf-extract' must be 'pdf'"):
        SkillArtifact.put("team/pdf", files=skill_files)


def test_a_folder_and_a_listing_write_the_same_files(local_stores, skill_files, tmp_path):
    from_folder = SkillArtifact.put("pdf-extract", skill_dir=tmp_path)
    from_listing = SkillArtifact.put("pdf-extract", files=skill_files)

    assert sorted(from_folder.get_skill_files().get_file_artifacts()) == sorted(
        from_listing.get_skill_files().get_file_artifacts())
    assert from_folder.skill_object_url != from_listing.skill_object_url


@pytest.mark.parametrize("sources", [{}, {"skill_dir": "x", "files": {}}], ids=["neither", "both"])
def test_a_put_takes_a_folder_or_a_listing(local_stores, sources):
    with pytest.raises(ValueError, match="exactly one of skill_dir or files"):
        SkillArtifact.put("pdf-extract", **sources)


def test_a_listing_without_skill_md_is_refused(local_stores, skill_files):
    with pytest.raises(ValueError, match="SKILL.md isn't among the skill's files"):
        SkillArtifact.put("pdf-extract", files={"scripts/extract.py": skill_files["scripts/extract.py"]})


def test_a_put_that_fails_after_uploading_its_files_can_be_retried(local_stores, skill_files, monkeypatch):
    real_put = FileArtifactUniverse.put

    def fail_once(*args, **kwargs):
        monkeypatch.setattr(FileArtifactUniverse, "put", real_put)
        raise RuntimeError("interrupted")

    monkeypatch.setattr(FileArtifactUniverse, "put", fail_once)
    with pytest.raises(RuntimeError, match="interrupted"):
        SkillArtifact.put("pdf-extract", files=skill_files)

    assert SkillArtifact.put("pdf-extract", files=skill_files).version == 1


def test_frontmatter_that_isnt_yaml_is_a_value_error(local_stores, skill_files):
    skill_files["SKILL.md"].write_text("---\nname: [pdf-extract\n---\n")

    with pytest.raises(ValueError, match="SKILL.md frontmatter isn't valid YAML"):
        SkillArtifact.put("pdf-extract", files=skill_files)
