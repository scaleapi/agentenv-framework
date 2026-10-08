"""``Skill.object_url`` and the neutral skill flags.

Stored add_skills steps carry ``s3_url``, so it is read forever and written beside ``object_url``. The ``s3_url``
keyword and attribute, the S3-named CLI flags and the private SKILL.md helper names are gone."""

import dataclasses

import pytest
from click.testing import CliRunner

from agent_env.artifact import FileArtifactUniverse, SkillArtifact
from agent_env.artifact.artifacts import skill as skill_module
from agent_env.cli import cli
from agent_env.task_step.task_steps.add_skills import Skill

_PREFIX = "mem://bucket/skills/review/"


def test_a_skill_named_by_object_url_is_stored_under_both_keys():
    skill = Skill(name="review", description="Review work", object_url=_PREFIX)
    assert skill.to_dict() == {"name": "review", "description": "Review work", "s3_url": _PREFIX, "object_url": _PREFIX}
    assert dataclasses.replace(skill) == skill


def test_an_inline_skill_is_stored_without_either_key():
    assert {"s3_url", "object_url"}.isdisjoint(Skill(name="review", description="Review work", body="Inline").to_dict())


def test_the_s3_url_keyword_and_attribute_are_gone():
    with pytest.raises(TypeError, match="unexpected keyword argument 's3_url'"):
        Skill(name="review", description="Review work", s3_url=_PREFIX)
    assert not hasattr(Skill(name="review", description="Review work", object_url=_PREFIX), "s3_url")


@pytest.mark.parametrize("stored", [
    {"s3_url": _PREFIX},
    {"object_url": _PREFIX},
    {"s3_url": _PREFIX, "object_url": "mem://bucket/stale/"},
], ids=["legacy-key", "neutral-key", "legacy-wins"])
def test_a_stored_skill_loads_keyed_either_way(stored):
    assert Skill.from_dict({"name": "review", "description": "Review work", **stored}).object_url == _PREFIX


def test_the_skill_md_helpers_are_public_only():
    assert not hasattr(skill_module, "_parse_skill_md") and not hasattr(skill_module, "_fetch_skill_md")
    frontmatter, body = skill_module.parse_skill_md(b"---\nname: demo\ndescription: A demo.\n---\nBody\n")
    assert (frontmatter["name"], body.strip()) == ("demo", "Body")


def _bundle_dir(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "a.txt").write_text("A")
    return str(tmp_path / "files")


def test_put_bundled_takes_its_prefix_under_prefix_url(local_stores, tmp_path):
    prefix = local_stores.get_object_store().object_url("bundles/cli/")
    result = CliRunner().invoke(cli, ["artifact", "file-artifact-universe", "put-bundled", "--id", "cli-bundle",
                                      "--file-dir", _bundle_dir(tmp_path), "--prefix-url", prefix])
    assert result.exit_code == 0, result.output
    assert f"bundle_object_url={prefix.rstrip('/')}/" in result.stdout
    assert FileArtifactUniverse.get("cli-bundle").bundle_object_url == prefix.rstrip("/") + "/"


def test_skill_put_reads_a_skill_from_a_prefix(local_stores):
    store = local_stores.get_object_store()
    store.put("skills/demo/SKILL.md", b"---\nname: demo\ndescription: A demo skill.\n---\nBody\n")
    result = CliRunner().invoke(cli, ["artifact", "skill", "put", "--id", "demo", "--prefix-url",
                                      store.object_url("skills/demo/")])
    assert result.exit_code == 0, result.output
    assert "skill_object_url=" in result.stdout
    assert SkillArtifact.get("demo").skill_name == "demo"


@pytest.mark.parametrize("args,old,new", [
    (["artifact", "file-artifact-universe", "put-bundled", "--id", "u", "--file-dir", ".", "--s3-url", _PREFIX],
     "--s3-url", "--prefix-url"),
    (["artifact", "skill", "put", "--id", "demo", "--s3-url", _PREFIX], "--s3-url", "--prefix-url"),
    (["a2a-agent", "add-skill", "--instance-id", "i", "--skill-s3-url", _PREFIX], "--skill-s3-url", "--skill-object-url"),
], ids=["put-bundled", "skill-put", "add-skill"])
def test_an_s3_named_flag_is_an_unknown_option_that_suggests_its_replacement(args, old, new):
    result = CliRunner().invoke(cli, args)
    assert result.exit_code == 2
    assert f"No such option '{old}'" in result.output and f"'{new}'" in result.output
