"""``Skill.object_url`` and the neutral skill flags replace their S3-named spellings.

Stored add_skills steps carry ``s3_url``, so it is read forever and written beside ``object_url`` while workers on
older versions read only it. The ``s3_url`` keyword and the S3-named CLI flags still work, with a warning."""

import dataclasses
import logging
import warnings

import pytest
from click.testing import CliRunner

from agent_env.artifact import FileArtifactUniverse, SkillArtifact
from agent_env.artifact.artifacts import skill as skill_module
from agent_env.cli import cli
from agent_env.task_step.task_steps.add_skills import Skill

_PREFIX = "mem://bucket/skills/review/"


def test_a_skill_named_by_object_url_carries_both_spellings_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        skill = Skill(name="review", description="Review work", object_url=_PREFIX)
    assert (skill.object_url, skill.s3_url) == (_PREFIX, _PREFIX)
    assert skill.to_dict() == {"name": "review", "description": "Review work", "s3_url": _PREFIX, "object_url": _PREFIX}


def test_the_s3_url_keyword_still_works_and_is_counted(caplog):
    with caplog.at_level(logging.WARNING, logger="agent_env.utils.deprecation"), \
            pytest.warns(DeprecationWarning, match=r"Skill\(s3_url=\)"):
        skill = Skill(name="review", description="Review work", s3_url=_PREFIX)
    assert (skill.object_url, skill.s3_url) == (_PREFIX, _PREFIX)
    assert [r.deprecated_symbol for r in caplog.records if getattr(r, "event", None) == "agent_env_deprecated_symbol"] == [
        "Skill(s3_url=)"
    ]


def test_an_inline_skill_reads_s3_url_as_none():
    """Readers outside agent-env check ``skill.s3_url is None``."""
    assert Skill(name="review", description="Review work", body="Inline").s3_url is None


def test_the_two_spellings_disagreeing_is_an_error_and_agreeing_is_a_copy():
    with pytest.raises(TypeError, match="got both object_url= and its deprecated spelling s3_url="):
        Skill(name="review", description="Review work", object_url=_PREFIX, s3_url="mem://bucket/other/")
    skill = Skill(name="review", description="Review work", object_url=_PREFIX)
    assert dataclasses.replace(skill) == skill


@pytest.mark.parametrize("stored", [
    {"s3_url": _PREFIX},
    {"object_url": _PREFIX},
    {"s3_url": _PREFIX, "object_url": "mem://bucket/stale/"},
], ids=["legacy-key", "neutral-key", "legacy-wins"])
def test_a_stored_skill_loads_keyed_either_way(stored):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        skill = Skill.from_dict({"name": "review", "description": "Review work", **stored})
    assert skill.object_url == _PREFIX


def test_the_skill_md_helpers_are_public_and_keep_their_private_names():
    assert skill_module._parse_skill_md is skill_module.parse_skill_md
    assert skill_module._fetch_skill_md is skill_module.fetch_skill_md
    frontmatter, body = skill_module.parse_skill_md(b"---\nname: demo\ndescription: A demo.\n---\nBody\n")
    assert (frontmatter["name"], body.strip()) == ("demo", "Body")


def _bundle_dir(tmp_path):
    (tmp_path / "files").mkdir()
    (tmp_path / "files" / "a.txt").write_text("A")
    return str(tmp_path / "files")


@pytest.mark.parametrize("flag,notice", [("--prefix-url", False), ("--s3-url", True)])
def test_put_bundled_takes_its_prefix_under_either_flag(local_stores, tmp_path, flag, notice):
    prefix = local_stores.get_object_store().object_url("bundles/cli/")
    result = CliRunner().invoke(cli, ["artifact", "file-artifact-universe", "put-bundled", "--id", "cli-bundle",
                                      "--file-dir", _bundle_dir(tmp_path), flag, prefix])
    assert result.exit_code == 0, result.output
    assert f"bundle_object_url={prefix.rstrip('/')}/" in result.stdout
    assert ("--s3-url is deprecated" in result.stderr) is notice
    assert FileArtifactUniverse.get("cli-bundle").bundle_object_url == prefix.rstrip("/") + "/"


def test_giving_a_flag_under_both_spellings_is_a_usage_error(local_stores, tmp_path):
    prefix = local_stores.get_object_store().object_url("bundles/cli/")
    result = CliRunner().invoke(cli, ["artifact", "file-artifact-universe", "put-bundled", "--id", "cli-bundle",
                                      "--file-dir", _bundle_dir(tmp_path), "--prefix-url", prefix, "--s3-url", prefix])
    assert result.exit_code == 2
    assert "--s3-url is the deprecated spelling of --prefix-url" in result.output


@pytest.mark.parametrize("flag", ["--prefix-url", "--s3-url"])
def test_skill_put_reads_a_skill_from_a_prefix_under_either_flag(local_stores, flag):
    store = local_stores.get_object_store()
    store.put("skills/demo/SKILL.md", b"---\nname: demo\ndescription: A demo skill.\n---\nBody\n")
    result = CliRunner().invoke(cli, ["artifact", "skill", "put", "--id", "demo", flag, store.object_url("skills/demo/")])
    assert result.exit_code == 0, result.output
    assert "skill_object_url=" in result.stdout
    assert SkillArtifact.get("demo").skill_name == "demo"
