"""The S3-named artifact keys get neutral twins, and the put helpers take only the neutral keywords.

A stored doc keyed either way loads, and every dump writes both keys (the additive wire contract: old readers keep
reading the legacy key)."""

import pytest

from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.skill import SkillArtifact

_CASES = {
    "file": (FileArtifact, {"description": "d", "filename": "a.json", "content_type": "application/json"},
             [("s3_url", "object_url")]),
    "docker_image": (DockerImageArtifact, {"description": "d", "image_name": "img:1"},
                     [("tar_gz_s3_url", "tar_gz_object_url"), ("build_context_s3_url", "build_context_object_url")]),
    "file_artifact_universe": (FileArtifactUniverse, {}, [("bundle_s3_url", "bundle_object_url")]),
    "skill": (SkillArtifact, {"skill_files_id": "s__files", "agent_skills_spec_version": "0.1", "skill_name": "s",
                              "description": "d"}, [("skill_s3_url", "skill_object_url")]),
    "cli": (CliArtifact, {"cli_files_id": "c__files", "entrypoint": "bin/c", "command_name": "c"},
            [("cli_s3_url", "cli_object_url")]),
}


@pytest.mark.parametrize("spelling", ["legacy", "neutral"])
@pytest.mark.parametrize("case", _CASES.values(), ids=_CASES.keys())
def test_a_doc_keyed_either_way_loads_and_dumps_both_keys(case, spelling):
    cls, fields, pairs = case
    urls = {neutral: f"mem://bucket/{neutral}" for _legacy, neutral in pairs}
    keyed = {(legacy if spelling == "legacy" else neutral): urls[neutral] for legacy, neutral in pairs}
    artifact = cls.model_validate({"id": "x", "version": 1, **fields, **keyed})

    assert {neutral: getattr(artifact, neutral) for _legacy, neutral in pairs} == urls
    for by_alias in (True, False):
        dumped = artifact.model_dump(by_alias=by_alias)
        assert all(dumped[legacy] == dumped[neutral] == urls[neutral] for legacy, neutral in pairs)


def test_the_legacy_key_wins_when_the_two_disagree():
    """Only a raw-doc writer outside the model can make them differ, and today's know only the legacy key."""
    doc = {"id": "x", "version": 1, "description": "d", "filename": "a", "content_type": "t"}
    assert FileArtifact.model_validate({**doc, "s3_url": "mem://b/new", "object_url": "mem://b/stale"}).object_url == "mem://b/new"


@pytest.mark.parametrize("cls", [*(case[0] for case in _CASES.values()), EnvironmentArtifact, EnvironmentUniverseArtifact],
                         ids=lambda cls: cls.__name__)
def test_the_serialization_schema_keeps_the_fields(cls):
    """A wrap serializer's return annotation would replace the schema; an annotated dict leaves no fields."""
    serialization = cls.model_json_schema(mode="serialization")
    assert set(serialization.get("properties", {})) == set(cls.model_json_schema(mode="validation")["properties"])


def test_an_excluded_field_stays_excluded():
    artifact = FileArtifact(id="x", description="d", filename="a", content_type="t", object_url="mem://b/a")
    assert {"s3_url", "object_url"}.isdisjoint(artifact.model_dump(exclude={"object_url"}))


def _bundle(tmp_path, local_stores):
    path = tmp_path / "a.txt"
    path.write_bytes(b"A")
    return {"a.txt": path}, local_stores.get_object_store().object_url("bundles/one/")


def test_the_put_helpers_take_their_object_store_keywords(local_stores, tmp_path):
    files, prefix = _bundle(tmp_path, local_stores)
    store = local_stores.get_object_store()
    image, context = store.object_url("images/i.tar.gz"), store.object_url("images/context.tar.gz")
    store.put("skills/demo/SKILL.md", b"---\nname: demo\ndescription: A demo skill.\n---\nBody\n")

    bundled = FileArtifactUniverse.put_bundled(id="bundled", files=files, prefix_url=prefix)
    existing = FileArtifactUniverse.put_existing(id="existing", prefix_url=prefix)
    put = FileArtifactUniverse.put(id="put", file_artifacts=bundled.get_file_artifacts(), bundle_object_url=prefix)
    image_artifact = DockerImageArtifact.put_tar("img", description="d", image_name="img:1", tar_gz_object_url=image,
                                                 build_context_object_url=context)
    SkillArtifact.validate(object_url=store.object_url("skills/demo"), expected_name="demo")

    assert bundled.bundle_object_url == existing.bundle_object_url == prefix.rstrip("/") + "/"
    assert put.bundle_object_url == prefix
    assert (image_artifact.tar_gz_object_url, image_artifact.build_context_object_url) == (image, context)


@pytest.mark.parametrize("call", [
    lambda url: FileArtifactUniverse.put_bundled(id="u", files={}, s3_url=url),
    lambda url: FileArtifactUniverse.put_existing(id="u", prefix_url=url, s3_url=url),
    lambda url: FileArtifactUniverse.put(id="u", file_artifacts={}, bundle_s3_url=url),
    lambda url: DockerImageArtifact.put_tar("img", description="d", image_name="img:1", tar_gz_object_url=url,
                                            tar_gz_s3_url=url),
    lambda url: DockerImageArtifact.put_tar("img", description="d", image_name="img:1", tar_gz_object_url=url,
                                            build_context_s3_url=url),
    lambda url: SkillArtifact.validate(s3_url=url, expected_name="demo"),
], ids=["put_bundled", "put_existing", "put", "put_tar-tar_gz", "put_tar-build_context", "skill-validate"])
def test_an_s3_named_keyword_is_refused(local_stores, call):
    with pytest.raises(TypeError, match="unexpected keyword argument '(s3_url|bundle_s3_url|tar_gz_s3_url|build_context_s3_url)'"):
        call(local_stores.get_object_store().object_url("a/"))


def test_the_object_url_a_helper_needs_is_required():
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'tar_gz_object_url'"):
        DockerImageArtifact.put_tar("img", description="d", image_name="img:1")
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'prefix_url'"):
        FileArtifactUniverse.put_existing(id="u")
