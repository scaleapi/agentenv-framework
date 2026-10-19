"""What ``bundle init`` does with each entity a task uses: writes it into a folder of its own, its image pruned from its
stored build context, downloading a context stored as one object once; and what a folder can't hold fails the import,
unless --include-scope user leaves it in the store, pinned."""

import io
import json
import os
import tarfile
import tomllib
from typing import ClassVar, Literal

import pytest
from click.testing import CliRunner

from agent_env.artifact import FileArtifact
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.environment import EnvironmentArtifact
from agent_env.artifact.artifacts.environment_universe import EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.artifacts.skill import SkillArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError
from agent_env.bundle.importer import import_task
from agent_env.bundle.resolve import build_step
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.entity.scopes import FieldScope
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.store.object_store.object_store import ObjectMetadata
from agent_env.task import Task

DEPLOY = {"id": "env", "type": "deploy_env", "env_id": "crm"}
USER = [FieldScope.USER]


@pytest.fixture
def home(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _files(root, files):
    root.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def _context_image(home, id, files=None, dockerfile="Dockerfile", folder=None):
    files = files or {dockerfile: "FROM python:3.12-slim\nCOPY app.py /app.py\n", "app.py": "print('hi')\n"}
    context = _files(home / "contexts" / (folder or id), files)
    return DockerImageArtifact.put_context(id, description=id, context_path=str(context), dockerfile_path=dockerfile)


def _task(steps, id="t"):
    return Task.put(id=id, steps=[build_step(dict(step)) for step in steps])


def _toml(root, path):
    return tomllib.loads((root / path).read_text())


def _tarball_image(id="crm-image", image_name="crm-image:v1"):
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description=id, image_name=image_name, tar_gz_s3_url=f"file:///nowhere/{id}.tar.gz"))


def test_an_image_stored_only_as_a_tarball_fails_an_import_naming_it_and_writes_nothing(home):
    MCPServerEnv.put(id="crm", docker_image_artifact=_tarball_image(), environment_name="crm")
    _task([DEPLOY])

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "t")

    assert caught.value.problems == (
        "envs/crm: image: artifact crm-image v1 can't be written into the folder: it's stored only as a docker save "
        "tarball, under crm-image:v1, which names no registry; with --include-scope user it stays in the store, for a "
        "folder run against the same store",)
    assert not (home / "t").exists() and not [path for path in home.iterdir() if ".import-" in path.name]


def test_with_the_user_scope_it_stays_in_the_store_pinned_and_the_report_and_readme_say_so(home):
    MCPServerEnv.put(id="crm", docker_image_artifact=_tarball_image(), environment_name="crm")
    _task([DEPLOY])

    imported = import_task("t", home / "t", include_scopes=USER)

    assert _toml(imported.root, "envs/crm/env.toml") == {"environment_name": "crm",
                                                          "image": {"artifact": "crm-image", "version": 1}}
    assert [(pin.kind.value, pin.id, pin.version, pin.where) for pin in imported.pinned] == [
        ("artifact", "crm-image", 1, "envs/crm: image")]
    readme = (imported.root / "README.md").read_text()
    assert "## Still needs the store" in readme and "except what *Still needs the store* lists" in readme
    assert json.loads((imported.root / "provenance.json").read_text())["store"][0]["id"] == "crm-image"


def test_the_cli_says_up_front_that_the_folder_still_needs_the_store(home):
    MCPServerEnv.put(id="crm", docker_image_artifact=_tarball_image(), environment_name="crm")
    _task([DEPLOY])

    result = CliRunner().invoke(cli, ["bundle", "init", str(home / "t"), "--from-task", "t", "--include-scope", "user"])

    assert result.exit_code == 0, result.output
    assert "  It still needs the store for 1 reference(s): see Still needs the store." in result.output.splitlines()


def test_an_image_its_registry_names_is_written_as_its_ref_and_reported_as_pulled(home):
    image = _tarball_image(image_name="registry.example.com/crm:v1")
    MCPServerEnv.put(id="crm", docker_image_artifact=image, environment_name="crm")
    _task([DEPLOY])

    imported = import_task("t", home / "t")

    assert _toml(imported.root, "envs/crm/env.toml")["image"] == {"ref": "registry.example.com/crm:v1"}
    assert [(pull.folder, pull.image) for pull in imported.pulled] == [("envs/crm", "registry.example.com/crm:v1")]
    assert imported.pinned == () and imported.images == ()
    assert ("- Pull access to the registries its pulled images are in, a `docker login` to each that's private: "
            "`registry.example.com`.") in (imported.root / "README.md").read_text().splitlines()


def test_contexts_stored_as_one_object_are_downloaded_once_and_pruned_for_each_image(home, monkeypatch):
    files = {"Dockerfile": "FROM python:3.12-slim\nCOPY app.py /app.py\n",
             "slack/Dockerfile": "FROM python:3.12-slim\nCOPY slack/ /slack/\n", "app.py": "app\n",
             "slack/main.py": "slack\n", "websites/huge.bin": "x" * 50_000}
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image", files, folder="shared"),
                     environment_name="crm")
    MCPServerEnv.put(id="slack", docker_image_artifact=_context_image(home, "slack-image", files, "slack/Dockerfile",
                                                                      folder="shared"),
                     environment_name="slack")
    _task([DEPLOY, {**DEPLOY, "id": "env2", "env_id": "slack"}])
    store = get_config().get_object_store()
    store_type = type(store)
    downloads = []
    real_download, real_head = store_type.download_to_file, store_type.get_object_metadata_at

    def head(self, url):
        metadata = real_head(self, url)
        same = url.endswith("build-context.tar.gz")  # both contexts are one object's bytes, as S3 would tell by ETag
        return ObjectMetadata(size=metadata.size, etag='"one-object"') if same and metadata else metadata

    def download(self, url, dest):
        downloads.append(url)
        return real_download(self, url, dest)

    monkeypatch.setattr(store_type, "get_object_metadata_at", head)
    monkeypatch.setattr(store_type, "download_to_file", download)

    imported = import_task("t", home / "t")

    assert len([url for url in downloads if url.endswith("build-context.tar.gz")]) == 1
    assert sorted(path.relative_to(imported.root).as_posix() for path in (imported.root / "envs").rglob("*")
                  if path.is_file()) == ["envs/crm/Dockerfile", "envs/crm/app.py", "envs/crm/env.toml",
                                         "envs/slack/env.toml", "envs/slack/slack/Dockerfile",
                                         "envs/slack/slack/main.py"]
    assert _toml(imported.root, "envs/slack/env.toml")["image"] == {"dockerfile": "slack/Dockerfile"}


def test_an_image_that_records_no_dockerfile_builds_the_one_its_context_holds_and_says_it_was_found(home):
    built = _context_image(home, "agent-built", {"claude_code/Dockerfile": "FROM node\nCOPY claude_code/ /app/\n",
                                                 "claude_code/run.sh": "run\n"}, "claude_code/Dockerfile")
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="agent-image", description="from GitHub", image_name="agent-image:v1",
        build_context_s3_url=built.build_context_object_url))
    MCPServerEnv.put(id="crm", docker_image_artifact=image, environment_name="crm")
    _task([DEPLOY])

    imported = import_task("t", home / "t")

    assert _toml(imported.root, "envs/crm/env.toml")["image"] == {"dockerfile": "claude_code/Dockerfile"}
    assert [(image.dockerfile, image.found) for image in imported.images] == [("claude_code/Dockerfile", True)]
    assert "(found: the image records no Dockerfile)" in (imported.root / "README.md").read_text()


@pytest.mark.parametrize("target", ["../../../etc", "/etc/hosts", "nowhere"], ids=["climbs", "absolute", "dangling"])
def test_a_context_holding_a_link_a_bundle_cant_follow_fails_and_with_the_user_scope_stays_in_the_store(home, target):
    context = _files(home / "contexts" / "linked", {"Dockerfile": "FROM python:3.12-slim\nCOPY . /app\n", "a": "a"})
    os.symlink(target, context / "out")
    image = DockerImageArtifact.put_context("crm-image", description="linked", context_path=str(context),
                                            dockerfile_path="Dockerfile")
    MCPServerEnv.put(id="crm", docker_image_artifact=image, environment_name="crm")
    _task([DEPLOY])

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "failed")
    kept = import_task("t", home / "kept", include_scopes=USER)

    assert f"its build context holds the link out -> {target}" in caught.value.problems[0]
    assert _toml(kept.root, "envs/crm/env.toml")["image"] == {"artifact": "crm-image", "version": 1}
    assert not (kept.root / "envs/crm/out").exists() and not os.path.lexists(kept.root / "envs/crm/out")


@pytest.mark.parametrize("name", ["../escape", "/etc", "a/../../b"])
def test_a_universe_name_that_would_leave_its_folder_is_refused_before_anything_is_written(home, name):
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    (home / "rows.json").write_text("{}")
    file = FileArtifact.put("rows", description="rows.json", file_path=str(home / "rows.json"))
    EnvironmentUniverseArtifact.put("world", environment_artifacts=[
        EnvironmentArtifact.put("world-env", environment_name=name, file_artifact=file)])
    _task([{**DEPLOY, "artifact_id": "world"}])

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "t")

    assert "would be written outside its folder" in caught.value.problems[0]
    assert not (home / "t").exists() and not (home / "escape").exists()


@pytest.mark.parametrize("names, why", [
    (["A.txt", "a.txt"], "by names that differ only in case"),
    ([".DS_Store", "b.txt"], "a bundle leaves out .DS_Store as the OS or Python's own"),
], ids=["case", "leavings"])
def test_a_file_universe_a_folder_cant_hold_as_it_is_fails_the_import(home, names, why):
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    files = {name: FileArtifact.put(f"doc-{n}", description=name, file_path=str(_files(home / f"u{n}", {name: name}) / name))
             for n, name in enumerate(names)}  # stored apart, so a case-insensitive disk holds both
    FileArtifactUniverse.put("docs", file_artifacts=files)
    _task([DEPLOY, {"id": "load", "type": "load_artifact", "env_id": "crm", "artifacts": [{"id": "docs"}]}])

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "t")

    assert why in caught.value.problems[0]


class _Unexportable(Artifact):
    """A plugin's artifact type with no to_toml."""

    type: Literal["unexportable_import_test"] = "unexportable_import_test"
    payload: str = "x"
    toml_keys: ClassVar[dict] = {}


def test_a_type_with_no_to_toml_fails_the_import_and_with_the_user_scope_stays_in_the_store(home, monkeypatch):
    artifacts = {**Config().artifact_registry(), _Unexportable.model_fields["type"].default: _Unexportable}
    monkeypatch.setattr(Config, "artifact_registry", lambda self: artifacts)
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    get_artifact_store().put_document(_Unexportable(id="plugin-data"))
    _task([DEPLOY, {"id": "load", "type": "load_artifact", "env_id": "crm", "artifacts": [{"id": "plugin-data"}]}])

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "failed")
    kept = import_task("t", home / "kept", include_scopes=USER)

    assert ("a unexportable_import_test artifact has no form a bundle folder holds" in caught.value.problems[0])
    assert json.loads((kept.root / "tasks/t.json").read_text())[1]["artifacts"][0] == {"id": "plugin-data",
                                                                                      "version": 1}
    assert [entity.folder for entity in kept.written] == ["envs/crm"]


def test_a_skill_whose_name_another_artifact_in_the_folder_has_fails_the_import(home):
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    (home / "notes.txt").write_text("notes")
    FileArtifact.put("notes", description="notes", file_path=str(home / "notes.txt"))
    skill = _files(home / "skill", {"SKILL.md": "---\nname: notes\ndescription: how notes are kept\n---\nKeep them.\n"})
    SkillArtifact.put("team/notes", skill_dir=skill)
    _task([DEPLOY, {"id": "load", "type": "load_artifact", "env_id": "crm", "artifacts": [{"id": "notes"}]},
           {"id": "skills", "type": "add_skills", "agent_name": "default-agent",
            "skills": [{"skill_artifact_id": "team/notes"}]}])

    with pytest.raises(BundleError) as caught:
        import_task("t", home / "t")

    assert "a skill's folder takes its skill's name, 'notes', which another artifact" in caught.value.problems[0]


def test_an_entity_two_steps_name_is_written_once_and_both_name_its_folder(home):
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    _task([DEPLOY, {"id": "agent", "type": "deploy_agent", "env_ids": ["crm"]}])

    imported = import_task("t", home / "t")

    steps = json.loads((imported.root / "tasks/t.json").read_text())
    assert [entity.folder for entity in imported.written] == ["envs/crm"]
    assert (steps[0]["env_id"], steps[1]["env_ids"]) == ("crm", ["crm"])


def test_provenance_names_each_written_entity_and_image_by_store_id_and_version_and_no_store_location(home):
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    _task([DEPLOY])

    imported = import_task("t", home / "t")

    text = (imported.root / "provenance.json").read_text()
    provenance = json.loads(text)
    assert provenance["from"] == {"task": "t", "version": 1}
    assert provenance["written"] == [{"folder": "envs/crm", "kind": "env", "type": "mcp_server", "id": "crm",
                                      "version": 1}]
    assert provenance["images"] == [{"folder": "envs/crm", "key": "image", "id": "crm-image", "version": 1,
                                     "dockerfile": "Dockerfile", "dockerfile_found": False}]
    assert "file://" not in text and str(home) not in text


def _saved_image(path, url, files):
    """A one-layer ``docker save`` tarball holding ``files``, stored at ``url``."""
    layer = io.BytesIO()
    with tarfile.open(fileobj=layer, mode="w") as tar:
        for name, text in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(text.encode())
            tar.addfile(info, io.BytesIO(text.encode()))
    with tarfile.open(path, "w:gz") as tar:
        for name, data in {"0/layer.tar": layer.getvalue(),
                           "manifest.json": json.dumps([{"Layers": ["0/layer.tar"]}]).encode()}.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return get_config().get_object_store_at(url).put_file_at(url, str(path), "application/gzip")


def test_an_image_the_store_holds_built_is_pinned_to_its_python_versions_and_the_report_says_so(home):
    context = _context_image(home, "context-only", {"Dockerfile": "FROM python:3.12-slim\nRUN pip install requests\n"})
    site = "usr/local/lib/python3.12/site-packages"
    url = context.build_context_object_url.rsplit("/", 1)[0] + "/saved.tar.gz"
    saved = _saved_image(home / "saved.tar.gz", url, {
        f"{site}/requests-2.31.0.dist-info/METADATA": "Name: requests\nVersion: 2.31.0\n",
        f"{site}/pip-24.0.dist-info/METADATA": "Name: pip\nVersion: 24.0\n"})
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="crm-image", description="built", image_name="crm-image:v1", tar_gz_s3_url=saved,
        build_context_s3_url=context.build_context_object_url, dockerfile_path="Dockerfile"))
    MCPServerEnv.put(id="crm", docker_image_artifact=image, environment_name="crm")
    MCPServerEnv.put(id="slack", docker_image_artifact=_context_image(home, "slack-image"), environment_name="slack")
    _task([DEPLOY, {"id": "slack", "type": "deploy_env", "env_id": "slack"}])

    imported = import_task("t", home / "t")

    assert (imported.root / "envs/crm/constraints.txt").read_text() == "requests==2.31.0\n"
    assert "ARG PIP_CONSTRAINT=/tmp/agent-env-constraints.txt\n" in (imported.root / "envs/crm/Dockerfile").read_text()
    assert not (imported.root / "envs/slack/constraints.txt").exists()
    readme = (imported.root / "README.md").read_text()
    assert ("pinned to the stored image: 1 Python package(s) by constraints.txt; left as the build resolves them: pip, "
            "which isolated builds install their own versions of") in readme
    assert "versions not pinned: the store holds no image to read its versions from" in readme


def test_a_field_naming_an_entity_it_cant_take_is_left_out_and_said_so_and_the_entity_still_written(home):
    """The store holds a step as it is, though the step can't read what it names; a folder can't hold it so."""
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    (home / "seed.json").write_text("{}")
    file = FileArtifact.put("seed", description="seed.json", file_path=str(home / "seed.json"))
    EnvironmentArtifact.put("vireo", environment_name="crm", file_artifact=file)
    _task([DEPLOY,
           {"id": "load", "type": "load_artifact", "env_id": "crm", "artifacts": [{"id": "vireo"}]},
           {"id": "snapshot", "type": "snapshot_agent_state", "artifact_id": "snap", "agent_name": "agent", "prompt_id": "ask",
            "env_id": "crm", "universe_artifact_id": "vireo"}])

    imported = import_task("t", home / "t")

    snapshot = json.loads((imported.root / "tasks/t.json").read_text())[2]
    assert snapshot.get("universe_artifact_id") is None
    assert ("tasks/t.json: step 'snapshot': universe_artifact_id", "dropped: 'vireo' is environment, and this field "
            "takes environment_universe, so the step can't read it") in [(d.where, d.why) for d in imported.dropped]
    assert _toml(imported.root, "artifacts/vireo/artifact.toml")["type"] == "environment"


def test_an_image_whose_context_cant_be_written_is_pulled_by_the_registry_ref_it_names_and_said_why(home):
    context = _context_image(home, "two-dockerfiles", {"mail/Dockerfile": "FROM python\nCOPY mail/ /app/\n",
                                                       "mail/website/Dockerfile": "FROM node\n", "mail/app.py": "app\n"},
                             "mail/Dockerfile")
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="mail-image", description="no dockerfile recorded", image_name="registry.example.com/mail:v1",
        build_context_s3_url=context.build_context_object_url))
    MCPServerEnv.put(id="crm", docker_image_artifact=image, environment_name="crm")
    _task([DEPLOY])

    imported = import_task("t", home / "t")

    assert _toml(imported.root, "envs/crm/env.toml")["image"] == {"ref": "registry.example.com/mail:v1"}
    assert imported.images == () and imported.pinned == ()
    [pulled] = imported.pulled
    assert pulled.why.startswith("its build context can't be written as a folder: the build context names no "
                                 "Dockerfile and holds 2: mail/Dockerfile, mail/website/Dockerfile")
    assert not (imported.root / "envs/crm/mail").exists()


def test_a_step_naming_no_version_of_an_env_another_step_pins_takes_that_version_and_its_folder(home):
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image"), environment_name="crm")
    MCPServerEnv.put(id="crm", docker_image_artifact=_context_image(home, "crm-image-2"), environment_name="crm")
    _task([{**DEPLOY, "env_version": 1},
           {"id": "cli", "type": "build_mcp_cli", "env_id": "crm", "command_name": "crm", "cli_artifact_id": "crm-cli",
            "depends_on": [{"task_step_id": "env"}]}])

    imported = import_task("t", home / "t")

    steps = json.loads((imported.root / "tasks/t.json").read_text())
    assert (steps[0]["env_id"], steps[0].get("env_version"), steps[1]["env_id"]) == ("crm", None, "crm")
    assert [(written.folder, written.version) for written in imported.written if written.kind.value == "env"] == [
        ("envs/crm", 1)]
    assert ("tasks/t.json: step 'cli': env_id", "pinned: it names no version, so it takes v1, the one the task pins "
            "elsewhere") in [(d.where, d.why) for d in imported.dropped]
