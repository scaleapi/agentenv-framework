"""A bundle writes an artifact type from its folder through the type's own ``bundle_folder`` and ``from_toml``, so a
plugin's type is planned, checked, written and reused as a core type is."""

import json
from typing import Literal

import pytest
from pydantic import Field

from agent_env.artifact import Artifact, FileArtifact
from agent_env.artifact.artifact import ArtifactFolder, FolderChild
from agent_env.artifact.ref import ArtifactRef
from agent_env.bundle import BundleError
from agent_env.bundle.materialize import materialize
from agent_env.config.runtime import Config
from tst.unit.bundle._support import layout, plan_of

ROOT = "@local/~/triage"
TOML = {"artifacts/rooms/artifact.toml": 'type = "rooms_folder_test"\n'}


class _Rooms(Artifact):
    """A plugin's type, no core type's subclass: each file seeds the environment it's named after, written as a file
    artifact of its own beside the rooms."""

    type: Literal["rooms_folder_test"] = "rooms_folder_test"
    rooms: dict[str, ArtifactRef] = Field(default_factory=dict)

    @classmethod
    def bundle_folder(cls, ctx):
        files = ctx.files()
        return ArtifactFolder(files, tuple(FolderChild(f"{ctx.id}-{key}", key, environment_name=key.split(".")[0])
                                           for key in files))

    @classmethod
    def from_toml(cls, data, ctx):
        folder = cls.bundle_folder(ctx)
        rooms = {child.environment_name: FileArtifact.put_attempt(
            child.id, description=child.path, file_path=str(folder.files[child.path]), filename=child.path).as_ref()
            for child in folder.children}
        return cls.put(id=ctx.id, rooms=rooms)


class _OwnFile(FileArtifact):
    """A plugin's file type with a from_toml and a bundle_folder of its own, keeping only the first line."""

    type: Literal["own_file_folder_test"] = "own_file_folder_test"

    @classmethod
    def bundle_folder(cls, ctx):
        return ArtifactFolder(ctx.only(ctx.files()))

    @classmethod
    def from_toml(cls, data, ctx):
        [(name, path)] = cls.bundle_folder(ctx).files.items()
        return cls.put_attempt(ctx.id, description=path.read_text().splitlines()[0], file_path=str(path),
                               filename=name)


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    artifacts = {**Config().artifact_registry(), **{cls.model_fields["type"].default: cls for cls in (_Rooms, _OwnFile)}}
    monkeypatch.setattr(Config, "artifact_registry", lambda self: artifacts)


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path / "triage"


def _sandbox_loading(files, *artifacts):
    steps = [{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm",
              "sandbox_type": "local"}]
    steps += [{"id": f"load-{name}", "type": "load_artifact", "sandbox_name": "box", "artifact_id": name,
               "destination_path": f"/app/{name}", "depends_on": ["box"]} for name in artifacts]
    return {**files, "tasks/t.json": json.dumps(steps)}


def _env_loading(files, artifact, env):
    return {**files, f"envs/{env}/Dockerfile": "FROM scratch\n", f"envs/{env}/env.toml": f'environment_name = "{env}"\n',
            "tasks/t.json": json.dumps([
                {"id": "env", "type": "deploy_env", "env_id": env},
                {"id": "load", "type": "load_artifact", "env_id": env, "artifact_id": artifact, "depends_on": ["env"]},
            ])}


def _summary(materialization):
    return {done.write.id: (done.version, done.reused) for done in materialization.writes}


def _problems(root):
    with pytest.raises(BundleError) as caught:
        plan_of(root)
    return caught.value.problems


def test_a_plugin_type_is_written_with_its_children_and_rewritten_only_when_a_file_it_reads_changes(bundle_dir):
    layout(bundle_dir, _sandbox_loading({**TOML, "artifacts/rooms/crm.json": "{}", "artifacts/rooms/slack.json": "[]"},
                                        "rooms"))

    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/rooms"] == (1, False)
    rooms = Artifact.get(f"{ROOT}/rooms")
    assert {name: ref.id for name, ref in rooms.rooms.items()} == {"crm": f"{ROOT}/rooms-crm.json",
                                                                   "slack": f"{ROOT}/rooms-slack.json"}
    assert FileArtifact.get(f"{ROOT}/rooms-slack.json").filename == "slack.json"
    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/rooms"] == (1, True)
    (bundle_dir / "artifacts/rooms/slack.json").write_text("[1]")
    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/rooms"] == (2, False)


def test_a_plugin_types_folder_problem_is_reported_before_anything_is_written(bundle_dir):
    layout(bundle_dir, _sandbox_loading(TOML, "rooms"))
    (bundle_dir / "artifacts/rooms/sub").mkdir()

    assert _problems(bundle_dir) == ("artifacts/rooms: has no files to write (.DS_Store, Thumbs.db, __pycache__, "
                                     "desktop.ini don't count)",)


def test_a_child_whose_id_another_write_claims_is_refused(bundle_dir):
    layout(bundle_dir, _sandbox_loading({**TOML, "artifacts/rooms/crm.json": "{}",
                                         "artifacts/rooms-crm.json/a.json": "{}"}, "rooms", "rooms-crm.json"))

    assert _problems(bundle_dir) == (f"artifacts/rooms-crm.json: '{ROOT}/rooms-crm.json' is also written by the "
                                     "artifact written from artifacts/rooms/crm.json; give each its own id",)


def test_a_child_seeding_no_server_of_the_env_it_loads_into_is_refused(bundle_dir):
    layout(bundle_dir, _env_loading({**TOML, "artifacts/rooms/crn.json": "{}"}, "rooms", "crm"))

    assert _problems(bundle_dir) == (
        "tasks/t.json: step 'load': artifacts/rooms/crn.json seeds an environment named 'crn', and envs/crm has no "
        "server by that name ('crm'); name it after the server it seeds, or leave it out",)


def test_two_children_seeding_one_environment_are_refused(bundle_dir):
    layout(bundle_dir, _sandbox_loading({**TOML, "artifacts/rooms/crm.json": "{}", "artifacts/rooms/crm.yaml": "{}"},
                                        "rooms"))

    assert _problems(bundle_dir) == ("artifacts/rooms/crm.yaml: environment_name 'crm' is also "
                                     "artifacts/rooms/crm.json's; a rooms_folder_test's environments need names of "
                                     "their own",)


def test_a_core_types_subclass_with_both_hooks_of_its_own_is_written_by_them(bundle_dir):
    layout(bundle_dir, _sandbox_loading({"artifacts/note/artifact.toml": 'type = "own_file_folder_test"\n',
                                         "artifacts/note/n.txt": "first\nsecond\n"}, "note"))

    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/note"] == (1, False)
    assert Artifact.get(f"{ROOT}/note").description == "first"
    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/note"] == (1, True)
