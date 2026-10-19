"""A bundle writes an artifact type from its folder through the type's own ``from_toml``, checked by its
``accept_toml``, with ``written_besides`` naming what else it writes and ``seeded_environments`` the environments its
folder holds, so a plugin's type is planned, checked, written and reused as a core type is."""

import json
from typing import ClassVar, Literal

import pytest
from pydantic import Field

from agent_env.artifact import Artifact, FileArtifact, SeedsEnvironments
from agent_env.artifact.artifact import FolderChild
from agent_env.artifact.ref import ArtifactRef
from agent_env.bundle import BundleError
from agent_env.bundle.materialize import materialize
from agent_env.config.runtime import Config
from tst.unit.bundle._support import layout, plan_of

ROOT = "@local/~/triage"
TOML = {"artifacts/rooms/artifact.toml": 'type = "rooms_folder_test"\n'}


class _Rooms(Artifact, SeedsEnvironments):
    """A plugin's type, no core type's subclass: each file seeds the environment it's named after, written as a file
    artifact of its own beside the rooms."""

    type: Literal["rooms_folder_test"] = "rooms_folder_test"
    rooms: dict[str, ArtifactRef] = Field(default_factory=dict)

    @classmethod
    def written_besides(cls, ctx):
        files = ctx.files()
        if strays := [key for key in files if not key.endswith((".json", ".yaml"))]:
            ctx.refuse([f"{key}: a room is a .json or .yaml file" for key in strays])
        return tuple(FolderChild(f"{ctx.id}-{key}", key) for key in files)

    @classmethod
    def seeded_environments(cls, ctx):
        return tuple((key, key.split(".")[0]) for key in ctx.files())

    @classmethod
    def from_toml(cls, data, ctx):
        files = ctx.files()
        rooms = {child.path.split(".")[0]: FileArtifact.put_attempt(
            child.id, description=child.path, file_path=str(files[child.path]), filename=child.path).as_ref()
            for child in cls.written_besides(ctx)}
        return cls.put(id=ctx.id, rooms=rooms)


class _OwnFile(FileArtifact):
    """A plugin's file type with only a from_toml of its own, describing the file by its first line."""

    type: Literal["own_file_folder_test"] = "own_file_folder_test"

    @classmethod
    def from_toml(cls, data, ctx):
        cls.accept_toml(data, ctx)
        name, path = ctx.file()
        return cls.put_attempt(ctx.id, description=path.read_text().splitlines()[0], file_path=str(path),
                               filename=name)


class _Setting(Artifact):
    """A plugin's type made only of its artifact.toml's values, so its folder holds nothing else."""

    type: Literal["setting_folder_test"] = "setting_folder_test"
    toml_keys: ClassVar[dict[str, type]] = {"value": str}
    value: str = ""

    @classmethod
    def from_toml(cls, data, ctx):
        return cls.put(id=ctx.id, value=ctx.accept(data, **cls.toml_keys).get("value", ""))


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    plugins = {cls.model_fields["type"].default: cls for cls in (_Rooms, _OwnFile, _Setting)}
    artifacts = {**Config().artifact_registry(), **plugins}
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
    return {**files, f"envs/{env}/Dockerfile": "FROM scratch\n",
            f"envs/{env}/env.toml": f'environment_name = "{env}"\n',
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


def test_a_folder_only_written_besides_refuses_is_reported_before_anything_is_written(bundle_dir):
    layout(bundle_dir, _sandbox_loading({**TOML, "artifacts/rooms/crm.json": "{}", "artifacts/rooms/notes.txt": "x"},
                                        "rooms"))

    assert _problems(bundle_dir) == ("artifacts/rooms: notes.txt: a room is a .json or .yaml file",)


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


def test_a_type_with_only_a_from_toml_of_its_own_is_written_by_it_and_rewritten_when_its_folder_changes(bundle_dir):
    layout(bundle_dir, _sandbox_loading({"artifacts/note/artifact.toml": 'type = "own_file_folder_test"\n',
                                         "artifacts/note/n.txt": "first\nsecond\n"}, "note"))

    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/note"] == (1, False)
    assert Artifact.get(f"{ROOT}/note").description == "first"
    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/note"] == (1, True)
    (bundle_dir / "artifacts/note/n.txt").write_text("changed\n")
    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/note"] == (2, False)
    assert Artifact.get(f"{ROOT}/note").description == "changed"


def test_a_type_with_only_a_from_toml_of_its_own_is_checked_by_the_accept_toml_it_inherits(bundle_dir):
    layout(bundle_dir, _sandbox_loading({"artifacts/note/artifact.toml": 'type = "own_file_folder_test"\n',
                                         "artifacts/note/a.txt": "a", "artifacts/note/b.txt": "b"}, "note"))

    assert _problems(bundle_dir) == ("artifacts/note: an own_file_folder_test artifact holds one file, and this folder "
                                     "has 2 ('a.txt', 'b.txt'); leave out its declared type to write the folder as a "
                                     "file_artifact_universe",)


def test_a_type_made_only_of_its_toml_is_written_from_a_folder_holding_nothing_else(bundle_dir):
    layout(bundle_dir, _sandbox_loading({"artifacts/mode/artifact.toml": 'type = "setting_folder_test"\nvalue = "a"\n'},
                                        "mode"))

    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/mode"] == (1, False)
    assert Artifact.get(f"{ROOT}/mode").value == "a"
    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/mode"] == (1, True)
    (bundle_dir / "artifacts/mode/artifact.toml").write_text('type = "setting_folder_test"\nvalue = "b"\n')
    assert _summary(materialize(plan_of(bundle_dir)))[f"{ROOT}/mode"] == (2, False)


@pytest.mark.parametrize("toml", ['type = "file"\n', 'type = "file_artifact_universe"\n',
                                  'type = "cli"\ncommand_name = "t"\nentrypoint = "t.sh"\n'],
                         ids=["file", "file-universe", "cli"])
def test_a_core_type_that_needs_a_file_refuses_an_empty_folder_before_anything_is_written(bundle_dir, toml):
    layout(bundle_dir, _sandbox_loading({"artifacts/x/artifact.toml": toml}, "x"))

    assert ("artifacts/x: has no files to write (.DS_Store, Thumbs.db, __pycache__, desktop.ini don't count)"
            in _problems(bundle_dir))
