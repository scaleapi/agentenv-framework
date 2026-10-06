"""Planning a resolved bundle: what a run writes and in what order, checked before anything is written."""

import json
import os
from typing import Literal

import pytest

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.vm_image import VMImageArtifact
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import BundleError, BundleKind, parse_bundle
from agent_env.bundle import plan as plan_module
from agent_env.bundle import resolve as resolve_module
from agent_env.bundle.plan import check_bundle, plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.config import configure
from agent_env.entity_refs import EntityKind, EntityRef
from agent_env.env.env import Env
from agent_env.env.store import get_env_store, reset_env_store
from agent_env.store import reset_config, set_document_store
from agent_env.task import Task
from agent_env.task.store import get_task_store, reset_task_store
from tst.unit.bundle._support import RefusingStore
from tst.unit.store.fakes import FakeDocumentStore

ROOT = "@local/~/triage"
LAYOUT = {
    "envs/tickets/Dockerfile": "FROM scratch\n",
    "envs/tickets/env.toml": 'environment_name = "tickets"\n',
    "agents/solver/Dockerfile": "FROM scratch\n",
    "artifacts/greeting/hello.txt": "hello\n",
    "artifacts/greeting/check.py": "print('ok')\n",
    "artifacts/base-mcp/Dockerfile": "FROM scratch\n",
    "skills/pdf/SKILL.md": "---\nname: pdf\n---\n",
}


class _Composite(Env):
    """An env type whose toml names other envs and images."""

    type = "composite_test"
    toml_refs = (EntityRef.env("mcp_server_envs[]"), EntityRef.artifact("image", artifact_type="docker_image"))


class _NoteEnv(Env):
    """An env in the store that is only a document."""

    type = "note_test"

    def to_dict(self):
        return {**super().to_dict(), "type": self.type}

    @classmethod
    def from_dict(cls, data):
        return cls(id=data["id"], version=data.get("version"))


class _PluginFile(FileArtifact):
    """A plugin's file type that writes as FileArtifact does."""

    type: Literal["plugin_file_test"] = "plugin_file_test"


class _OwnFile(FileArtifact):
    """A plugin's file type with its own from_toml, a static method."""

    type: Literal["own_file_test"] = "own_file_test"

    @staticmethod
    def from_toml(data, ctx):
        raise NotImplementedError


class _UnreadableStep:
    """A step type whose stored config no longer reads."""

    @classmethod
    def from_dict(cls, data):
        raise KeyError("env_id")


@pytest.fixture(autouse=True)
def stores(monkeypatch):
    set_document_store(FakeDocumentStore())
    envs = resolve_module.get_env_registry()
    monkeypatch.setattr(resolve_module, "get_env_registry", lambda: {**envs, _Composite.type: _Composite})
    monkeypatch.setattr("agent_env.env.registry.get_env_registry", lambda: {_NoteEnv.type: _NoteEnv})
    yield
    reset_artifact_store()
    reset_env_store()
    reset_task_store()
    reset_config()


@pytest.fixture
def make(tmp_path, monkeypatch):
    """A bundle at ``~/triage`` with the fixed layout, plus the given tasks, evals and files."""
    monkeypatch.setenv("HOME", str(tmp_path))

    def make(tasks=None, evals=None, files=None):
        spec = {**LAYOUT, **(files or {})}
        spec.update({f"tasks/{name}.json": json.dumps(steps) for name, steps in (tasks or {}).items()})
        spec.update({f"evals/{name}.toml": text for name, text in (evals or {}).items()})
        for rel, text in spec.items():
            path = tmp_path / "triage" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        return resolve_bundle(parse_bundle(tmp_path / "triage"))

    return make


def deploy(env_id, step_id="deploy", **fields):
    return {"id": step_id, "type": "deploy_env", "env_id": env_id, **fields}


def snapshot(snapshot_id, step_id="snap"):
    return {"id": step_id, "type": "snapshot_env", "env_id": "tickets", "snapshot_id": snapshot_id}


def load(artifact_id, step_id="load"):
    return {"id": step_id, "type": "load_artifact", "env_id": "tickets", "artifact_id": artifact_id}


def composite(**keys):
    return "".join([f'type = "{_Composite.type}"\n', *(f"{key} = {json.dumps(value)}\n" for key, value in keys.items())])


def put_env(env_id):
    get_env_store().put_document(_NoteEnv(id=env_id, version=None))


def writes(plan):
    return [(write.kind, write.id) for write in plan.writes]


def names(entries):
    return [entry.entry.name for entry in entries]


def problems(resolved, **selection) -> tuple[str, ...]:
    with pytest.raises(BundleError) as caught:
        plan_bundle(resolved, **selection)
    return caught.value.problems


# Selection


def test_a_bundle_without_evals_runs_every_task_and_writes_only_what_they_reach(make):
    plan = plan_bundle(make(tasks={
        "t": [deploy("tickets"), load("greeting"),
              {"id": "agent", "type": "deploy_agent", "env_ids": ["tickets"], "a2a_agent_id": "solver"}],
    }))
    assert writes(plan) == [(BundleKind.ARTIFACT, f"{ROOT}/tickets__env_image"), (BundleKind.ENV, f"{ROOT}/tickets"),
                            (BundleKind.ARTIFACT, f"{ROOT}/solver__agent_image"),
                            (BundleKind.AGENT, f"{ROOT}/solver"), (BundleKind.ARTIFACT, f"{ROOT}/greeting"),
                            (BundleKind.TASK, f"{ROOT}/t")]
    assert set(plan.writes[-1].needs) == {("env", f"{ROOT}/tickets"), ("agent", f"{ROOT}/solver"),
                                          ("artifact", f"{ROOT}/greeting")}
    assert (names(plan.tasks), plan.evals, plan.store_refs) == (["t"], (), ())


def test_a_bundle_with_evals_runs_every_eval_and_writes_the_tasks_they_name(make):
    plan = plan_bundle(make(tasks={"t": [deploy("tickets")], "u": [deploy("tickets")]},
                            evals={"regression": 'tasks = ["t"]'}))
    assert writes(plan) == [(BundleKind.ARTIFACT, f"{ROOT}/tickets__env_image"), (BundleKind.ENV, f"{ROOT}/tickets"),
                            (BundleKind.TASK, f"{ROOT}/t"),
                            (BundleKind.EVAL, f"{ROOT}/regression")]
    assert plan.writes[-1].needs == (("task", f"{ROOT}/t"),)
    assert (plan.tasks, names(plan.evals)) == ((), ["regression"])


def test_tasks_and_evals_are_selected_by_name_or_id_once_each(make):
    resolved = make(tasks={"t": [deploy("tickets")], "u": []}, evals={"regression": 'tasks = ["t"]'})
    plan = plan_bundle(resolved, tasks=["u", f"{ROOT}/u", "t"], evals=[f"{ROOT}/regression"])
    assert (names(plan.tasks), names(plan.evals)) == (["u", "t"], ["regression"])
    assert [write.id for write in plan.writes if write.kind is BundleKind.TASK] == [f"{ROOT}/t", f"{ROOT}/u"]


@pytest.mark.parametrize(("selection", "problem"), [
    ({"tasks": ["T"]}, "--task 'T': isn't in this bundle, but 't' is; names are case-sensitive"),
    ({"tasks": ["nope"]}, "--task 'nope': this bundle has no task with that name or id; its tasks are 't', 'u'"),
    ({"evals": ["regression"]}, "--eval 'regression': this bundle has no eval with that name or id; it has no evals"),
])
def test_a_selection_outside_the_bundle_is_refused(make, selection, problem):
    assert problems(make(tasks={"t": [], "u": []}), **selection) == (problem,)


def test_a_bundle_with_nothing_to_run_is_refused(make):
    assert problems(make()) == ("this bundle has no tasks or evals to run",)
    with pytest.raises(BundleError) as caught:
        check_bundle(make())
    assert caught.value.problems == ("this bundle has no tasks or evals to run",)


# Order


def test_each_write_comes_after_what_it_references(make):
    plan = plan_bundle(make(tasks={"t": [deploy("both")]}, files={
        "envs/both/Dockerfile": "FROM scratch\n", "envs/both/env.toml": composite(mcp_server_envs=["tickets"]),
    }))
    assert writes(plan) == [(BundleKind.ARTIFACT, f"{ROOT}/both__env_image"),
                            (BundleKind.ARTIFACT, f"{ROOT}/tickets__env_image"), (BundleKind.ENV, f"{ROOT}/tickets"),
                            (BundleKind.ENV, f"{ROOT}/both"), (BundleKind.TASK, f"{ROOT}/t")]
    assert plan.writes[0].needs == ()
    assert plan.writes[3].needs == (("artifact", f"{ROOT}/both__env_image"), ("env", f"{ROOT}/tickets"))


@pytest.mark.parametrize(("files", "problem"), [
    ({"envs/a/env.toml": composite(mcp_server_envs=["b"], image="base-mcp"),
      "envs/b/env.toml": composite(mcp_server_envs=["a"], image="base-mcp")},
     "envs/a: references form a loop: envs/a -> envs/b -> envs/a"),
    ({"envs/a/env.toml": composite(mcp_server_envs=["a"], image="base-mcp")},
     "envs/a: references form a loop: envs/a -> envs/a"),
])
def test_a_reference_loop_is_refused_even_outside_the_selection(make, files, problem):
    assert problems(make(tasks={"t": [deploy("tickets")]}, files=files), tasks=["t"]) == (problem,)


# One writer per id


@pytest.mark.parametrize(("tasks", "files", "problem"), [
    ({"t": [snapshot("snap")]}, {"artifacts/report/artifact.toml": f'id = "{ROOT}/t/snap"\n',
                                 "artifacts/report/data.txt": "x"},
     f"tasks/t.json: step 'snap': '{ROOT}/t/snap' is also written by artifacts/report; give each its own id"),
    ({"t": [snapshot("@local/shared/snap")], "u": [snapshot("@local/shared/snap")]}, {},
     "tasks/u.json: step 'snap': '@local/shared/snap' is also written by step 'snap' of tasks/t.json; give each its "
     "own id"),
    ({"t": [snapshot("snap"), snapshot(f"{ROOT}/t/snap", "again")]}, {},
     f"tasks/t.json: step 'again': '{ROOT}/t/snap' is also written by step 'snap' of tasks/t.json; give each its own "
     "id"),
])
def test_an_id_has_one_writer(make, tasks, files, problem):
    assert problems(make(tasks=tasks, files=files)) == (problem,)


# Store references


def test_store_ids_the_selection_reaches_are_read_once_and_kept(make, monkeypatch):
    put_env("crm")
    reads = []
    monkeypatch.setitem(plan_module._GETTERS, EntityKind.ENV, lambda *key: reads.append(key) or Env.get(*key))
    plan = plan_bundle(make(tasks={"t": [deploy("crm")], "u": [deploy("crm")]}))
    assert reads == [("crm", None)]
    ((kind, env_id, version, where),) = [(r.kind, r.id, r.version, r.where) for r in plan.store_refs]
    assert (kind, env_id, version, where) == (EntityKind.ENV, "crm", None, "step 'deploy': env_id")
    assert plan.store_latest == {(EntityKind.ENV, "crm"): 1}


@pytest.mark.parametrize(("steps", "problem"), [
    ([deploy("missing-env")], "step 'deploy': env_id: there is no env 'missing-env' in the store"),
    ([deploy("crm", env_version=2)], "step 'deploy': env_id: there is no env 'crm' version 2 in the store"),
    ([load("tickets")], "step 'load': artifact_id: there is no artifact 'tickets' in the store; this bundle's env "
     "'tickets' has that name"),
    ([load(f"{ROOT}/tickets")], f"step 'load': artifact_id: '{ROOT}/tickets' is this bundle's env 'tickets', but "
     "this field takes an artifact"),
    ([deploy(f"{ROOT}/tickts")], f"step 'deploy': env_id: '{ROOT}/tickts' isn't in this bundle"),
])
def test_a_store_id_must_exist(make, steps, problem):
    put_env("crm")
    assert problems(make(tasks={"t": steps})) == (f"tasks/t.json: {problem}",)


def test_another_bundles_local_id_is_read_from_the_local_namespace_like_a_store_id(make, local_stores, cli_routing):
    put_env("@local/~/other/crm")
    VMImageArtifact.put(id="@local/~/other/golden", description="golden", ecr_url="ecr/x")
    configure(document_store=RefusingStore())

    plan = plan_bundle(make(tasks={"t": [deploy("@local/~/other/crm")]}))

    assert [(ref.kind, ref.id, ref.version) for ref in plan.store_refs] == [(EntityKind.ENV, "@local/~/other/crm", None)]
    assert plan.store_latest == {(EntityKind.ENV, "@local/~/other/crm"): 1}
    assert problems(make(tasks={"t": [deploy("@local/~/other/gone")]})) == (
        "tasks/t.json: step 'deploy': env_id: there is no env '@local/~/other/gone' in the store",
    )
    assert problems(make(tasks={"t": [deploy("both")]}, files={
        "envs/both/env.toml": composite(image="@local/~/other/golden")})) == (
        "envs/both: image: '@local/~/other/golden' is a vm_image in the store, but this field takes docker_image",
    )


def test_an_id_under_the_bundles_own_root_that_it_doesnt_write_is_refused_without_a_read(
        make, local_stores, cli_routing, monkeypatch):
    put_env(f"{ROOT}/crm")

    def read(*_):
        raise AssertionError("the plan read a store")

    monkeypatch.setattr(plan_module, "_GETTERS", dict.fromkeys(plan_module._GETTERS, read))
    assert problems(make(tasks={"t": [deploy(f"{ROOT}/crm")]})) == (
        f"tasks/t.json: step 'deploy': env_id: '{ROOT}/crm' isn't in this bundle",
    )


def test_planning_a_read_of_another_bundles_local_id_without_namespace_routing_is_refused(make):
    with pytest.raises(RuntimeError, match="reading an @local id from the store needs namespace routing"):
        plan_bundle(make(tasks={"t": [deploy("@local/~/other/crm")]}))


def test_an_output_is_only_readable_in_its_own_task(make):
    resolved = make(tasks={"t": [snapshot("snap")], "u": [load(f"{ROOT}/t/snap")]})
    assert problems(resolved, tasks=["u"]) == (
        f"tasks/u.json: step 'load': artifact_id: '{ROOT}/t/snap' is written by step 'snap' of tasks/t.json; refer "
        "to an output by its name, in the task that writes it",
    )


def test_a_store_artifact_must_have_the_type_its_field_takes(make):
    VMImageArtifact.put(id="golden-image", description="golden", ecr_url="ecr/x")
    resolved = make(tasks={"t": [deploy("both")]}, files={"envs/both/env.toml": composite(image="golden-image")})
    assert problems(resolved) == (
        "envs/both: image: 'golden-image' is a vm_image in the store, but this field takes docker_image",
    )


def test_an_eval_may_name_a_store_task_that_must_exist(make):
    assert problems(make(evals={"regression": 'tasks = [{ task = "shared-task", version = 3 }]'})) == (
        "evals/regression.toml: tasks[0].task: there is no task 'shared-task' version 3 in the store",
    )


def test_a_store_id_that_cant_be_read_is_reported(make, monkeypatch):
    put_env("crm")
    monkeypatch.setattr("agent_env.env.registry.get_env_registry", lambda: {})
    (problem,) = problems(make(tasks={"t": [deploy("crm")]}))
    assert problem.startswith("tasks/t.json: step 'deploy': env_id: env 'crm' can't be read (ValueError: Unknown env "
                              "type: note_test")


def test_a_store_task_whose_steps_cant_be_read_is_reported(make, monkeypatch):
    get_task_store().put_document(Task.from_dict({"id": "shared-task", "steps": [deploy("tickets")]}))
    monkeypatch.setattr("agent_env.task_step.registry.get_task_step_registry", lambda: {"deploy_env": _UnreadableStep})
    assert problems(make(evals={"regression": 'tasks = ["shared-task"]'})) == (
        "evals/regression.toml: tasks[0]: task 'shared-task' can't be read (KeyError: 'env_id')",
    )


# Checked without a store


def test_a_check_covers_every_task_and_eval_and_reads_no_store(make, monkeypatch):
    def read(*_):
        raise AssertionError("the check read a store")

    monkeypatch.setattr(plan_module, "_GETTERS", dict.fromkeys(plan_module._GETTERS, read))
    check_bundle(make(tasks={"t": [deploy("tickets")], "u": [deploy("crm")]}, evals={"e": 'tasks = ["t"]\n'}))

    resolved = make(tasks={"u": [deploy("tickets", "d"), deploy("tickets", "d")], "v": [deploy("@local/~/other/crm")]},
                    files={"artifacts/note/n.txt": "n", "artifacts/note/artifact.toml": 'descripton = "typo"\n'})
    with pytest.raises(BundleError) as caught:
        check_bundle(resolved)
    assert sorted(caught.value.problems) == [
        "artifacts/note: artifact.toml: unknown key 'descripton'; a file artifact takes description, type and id",
        "tasks/u.json: Duplicate step id 'd' at position 1; step ids must be unique within a task",
    ]


# Tasks built before any write


@pytest.mark.parametrize(("steps", "problem"), [
    ([{"id": "cli", "type": "build_mcp_cli", "env_id": "tickets", "cli_artifact_id": "tickets-cli"}],
     "step 'cli': build_mcp_cli can't read it (KeyError: 'command_name')"),
    ([deploy("tickets", "d"), deploy("tickets", "d")],
     "Duplicate step id 'd' at position 1; step ids must be unique within a task"),
])
def test_a_task_whose_steps_dont_build_is_refused(make, steps, problem):
    assert problems(make(tasks={"t": steps})) == (f"tasks/t.json: {problem}",)


def test_a_file_artifact_the_selection_reaches_is_listed_before_any_write(make, tmp_path):
    resolved = make(tasks={"t": [load("greeting")], "u": [deploy("tickets")]})
    outside = tmp_path / "outside.txt"
    outside.write_text("s")
    (tmp_path / "triage" / "artifacts" / "greeting" / "sub").mkdir()
    (tmp_path / "triage" / "artifacts" / "greeting" / "sub" / "leak.txt").symlink_to(outside)
    assert problems(resolved, tasks=["t"]) == (
        f"artifacts/greeting/sub/leak.txt: links to {os.path.realpath(outside)}, outside the bundle; copy what it "
        "points to into the bundle, or put it in a store and refer to it by id",
    )
    assert names(plan_bundle(resolved, tasks=["u"]).tasks) == ["u"]


def test_a_plugin_file_type_is_listed_when_it_writes_as_file_artifact_does(make, monkeypatch, tmp_path):
    artifacts = resolve_module.get_artifact_registry()
    registry = {**artifacts, _PluginFile.model_fields["type"].default: _PluginFile,
                _OwnFile.model_fields["type"].default: _OwnFile}
    monkeypatch.setattr(resolve_module, "get_artifact_registry", lambda: registry)
    monkeypatch.setattr(plan_module, "get_artifact_registry", lambda: registry)
    resolved = make(tasks={"t": [load("mine", "l1"), load("theirs", "l2")]}, files={
        "artifacts/mine/artifact.toml": 'type = "plugin_file_test"\n', "artifacts/mine/a.txt": "a",
        "artifacts/theirs/artifact.toml": 'type = "own_file_test"\n', "artifacts/theirs/a.txt": "a",
        "artifacts/theirs/b.txt": "b",
    })
    (tmp_path / "triage" / "artifacts" / "mine" / "b.txt").write_text("b")
    assert problems(resolved) == (
        "artifacts/mine: a plugin_file_test artifact holds one file, and this folder has 2 ('a.txt', 'b.txt'); "
        "leave out its declared type to write the folder as a file_artifact_universe",
    )


def test_a_declared_file_artifact_holding_more_than_one_file_is_refused_before_any_write(make):
    resolved = make(tasks={"t": [load("pair")]}, files={
        "artifacts/pair/artifact.toml": 'type = "file"\n', "artifacts/pair/a.txt": "a", "artifacts/pair/b.txt": "b",
    })
    assert problems(resolved) == (
        "artifacts/pair: a file artifact holds one file, and this folder has 2 ('a.txt', 'b.txt'); leave out its "
        "declared type to write the folder as a file_artifact_universe",
    )


def test_every_artifact_toml_key_a_type_doesnt_take_is_refused_before_any_write(make):
    resolved = make(tasks={"t": [load("note", "load-note"), load("count", "load-count"), load("greeting")]}, files={
        "artifacts/note/n.txt": "n", "artifacts/note/artifact.toml": 'descripton = "typo"\n',
        "artifacts/count/c.txt": "c", "artifacts/count/artifact.toml": "description = 3\n",
        "artifacts/greeting/artifact.toml": 'description = "greetings"\n',
    })
    assert sorted(problems(resolved)) == [
        "artifacts/count: artifact.toml: description must be a string, not 3",
        "artifacts/greeting: artifact.toml: unknown key 'description'; a file_artifact_universe artifact takes type "
        "and id",
        "artifacts/note: artifact.toml: unknown key 'descripton'; a file artifact takes description, type and id",
    ]


def test_store_ids_and_steps_outside_the_selection_are_not_checked(make):
    resolved = make(tasks={"t": [deploy("tickets")], "u": [deploy("missing-env"), deploy("tickets")]})
    assert names(plan_bundle(resolved, tasks=["t"]).tasks) == ["t"]
    assert problems(resolved, tasks=["u"]) == (
        "tasks/u.json: Duplicate step id 'deploy' at position 1; step ids must be unique within a task",
        "tasks/u.json: step 'deploy': env_id: there is no env 'missing-env' in the store",
    )


def test_every_problem_is_reported_together(make):
    resolved = make(tasks={"t": [deploy("missing-env")], "u": [snapshot("@local/s")], "w": [snapshot("@local/s")]},
                    files={"envs/a/env.toml": composite(mcp_server_envs=["a"], image="base-mcp")})
    assert problems(resolved, tasks=["t", "nope"]) == (
        "--task 'nope': this bundle has no task with that name or id; its tasks are 't', 'u', 'w'",
        "envs/a: references form a loop: envs/a -> envs/a",
        "tasks/t.json: step 'deploy': env_id: there is no env 'missing-env' in the store",
        "tasks/w.json: step 'snap': '@local/s' is also written by step 'snap' of tasks/u.json; give each its own id",
    )
