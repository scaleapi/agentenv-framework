"""Materializing a planned bundle: what gets written, what is reused, and what is refused before any write."""

import contextlib
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Literal

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import Artifact, EnvironmentArtifact, EnvironmentUniverseArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.artifact.artifacts.file_artifact_universe import FileArtifactUniverse
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError
from agent_env.bundle import materialize as materialize_module
from agent_env.bundle.ledger import LEDGER_COLLECTION, Ledger
from agent_env.bundle.materialize import materialize
from agent_env.bundle.parse import BundleKind
from agent_env.bundle.run import _written
from agent_env.config import configure
from agent_env.config.paths import state_root
from agent_env.config.runtime import Config
from agent_env.entity_refs import EntityRef, RefRole
from agent_env.env.env import Env
from agent_env.eval import Eval, EvalTask
from agent_env.store import Filter
from agent_env.store.ids import image_repository
from agent_env.store.routing import disable_namespace_routing
from agent_env.task import Task
from agent_env.task_step.task_step import TaskStep
from agent_env.utils.build_context import BuildContext
from tst.unit.bundle._support import RefusingStore, layout, local_store, plan_of

ROOT = "@local/~/triage"
LAYOUT = {
    "artifacts/greeting/hello.txt": "hello\n",
    "artifacts/docs/a.md": "a\n",
    "artifacts/docs/b.md": "b\n",
    "tasks/t.json": json.dumps([
        {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"},
        {"id": "greet", "type": "load_artifact", "sandbox_name": "box", "artifact_id": "greeting"},
        {"id": "docs", "type": "load_artifact", "sandbox_name": "box", "artifact_id": "docs"},
    ]),
}


class _Checked(TaskStep):
    """A step reading ``artifact_id``, whose preflight reports whatever ``problems`` holds."""

    type = "checked_materialize_test"
    entity_refs = (EntityRef.artifact("artifact_id"),)
    problems: list[str] = []

    def __init__(self, artifact_id=None, **base):
        super().__init__(**base)
        self.artifact_id = artifact_id

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), artifact_id=data.get("artifact_id"))

    def to_dict(self):
        return {**super().to_dict(), "artifact_id": self.artifact_id}

    def preflight(self):
        return list(self.problems)

    async def execute(self, context):
        return context


class _Writes(_Checked):
    """A step writing ``artifact_id``."""

    type = "writes_materialize_test"
    entity_refs = (EntityRef.artifact("artifact_id", role=RefRole.OUTPUT),)


class _Consuming(_Checked):
    """A step whose from_dict takes ``note`` out of the dict it's given."""

    type = "consuming_materialize_test"
    entity_refs = ()

    @classmethod
    def from_dict(cls, data):
        data.pop("note")
        return super().from_dict(data)


class _PluginFile(FileArtifact):
    """A plugin's file type that writes as FileArtifact does."""

    type: Literal["plugin_file_materialize_test"] = "plugin_file_materialize_test"


class _OwnFile(FileArtifact):
    """A plugin's file type with a from_toml of its own."""

    type: Literal["own_file_materialize_test"] = "own_file_materialize_test"

    @classmethod
    def from_toml(cls, data, ctx):
        raise NotImplementedError


class _OwnEnvironment(EnvironmentArtifact):
    """A plugin's environment type with a from_toml of its own."""

    type: Literal["own_environment_materialize_test"] = "own_environment_materialize_test"

    @classmethod
    def from_toml(cls, data, ctx):
        raise NotImplementedError


class _OwnUniverse(EnvironmentUniverseArtifact):
    """A plugin's universe type with a from_toml of its own."""

    type: Literal["own_universe_materialize_test"] = "own_universe_materialize_test"

    @classmethod
    def from_toml(cls, data, ctx):
        raise NotImplementedError


class _DocumentOnly(Artifact):
    """A plugin's artifact type that is only a document, which a bundle has no writer for yet."""

    type: Literal["document_only_materialize_test"] = "document_only_materialize_test"


class _OwnEnv(Env):
    """A plugin's env type with a from_toml of its own."""

    type = "own_env_materialize_test"

    @classmethod
    def from_toml(cls, data, ctx):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def registries(monkeypatch):
    steps = {**Config().task_step_registry(), **{cls.type: cls for cls in (_Checked, _Writes, _Consuming)}}
    envs = {**Config().env_registry(), _OwnEnv.type: _OwnEnv}
    plugins = {cls.model_fields["type"].default: cls
               for cls in (_PluginFile, _OwnFile, _OwnEnvironment, _OwnUniverse, _DocumentOnly)}
    artifacts = {**Config().artifact_registry(), **plugins}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    monkeypatch.setattr(Config, "env_registry", lambda self: envs)
    monkeypatch.setattr(Config, "artifact_registry", lambda self: artifacts)
    monkeypatch.setattr(_Checked, "problems", [])


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch, local_stores, cli_routing):
    monkeypatch.setenv("HOME", str(tmp_path))
    return layout(tmp_path / "triage", LAYOUT)


_RUN_OR_DRY_RUN = pytest.mark.parametrize("dry_run", [False, True], ids=["run", "dry-run"])


def _steps(root, steps):
    (root / "tasks/t.json").write_text(json.dumps(steps))


def _run(root, dry_run=False):
    return materialize(plan_of(root), dry_run=dry_run)


def _summary(materialization):
    return {done.write.id: (done.version, done.reused, done.reasons) for done in materialization.writes}


def _problems(call):
    with pytest.raises(BundleError) as caught:
        call()
    return caught.value.problems


def test_a_bundle_is_written_to_the_local_namespace_and_nothing_reaches_the_configured_store(bundle_dir):
    plan = plan_of(bundle_dir)
    configure(document_store=RefusingStore())

    first = materialize(plan)

    assert _summary(first) == {id: (1, False, ("new",)) for id in (f"{ROOT}/docs", f"{ROOT}/greeting", f"{ROOT}/t")}
    task = Task.get(f"{ROOT}/t", first.version_of("task", f"{ROOT}/t"))
    assert [step.artifacts[0]["id"] for step in task.steps[1:]] == [f"{ROOT}/greeting", f"{ROOT}/docs"]
    assert len(FileArtifactUniverse.get(f"{ROOT}/docs").get_file_artifacts()) == 2


def test_a_rerun_reuses_every_version_and_an_edit_rewrites_only_what_it_changed(bundle_dir):
    first = _run(bundle_dir)
    assert {done.reused for done in _run(bundle_dir).writes} == {True}
    (bundle_dir / "artifacts/greeting/hello.txt").write_text("hello again\n")

    third = _run(bundle_dir)

    assert _summary(third) == {
        f"{ROOT}/docs": (1, True, ()),
        f"{ROOT}/greeting": (2, False, ("files changed: hello.txt",)),
        f"{ROOT}/t": (1, True, ()),
    }
    assert third.version_of("task", f"{ROOT}/t") == first.version_of("task", f"{ROOT}/t")


@_RUN_OR_DRY_RUN
def test_what_has_no_writer_yet_is_refused_before_anything_is_written(bundle_dir, dry_run):
    layout(bundle_dir, {
        "envs/tickets/Dockerfile": "FROM scratch\n",
        "envs/tickets/env.toml": 'environment_name = "tickets"\n',
        "envs/own/env.toml": 'type = "own_env_materialize_test"\n',
        "agents/solver/Dockerfile": "FROM scratch\n",
        "skills/pdf/SKILL.md": "---\nname: pdf\n---\n",
        "artifacts/base-mcp/Dockerfile": "FROM scratch\n",
        "artifacts/doc/artifact.toml": 'type = "document_only_materialize_test"\n',
        "evals/regression.toml": 'tasks = ["t"]\n',
    })
    _steps(bundle_dir, [
        {"id": "tickets", "type": "deploy_env", "env_id": "tickets"},
        {"id": "own", "type": "deploy_env", "env_id": "own"},
        {"id": "agent", "type": "deploy_agent", "env_ids": ["tickets"], "a2a_agent_id": "solver"},
        {"id": "pdf", "type": "load_artifact", "env_id": "tickets", "artifact_id": "pdf"},
        {"id": "image", "type": "load_artifact", "env_id": "tickets", "artifact_id": "base-mcp"},
        {"id": "doc", "type": "load_artifact", "env_id": "tickets", "artifact_id": "doc"},
    ])

    assert sorted(_problems(lambda: _run(bundle_dir, dry_run))) == [
        "artifacts/base-mcp: writing a docker_image artifact isn't supported yet",
        "artifacts/doc: writing a document_only_materialize_test artifact isn't supported yet",
        "envs/own: writing an own_env_materialize_test env isn't supported yet",
        "skills/pdf: writing a skill isn't supported yet",
    ]
    assert not local_store().path.exists()


AGENT_TOML = """\
image = { artifact = "claude-image", version = 1 }
default_env_vars = { LOG_LEVEL = "debug" }

[metadata]
default_model = "claude-sonnet-4-6"
min_disk_size_gb = 20
"""


def test_an_agent_is_written_from_its_agent_toml_over_a_store_image(bundle_dir):
    _put_image("claude-image")
    layout(bundle_dir, {"agents/solver/agent.toml": AGENT_TOML})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    assert _summary(_run(bundle_dir))[f"{ROOT}/solver"] == (1, False, ("new",))
    agent = A2AAgent.get(f"{ROOT}/solver")
    assert (agent.docker_image_artifact.id, agent.docker_image_artifact.version) == ("claude-image", 1)
    assert (agent.default_env_vars, agent.metadata) == (
        {"LOG_LEVEL": "debug"}, {"default_model": "claude-sonnet-4-6", "min_disk_size_gb": 20})
    assert Task.get(f"{ROOT}/t").steps[0].a2a_agent_id == f"{ROOT}/solver"
    assert _summary(_run(bundle_dir))[f"{ROOT}/solver"] == (1, True, ())


@pytest.mark.parametrize(("toml", "problem"), [
    ('image = "claude-image"\nowner = "me"\n',
     "agent.toml: unknown key 'owner'; an a2a_agent agent takes image, default_env_vars, metadata, type and id"),
    ('image = "claude-image"\ndefault_env_vars = "LOG_LEVEL=debug"\n',
     "agent.toml: default_env_vars must be a table, not 'LOG_LEVEL=debug'"),
    ('image = "claude-image"\ndefault_env_vars = { PORT = 8080 }\n',
     "agent.toml: default_env_vars.PORT must be a string, not 8080"),
    ('image = "claude-image"\n[metadata]\nowner = "me"\n',
     "agent.toml: unknown [metadata] key 'owner'; an agent's [metadata] takes default_model and min_disk_size_gb"),
    ('image = "claude-image"\n[metadata]\nmin_disk_size_gb = "20"\n',
     "agent.toml: metadata.min_disk_size_gb must be a number, not '20'"),
    ('image = "claude-image"\n[metadata]\nmin_disk_size_gb = true\n',
     "agent.toml: metadata.min_disk_size_gb must be a number, not True"),
    ('image = "no-such-image"\n', "image: there is no artifact 'no-such-image' in the store"),
    ('image = "greeting"\n', "image: 'greeting' is this bundle's file, but this field takes docker_image"),
])
def test_an_agent_toml_the_agent_cant_take_is_refused_before_anything_is_written(bundle_dir, toml, problem):
    _put_image("claude-image")
    layout(bundle_dir, {"agents/solver/agent.toml": toml})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    assert _problems(lambda: _run(bundle_dir)) == (f"agents/solver: {problem}",)
    assert not local_store().path.exists()


def test_every_problem_in_an_agent_toml_is_refused_together(bundle_dir):
    layout(bundle_dir, {"agents/solver/agent.toml": 'image = "claude-image"\nowner = "me"\n'
                        'default_env_vars = { PORT = 8080 }\n[metadata]\ncolor = "red"\n'})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    _put_image("claude-image")

    assert sorted(_problems(lambda: _run(bundle_dir))) == [
        "agents/solver: agent.toml: default_env_vars.PORT must be a string, not 8080",
        "agents/solver: agent.toml: unknown [metadata] key 'color'; an agent's [metadata] takes default_model and "
        "min_disk_size_gb",
        "agents/solver: agent.toml: unknown key 'owner'; an a2a_agent agent takes image, default_env_vars, "
        "metadata, type and id",
    ]


@pytest.mark.parametrize("image", ['"claude-image"', '{ artifact = "claude-image" }'])
def test_an_unpinned_store_image_is_written_at_the_version_the_plan_checked(bundle_dir, image):
    _put_image("claude-image")
    layout(bundle_dir, {"agents/solver/agent.toml": f"image = {image}\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    plan = plan_of(bundle_dir)
    _put_image("claude-image")

    materialize(plan)

    assert A2AAgent.get(f"{ROOT}/solver").docker_image_artifact.version == 1


def test_another_bundles_entities_are_read_like_store_ids_and_an_agent_pins_the_version_it_read(bundle_dir):
    other = layout(bundle_dir.parent / "other", LAYOUT)
    _run(other)
    image = "@local/~/other/solver-image"
    _put_image(image)
    box, greet, _ = json.loads(LAYOUT["tasks/t.json"])
    layout(bundle_dir, {"agents/solver/agent.toml": f'image = "{image}"\n'})
    _steps(bundle_dir, [box, {**greet, "artifact_id": "@local/~/other/greeting"},
                        {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    assert _summary(_run(bundle_dir)) == {f"{ROOT}/solver": (1, False, ("new",)), f"{ROOT}/t": (1, False, ("new",))}
    assert Task.get(f"{ROOT}/t").steps[1].artifacts == [{"id": "@local/~/other/greeting", "version": None}]
    _put_image(image)
    (other / "artifacts/greeting/hello.txt").write_text("hello again\n")
    _run(other)

    assert _summary(_run(bundle_dir)) == {
        f"{ROOT}/solver": (2, False, (f"artifact {image} has a new version in the store (v1 → v2)",)),
        f"{ROOT}/t": (1, True, ()),
    }
    assert A2AAgent.get(f"{ROOT}/solver").docker_image_artifact.version == 2


def test_an_agent_folder_with_a_dockerfile_is_built_and_the_agent_written_over_its_image(bundle_dir, builds):
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\nCOPY run.sh /\n", "agents/solver/run.sh": "echo hi\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    image = f"{ROOT}/solver__agent_image"
    announced = []

    first = materialize(plan_of(bundle_dir), on_build=lambda write: announced.append(write.id))

    tag = f"{image_repository(image)}:bundle"
    assert builds == [{"build": ("Dockerfile", ["Dockerfile", "run.sh"], tag, None),
                       "put": (image, tag, ["Dockerfile", "run.sh"], "Dockerfile")}]
    assert announced == [image]
    assert {id: summary[:2] for id, summary in _summary(first).items() if "solver" in id} == {
        image: (1, False), f"{ROOT}/solver": (1, False)}
    agent = A2AAgent.get(f"{ROOT}/solver")
    assert (agent.docker_image_artifact.id, agent.docker_image_artifact.version) == (image, 1)

    again = materialize(plan_of(bundle_dir), on_build=lambda write: announced.append(write.id))

    assert len(builds) == 1 and announced == [image]
    assert _summary(again)[image] == (1, True, ()) and _summary(again)[f"{ROOT}/solver"] == (1, True, ())


def test_an_image_is_built_from_the_files_the_ledger_hashes_so_a_reused_one_matches_them(bundle_dir, builds):
    layout(bundle_dir, {
        "agents/solver/Dockerfile": "FROM scratch\nCOPY . /agent\n",
        "agents/solver/agent.toml": '[metadata]\ndefault_model = "claude-sonnet-4-6"\n',
        "agents/solver/.dockerignore": "notes.md\n",
        "agents/solver/lib/run.sh": "echo hi\n",
        "agents/solver/__pycache__/run.cpython-312.pyc": "compiled",
        "agents/solver/.DS_Store": "finder",
    })
    (bundle_dir / "agents/solver/run.sh").symlink_to("lib/run.sh")
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    _run(bundle_dir)
    (bundle_dir / "agents/solver/__pycache__/run.cpython-312.pyc").write_text("recompiled")
    (bundle_dir / "agents/solver/.DS_Store").write_text("moved")
    again = _summary(_run(bundle_dir))

    context = [".dockerignore", "Dockerfile", "agent.toml", "lib/run.sh", "run.sh"]
    assert [call["build"][1] for call in builds] == [context]
    assert builds[0]["put"][2] == context
    assert again[f"{ROOT}/solver__agent_image"] == (1, True, ())


def test_a_changed_build_context_rebuilds_the_image_and_rewrites_its_agent(bundle_dir, builds):
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\nCOPY run.sh /\n", "agents/solver/run.sh": "echo hi\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    _run(bundle_dir)
    (bundle_dir / "agents/solver/run.sh").write_text("echo bye\n")

    rebuilt = _summary(_run(bundle_dir))

    assert len(builds) == 2
    assert rebuilt[f"{ROOT}/solver__agent_image"][:2] == (2, False)
    assert rebuilt[f"{ROOT}/solver"][:2] == (2, False)
    assert A2AAgent.get(f"{ROOT}/solver").docker_image_artifact.version == 2


def test_an_image_written_as_its_build_context_is_uploaded_with_no_build_and_no_docker(bundle_dir, builds,
                                                                                       monkeypatch):
    monkeypatch.setattr(materialize_module.shutil, "which", lambda name: None)
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\nCOPY run.sh /\n", "agents/solver/run.sh": "echo hi\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    image = f"{ROOT}/solver__agent_image"
    announced = []

    first = materialize(plan_of(bundle_dir), contexts=frozenset({image}), on_build=lambda write: announced.append(write.id))

    written = DockerImageArtifact.get(image, 1)
    folder = bundle_dir / "agents/solver"
    assert builds == [] and announced == [] and first.contexts == {image}
    assert written.context_only and (written.dockerfile_path, written.platform) == ("Dockerfile", "linux/amd64")
    assert written.source_digest == BuildContext.of(folder, folder / "Dockerfile").source_digest("linux/amd64")
    assert A2AAgent.get(f"{ROOT}/solver").docker_image_artifact.version == 1
    assert _summary(materialize(plan_of(bundle_dir), contexts=frozenset({image})))[image] == (1, True, ())


def test_an_image_switched_between_built_here_and_a_build_context_is_rewritten_with_its_agent(bundle_dir, builds):
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\nCOPY run.sh /\n", "agents/solver/run.sh": "echo hi\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    image = f"{ROOT}/solver__agent_image"
    _run(bundle_dir)

    as_context = _summary(materialize(plan_of(bundle_dir), contexts=frozenset({image})))
    built_again = _summary(_run(bundle_dir))

    assert as_context[image] == (2, False, ("now written as a build context, built where it runs, not an image built "
                                            "on this machine",))
    assert as_context[f"{ROOT}/solver"] == (2, False, (f"artifact {image} is written anew (v1 → v2)",))
    assert built_again[image] == (3, False, ("now written as an image built on this machine, not a build context, "
                                             "built where it runs",))
    assert len(builds) == 2


def test_a_dry_run_predicts_a_rebuild_and_its_agents_rewrite_without_building(bundle_dir, builds):
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\nCOPY run.sh /\n", "agents/solver/run.sh": "echo hi\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    _run(bundle_dir)
    (bundle_dir / "agents/solver/run.sh").write_text("echo bye\n")
    image = f"{ROOT}/solver__agent_image"

    predicted = _summary(_run(bundle_dir, dry_run=True))

    assert len(builds) == 1
    assert predicted[image] == (2, False, ("files changed: run.sh",))
    assert predicted[f"{ROOT}/solver"] == (2, False, (f"artifact {image} is written anew (v1 → v2)",))
    assert _summary(_run(bundle_dir)) == predicted
    assert len(builds) == 2


def test_an_agent_toml_edit_rebuilds_the_image_too_since_a_dockerfile_can_copy_it(bundle_dir, builds):
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\nCOPY . /agent\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    _run(bundle_dir)
    (bundle_dir / "agents/solver/agent.toml").write_text('[metadata]\ndefault_model = "claude-sonnet-4-6"\n')

    edited = _summary(_run(bundle_dir))

    assert len(builds) == 2
    assert edited[f"{ROOT}/solver__agent_image"][:3] == (2, False, ("files added: agent.toml",))
    assert edited[f"{ROOT}/solver"][:2] == (2, False)


def test_a_failed_build_is_a_bundle_problem_naming_the_agent(bundle_dir, builds, monkeypatch):
    def fail(dockerfile, context, tag, *, platform):
        output = "\n".join(f"#{n} step" for n in range(60))
        raise materialize_module.DockerBuildError(f"docker build of {tag} failed (exit 1):\n{output}\nERROR: failed to solve")

    monkeypatch.setattr(materialize_module, "build_image", fail)
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    (problem,) = _problems(lambda: _run(bundle_dir))
    lines = problem.splitlines()
    assert lines[0].startswith("agents/solver: docker build of ") and lines[0].endswith(" failed (exit 1):")
    assert lines[1:] == [*(f"#{n} step" for n in range(21, 60)), "ERROR: failed to solve"]


def test_a_failed_push_is_a_bundle_problem_naming_the_agent(bundle_dir, builds, monkeypatch):
    def fail(id, **kwargs):
        raise RuntimeError("could not start the local registry on port 5000: port is already allocated")

    monkeypatch.setattr(materialize_module.DockerImageArtifact, "put", fail)
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    assert _problems(lambda: _run(bundle_dir)) == (
        "agents/solver: could not start the local registry on port 5000: port is already allocated",)


@_RUN_OR_DRY_RUN
def test_an_image_to_build_without_docker_on_path_is_refused_before_anything_is_written(
    bundle_dir, monkeypatch, dry_run,
):
    monkeypatch.setattr(materialize_module.shutil, "which", lambda name: None)
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    assert _problems(lambda: _run(bundle_dir, dry_run)) == (
        "agents/solver: building its image from Dockerfile needs docker, and it isn't on PATH",)
    assert not local_store().path.exists()


def test_whether_an_image_needs_docker_is_decided_once_another_run_writing_it_is_done(bundle_dir, builds, monkeypatch):
    """Another run may be building the image; once its lock is released, the ledger can reuse what it built."""
    events = []
    locked = materialize_module.materializing

    @contextlib.contextmanager
    def materializing(bundle, on_wait=None):
        with locked(bundle, on_wait):
            events.append("locked")
            yield

    monkeypatch.setattr(materialize_module, "materializing", materializing)
    check = materialize_module._refuse_builds_without_docker
    monkeypatch.setattr(materialize_module, "_refuse_builds_without_docker",
                        lambda plan, ledger, contexts: events.append("docker checked") or check(plan, ledger, contexts))
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])

    _run(bundle_dir)

    assert events == ["locked", "docker checked"]


@_RUN_OR_DRY_RUN
def test_an_image_the_ledger_reuses_needs_no_docker(bundle_dir, builds, monkeypatch, dry_run):
    layout(bundle_dir, {"agents/solver/Dockerfile": "FROM scratch\n"})
    _steps(bundle_dir, [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "solver"}])
    _run(bundle_dir)
    monkeypatch.setattr(materialize_module.shutil, "which", lambda name: None)

    assert _summary(_run(bundle_dir, dry_run))[f"{ROOT}/solver__agent_image"] == (1, True, ())
    assert len(builds) == 1


def _put_image(id):
    get_artifact_store().put_document(DockerImageArtifact(
        id=id, description=id, image_name=f"{id}:v1", tar_gz_s3_url=f"file:///{id}.tar.gz"))


def test_a_plugin_file_type_is_written_when_it_writes_as_file_artifact_does(bundle_dir):
    (bundle_dir / "artifacts/greeting/artifact.toml").write_text('type = "plugin_file_materialize_test"\n')

    assert _summary(_run(bundle_dir))[f"{ROOT}/greeting"] == (1, False, ("new",))
    assert Artifact.get(f"{ROOT}/greeting").type == "plugin_file_materialize_test"
    assert _summary(_run(bundle_dir))[f"{ROOT}/greeting"] == (1, True, ())


@pytest.mark.parametrize("type_, written", [
    ("own_file_materialize_test", "an own_file_materialize_test artifact"),
    ("own_environment_materialize_test", "an own_environment_materialize_test artifact"),
    ("own_universe_materialize_test", "an own_universe_materialize_test artifact"),
])
def test_a_plugin_type_with_a_from_toml_of_its_own_is_refused(bundle_dir, type_, written):
    (bundle_dir / "artifacts/greeting/artifact.toml").write_text(f'type = "{type_}"\n')

    assert _problems(lambda: _run(bundle_dir)) == (f"artifacts/greeting: writing {written} isn't supported yet",)


def test_an_eval_is_rewritten_only_when_it_changes(bundle_dir):
    for _ in range(2):
        Task.put(id="shared", steps=[])
    layout(bundle_dir, {"evals/regression.toml": 'tasks = ["t", { task = "shared", version = 1 }]\n'})

    assert _summary(_run(bundle_dir))[f"{ROOT}/regression"] == (1, False, ("new",))
    assert Eval.get(f"{ROOT}/regression").tasks == [EvalTask(f"{ROOT}/t"), EvalTask("shared", 1)]
    steps = json.loads(LAYOUT["tasks/t.json"])
    _steps(bundle_dir, [*steps[:-1], {**steps[-1], "id": "load-docs"}])
    rewritten = _summary(_run(bundle_dir))
    assert (rewritten[f"{ROOT}/t"][:2], rewritten[f"{ROOT}/regression"]) == ((2, False), (1, True, ()))
    (bundle_dir / "evals/regression.toml").write_text('tasks = ["t", { task = "shared", version = 2 }]\n')
    assert _summary(_run(bundle_dir))[f"{ROOT}/regression"] == (2, False, ("config changed",))
    assert Eval.get(f"{ROOT}/regression").tasks == [EvalTask(f"{ROOT}/t"), EvalTask("shared", 2)]


def test_evals_are_written_after_the_tasks(bundle_dir, monkeypatch):
    layout(bundle_dir, {"evals/regression.toml": 'tasks = ["t"]\n'})
    order = []
    put_task, put_eval = Task.put.__func__, Eval.put.__func__
    monkeypatch.setattr(Task, "put", classmethod(lambda cls, **fields: order.append("task") or put_task(cls, **fields)))
    monkeypatch.setattr(Eval, "put", classmethod(lambda cls, **fields: order.append("eval") or put_eval(cls, **fields)))

    _run(bundle_dir)

    assert order == ["task", "eval"]


def test_no_eval_is_written_when_a_task_fails_preflight(bundle_dir, monkeypatch):
    _steps(bundle_dir, [{"id": "check", "type": "checked_materialize_test", "artifact_id": "greeting"}])
    layout(bundle_dir, {"evals/regression.toml": 'tasks = ["t"]\n'})
    monkeypatch.setattr(_Checked, "problems", ["the script isn't there"])

    assert _problems(lambda: _run(bundle_dir)) == ("tasks/t.json: the script isn't there",)
    assert local_store().query("evals", Filter()) == []
    assert {row["id"] for row in local_store().query(LEDGER_COLLECTION, Filter())} == {f"{ROOT}/greeting"}


def test_an_eval_may_share_its_tasks_name(bundle_dir):
    layout(bundle_dir, {"evals/t.toml": 'tasks = ["t"]\n'})

    done = _run(bundle_dir)

    assert (done.version_of("task", f"{ROOT}/t"), done.version_of("eval", f"{ROOT}/t")) == (1, 1)
    assert Eval.get(f"{ROOT}/t").tasks == [EvalTask(f"{ROOT}/t")]


@_RUN_OR_DRY_RUN
def test_without_namespace_routing_nothing_is_written(bundle_dir, dry_run):
    plan = plan_of(bundle_dir)
    disable_namespace_routing()

    with pytest.raises(RuntimeError, match="needs namespace routing"):
        materialize(plan, dry_run=dry_run)
    assert not local_store().path.exists()


def test_tasks_are_preflighted_together_after_the_entities_and_none_is_written_if_one_fails(bundle_dir, monkeypatch):
    _steps(bundle_dir, [{"id": "check", "type": "checked_materialize_test", "artifact_id": "greeting"}])
    (bundle_dir / "tasks/u.json").write_text(json.dumps([{"id": "check", "type": "checked_materialize_test"}]))
    monkeypatch.setattr(_Checked, "problems", ["the script isn't there"])

    assert _problems(lambda: _run(bundle_dir)) == (
        "tasks/t.json: the script isn't there", "tasks/u.json: the script isn't there",
    )
    ledgered = {row["id"] for row in local_store().query(LEDGER_COLLECTION, Filter.of(status="done"))}
    assert ledgered == {f"{ROOT}/greeting"}
    assert local_store().query("tasks", Filter()) == []

    monkeypatch.setattr(_Checked, "problems", [])
    fixed = _summary(_run(bundle_dir))
    assert fixed[f"{ROOT}/greeting"] == (1, True, ())
    assert fixed[f"{ROOT}/t"] == (1, False, ("new",))


@_RUN_OR_DRY_RUN
def test_a_task_the_ledger_would_reuse_is_still_preflighted(bundle_dir, monkeypatch, dry_run):
    """In a dry run too: the store holds the reused artifact the step reads, as the run will read it."""
    _steps(bundle_dir, [{"id": "check", "type": "checked_materialize_test", "artifact_id": "greeting"}])
    _run(bundle_dir)
    monkeypatch.setattr(_Checked, "problems", ["the script is gone"])

    assert _problems(lambda: _run(bundle_dir, dry_run)) == ("tasks/t.json: the script is gone",)


def test_a_step_reading_its_own_tasks_output_is_not_preflighted(bundle_dir, monkeypatch):
    _steps(bundle_dir, [
        {"id": "write", "type": "writes_materialize_test", "artifact_id": "made"},
        {"id": "read", "type": "checked_materialize_test", "artifact_id": "made",
         "depends_on": [{"task_step_id": "write"}]},
    ])
    monkeypatch.setattr(_Writes, "preflight", lambda self: [])
    monkeypatch.setattr(_Checked, "problems", ["the output doesn't exist yet"])

    done = _run(bundle_dir)

    task = Task.get(f"{ROOT}/t", done.version_of("task", f"{ROOT}/t"))
    assert task.steps[1].artifact_id == f"{ROOT}/t/made"


def test_a_bare_step_id_in_depends_on_is_written_in_the_object_form_and_reused(bundle_dir):
    box, greet, docs = json.loads(LAYOUT["tasks/t.json"])
    _steps(bundle_dir, [box, {**greet, "depends_on": ["box"]}, {**docs, "depends_on": ["greet", {"task_step_id": "box"}]}])
    _run(bundle_dir)

    stored = local_store().find_one("tasks", Filter.of(id=f"{ROOT}/t"))
    assert [step.get("depends_on") for step in stored["steps"]] == [
        None, [{"task_step_id": "box"}], [{"task_step_id": "greet"}, {"task_step_id": "box"}]]
    assert {done.reused for done in _run(bundle_dir).writes} == {True}


def test_a_step_writing_an_output_is_still_preflighted(bundle_dir, monkeypatch):
    _steps(bundle_dir, [{"id": "write", "type": "writes_materialize_test", "artifact_id": "made"}])
    monkeypatch.setattr(_Checked, "problems", ["the writer's script isn't there"])

    assert _problems(lambda: _run(bundle_dir)) == ("tasks/t.json: the writer's script isn't there",)


@_RUN_OR_DRY_RUN
def test_a_preflight_that_raises_names_its_task(bundle_dir, monkeypatch, dry_run):
    _steps(bundle_dir, [{"id": "check", "type": "checked_materialize_test"}])

    def unreachable(self):
        raise ConnectionError("the store is unreachable")

    monkeypatch.setattr(_Checked, "preflight", unreachable)
    with pytest.raises(ConnectionError) as caught:
        _run(bundle_dir, dry_run)
    assert caught.value.__notes__ == [f"while preflighting tasks/t.json ({ROOT}/t)"]


def test_a_step_that_changes_the_dict_it_reads_still_has_its_task_reused_and_rewritten_when_edited(bundle_dir):
    _steps(bundle_dir, [{"id": "consume", "type": "consuming_materialize_test", "note": "kept"}])
    _run(bundle_dir)
    assert _summary(_run(bundle_dir))[f"{ROOT}/t"] == (1, True, ())
    _steps(bundle_dir, [{"id": "consume", "type": "consuming_materialize_test", "note": "changed"}])

    assert _summary(_run(bundle_dir))[f"{ROOT}/t"] == (2, False, ("config changed",))


def test_a_failed_write_names_its_entry_and_leaves_the_earlier_writes_reusable(bundle_dir, monkeypatch):
    assert [write.id for write in plan_of(bundle_dir).writes][:2] == [f"{ROOT}/docs", f"{ROOT}/greeting"]
    full = [True]
    write_file = FileArtifact.from_toml.__func__

    def from_toml(cls, data, ctx):
        if full[0]:
            raise OSError("disk full")
        return write_file(cls, data, ctx)

    monkeypatch.setattr(FileArtifact, "from_toml", classmethod(from_toml))
    with pytest.raises(OSError) as caught:
        _run(bundle_dir)
    assert caught.value.__notes__ == [f"while writing artifacts/greeting ({ROOT}/greeting)"]
    full[0] = False

    after = _summary(_run(bundle_dir))
    assert (after[f"{ROOT}/docs"], after[f"{ROOT}/greeting"][:2]) == ((1, True, ()), (1, False))


@_RUN_OR_DRY_RUN
def test_each_write_is_reported_reused_or_not(bundle_dir, dry_run):
    reported = []
    (bundle_dir / "artifacts/greeting/hello.txt").write_text("hello again\n")
    _run(bundle_dir)
    (bundle_dir / "artifacts/greeting/hello.txt").write_text("hello\n")

    done = materialize(plan_of(bundle_dir), dry_run=dry_run, on_write=reported.append)

    assert sorted(reported, key=lambda item: item.write.id) == sorted(done.writes, key=lambda item: item.write.id)
    assert [(item.write.id, item.reused) for item in reported] == [
        (f"{ROOT}/docs", True), (f"{ROOT}/greeting", False), (f"{ROOT}/t", True)]


def test_a_dry_run_predicts_what_materializing_then_writes(bundle_dir):
    def agree():
        predicted = _summary(_run(bundle_dir, dry_run=True))
        assert predicted == _summary(_run(bundle_dir))
        return predicted

    assert set(agree().values()) == {(1, False, ("new",))}
    assert {reused for _, reused, _ in agree().values()} == {True}
    (bundle_dir / "artifacts/docs/b.md").write_text("b2\n")
    assert agree()[f"{ROOT}/docs"] == (2, False, ("files changed: b.md",))
    FileArtifact.put_bytes(f"{ROOT}/greeting", description="by hand", filename="hello.txt", content=b"by hand")
    assert agree()[f"{ROOT}/greeting"] == (3, False, ("the store's latest, v2, wasn't recorded by this bundle",))


def test_a_version_a_run_wrote_before_it_was_interrupted_is_kept_by_the_next_run_and_its_dry_run(
    bundle_dir, monkeypatch,
):
    interrupted, write_artifact, digest = [], materialize_module._WRITERS[BundleKind.ARTIFACT], Ledger.digest

    def written(plan, write):  # the greeting's write lands, and then the run is interrupted
        version = write_artifact(plan, write)
        interrupted.append(write.id == f"{ROOT}/greeting")
        return version

    def digest_until_interrupted(self, *args):
        if any(interrupted):
            raise KeyboardInterrupt
        return digest(self, *args)

    monkeypatch.setitem(materialize_module._WRITERS, BundleKind.ARTIFACT, written)
    monkeypatch.setattr(Ledger, "digest", digest_until_interrupted)
    with pytest.raises(KeyboardInterrupt):
        _run(bundle_dir)
    monkeypatch.setitem(materialize_module._WRITERS, BundleKind.ARTIFACT, write_artifact)
    monkeypatch.setattr(Ledger, "digest", digest)

    kept = (1, True, ("written by an interrupted run that didn't record it",))
    assert _summary(_run(bundle_dir, dry_run=True))[f"{ROOT}/greeting"] == kept
    done = _run(bundle_dir)
    assert _summary(done)[f"{ROOT}/greeting"] == kept
    greeting = next(item for item in done.writes if item.write.id == f"{ROOT}/greeting")
    assert _written(done.plan, greeting) == (
        "artifacts/greeting: v1, unchanged (written by an interrupted run that didn't record it)")
    assert _summary(_run(bundle_dir))[f"{ROOT}/greeting"] == (1, True, ())
    assert [d["version"] for d in local_store().query("artifacts", Filter.of(id=f"{ROOT}/greeting"))] == [1]


def test_a_dry_run_calls_no_writer_and_takes_no_lock(bundle_dir, monkeypatch):
    layout(bundle_dir, {"evals/regression.toml": 'tasks = ["t"]\n'})

    def refused(*_, **__):
        raise AssertionError("a dry run wrote")

    monkeypatch.setattr(Ledger, "record", refused)
    monkeypatch.setattr(materialize_module, "materializing", refused)
    monkeypatch.setattr(Task, "put", refused)
    monkeypatch.setattr(Eval, "put", refused)

    dry = _run(bundle_dir, dry_run=True)

    assert {done.version for done in dry.writes} == {1}
    assert list(state_root().rglob("*")) == []


def test_a_dry_run_doesnt_preflight_a_step_reading_what_it_would_write_and_lists_it(bundle_dir, monkeypatch):
    box, greet, _ = json.loads(LAYOUT["tasks/t.json"])
    _steps(bundle_dir, [
        box, greet, {"id": "check", "type": "checked_materialize_test", "artifact_id": "greeting"},
        {"id": "write", "type": "writes_materialize_test", "artifact_id": "made"},
        {"id": "read", "type": "checked_materialize_test", "artifact_id": "made", "depends_on": [{"task_step_id": "write"}]},
    ])
    monkeypatch.setattr(_Writes, "preflight", lambda self: [])
    monkeypatch.setattr(_Checked, "problems", ["greeting isn't in the store yet"])
    new = _run(bundle_dir, dry_run=True)
    monkeypatch.setattr(_Checked, "problems", [])
    assert _run(bundle_dir).not_preflighted == ()
    (bundle_dir / "artifacts/greeting/hello.txt").write_text("hello again\n")
    monkeypatch.setattr(_Checked, "problems", ["the store holds the greeting it is about to replace"])

    rewritten = _run(bundle_dir, dry_run=True)

    assert [(write.id, step.id) for write, step in new.not_preflighted] == [(f"{ROOT}/t", "check")]
    assert [(write.id, step.id) for write, step in rewritten.not_preflighted] == [(f"{ROOT}/t", "check")]


def test_a_second_run_of_the_bundle_waits_for_the_first_to_finish_writing(bundle_dir, tmp_path):
    held, done, waited = tmp_path / "held", tmp_path / "done", []
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import pathlib, time
        from agent_env.bundle import parse_bundle
        from agent_env.bundle.ledger import materializing
        with materializing(parse_bundle(pathlib.Path({str(bundle_dir)!r}))):
            pathlib.Path({str(held)!r}).write_text("held")
            time.sleep(1.5)
            pathlib.Path({str(done)!r}).write_text("done")
    """)])
    try:
        for _ in range(200):
            if held.exists():
                break
            time.sleep(0.05)
        materialize(plan_of(bundle_dir), on_wait=lambda: waited.append(done.exists()))
        assert done.exists()
        assert waited == [False]
    finally:
        holder.wait()


def test_version_of_names_what_isnt_a_write(bundle_dir):
    with pytest.raises(KeyError, match="isn't one of this plan's writes"):
        _run(bundle_dir).version_of("env", f"{ROOT}/nothing")


def test_a_dry_run_preflights_a_step_over_a_reused_artifact_named_like_a_rewritten_agent(bundle_dir, monkeypatch):
    _put_image("solver-image")
    layout(bundle_dir, {"agents/greeting/agent.toml": 'image = "solver-image"\n'})
    _steps(bundle_dir, [{"id": "check", "type": "checked_materialize_test", "artifact_id": "greeting"},
                        {"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": "greeting"}])
    _run(bundle_dir)
    (bundle_dir / "agents/greeting/agent.toml").write_text('image = "solver-image"\n[default_env_vars]\nA = "1"\n')
    monkeypatch.setattr(_Checked, "problems", ["the script is gone"])

    assert _problems(lambda: _run(bundle_dir, dry_run=True)) == ("tasks/t.json: the script is gone",)
    assert _problems(lambda: _run(bundle_dir)) == ("tasks/t.json: the script is gone",)
