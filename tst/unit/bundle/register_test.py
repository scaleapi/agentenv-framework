"""Registering a bundle: each entry under a registry id, each task and eval naming the versions it reads, a ledger in
the target store so registering again writes only what changed, and every problem refused before anything is
written."""

import json
import subprocess
from dataclasses import fields
from pathlib import Path

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import Artifact, FileArtifact
from agent_env.artifact.artifacts import docker_image as docker_image_module
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact, push_context_image
from agent_env.bundle import BundleError
from agent_env.bundle import materialize as materialize_module
from agent_env.bundle import register as register_module
from agent_env.bundle.register import REGISTRATIONS, _remote_name, register_bundle
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.entity.refs import EntityRef, RefRole
from agent_env.env.env import Env
from agent_env.eval import Eval
from agent_env.bundle.parse import BundleKind, parse_bundle
from agent_env.bundle.plan import Plan, plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.store import ConcurrentModificationError, Filter
from agent_env.store.base import NotFoundError
from agent_env.store.document_store import expecting_version
from agent_env.task import Task
from agent_env.task_step.task_step import TaskStep
from tst.unit.bundle._support import DIGEST, layout

BOX = {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"}
# Not a credential: what a remote URL's user info holds in the tests that its name drops, joined in at run time.
_FAKE_TOKEN = "not-a-token"


def _load(step_id, artifact):
    return {"id": step_id, "type": "load_artifact", "sandbox_name": "box", "artifact_id": artifact}


TRIAGE = {
    "artifacts/greeting/hello.txt": "hello\n",
    "artifacts/docs/a.md": "a\n",
    "tasks/t.json": json.dumps([BOX, _load("greet", "greeting"), _load("docs", "docs")]),
}


class _Writes(TaskStep):
    """A step writing the artifact ``artifact_id``."""

    type = "writes_register_test"
    entity_refs = (EntityRef.artifact("artifact_id", role=RefRole.OUTPUT),)

    def __init__(self, artifact_id=None, **base):
        super().__init__(**base)
        self.artifact_id = artifact_id

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), artifact_id=data.get("artifact_id"))

    def to_dict(self):
        return {**super().to_dict(), "artifact_id": self.artifact_id}

    async def execute(self, context):
        return context


@pytest.fixture(autouse=True)
def steps(monkeypatch):
    registry = {**Config().task_step_registry(), _Writes.type: _Writes}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: registry)


_GIT = register_module._git


@pytest.fixture(autouse=True)
def outside_git(monkeypatch):
    """Each folder is outside git, but the test that says otherwise: none reads a repository it didn't make."""
    monkeypatch.setattr(register_module, "_git", lambda root, *args: None)


@pytest.fixture
def home(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def _steps(task_id, version=None):
    return Task.get(task_id, version).to_dict()["steps"]


def _rows():
    return get_config().get_document_store().query(REGISTRATIONS, Filter.of(status="done"))


def test_each_entry_takes_its_prefix_and_name_and_each_task_names_the_versions_it_reads(home):
    registered = register_bundle(layout(home / "triage", TRIAGE))

    assert {done.write.id: done.version for done in registered.materialization.writes} == {
        "triage-docs": 1, "triage-greeting": 1, "triage-t": 1}
    steps = _steps("triage-t")
    assert [step["artifacts"] for step in steps[1:]] == [[{"id": "triage-greeting", "version": 1}],
                                                         [{"id": "triage-docs", "version": 1}]]
    assert Artifact.get("triage-docs").version == 1


def test_registering_again_writes_nothing_and_an_edit_rewrites_only_what_it_reaches(home):
    root = layout(home / "triage", TRIAGE)
    register_bundle(root)

    again = register_bundle(root)
    (root / "artifacts/docs/a.md").write_text("changed\n")
    edited = register_bundle(root)

    assert all(done.reused for done in again.materialization.writes)
    assert {done.write.id: (done.version, done.reused) for done in edited.materialization.writes} == {
        "triage-docs": (2, False), "triage-greeting": (1, True), "triage-t": (2, False)}
    assert _steps("triage-t")[2]["artifacts"] == [{"id": "triage-docs", "version": 2}]
    assert _steps("triage-t", 1)[2]["artifacts"] == [{"id": "triage-docs", "version": 1}]


def test_a_dry_run_says_what_it_would_write_and_writes_nothing(home):
    planned = register_bundle(layout(home / "triage", TRIAGE), dry_run=True)

    assert {done.write.id: done.version for done in planned.materialization.writes} == {
        "triage-docs": 1, "triage-greeting": 1, "triage-t": 1}
    assert get_config().get_document_store().count("tasks", Filter()) == 0 and _rows() == []


def test_the_ledger_rows_are_in_the_target_store_and_name_the_bundle_by_its_remote_without_credentials(
        local_stores, tmp_path, monkeypatch):
    monkeypatch.setattr(register_module, "_git", _GIT)
    repo = tmp_path / "repo"
    layout(repo / "bundles" / "triage", TRIAGE)
    remote = "https://user:" + _FAKE_TOKEN + "@github.com/team/repo.git"
    for args in (["init", "-q"], ["remote", "add", "origin", remote],
                 ["add", "."], ["-c", "user.email=a@b.c", "-c", "user.name=a", "commit", "-q", "-m", "x"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)

    registered = register_bundle(repo / "bundles" / "triage")

    assert registered.bundle == "github.com/team/repo:bundles/triage"
    rows = _rows()
    assert {row["id"] for row in rows} == {"triage-docs", "triage-greeting", "triage-t"}
    assert {row["bundle"] for row in rows} == {"github.com/team/repo:bundles/triage"}
    assert all(row["commit"] and row["dirty"] is False for row in rows)
    assert _FAKE_TOKEN not in json.dumps(rows, default=str)


@pytest.mark.parametrize(("remote", "name"), [
    ("git@github.com:team/repo.git", "github.com/team/repo"),
    ("https://x-access-token:" + _FAKE_TOKEN + "@github.com/team/repo", "github.com/team/repo"),
    ("ssh://git@gitlab.example.com:2222/group/sub/repo.git", "gitlab.example.com/group/sub/repo"),
])
def test_a_remote_is_named_by_its_host_and_path_alone(remote, name):
    assert _remote_name(remote) == name


def test_an_id_the_store_holds_takes_the_bundle_s_next_version_whoever_wrote_it(home):
    (home / "seed.txt").write_text("seed")
    FileArtifact.put("triage-docs", description="someone else's", file_path=str(home / "seed.txt"))
    root = layout(home / "triage", TRIAGE)

    first = register_bundle(root)
    FileArtifact.put("triage-docs", description="someone else's again", file_path=str(home / "seed.txt"))
    (root / "artifacts/docs/a.md").write_text("changed\n")
    second = register_bundle(root)
    elsewhere = register_bundle(layout(home / "b" / "triage", TRIAGE))

    assert {done.write.id: done.version for done in first.materialization.writes}["triage-docs"] == 2
    assert {done.write.id: done.version for done in second.materialization.writes}["triage-docs"] == 4
    assert {done.write.id: done.version for done in elsewhere.materialization.writes}["triage-docs"] == 5


def test_a_prefix_registers_under_other_ids(home):
    registered = register_bundle(layout(home / "triage", TRIAGE), prefix="team.")

    assert {done.write.id for done in registered.materialization.writes} == {"team.docs", "team.greeting", "team.t"}


def test_a_name_an_id_cant_hold_is_made_one_and_names_that_collide_are_refused(home):
    files = {"artifacts/My Docs/a.md": "a\n", "artifacts/my-docs/b.md": "b\n",
             "tasks/T.json": json.dumps([BOX, _load("one", "My Docs"), _load("two", "my-docs")])}

    with pytest.raises(BundleError) as refused:
        register_bundle(layout(home / "Big Triage", files))

    assert refused.value.problems == ("@local/~/Big Triage/My Docs, @local/~/Big Triage/my-docs would all be "
                                      "registered as 'big-triage-my-docs'; rename one",)


def test_entries_whose_names_make_one_id_register_in_their_own_stores(home):
    files = {"artifacts/Echo/a.md": "a\n", "tasks/echo.json": json.dumps([BOX, _load("greet", "Echo")])}

    registered = register_bundle(layout(home / "triage", files))

    assert {(done.write.kind.store, done.write.id) for done in registered.materialization.writes} == {
        ("artifact", "triage-echo"), ("task", "triage-echo")}


def test_a_translated_plan_renames_every_field_of_a_plan_but_those_that_hold_no_local_id(home):
    plan = plan_bundle(resolve_bundle(parse_bundle(layout(home / "triage", TRIAGE))), tasks=["t"])

    translations = register_module._translations(plan, {})

    assert translations.keys() | register_module._KEPT == {field.name for field in fields(Plan)}
    assert not translations.keys() & register_module._KEPT


def test_a_bad_prefix_is_refused(home):
    with pytest.raises(BundleError) as refused:
        register_bundle(layout(home / "triage", TRIAGE), prefix="Team/")

    assert "--prefix 'Team/'" in refused.value.problems[0]


def test_a_task_output_is_named_after_its_task_and_isnt_pinned(home):
    files = {**TRIAGE, "tasks/t.json": json.dumps([BOX, {"id": "make", "type": _Writes.type, "artifact_id": "report"},
                                                   {**_load("read", "report"), "depends_on": [{"task_step_id": "make"}]}])}

    registered = register_bundle(layout(home / "triage", files))

    assert registered.ids["@local/~/triage/t/report"] == "triage-t-report"
    steps = _steps("triage-t")
    assert steps[1]["artifact_id"] == "triage-t-report"
    assert steps[2]["artifacts"] == [{"id": "triage-t-report", "version": None}]


def test_an_eval_names_the_version_of_each_task(home):
    files = {**TRIAGE, "evals/nightly.toml": 'tasks = ["t"]\n'}
    root = layout(home / "triage", files)
    register_bundle(root)
    (root / "artifacts/docs/a.md").write_text("changed\n")

    register_bundle(root)

    assert [(task.task_id, task.task_version) for task in Eval.get("triage-nightly").tasks] == [("triage-t", 2)]
    assert [(task.task_id, task.task_version) for task in Eval.get("triage-nightly", 1).tasks] == [("triage-t", 1)]


def test_a_store_reference_is_pinned_to_the_version_the_plan_read(home):
    (home / "seed.txt").write_text("seed")
    FileArtifact.put("shared-seed", description="seed", file_path=str(home / "seed.txt"))
    FileArtifact.put("shared-seed", description="seed", file_path=str(home / "seed.txt"))
    files = {"tasks/t.json": json.dumps([BOX, _load("seed", "shared-seed")])}

    register_bundle(layout(home / "triage", files))

    assert _steps("triage-t")[1]["artifacts"] == [{"id": "shared-seed", "version": 2}]


def test_a_step_that_leaves_its_agent_unset_is_registered_so_and_writes_no_default_agent(home):
    files = {"tasks/t.json": json.dumps([BOX, {"id": "agent", "type": "deploy_agent", "env_ids": []}])}

    registered = register_bundle(layout(home / "triage", files))

    assert [done.write.id for done in registered.materialization.writes] == ["triage-t"]
    assert _steps("triage-t")[1].get("a2a_agent_id") is None


def test_a_shipped_agent_named_by_its_alias_is_registered_under_the_alias(home, agents_published):
    files = {"tasks/t.json": json.dumps([BOX, {"id": "agent", "type": "deploy_agent", "env_ids": [],
                                               "a2a_agent_id": "claude-code"}])}

    registered = register_bundle(layout(home / "triage", files))

    written = {done.write.id: done.version for done in registered.materialization.writes}
    assert written == {"claude-code__agent_image": 1, "claude-code": 1, "triage-t": 1}
    step = _steps("triage-t")[1]
    assert (step["a2a_agent_id"], step["a2a_agent_version"]) == ("claude-code", 1)
    image = DockerImageArtifact.get("claude-code__agent_image")
    assert (image.image_name, image.image_digest) == (agents_published, DIGEST)
    assert A2AAgent.get("claude-code").docker_image_artifact.id == "claude-code__agent_image"


def test_an_image_the_bundle_builds_is_pushed_and_registered_by_its_digest_with_its_context(home, monkeypatch):
    pushed = []

    def push(entity_id, context_path, dockerfile_path, before=None):
        pushed.append((entity_id, dockerfile_path))
        return f"registry.example.com/{entity_id}:src-1", DIGEST

    monkeypatch.setattr(materialize_module, "push_context_image", push)
    files = {"envs/crm/Dockerfile": "FROM python:3.12-slim\n", "envs/crm/env.toml": 'environment_name = "crm"\n',
             "tasks/t.json": json.dumps([{"id": "env", "type": "deploy_env", "env_id": "crm"}])}

    register_bundle(layout(home / "triage", files))

    assert [(entity_id, Path(dockerfile).name) for entity_id, dockerfile in pushed] == [
        ("triage-crm__env_image", "Dockerfile")]
    image = DockerImageArtifact.get("triage-crm__env_image")
    assert (image.image_name, image.image_digest) == ("registry.example.com/triage-crm__env_image:src-1", DIGEST)
    assert image.build_context_object_url and image.dockerfile_path == "Dockerfile"
    assert Env.get("triage-crm").docker_image_artifact.id == "triage-crm__env_image"
    step = _steps("triage-t")[0]
    assert (step["env_id"], step["env_version"]) == ("triage-crm", 1)


def test_an_at_local_id_left_in_what_would_be_written_is_refused(home):
    files = {"tasks/t.json": json.dumps([{**BOX, "metadata": {"from": "@local/~/elsewhere/x"}}])}

    with pytest.raises(BundleError) as refused:
        register_bundle(layout(home / "triage", files))

    assert refused.value.problems == (
        "tasks/t.json: [0].metadata.from names '@local/~/elsewhere/x', which only this machine's stores hold",)


def test_a_shared_store_refuses_a_credential_or_reference_in_env_vars_and_a_url_only_this_machine_reaches(
        home, monkeypatch):
    monkeypatch.setattr(register_module, "shared_store", lambda store: True)
    files = {"tasks/t.json": json.dumps([
        {**BOX, "env_vars": {"API_KEY": "abc", "TOKEN": "secret:team_token", "MODE": "fast", "DB_URL": "env:DB_URL"}},
        {"id": "agent", "type": "deploy_agent", "env_ids": [], "metadata": {"wiki": "http://localhost:8080/x"}}])}

    with pytest.raises(BundleError) as refused:
        register_bundle(layout(home / "triage", files))

    assert refused.value.problems == (
        "tasks/t.json: [0].env_vars.API_KEY: its name looks like a secret's and it's set, which a shared store shows "
        "to anyone who reads it; leave it out, for whatever runs the task to give it",
        "tasks/t.json: [0].env_vars.DB_URL: it's a reference, and nothing resolves one in a step's env vars, so the run "
        "would get the reference itself; leave it out, for whatever runs the task to give it",
        "tasks/t.json: [0].env_vars.TOKEN: it's a reference, and nothing resolves one in a step's env vars, so the run "
        "would get the reference itself; leave it out, for whatever runs the task to give it",
        "tasks/t.json: [1].metadata.wiki names http://localhost:8080, which only this machine reaches")


def test_the_cli_says_what_it_would_write_asks_before_a_shared_store_and_dry_run_writes_nothing(home, monkeypatch):
    root = layout(home / "triage", TRIAGE)
    monkeypatch.setattr(register_module, "shared_store", lambda store: True)

    dry = CliRunner().invoke(cli, ["bundle", "register", str(root), "--dry-run"])
    declined = CliRunner().invoke(cli, ["bundle", "register", str(root)], input="n\n")
    written = CliRunner().invoke(cli, ["bundle", "register", str(root), "--yes"])
    again = CliRunner().invoke(cli, ["bundle", "register", str(root), "--yes"])

    assert dry.exit_code == 0, dry.output
    assert "  artifacts/docs -> artifact triage-docs v1 (new)" in dry.output
    assert dry.output.rstrip().endswith("Nothing was written: --dry-run.")
    assert declined.exit_code == 1 and "Write these to the stores above?" in declined.output
    assert written.exit_code == 0, written.output
    assert "Run a registered task with: agent-env task run --id triage-t --version 1" in written.output
    assert "Nothing to write: every version is registered already." in again.output
    assert [task.version for task in [Task.get("triage-t")]] == [1]


def test_a_bundle_with_no_tasks_registers_every_entity(home):
    registered = register_bundle(layout(home / "library", {"artifacts/greeting/hello.txt": "hello\n"}))

    assert [done.write.id for done in registered.materialization.writes] == ["library-greeting"]


def test_a_failed_write_keeps_what_was_written_and_the_next_run_reuses_it(home, monkeypatch):
    root = layout(home / "triage", TRIAGE)
    real = Task.put

    def failing(*args, **kwargs):
        raise RuntimeError("the store went away")

    monkeypatch.setattr(Task, "put", failing)
    with pytest.raises(RuntimeError):
        register_bundle(root)
    monkeypatch.setattr(Task, "put", real)

    retried = register_bundle(root)

    assert {done.write.id: (done.version, done.reused) for done in retried.materialization.writes} == {
        "triage-docs": (1, True), "triage-greeting": (1, True), "triage-t": (1, False)}
    assert Artifact.get("triage-docs").version == 1


class _Store:
    """An image store that records each repository it's asked to set up."""

    def __init__(self):
        self.repositories = []

    def image_ref(self, repository, tag):
        return f"registry.example.com/{repository}:{tag}"

    def ensure_repository(self, repository):
        self.repositories.append(repository)

    def auth(self, ref):
        return None


def test_an_image_is_built_and_pushed_once_under_the_tag_of_its_sources(home, monkeypatch):
    store, built, pushed, served = _Store(), [], [], set()

    def pin(ref, auth):
        if ref not in served:
            raise docker_image_module.NoSuchImageError(f"{ref}: no such image")
        return f"{ref}@{DIGEST}"

    monkeypatch.setattr(get_config(), "get_image_store_for", lambda entity_id: store)
    monkeypatch.setattr(docker_image_module, "pin_digest", pin)
    monkeypatch.setattr(docker_image_module, "build_image",
                        lambda dockerfile, context, tag, platform: built.append((tag, platform)))
    monkeypatch.setattr(docker_image_module, "push_local_image",
                        lambda local, ref, image_store: (pushed.append(ref), served.add(ref)))
    context = layout(home / "context", {"Dockerfile": "FROM python:3.12-slim\n"})

    first = push_context_image("triage-crm__env_image", str(context), "Dockerfile")
    again = push_context_image("triage-crm__env_image", str(context), "Dockerfile")

    ref = first[0]
    assert first == again == (ref, DIGEST)
    assert ref.startswith("registry.example.com/triage-crm-env_image:src-") and "__" not in ref
    assert [platform for _, platform in built] == ["linux/amd64"] and pushed == [ref]
    assert store.repositories == ["triage-crm-env_image", "triage-crm-env_image"]


def test_a_put_held_to_a_version_writes_exactly_it_or_fails(home):
    (home / "seed.txt").write_text("seed")

    with expecting_version("held", 1):
        held = FileArtifact.put("held", description="first", file_path=str(home / "seed.txt"))
    with expecting_version("held", 1), pytest.raises(ConcurrentModificationError):
        FileArtifact.put("held", description="late", file_path=str(home / "seed.txt"))
    with expecting_version("held-task", 2), pytest.raises(ConcurrentModificationError):
        Task.put(id="held-task", steps=[])

    assert held.version == 1 and Artifact.get("held").description == "first"


def test_a_registration_that_loses_a_race_for_a_version_fails_rather_than_write_the_next_one(home, monkeypatch):
    """As a registration of the same bundle on another machine, which no local lock waits for, would."""
    root = layout(home / "triage", TRIAGE)
    writer = materialize_module._WRITERS[BundleKind.ARTIFACT]

    def raced(plan, write):
        if write.id == "triage-docs" and not raced.done:
            raced.done = True
            (home / "other.txt").write_text("the other machine's")
            with expecting_version("triage-docs", 1):
                FileArtifact.put("triage-docs", description="won the race", file_path=str(home / "other.txt"))
        return writer(plan, write)

    raced.done = False
    monkeypatch.setitem(materialize_module._WRITERS, BundleKind.ARTIFACT, raced)

    with pytest.raises(ConcurrentModificationError):
        register_bundle(root)

    assert Artifact.get("triage-docs", 1).description == "won the race"
    with pytest.raises(NotFoundError):
        Artifact.get("triage-docs", 2)


def test_an_installed_env_a_task_names_by_its_alias_is_refused_before_anything_is_written(home):
    files = {"tasks/t.json": json.dumps([{"id": "env", "type": "deploy_env", "env_id": "mail"}])}

    with pytest.raises(BundleError) as refused:
        register_bundle(layout(home / "triage", files))

    assert "comes with an installed bundle, and only an agent with an alias is registered from one" in \
        refused.value.problems[0]
    assert get_config().get_document_store().count("envs", Filter()) == 0


def test_a_write_that_fails_leaves_no_pending_row(home, monkeypatch):
    def failing(*args, **kwargs):
        raise RuntimeError("the store went away")

    monkeypatch.setattr(Task, "put", failing)
    with pytest.raises(RuntimeError):
        register_bundle(layout(home / "triage", TRIAGE))

    assert get_config().get_document_store().query(REGISTRATIONS, Filter.of(status="pending")) == []


def test_a_step_naming_no_version_takes_the_latest_even_where_another_step_pins_one(home):
    (home / "seed.txt").write_text("seed")
    FileArtifact.put("shared-seed", description="seed", file_path=str(home / "seed.txt"))
    FileArtifact.put("shared-seed", description="seed", file_path=str(home / "seed.txt"))
    files = {"tasks/t.json": json.dumps([BOX, _load("latest", "shared-seed"),
                                         {**_load("pinned", "shared-seed"), "artifact_version": 1}])}

    register_bundle(layout(home / "triage", files))

    assert [step["artifacts"] for step in _steps("triage-t")[1:]] == [[{"id": "shared-seed", "version": 2}],
                                                                       [{"id": "shared-seed", "version": 1}]]


class _Killed(BaseException):
    """Stands in for the process dying."""


def test_a_registration_killed_after_a_write_and_before_its_row_is_done_resumes(home, monkeypatch):
    root = layout(home / "triage", TRIAGE)
    digest = register_module.Ledger.digest
    calls = []

    def dies_after_writing_docs(self, write, needs):
        calls.append(write.id)
        if write.id == "triage-docs" and calls.count("triage-docs") == 2:  # the check's, then the one after the write
            raise _Killed
        return digest(self, write, needs)

    monkeypatch.setattr(register_module.Ledger, "digest", dies_after_writing_docs)
    with pytest.raises(_Killed):
        register_bundle(root)
    monkeypatch.setattr(register_module.Ledger, "digest", digest)

    resumed = register_bundle(root)

    assert {done.write.id: (done.version, done.reused) for done in resumed.materialization.writes} == {
        "triage-docs": (1, True), "triage-greeting": (1, False), "triage-t": (1, False)}
