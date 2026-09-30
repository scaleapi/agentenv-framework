"""The bundle ledger: a re-run reuses every version whose inputs haven't changed, and says why the rest
are written anew."""

import json
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Literal

import pytest

from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.bundle import parse_bundle
from agent_env.bundle import plan as plan_module
from agent_env.bundle import resolve as resolve_module
from agent_env.bundle.ledger import LEDGER_COLLECTION, Ledger, materializing
from agent_env.config import configure, get_config
from agent_env.entity_refs import EntityRef
from agent_env.env.env import Env
from agent_env.store import Filter, Sort, UpdateSpec
from tst.unit.bundle._support import RefusingStore, local_store, plan_of

ROOT = "@local/~/triage"
GREETING = f"{ROOT}/greeting"
LAYOUT = {
    "envs/tickets/Dockerfile": "FROM scratch\n",
    "envs/shelf/env.toml": 'type = "shelf_ledger_test"\ndata = "greeting"\n',
    "artifacts/greeting/hello.txt": "hello\n",
    "tasks/t.json": json.dumps([
        {"id": "load", "type": "load_artifact", "env_id": "tickets", "artifact_id": "greeting"},
        {"id": "shelf", "type": "deploy_env", "env_id": "shelf"},
    ]),
}
_COLLECTIONS = {"env": "envs", "agent": "a2a_agents", "artifact": "artifacts", "task": "tasks"}
UNTRACKED = ("its inputs aren't tracked yet, so it is written every run",)


class _Shelf(Env):
    """An env whose document records the version of the artifact it names, as an image-built env does."""

    type = "shelf_ledger_test"
    toml_refs = (EntityRef.artifact("data"),)


class _Rack(Env):
    """An env whose document records the version of each env it names, as a multi env's does."""

    type = "rack_ledger_test"
    toml_refs = (EntityRef.env("envs[]"),)


class _OwnEnv(Env):
    """A plugin's env type whose from_toml is its own."""

    type = "own_env_ledger_test"

    @classmethod
    def from_toml(cls, data, ctx):
        raise NotImplementedError


class _Imaged(Env):
    """An env type whose image is built from the folder's Dockerfile."""

    type = "imaged_ledger_test"
    toml_refs = (EntityRef.artifact("image", artifact_type="docker_image"),)


class _NoteEnv(Env):
    """A registry env that is only a document."""

    type = "note_ledger_test"

    @classmethod
    def from_dict(cls, data):
        return cls(id=data["id"], version=data.get("version"))


class _OwnFile(FileArtifact):
    """A plugin's file type whose from_toml is its own, so its inputs aren't the ledger's to list."""

    type: Literal["own_file_ledger_test"] = "own_file_ledger_test"

    @staticmethod
    def from_toml(data, ctx):
        raise NotImplementedError


@pytest.fixture
def bundle_dir(tmp_path, monkeypatch):
    envs = {**resolve_module.get_env_registry(), **{cls.type: cls for cls in (_Shelf, _Rack, _OwnEnv, _Imaged)}}
    monkeypatch.setattr(resolve_module, "get_env_registry", lambda: envs)
    monkeypatch.setattr("agent_env.bundle.ledger.get_env_registry", lambda: envs)
    artifacts = {**resolve_module.get_artifact_registry(), "own_file_ledger_test": _OwnFile}
    monkeypatch.setattr(resolve_module, "get_artifact_registry", lambda: artifacts)
    monkeypatch.setattr(plan_module, "get_artifact_registry", lambda: artifacts)
    monkeypatch.setattr("agent_env.bundle.ledger.get_artifact_registry", lambda: artifacts)
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "triage"
    for rel, text in LAYOUT.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def _latest_version(collection, id):
    return local_store().find_one(collection, Filter.of(id=id), sort=Sort.by("version"))["version"]


def _written(write):
    """Stands in for the materializer's write: the next version of the entity, in the @local namespace's store."""
    collection = _COLLECTIONS[write.kind.store]
    latest = local_store().find_one(collection, Filter.of(id=write.id), sort=Sort.by("version"))
    version = latest["version"] + 1 if latest else 1
    local_store().insert(collection, {"id": write.id, "version": version})
    return version


def _record_changed(ledger, writes):
    """Check and record ``writes`` in order, as the materializer does; the checks, by id."""
    checks, versions = {}, {}
    for write in writes:
        check = ledger.check(write, versions)
        version = check.version if check.unchanged else ledger.record(check, lambda: _written(write))
        versions[write.kind.store, write.id] = version
        checks[write.id] = check
    return checks


def _checked(ledger, writes):
    """Check ``writes`` in order without recording any, as a dry run does; the checks, by id."""
    checks, versions = {}, {}
    for write in writes:
        check = ledger.check(write, versions)
        versions[write.kind.store, write.id] = check.version if check.unchanged else check.next_version
        checks[write.id] = check
    return checks


def _run(root):
    plan = plan_of(root)
    return _record_changed(Ledger.for_plan(plan), plan.writes)


def _run_rooted(root, id_root):
    plan = plan_module.plan_bundle(resolve_module.resolve_bundle(parse_bundle(root, id_root=id_root)))
    return _record_changed(Ledger.for_plan(plan), plan.writes)


def _steps(*extra):
    return json.dumps([*json.loads(LAYOUT["tasks/t.json"]), *extra])


def test_a_rerun_reuses_every_version_whose_inputs_havent_changed(bundle_dir):
    first = _run(bundle_dir)
    assert {id: check.reasons for id, check in first.items()} == dict.fromkeys(
        [f"{ROOT}/tickets", GREETING, f"{ROOT}/shelf", f"{ROOT}/t"], ("new",))

    second = _run(bundle_dir)

    assert {id: (check.unchanged, check.version) for id, check in second.items()} == dict.fromkeys(second, (True, 1))


def test_a_rewritten_dependency_rewrites_an_env_that_records_its_version_but_not_a_task(bundle_dir):
    _run(bundle_dir)
    (bundle_dir / "artifacts/greeting/hello.txt").write_text("hello again\n")

    checks = _run(bundle_dir)

    assert checks[GREETING].reasons == ("files changed: hello.txt",)
    assert checks[f"{ROOT}/shelf"].reasons == (f"artifact {GREETING} is written anew (v1 → v2)",)
    assert checks[f"{ROOT}/t"].unchanged


def test_a_dry_check_hashes_the_versions_its_needs_would_get_so_it_predicts_the_run(bundle_dir):
    _run(bundle_dir)
    (bundle_dir / "artifacts/greeting/hello.txt").write_text("hello again\n")
    plan = plan_of(bundle_dir)

    dry = _checked(Ledger.for_plan(plan), plan.writes)
    real = _record_changed(Ledger.for_plan(plan), plan.writes)

    assert dry[f"{ROOT}/shelf"].reasons == (f"artifact {GREETING} is written anew (v1 → v2)",)
    assert {id: (check.unchanged, check.reasons) for id, check in dry.items()} == {
        id: (check.unchanged, check.reasons) for id, check in real.items()}
    assert {id: check.version if check.unchanged else check.next_version for id, check in dry.items()} == {
        id: _latest_version(_COLLECTIONS[check.write.kind.store], id) for id, check in real.items()}


def test_a_need_is_hashed_as_its_version_string_so_the_rows_a_run_recorded_stay_reused(bundle_dir):
    _run(bundle_dir)
    plan = plan_of(bundle_dir)

    checks = _checked(Ledger.for_plan(plan), plan.writes)

    assert checks[f"{ROOT}/shelf"].digest.inputs["needs"] == {f"artifact {GREETING}": "1"}
    assert all(check.unchanged for check in checks.values())


def test_a_second_file_flips_the_inferred_type_and_says_so(bundle_dir):
    _run(bundle_dir)
    (bundle_dir / "artifacts/greeting/extra.txt").write_text("x")

    assert _run(bundle_dir)[GREETING].reasons == (
        "type changed: file → file_artifact_universe", "files added: extra.txt",
    )


def test_a_config_change_says_so_and_key_order_is_not_a_change(bundle_dir):
    steps = json.loads(LAYOUT["tasks/t.json"])
    _run(bundle_dir)
    (bundle_dir / "tasks/t.json").write_text(json.dumps([dict(reversed(list(step.items()))) for step in steps]))
    assert _run(bundle_dir)[f"{ROOT}/t"].unchanged

    (bundle_dir / "tasks/t.json").write_text(json.dumps([{**steps[0], "id": "load-greeting"}, steps[1]]))
    assert _run(bundle_dir)[f"{ROOT}/t"].reasons == ("config changed",)


def test_a_lone_surrogate_in_a_tasks_json_is_hashed(bundle_dir):
    steps = json.loads(LAYOUT["tasks/t.json"])
    (bundle_dir / "tasks/t.json").write_text(json.dumps([{**steps[0], "id": "load-\ud800"}, steps[1]]))
    _run(bundle_dir)

    assert _run(bundle_dir)[f"{ROOT}/t"].unchanged


def test_a_revert_is_written_as_a_new_version(bundle_dir):
    hello = bundle_dir / "artifacts/greeting/hello.txt"
    _run(bundle_dir)
    hello.write_text("changed\n")
    _run(bundle_dir)
    hello.write_text("hello\n")

    reverted = _run(bundle_dir)[GREETING]

    assert reverted.reasons == ("files changed: hello.txt",)
    assert _latest_version("artifacts", GREETING) == 3


def test_a_version_the_bundle_didnt_record_is_written_over(bundle_dir):
    local_store().insert("artifacts", {"id": GREETING, "version": 1})
    assert _run(bundle_dir)[GREETING].reasons == ("the store's latest, v1, wasn't recorded by this bundle",)
    local_store().insert("artifacts", {"id": GREETING, "version": 3})

    check = _run(bundle_dir)[GREETING]

    assert check.reasons == ("the store's latest, v3, wasn't recorded by this bundle",)
    assert check.next_version == _latest_version("artifacts", GREETING) == 4


def test_a_version_written_by_another_bundle_is_written_over(bundle_dir):
    _run(bundle_dir)
    elsewhere = UpdateSpec(set={"bundle": "@local/~/elsewhere/triage"})
    local_store().update(LEDGER_COLLECTION, Filter.of(id=GREETING, status="done"), elsewhere)

    assert _run(bundle_dir)[GREETING].reasons == ("last written by the bundle @local/~/elsewhere/triage",)


def test_the_same_bundle_from_another_folder_reuses_what_it_wrote(bundle_dir, tmp_path):
    """An installed bundle has one id root wherever its package is installed, such as in two venvs."""
    id_root = "@local/demo-bundles/triage"
    copy = tmp_path / "venv2" / "triage"
    shutil.copytree(bundle_dir, copy)
    _run_rooted(bundle_dir, id_root)

    again = _run_rooted(copy, id_root)

    assert {(check.unchanged, check.version) for check in again.values()} == {(True, 1)}


def test_a_file_changed_during_its_write_is_written_again_even_once_changed_back(bundle_dir):
    hello = bundle_dir / "artifacts/greeting/hello.txt"
    plan = plan_of(bundle_dir)
    ledger = Ledger.for_plan(plan)
    write = next(w for w in plan.writes if w.id == GREETING)

    def edited_meanwhile():
        hello.write_text("edited\n")
        return _written(write)

    ledger.record(ledger.check(write, {}), edited_meanwhile)
    hello.write_text("hello\n")

    assert ledger.check(write, {}).reasons == ("the ledger doesn't know what its last version was made from",)


def test_a_new_digest_scheme_rewrites_everything_and_says_only_that(bundle_dir, monkeypatch):
    _run(bundle_dir)
    monkeypatch.setattr("agent_env.bundle.ledger.SCHEME", 2)

    assert {check.reasons for check in _run(bundle_dir).values()} == {("the ledger's digest scheme changed",)}


def test_a_crashed_write_leaves_a_pending_row_the_bundles_next_write_drops(bundle_dir):
    plan = plan_of(bundle_dir)
    ledger = Ledger.for_plan(plan)
    write = next(w for w in plan.writes if w.id == GREETING)
    local_store().insert(LEDGER_COLLECTION, {"store": "artifact", "id": GREETING, "bundle": "@local/~/elsewhere/triage",
                                        "status": "pending"})

    def crash():
        raise RuntimeError("the write crashed")

    with pytest.raises(RuntimeError):
        ledger.record(ledger.check(write, {}), crash)
    assert ledger.check(write, {}).reasons == ("new",)

    ledger.record(ledger.check(write, {}), lambda: _written(write))
    rows = local_store().query(LEDGER_COLLECTION, Filter.of(id=GREETING))
    assert sorted((row["bundle"], row["status"], row.get("version")) for row in rows) == [
        ("@local/~/elsewhere/triage", "pending", None), (plan.bundle.bundle.id_root, "done", 1),
    ]


def test_bare_refs_stay_bare(bundle_dir, monkeypatch):
    monkeypatch.setattr("agent_env.env.registry.get_env_registry", lambda: {_NoteEnv.type: _NoteEnv})
    registry = get_config().get_document_store()
    registry.insert("envs", {"id": "rocket", "version": 1, "type": _NoteEnv.type})
    (bundle_dir / "tasks/t.json").write_text(json.dumps([{"id": "deploy", "type": "deploy_env", "env_id": "rocket"}]))
    _run(bundle_dir)
    registry.insert("envs", {"id": "rocket", "version": 2, "type": _NoteEnv.type})

    assert _run(bundle_dir)[f"{ROOT}/t"].unchanged


def test_an_env_naming_a_store_env_without_a_version_is_rewritten_when_the_store_has_a_new_one(bundle_dir, monkeypatch):
    monkeypatch.setattr("agent_env.env.registry.get_env_registry", lambda: {_NoteEnv.type: _NoteEnv})
    registry = get_config().get_document_store()
    registry.insert("envs", {"id": "rocket", "version": 1, "type": _NoteEnv.type})
    (bundle_dir / "envs/rack").mkdir()
    (bundle_dir / "envs/rack/env.toml").write_text('type = "rack_ledger_test"\nenvs = ["rocket"]\n')
    (bundle_dir / "tasks/t.json").write_text(_steps({"id": "rack", "type": "deploy_env", "env_id": "rack"}))
    _run(bundle_dir)
    assert _run(bundle_dir)[f"{ROOT}/rack"].unchanged
    registry.insert("envs", {"id": "rocket", "version": 2, "type": _NoteEnv.type})

    checks = _run(bundle_dir)

    assert checks[f"{ROOT}/rack"].reasons == ("env rocket has a new version in the store (v1 → v2)",)
    assert checks[f"{ROOT}/t"].unchanged


def test_an_agent_is_rewritten_when_its_toml_changes_or_its_unpinned_store_image_does(bundle_dir):
    registry = get_config().get_document_store()
    registry.insert("artifacts", _image_document(1))
    toml = bundle_dir / "agents/solver/agent.toml"
    toml.parent.mkdir(parents=True)
    toml.write_text('image = "claude-image"\n')
    (bundle_dir / "tasks/t.json").write_text(
        _steps({"id": "agent", "type": "deploy_agent", "env_ids": ["tickets"], "a2a_agent_id": "solver"}))

    assert _run(bundle_dir)[f"{ROOT}/solver"].reasons == ("new",)
    assert _run(bundle_dir)[f"{ROOT}/solver"].unchanged
    registry.insert("artifacts", _image_document(2))
    assert _run(bundle_dir)[f"{ROOT}/solver"].reasons == (
        "artifact claude-image has a new version in the store (v1 → v2)",)
    toml.write_text('image = "claude-image"\ndefault_env_vars = { LOG_LEVEL = "debug" }\n')
    checks = _run(bundle_dir)
    assert checks[f"{ROOT}/solver"].reasons == ("config changed",)
    assert checks[f"{ROOT}/t"].unchanged


def _image_document(version):
    return {"id": "claude-image", "version": version, "type": "docker_image", "description": "claude",
            "image_name": f"claude:v{version}", "tar_gz_s3_url": f"file:///claude-v{version}.tar.gz"}


def test_reads_never_create_the_local_namespaces_store(bundle_dir):
    plan = plan_of(bundle_dir)
    _checked(Ledger.for_plan(plan), plan.writes)

    assert not local_store().path.exists()


def test_the_ledger_never_touches_the_configured_store(bundle_dir, cli_routing):
    plan = plan_of(bundle_dir)
    configure(document_store=RefusingStore())
    ledger = Ledger.for_plan(plan)

    _record_changed(ledger, plan.writes)
    assert all(check.unchanged for check in _checked(ledger, plan.writes).values())
    (bundle_dir / "artifacts/greeting/hello.txt").write_text("hello again\n")
    rewritten = _record_changed(ledger, plan.writes)

    assert [id for id, check in rewritten.items() if not check.unchanged] == [GREETING, f"{ROOT}/shelf"]
    assert _latest_version("envs", f"{ROOT}/shelf") == 2


@pytest.mark.parametrize("files, step, id", [
    ({"artifacts/greeting/artifact.toml": 'type = "own_file_ledger_test"\n'}, None, GREETING),
    ({"envs/own/env.toml": 'type = "own_env_ledger_test"\n'},
     {"id": "own", "type": "deploy_env", "env_id": "own"}, f"{ROOT}/own"),
    ({"envs/imaged/env.toml": 'type = "imaged_ledger_test"\n', "envs/imaged/Dockerfile": "FROM scratch\n"},
     {"id": "imaged", "type": "deploy_env", "env_id": "imaged"}, f"{ROOT}/imaged__env_image"),
    ({"skills/pdf/SKILL.md": "---\nname: pdf\n---\n"},
     {"id": "pdf", "type": "load_artifact", "env_id": "tickets", "artifact_id": "pdf"}, f"{ROOT}/pdf"),
    ({"agents/solver/Dockerfile": "FROM scratch\n"},
     {"id": "agent", "type": "deploy_agent", "env_ids": ["tickets"], "a2a_agent_id": "solver"},
     f"{ROOT}/solver__agent_image"),
], ids=["artifact-with-own-from_toml", "env-with-own-from_toml", "built-image", "skill", "agent-image"])
def test_a_write_whose_inputs_arent_tracked_is_written_every_run(bundle_dir, files, step, id):
    for rel, text in files.items():
        (bundle_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle_dir / rel).write_text(text)
    if step is not None:
        (bundle_dir / "tasks/t.json").write_text(_steps(step))

    assert _run(bundle_dir)[id].reasons == UNTRACKED
    assert _run(bundle_dir)[id].reasons == UNTRACKED


@pytest.mark.parametrize("other", [
    "the-same-folder", "another-folder-with-its-id-root", "another-bundle-declaring-its-id",
])
def test_a_second_run_writing_any_of_the_same_ids_waits_for_the_first_to_finish(bundle_dir, tmp_path, other):
    first, second, root = bundle_dir, bundle_dir, {}
    if other == "another-folder-with-its-id-root":
        first, second = tmp_path / "venv-a/triage", tmp_path / "venv-b/triage"
        shutil.copytree(bundle_dir, first)
        shutil.copytree(bundle_dir, second)
        root = {"id_root": "@local/agentenv-framework/triage", "name": "triage"}
    elif other == "another-bundle-declaring-its-id":
        (bundle_dir / "artifacts/greeting/artifact.toml").write_text('id = "@local/shared/greeting"\n')
        second = tmp_path / "other"
        (second / "artifacts/hello").mkdir(parents=True)
        (second / "artifacts/hello/hello.txt").write_text("hello\n")
        (second / "artifacts/hello/artifact.toml").write_text('id = "@local/shared/greeting"\n')
    held, done = tmp_path / "held", tmp_path / "done"
    holder = subprocess.Popen([sys.executable, "-c", textwrap.dedent(f"""
        import pathlib, time
        from agent_env.bundle import parse_bundle
        from agent_env.bundle.ledger import materializing
        with materializing(parse_bundle(pathlib.Path({str(first)!r}), **{root!r})):
            pathlib.Path({str(held)!r}).write_text("held")
            time.sleep(1.5)
            pathlib.Path({str(done)!r}).write_text("done")
    """)])
    try:
        for _ in range(200):
            if held.exists():
                break
            time.sleep(0.05)
        waited = []
        with materializing(parse_bundle(second, **root), on_wait=lambda: waited.append(True)):
            assert done.exists()
        assert waited == [True]
    finally:
        holder.wait()
