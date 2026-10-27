"""Registering a folder ``bundle init`` wrote back into the store it came from: each entity under the id it was
imported from, nothing written while the folder holds it as imported, an edit re-versioning only what it reaches, and
the user-scoped fields the import left out given back from the stored task."""

import json

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import Artifact, FileArtifact
from agent_env.bundle import BundleError
from agent_env.bundle import _imported as imported_module
from agent_env.bundle import materialize as materialize_module
from agent_env.bundle import register as register_module
from agent_env.bundle.importer import import_task
from agent_env.bundle.register import REGISTRATIONS, register_bundle
from agent_env.bundle.resolve import build_step
from agent_env.config import get_config
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.store import Filter
from agent_env.task import Task
from tst.unit.bundle._support import DIGEST, context_image

STEPS = [
    {"id": "env", "type": "deploy_env", "env_id": "crm-env"},
    {"id": "load", "type": "load_artifact", "env_id": "crm-env", "artifacts": [{"id": "crm-data"}]},
    {"id": "agent", "type": "deploy_agent", "env_ids": ["crm-env"], "a2a_agent_id": "solver",
     "env_vars": {"API_KEY": "abc", "MODE": "fast"}},
]


@pytest.fixture(autouse=True)
def outside_git(monkeypatch):
    monkeypatch.setattr(register_module, "_git", lambda root, *args: None)


@pytest.fixture
def imported(local_stores, tmp_path, monkeypatch):
    """The store's task triage, which names crm-env v1, crm-data v1 and the agent solver, imported into a folder."""
    monkeypatch.setenv("HOME", str(tmp_path))
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image"), environment_name="crm")
    A2AAgent.put(id="solver", docker_image_artifact=context_image(tmp_path, "solver-image"))
    (tmp_path / "crm.json").write_text("{}")
    FileArtifact.put("crm-data", description="crm seed", file_path=str(tmp_path / "crm.json"))
    Task.put(id="triage", steps=[build_step(dict(step)) for step in STEPS])
    return import_task("triage", tmp_path / "triage").root


def _counts():
    documents = get_config().get_document_store()
    return {name: documents.count(name, Filter()) for name in ("tasks", "envs", "a2a_agents", "artifacts")}


def _written(registration):
    return {done.write.id: (done.version, done.reused) for done in registration.materialization.writes}


def test_a_folder_registered_as_it_was_imported_writes_nothing_and_keeps_each_version(imported):
    before = _counts()

    registered = register_bundle(imported)

    assert _written(registered) == {"crm-image": (1, True), "crm-env": (1, True), "crm-data": (1, True),
                                    "triage": (1, True)}
    assert _counts() == before
    rows = get_config().get_document_store().query(REGISTRATIONS, Filter.of(status="done"))
    assert {(row["id"], row["version"]) for row in rows} == {("crm-image", 1), ("crm-env", 1), ("crm-data", 1),
                                                             ("triage", 1)}
    assert _written(register_bundle(imported)) == _written(registered)


def test_an_edit_rewrites_only_what_it_reaches_and_gives_back_what_the_import_left_out(imported):
    (imported / "artifacts/crm-data/crm.json").write_text('{"edited": true}')

    registered = register_bundle(imported)

    assert _written(registered) == {"crm-image": (1, True), "crm-env": (1, True), "crm-data": (2, False),
                                    "triage": (2, False)}
    steps = Task.get("triage", 2).to_dict()["steps"]
    assert (steps[0]["env_id"], steps[0]["env_version"]) == ("crm-env", 1)
    assert steps[1]["artifacts"] == [{"id": "crm-data", "version": 2}]
    assert (steps[2]["a2a_agent_id"], steps[2]["a2a_agent_version"]) == ("solver", 1)
    assert steps[2]["env_vars"] == {"API_KEY": "abc", "MODE": "fast"}
    assert Artifact.get("crm-data", 2).description == "crm seed"


def test_a_given_back_field_the_store_holds_is_no_credential_written_out_to_a_shared_store(imported, monkeypatch):
    monkeypatch.setattr(register_module, "shared_store", lambda store: True)
    (imported / "artifacts/crm-data/crm.json").write_text('{"edited": true}')

    registered = register_bundle(imported)

    assert _written(registered)["triage"] == (2, False)


def test_an_edit_to_what_the_store_has_since_written_anew_is_refused_unless_adopted(imported, tmp_path):
    FileArtifact.put("crm-data", description="someone else's", file_path=str(tmp_path / "crm.json"))
    kept = register_bundle(imported)
    (imported / "artifacts/crm-data/crm.json").write_text('{"edited": true}')

    with pytest.raises(BundleError) as refused:
        register_bundle(imported)
    adopted = register_bundle(imported, adopt=["crm-data"])

    assert _written(kept)["crm-data"] == (1, True)
    assert refused.value.problems == ("artifacts/crm-data: 'crm-data' is at v2 in the store, newer than the v1 this "
                                      "bundle last registered; pass --adopt crm-data to write over it",)
    assert _written(adopted)["crm-data"] == (3, False)


def test_an_image_whose_build_context_is_edited_is_pushed_and_its_env_rewritten(imported, monkeypatch):
    monkeypatch.setattr(materialize_module, "push_context_image",
                        lambda entity_id, context, dockerfile, before=None: (f"registry.example.com/{entity_id}:x",
                                                                             DIGEST))
    dockerfile = imported / "envs/crm-env/Dockerfile"
    dockerfile.write_text(dockerfile.read_text() + "ENV EDITED=1\n")

    registered = register_bundle(imported)

    assert _written(registered) == {"crm-image": (2, False), "crm-env": (2, False), "crm-data": (1, True),
                                    "triage": (2, False)}


def test_a_folder_imported_from_another_store_registers_as_a_new_bundle(imported, monkeypatch):
    monkeypatch.setattr(imported_module, "store_fingerprint", lambda: "sha256:another-store")
    said = []

    registered = register_bundle(imported, dry_run=True, on_progress=said.append)

    assert "  it was imported from another store, so its entries take new ids" in said
    assert set(_written(registered)) == {"triage-crm-env__env_image", "triage-crm-env", "triage-crm-data",
                                         "triage-triage"}


def test_a_record_from_an_older_agent_env_registers_the_folder_as_a_new_bundle(imported):
    record = json.loads((imported / "provenance.json").read_text())
    (imported / "provenance.json").write_text(json.dumps({**record, "schema": 1}))
    said = []

    registered = register_bundle(imported, dry_run=True, on_progress=said.append)

    assert "  provenance.json is from an older agent-env, so the folder registers as a new bundle" in said
    assert set(_written(registered)) == {"triage-crm-env__env_image", "triage-crm-env", "triage-crm-data",
                                         "triage-triage"}


def test_an_imported_task_with_an_output_keeps_the_output_s_name(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image"), environment_name="crm")
    Task.put(id="cli", steps=[build_step({"id": "env", "type": "deploy_env", "env_id": "crm-env"}),
                              build_step({"id": "cli", "type": "build_mcp_cli", "env_id": "crm-env",
                                          "command_name": "crm", "cli_artifact_id": "crm-cli"})])
    folder = import_task("cli", tmp_path / "cli").root
    (folder / "tasks/cli.json").write_text((folder / "tasks/cli.json").read_text().replace('"crm"', '"crm2"'))

    registered = register_bundle(folder)

    assert Task.get("cli", 2).to_dict()["steps"][1]["cli_artifact_id"] == "crm-cli"
    assert _written(registered)["cli"] == (2, False)
