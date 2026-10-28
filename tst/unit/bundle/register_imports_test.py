"""Registering a folder ``bundle init`` wrote back into the store it came from: each entity under the id it was
imported from, nothing written while the folder holds it as imported, an edit re-versioning only what it reaches, and
the user-scoped fields the import left out given back from the stored task."""

import json

import pytest

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import Artifact, FileArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError
from agent_env.bundle import _import_report as import_report_module
from agent_env.bundle import _imported as imported_module
from agent_env.bundle import materialize as materialize_module
from agent_env.bundle import register as register_module
from agent_env.bundle._imported import Imported, ImportedImage, Source, source_of
from agent_env.bundle.importer import import_task
from agent_env.bundle.installed import InstalledBundle
from agent_env.bundle.parse import parse_bundle
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.register import REGISTRATIONS, register_bundle
from agent_env.bundle.resolve import build_step, resolve_bundle
from agent_env.config import get_config
from agent_env.entity.scopes import FieldScope
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.store import Filter
from agent_env.task import Task
from tst.unit.bundle._support import DIGEST, context_image, layout

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
    (imported / "envs/crm-env/.DS_Store").write_bytes(b"left by Finder")
    assert _written(register_bundle(imported)) == _written(registered)
    assert len(get_config().get_document_store().query(REGISTRATIONS, Filter())) == len(rows)


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


def test_an_untouched_entry_keeps_its_imported_version_and_an_edit_writes_past_what_the_store_has_since(imported,
                                                                                                   tmp_path):
    FileArtifact.put("crm-data", description="someone else's", file_path=str(tmp_path / "crm.json"))
    kept = register_bundle(imported)
    (imported / "artifacts/crm-data/crm.json").write_text('{"edited": true}')

    edited = register_bundle(imported)

    assert _written(kept)["crm-data"] == (1, True)
    assert _written(edited)["crm-data"] == (3, False)


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


def _pushed(monkeypatch):
    monkeypatch.setattr(materialize_module, "push_context_image",
                        lambda entity_id, context, dockerfile, before=None: (f"registry.example.com/{entity_id}:x",
                                                                             DIGEST))


@pytest.fixture
def imported_with_user(local_stores, tmp_path, monkeypatch):
    """The store's task triage imported with its user-scoped fields: the agent solver written into the folder and named
    there, and the secret env var dropped."""
    monkeypatch.setenv("HOME", str(tmp_path))
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image"), environment_name="crm")
    A2AAgent.put(id="solver", docker_image_artifact=context_image(tmp_path, "solver-image"))
    (tmp_path / "crm.json").write_text("{}")
    FileArtifact.put("crm-data", description="crm seed", file_path=str(tmp_path / "crm.json"))
    Task.put(id="triage", steps=[build_step(dict(step)) for step in STEPS])
    return import_task("triage", tmp_path / "triage", include_scopes={FieldScope.USER}).root


def test_a_folder_imported_with_its_user_fields_registers_as_imported(imported_with_user):
    step = json.loads((imported_with_user / "tasks/triage.json").read_text())[2]
    assert (step["a2a_agent_id"], step.get("a2a_agent_version"), step["env_vars"]) == ("solver", None, {"MODE": "fast"})

    registered = register_bundle(imported_with_user)

    assert all(reused for _, reused in _written(registered).values())


def test_an_edit_keeps_the_env_vars_the_import_dropped_and_they_are_no_credential_written_out(imported_with_user,
                                                                                            monkeypatch):
    monkeypatch.setattr(register_module, "shared_store", lambda store: True)
    (imported_with_user / "artifacts/crm-data/crm.json").write_text('{"edited": true}')

    registered = register_bundle(imported_with_user)

    assert _written(registered)["triage"] == (2, False)
    assert Task.get("triage", 2).to_dict()["steps"][2]["env_vars"] == {"API_KEY": "abc", "MODE": "fast"}


def test_an_imported_skill_takes_back_the_source_the_import_left_out_unless_it_was_edited(local_stores, tmp_path,
                                                                                         monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    notes = {"name": "notes", "description": "how notes are kept", "object_url": "s3://their-bucket/skills/notes/"}
    Task.put(id="skilled", steps=[build_step({"id": "skills", "type": "add_skills", "agent_name": "default-agent",
                                              "skills": [notes]})])
    folder = import_task("skilled", tmp_path / "skilled").root
    task_file = folder / "tasks/skilled.json"
    task_file.write_text(task_file.read_text().replace('"default-agent"', '"helper"'))

    registered = register_bundle(folder)
    task_file.write_text(task_file.read_text().replace("how notes are kept", "how notes are filed"))
    with pytest.raises(BundleError) as refused:
        register_bundle(folder)

    assert _written(registered)["skilled"] == (2, False)
    skill = Task.get("skilled", 2).to_dict()["steps"][0]["skills"][0]
    assert (skill["description"], skill["object_url"]) == ("how notes are kept", notes["object_url"])
    assert "skill 'notes' names no source" in refused.value.problems[0]


def test_envs_sharing_one_stored_image_register_as_imported_and_an_edit_to_one_copy_is_refused(local_stores, tmp_path,
                                                                                              monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    image = context_image(tmp_path, "shared-image")
    MCPServerEnv.put(id="crm-env", docker_image_artifact=image, environment_name="crm")
    MCPServerEnv.put(id="mail-env", docker_image_artifact=image, environment_name="mail")
    Task.put(id="pair", steps=[build_step({"id": "crm", "type": "deploy_env", "env_id": "crm-env"}),
                               build_step({"id": "mail", "type": "deploy_env", "env_id": "mail-env"})])
    folder = import_task("pair", tmp_path / "pair").root

    kept = register_bundle(folder)
    dockerfile = folder / "envs/mail-env/Dockerfile"
    dockerfile.write_text(dockerfile.read_text() + "ENV EDITED=1\n")
    with pytest.raises(BundleError) as refused:
        register_bundle(folder)

    assert all(reused for _, reused in _written(kept).values())
    assert refused.value.problems == ("envs/crm-env, envs/mail-env hold copies of 'shared-image' as imported, and "
                                      "envs/mail-env changed it; a registration writes each id once, so undo the change",)


def test_an_image_whose_toml_names_another_dockerfile_is_pushed(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _pushed(monkeypatch)
    files = {"Dockerfile": "FROM python:3.12-slim\nCOPY . /app\n", "app.py": "print(1)\n",
             "Dockerfile.slim": "FROM python:3.12-alpine\nCOPY . /app\n"}
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image", files),
                     environment_name="crm")
    Task.put(id="one", steps=[build_step({"id": "env", "type": "deploy_env", "env_id": "crm-env"})])
    folder = import_task("one", tmp_path / "one").root
    toml = folder / "envs/crm-env/env.toml"
    assert 'dockerfile = "Dockerfile"' in toml.read_text()
    toml.write_text(toml.read_text().replace('dockerfile = "Dockerfile"', 'dockerfile = "Dockerfile.slim"'))

    registered = register_bundle(folder)

    assert _written(registered)["crm-image"] == (2, False)


def test_an_imported_task_and_an_env_of_one_name_but_two_ids_keep_their_own(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    MCPServerEnv.put(id="crm", docker_image_artifact=context_image(tmp_path, "crm-image"), environment_name="crm")
    Task.put(id="ops/crm", steps=[build_step({"id": "env", "type": "deploy_env", "env_id": "crm"})])
    folder = import_task("ops/crm", tmp_path / "ops").root

    registered = register_bundle(folder)
    (folder / "envs/crm-2").rename(folder / "envs/crm")
    task_file = folder / "tasks/crm.json"
    task_file.write_text(task_file.read_text().replace('"crm-2"', '"crm"'))
    with pytest.raises(BundleError) as refused:
        register_bundle(folder)

    assert _written(registered) == {"crm-image": (1, True), "crm": (1, True), "ops/crm": (1, True)}
    assert refused.value.problems == ("tasks/crm.json: shares its name with what's registered as 'ops-crm', so it "
                                      "can't keep the id it was imported as, 'ops/crm'; rename one",)


def test_an_imported_task_s_output_under_a_folder_keeps_its_name(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image"), environment_name="crm")
    Task.put(id="cli", steps=[build_step({"id": "env", "type": "deploy_env", "env_id": "crm-env"}),
                              build_step({"id": "cli", "type": "build_mcp_cli", "env_id": "crm-env",
                                          "command_name": "crm", "cli_artifact_id": "tools/crm-cli"})])
    folder = import_task("cli", tmp_path / "cli").root
    (folder / "tasks/cli.json").write_text((folder / "tasks/cli.json").read_text().replace('"crm"', '"crm2"'))

    registered = register_bundle(folder)

    assert Task.get("cli", 2).to_dict()["steps"][1]["cli_artifact_id"] == "tools/crm-cli"
    assert _written(registered)["cli"] == (2, False)


def test_registering_reads_only_the_files_of_what_the_import_wrote(imported, monkeypatch):
    read = []
    hash_file = import_report_module.file_sha256
    monkeypatch.setattr(import_report_module, "file_sha256", lambda path: read.append(path) or hash_file(path))
    layout(imported, {".venv/lib/big.py": "x = 1\n", "notes/todo.md": "later\n"})

    register_bundle(imported)

    assert read and not [path for path in read if ".venv" in path.parts or "notes" in path.parts]


def test_an_installed_bundle_s_agent_takes_no_source_from_the_folder_s_record(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    kit_root = layout(tmp_path / "site" / "kit", {"agents/solver/Dockerfile": "FROM scratch\n"})
    kit = InstalledBundle("kit", "demo-dist", "1.0", "demo", kit_root, None, None)
    folder = layout(tmp_path / "triage", {"tasks/t.json": json.dumps([
        {"id": "agent", "type": "deploy_agent", "a2a_agent_id": "@local/demo-dist/kit/solver"}])})
    plan = plan_bundle(resolve_bundle(parse_bundle(folder)), installed=lambda: (kit,))
    record = Imported({"agents/solver": Source("solver", 3)},
                      {("agents/solver", "image"): ImportedImage(Source("solver-image", 3))}, {})

    installed = [write for write in plan.writes if write.source.entry in plan.origins]

    assert installed and [source_of(record, plan, write) for write in installed] == [None] * len(installed)


def test_an_env_whose_image_is_pulled_from_its_registry_registers_as_imported(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    image = get_artifact_store().put_document(DockerImageArtifact(
        id="crm-image", description="crm", image_name="registry.example.com/crm:v1",
        tar_gz_s3_url="file:///nowhere/crm-image.tar.gz"))
    MCPServerEnv.put(id="crm-env", docker_image_artifact=image, environment_name="crm")
    Task.put(id="pulled", steps=[build_step({"id": "env", "type": "deploy_env", "env_id": "crm-env"})])
    folder = import_task("pulled", tmp_path / "pulled").root

    registered = register_bundle(folder)

    assert _written(registered) == {"crm-image": (1, True), "crm-env": (1, True), "pulled": (1, True)}


@pytest.mark.parametrize("source", [{"object_url": "s3://their-bucket/skills/notes-v2/"},
                                    {"body": "Keep notes in notes.md."}])
def test_a_skill_the_folder_gives_a_source_of_its_own_keeps_it(local_stores, tmp_path, monkeypatch, source):
    monkeypatch.setenv("HOME", str(tmp_path))
    notes = {"name": "notes", "description": "how notes are kept", "object_url": "s3://their-bucket/skills/notes/"}
    Task.put(id="skilled", steps=[build_step({"id": "skills", "type": "add_skills", "agent_name": "default-agent",
                                              "skills": [notes]})])
    folder = import_task("skilled", tmp_path / "skilled").root
    task_file = folder / "tasks/skilled.json"
    steps = json.loads(task_file.read_text())
    steps[0]["skills"][0].update(source)
    task_file.write_text(json.dumps(steps))

    register_bundle(folder)

    skill = Task.get("skilled", 2).to_dict()["steps"][0]["skills"][0]
    assert {key: skill.get(key) for key in ("object_url", "s3_url", "body")} == {
        "object_url": source.get("object_url"), "s3_url": source.get("object_url"), "body": source.get("body")}


def test_a_link_into_an_imported_image_s_folder_is_an_edit_to_its_image(local_stores, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    _pushed(monkeypatch)
    files = {"Dockerfile": "FROM python:3.12-slim\nCOPY . /app\n", "app.py": "print(1)\n"}
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image", files),
                     environment_name="crm")
    Task.put(id="one", steps=[build_step({"id": "env", "type": "deploy_env", "env_id": "crm-env"})])
    folder = import_task("one", tmp_path / "one").root
    (folder / "envs/crm-env/tasks").symlink_to("../../tasks", target_is_directory=True)

    registered = register_bundle(folder)

    assert _written(registered)["crm-image"] == (2, False)


def _downloads(tmp_path, monkeypatch, names):
    """The stored task fetch, downloading one file for each of ``names`` from that user's URL, which the import masks
    to the same text, imported with its user-scoped fields; the folder, and the urls the task stores."""
    monkeypatch.setenv("HOME", str(tmp_path))
    # Not credentials: users' URLs for one file, joined in at run time.
    urls = [{"url": "https://" + name + ":" + "not-a-token@data.example.com/rows.csv", "filename": f"{name}.csv"}
            for name in names]
    Task.put(id="fetch", steps=[build_step({"id": "box", "type": "deploy_sandbox", "sandbox_name": "box",
                                            "sandbox_mode": "vm"}),
                                build_step({"id": "fetch", "type": "load_artifact", "sandbox_name": "box",
                                            "urls": urls, "destination_path": "/data"})])
    return import_task("fetch", tmp_path / "fetch", include_scopes={FieldScope.USER}).root, urls


def _edit_urls(folder, edit):
    task_file = folder / "tasks/fetch.json"
    steps = json.loads(task_file.read_text())
    edit(steps[1]["urls"])
    task_file.write_text(json.dumps(steps))


def test_reordered_downloads_each_take_back_their_own_credentials(local_stores, tmp_path, monkeypatch):
    folder, urls = _downloads(tmp_path, monkeypatch, ["ann", "bob"])
    _edit_urls(folder, list.reverse)

    register_bundle(folder)

    assert Task.get("fetch", 2).to_dict()["steps"][1]["urls"] == list(reversed(urls))


def test_a_renamed_download_takes_back_its_credentials(local_stores, tmp_path, monkeypatch):
    folder, urls = _downloads(tmp_path, monkeypatch, ["ann"])
    _edit_urls(folder, lambda entries: entries[0].update(filename="rows.csv"))

    register_bundle(folder)

    assert Task.get("fetch", 2).to_dict()["steps"][1]["urls"] == [{"url": urls[0]["url"], "filename": "rows.csv"}]


def test_a_masked_value_that_cant_be_told_from_another_is_refused_rather_than_written(local_stores, tmp_path,
                                                                                      monkeypatch):
    folder, _ = _downloads(tmp_path, monkeypatch, ["ann", "bob"])
    _edit_urls(folder, lambda entries: [entry.update(filename=f"renamed-{index}.csv")
                                        for index, entry in enumerate(entries)])

    with pytest.raises(BundleError) as refused:
        register_bundle(folder)

    assert refused.value.problems == tuple(
        f"tasks/fetch.json: step 'fetch': urls[{index}].url still holds what the import masked, and the folder "
        "changed what it was masked from; set its real value" for index in (0, 1))
    assert Task.get("fetch").version == 1


def test_a_masked_value_in_a_task_not_registered_refuses_nothing(local_stores, tmp_path, monkeypatch):
    folder, _ = _downloads(tmp_path, monkeypatch, ["ann", "bob"])
    _edit_urls(folder, lambda entries: [entry.update(filename=f"renamed-{index}.csv")
                                        for index, entry in enumerate(entries)])
    (folder / "tasks/box.json").write_text(json.dumps([{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box",
                                                        "sandbox_mode": "vm"}]))

    registered = register_bundle(folder, tasks=["box"])

    assert _written(registered) == {"fetch-box": (1, False)}
