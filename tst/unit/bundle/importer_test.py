"""Importing a stored task, or an eval and its tasks, as a bundle: each step as the store holds it, every store
reference pinned at the version the store holds now, an id a step of the task writes left unpinned, secrets left
out, and a folder that plans and materializes back to the stored steps. A failed import writes nothing."""

import json
import os
import signal
import tomllib
from typing import ClassVar

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import FileArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle import importer as importer_module
from agent_env.bundle.importer import ImportedEval, import_eval, import_task
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.config.runtime import Config
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.eval import Eval
from agent_env.eval.eval import EvalTask
from agent_env.store import Filter
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task_step.task_step import TaskStep

STEPS = [
    {"id": "env", "type": "deploy_env", "env_id": "crm-env"},
    {"id": "load", "type": "load_artifact", "env_id": "crm-env", "artifacts": [{"id": "crm-data"}]},
    {"id": "agent", "type": "deploy_agent", "env_ids": ["crm-env"], "a2a_agent_id": "solver",
     "env_vars": {"LITELLM_API_KEY": "sk-1", "MODE": "fast", "DB": "postgres://admin:pw@db/crm"}},
    {"id": "helper", "type": "deploy_agent", "agent_name": "helper", "env_ids": []},
    {"id": "judge", "type": "rubrics_verifier", "verifier_id": "judge", "prompt_id": "agent",
     "judge_a2a_agent_id": "solver",
     "criteria": [{"id": "c1", "description": "cites https://user:hunter2@wiki.example.com/x"},
                  {"id": "c2", "description": "clones ssh://git@github.com/org/repo"}]},
    {"id": "cli", "type": "build_mcp_cli", "env_id": "crm-env", "command_name": "crm", "cli_artifact_id": "crm-cli"},
    {"id": "skills", "type": "add_skills", "agent_name": "default-agent", "cli_artifact_ids": ["crm-cli"],
     "depends_on": [{"task_step_id": "cli"}]},
]


def _image(id):
    return get_artifact_store().put_document(DockerImageArtifact(
        id=id, description=id, image_name=f"registry.example.com/{id}:v1", tar_gz_s3_url=f"file:///{id}.tar.gz"))


@pytest.fixture
def store(local_stores, tmp_path, monkeypatch):
    """A registry of bare ids in the configured local store, holding the entities STEPS names: crm-env at v2."""
    monkeypatch.setenv("HOME", str(tmp_path))
    MCPServerEnv.put(id="crm-env", docker_image_artifact=_image("crm-image"), environment_name="crm")
    MCPServerEnv.put(id="crm-env", docker_image_artifact=_image("crm-image"), environment_name="crm")
    A2AAgent.put(id="solver", docker_image_artifact=_image("solver-image"))
    (tmp_path / "crm.json").write_text("{}")
    FileArtifact.put("crm-data", description="crm seed", file_path=str(tmp_path / "crm.json"))
    _task("triage", STEPS)
    return tmp_path


def _task(task_id, steps):
    from agent_env.bundle.resolve import build_step
    return Task.put(id=task_id, steps=[build_step(dict(step)) for step in steps])


def _steps(root, name):
    return json.loads((root / "tasks" / f"{name}.json").read_text())


def _left(root):
    """Everything an import left beside its destination, staging folders included."""
    return sorted(path.name for path in root.iterdir())


def test_a_task_imports_as_its_stored_steps_with_each_store_reference_pinned_at_its_version_now(store):
    imported = import_task("triage", store / "triage")
    steps = _steps(store / "triage", "triage")
    stored = Task.get("triage").to_dict()["steps"]

    assert imported.files == ("README.md", "tasks/triage.json")
    assert [(step["id"], step["type"]) for step in steps] == [(step["id"], step["type"]) for step in stored]
    assert steps[0]["env_version"] == 2 and stored[0]["env_version"] is None
    assert steps[1]["artifacts"] == [{"id": "crm-data", "version": 1}]
    assert steps[2]["a2a_agent_version"] == 1
    assert {k: v for k, v in steps[4].items() if k != "criteria"} == {
        k: v for k, v in stored[4].items() if k != "criteria"}
    assert sorted((pin.kind.value, pin.id, pin.version, pin.stored) for pin in imported.pinned) == [
        ("agent", "solver", 1, False), ("artifact", "crm-data", 1, False), ("env", "crm-env", 2, False)]


def test_an_id_a_step_of_the_task_writes_stays_unpinned_and_unchanged(store):
    steps = _steps(import_task("triage", store / "triage").root, "triage")

    assert steps[5]["cli_artifact_id"] == "crm-cli"
    assert steps[6]["cli_artifact_ids"] == ["crm-cli"]


def test_a_reference_that_takes_no_version_and_a_step_that_names_no_agent_are_reported_as_floating(store):
    imported = import_task("triage", store / "triage")

    assert sorted((ref.where, ref.kind.value, ref.id) for ref in imported.floating) == [
        ("tasks/triage.json: step 'helper': a2a_agent_id", "agent", "a2a-default"),
        ("tasks/triage.json: step 'judge': judge_a2a_agent_id", "agent", "solver"),
    ]


def test_env_vars_that_look_like_secrets_are_left_out_and_a_urls_credentials_masked_elsewhere(store):
    imported = import_task("triage", store / "triage")
    steps = _steps(imported.root, "triage")

    assert steps[2]["env_vars"] == {"MODE": "fast"}
    assert [c["description"] for c in steps[4]["criteria"]] == [
        "cites https://***@wiki.example.com/x", "clones ssh://git@github.com/org/repo"]
    assert [(drop.where, drop.why) for drop in imported.dropped] == [
        ("tasks/triage.json: step 'agent': env_vars.LITELLM_API_KEY", "dropped: its name looks like a secret"),
        ("tasks/triage.json: step 'agent': env_vars.DB", "dropped: its value holds a URL's credentials"),
        ("tasks/triage.json: step 'judge': criteria[0].description", "masked: a URL's credentials"),
    ]
    written = (imported.root / "README.md").read_text() + (imported.root / "tasks/triage.json").read_text()
    assert "sk-1" not in written and "hunter2" not in written and "admin:pw" not in written


# Each in the shape its issuer gives it, joined at run time so no scanner takes it for a real one.
TOKENS = {
    "an API key": "sk" + "-proj-" + "a1B2" * 6,
    "a GitHub token": "gh" + "p_" + "a1B2c3" * 6,
    "an AWS access key": "AK" + "IA" + "Q4ZX7" * 3 + "W",
    "a Slack token": "xo" + "xb-" + "1234-5678-abcd",
    "a Google API key": "AI" + "za" + "Sy" + "a1B2c" * 6 + "d3E",
    "a Hugging Face token": "hf" + "_" + "a1B2c3" * 6,
    "a JWT": "ey" + "JhbGciOiJIUzI1NiJ9." + "ey" + "JzdWIiOiIxMjM0In0." + "c2lnbmF0dXJlMTIz",
    "a private key": "-----BEGIN " + "RSA PRIVATE KEY-----\nMIIBOgIBAAJBAK\n-----END RSA PRIVATE KEY-----",
}


@pytest.mark.parametrize("what", TOKENS)
def test_a_credential_by_its_shape_is_dropped_from_an_env_var_and_masked_anywhere_else(store, what):
    token = TOKENS[what]
    _task("creds", [STEPS[0], STEPS[1], {**STEPS[2], "env_vars": {"SETTING": token}},
                    {**STEPS[4], "criteria": [{"id": "c1", "description": f"logs in with {token} today"}]}])

    imported = import_task("creds", store / "creds")
    steps = _steps(imported.root, "creds")

    assert steps[2]["env_vars"] == {}
    assert steps[3]["criteria"][0]["description"] == "logs in with *** today"
    assert [(drop.where, drop.why) for drop in imported.dropped] == [
        ("tasks/creds.json: step 'agent': env_vars.SETTING", f"dropped: its value holds {what}"),
        ("tasks/creds.json: step 'judge': criteria[0].description", f"masked: {what}"),
    ]
    assert not any(token in path.read_text() for path in imported.root.rglob("*") if path.is_file())


def test_text_that_only_resembles_a_credential_is_kept(store):
    kept = ["sk-learn-is-a-python-library", "max_tokens=4096", "eyJ is how a JWT starts", "ssh://git@github.com/o/r",
            "commit 5a3c2bf0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6"]
    _task("kept", [STEPS[0], STEPS[1], {**STEPS[2], "env_vars": {f"V{i}": text for i, text in enumerate(kept)}},
                   {**STEPS[4], "criteria": [{"id": f"c{i}", "description": text} for i, text in enumerate(kept)]}])

    imported = import_task("kept", store / "kept")
    steps = _steps(imported.root, "kept")

    assert list(steps[2]["env_vars"].values()) == kept
    assert [criterion["description"] for criterion in steps[3]["criteria"]] == kept
    assert imported.dropped == ()


def test_the_readme_says_where_the_folder_came_from_and_how_each_reference_was_written(store):
    readme = (import_task("triage", store / "triage").root / "README.md").read_text()

    assert readme.splitlines()[0] == "Imported from the store's task triage v1."
    assert "- env crm-env v2: tasks/triage.json: step 'env': env_id" in readme
    assert "## Floating" in readme and "## Left out" in readme
    assert parse_bundle(store / "triage").description == "Imported from the store's task triage v1."


def test_the_folder_materializes_back_to_the_stored_steps(store):
    imported = import_task("triage", store / "triage")
    bundle = parse_bundle(imported.root)
    with namespace_routing():
        materialization = materialize(plan_bundle(resolve_bundle(bundle)))
        written = Task.get(f"{bundle.id_root}/triage", materialization.version_of("task", f"{bundle.id_root}/triage"))
    expected = _steps(imported.root, "triage")
    output = f"{bundle.id_root}/triage/crm-cli"
    expected[5]["cli_artifact_id"], expected[6]["cli_artifact_ids"] = output, [output]

    assert written.to_dict()["steps"] == expected


def test_an_id_a_bundle_named_under_its_task_is_renamed_so_the_import_names_it_under_its_own(store):
    old = "@local/~/old/triage"
    with namespace_routing():
        _task(old, [STEPS[0], {**STEPS[5], "cli_artifact_id": f"{old}/crm-cli"},
                    {**STEPS[6], "cli_artifact_ids": [f"{old}/crm-cli"]}])
        imported = import_task(old, store / "again")
    steps = _steps(imported.root, "triage")

    assert (steps[1]["cli_artifact_id"], steps[2]["cli_artifact_ids"]) == ("crm-cli", ["crm-cli"])


def test_a_renamed_output_is_numbered_when_another_output_or_a_store_reference_has_its_name(store):
    old = "@local/~/old/triage"
    with namespace_routing():
        _task(old, [STEPS[0], STEPS[1],
                    {**STEPS[5], "cli_artifact_id": f"{old}/crm-cli"},
                    {**STEPS[5], "id": "cli2", "cli_artifact_id": "@local/~/other/triage/crm-cli"},
                    {**STEPS[5], "id": "cli3", "cli_artifact_id": f"{old}/crm-data"},
                    {**STEPS[6], "cli_artifact_ids": [f"{old}/crm-cli", "@local/~/other/triage/crm-cli"],
                     "depends_on": [{"task_step_id": "cli"}, {"task_step_id": "cli2"}]}])
        imported = import_task(old, store / "again")
    steps = _steps(imported.root, "triage")

    assert [step.get("cli_artifact_id") for step in steps[2:5]] == ["crm-cli", "crm-cli-2", "crm-data-2"]
    assert steps[5]["cli_artifact_ids"] == ["crm-cli", "crm-cli-2"]
    assert steps[1]["artifacts"] == [{"id": "crm-data", "version": 1}]


def test_an_env_no_step_of_the_task_deploys_floats(store):
    _task("live", [{"id": "agent", "type": "deploy_agent", "env_ids": ["crm-env"], "a2a_agent_id": "solver"}])

    imported = import_task("live", store / "live")

    assert [(ref.kind.value, ref.id, ref.where) for ref in imported.floating] == [
        ("env", "crm-env", "tasks/live.json: step 'agent': env_ids[0]")]


def test_a_pin_the_task_already_has_is_kept_when_the_store_holds_it_and_refused_when_it_doesnt(store):
    _task("pinned", [{"id": "env", "type": "deploy_env", "env_id": "crm-env", "env_version": 1}])
    _task("gone", [{"id": "env", "type": "deploy_env", "env_id": "crm-env", "env_version": 9}])

    kept = import_task("pinned", store / "pinned")
    with pytest.raises(BundleError) as e:
        import_task("gone", store / "gone")

    assert [(pin.id, pin.version, pin.stored) for pin in kept.pinned] == [("crm-env", 1, True)]
    assert e.value.problems == ("tasks/gone.json: step 'env': env_id: there is no env 'crm-env' v9 in the store",)


def test_the_current_folder_imports_when_it_is_empty(store, monkeypatch):
    (store / "here").mkdir()
    monkeypatch.chdir(store / "here")
    before = _left(store)

    result = CliRunner().invoke(cli, ["bundle", "init", ".", "--from-task", "triage"])

    assert result.exit_code == 0, result.output
    assert (store / "here" / "tasks" / "triage.json").exists() and _left(store) == before


def test_an_eval_imports_with_each_task_it_names_once_under_names_taken_from_their_ids(store):
    _task("team/run", [STEPS[0]])
    _task("other/run", [STEPS[0]])
    _task("RUN", [STEPS[0]])
    Eval.put(id="nightly/regression", tasks=[EvalTask("team/run"), EvalTask("other/run"), EvalTask("RUN"),
                                             EvalTask("team/run"), EvalTask("triage", 1)])

    imported = import_eval("nightly/regression", store / "nightly")

    assert imported.files == ("README.md", "evals/regression.toml", "tasks/RUN-3.json", "tasks/run-2.json",
                              "tasks/run.json", "tasks/triage.json")
    assert tomllib.loads((imported.root / "evals/regression.toml").read_text()) == {
        "tasks": ["run", "run-2", "RUN-3", "triage"]}
    assert imported.duplicates == ("task team/run v1",)
    assert imported.eval == ImportedEval("evals/regression.toml", "nightly/regression", 1)
    plan = plan_bundle(resolve_bundle(parse_bundle(imported.root)))
    assert [entry.entry.name for entry in plan.evals] == ["regression"]


def test_an_eval_naming_a_task_the_store_doesnt_hold_is_refused_and_writes_nothing(store):
    Eval.put(id="broken", tasks=[EvalTask("triage"), EvalTask("gone"), EvalTask("triage", 7)])
    before = _left(store)

    with pytest.raises(BundleError) as caught:
        import_eval("broken", store / "broken")

    assert caught.value.problems == ("eval broken v1 names task gone, which isn't in the store",
                                     "eval broken v1 names task triage v7, which isn't in the store")
    assert _left(store) == before


def test_an_older_version_imports_when_asked_for(store):
    _task("triage", [STEPS[0]])

    old, new = import_task("triage", store / "old", version=1), import_task("triage", store / "new")

    assert [(task.version, task.steps) for task in old.tasks + new.tasks] == [(1, len(STEPS)), (2, 1)]
    assert (old.root / "README.md").read_text().splitlines()[0] == "Imported from the store's task triage v1."


@pytest.mark.parametrize("task_id, name", [
    ("crm#triage", "crm-triage"), ("..hidden", "hidden"), ("  edge  ", "edge"), ("org/", "task"), ("😀", "task"),
    ("a" * 300, "a" * 200),
])
def test_a_name_is_the_ids_last_segment_made_valid(store, task_id, name):
    _task(task_id, [STEPS[0]])

    assert import_task(task_id, store / "named").tasks[0].path == f"tasks/{name}.json"


@pytest.mark.parametrize("prepare, problem", [
    (lambda dest: (dest.mkdir(), (dest / "x").write_text("x")), "not empty; import into a new or empty folder"),
    (lambda dest: dest.write_text("x"), "not a folder; import into a new or empty folder"),
])
def test_a_destination_that_holds_anything_is_refused(store, prepare, problem):
    dest = store / "dest"
    prepare(dest)

    with pytest.raises(BundleError) as e:
        import_task("triage", dest)

    assert e.value.problems == (f"{dest}: {problem}",)


def test_a_destination_whose_folder_gives_no_id_or_has_no_parent_is_refused(store):
    with pytest.raises(BundleError, match="rename the folder"):
        import_task("triage", store / "bad​name")
    with pytest.raises(BundleError, match="no such folder"):
        import_task("triage", store / "missing" / "dest")
    assert not (store / "missing").exists()


def test_an_empty_destination_is_filled(store):
    (store / "dest").mkdir()

    assert import_task("triage", store / "dest").files == ("README.md", "tasks/triage.json")


def test_a_reference_the_store_doesnt_hold_is_refused_with_every_problem_and_nothing_is_written(store):
    _task("broken", [{"id": "env", "type": "deploy_env", "env_id": "gone"},
                     {"id": "load", "type": "load_artifact", "env_id": "gone", "artifacts": [{"id": "lost"}]}])
    before = _left(store)

    with pytest.raises(BundleError) as e:
        import_task("broken", store / "broken")

    assert e.value.problems == (
        "tasks/broken.json: step 'env': env_id: there is no env 'gone' in the store",
        "tasks/broken.json: step 'load': artifacts[0].id: there is no artifact 'lost' in the store",
    )
    assert _left(store) == before


def test_a_folder_that_fails_its_check_is_not_moved_into_place_and_its_staging_is_removed(store, monkeypatch):
    (store / "dest").mkdir()
    before = _left(store)

    def refuse(resolved):
        raise BundleError(["refused"])

    monkeypatch.setattr(importer_module, "check_bundle", refuse)
    with pytest.raises(BundleError):
        import_task("triage", store / "dest")

    assert _left(store) == before and not any((store / "dest").iterdir())


def test_an_empty_destination_is_put_back_when_the_move_fails(store, monkeypatch):
    (store / "dest").mkdir()
    before = _left(store)

    def fail(src, dst):
        raise OSError("disk gone")

    monkeypatch.setattr(importer_module.os, "replace", fail)
    with pytest.raises(BundleError) as caught:
        import_task("triage", store / "dest")

    assert caught.value.problems == (f"{store / 'dest'}: can't move the import into place: disk gone",)
    assert _left(store) == before and (store / "dest").is_dir()


def test_an_empty_destination_is_put_back_when_an_interrupt_stops_the_move(store, monkeypatch):
    (store / "dest").mkdir()
    before = _left(store)

    def interrupt(src, dst):
        raise KeyboardInterrupt

    monkeypatch.setattr(importer_module.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        import_task("triage", store / "dest")

    assert _left(store) == before and (store / "dest").is_dir()


@pytest.mark.parametrize("empty", [False, True])
def test_a_destination_something_else_writes_during_the_import_is_left_as_it_wrote_it(store, monkeypatch, empty):
    dest = store / "dest"
    if empty:
        dest.mkdir()

    def land(resolved):
        dest.mkdir(exist_ok=True)
        (dest / "theirs.txt").write_text("theirs")

    monkeypatch.setattr(importer_module, "check_bundle", land)
    with pytest.raises(BundleError) as caught:
        import_task("triage", dest)

    assert caught.value.problems == (f"{dest}: not empty; import into a new or empty folder",)
    assert sorted(path.name for path in dest.iterdir()) == ["theirs.txt"]
    assert not [name for name in _left(store) if ".import-" in name]


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() == 0, reason="root writes anywhere")
def test_a_folder_the_import_cant_write_in_is_refused_in_one_line(store):
    (store / "ro").mkdir()
    (store / "ro").chmod(0o555)
    try:
        with pytest.raises(BundleError) as caught:
            import_task("triage", store / "ro" / "dest")
        left = list((store / "ro").iterdir())
    finally:
        (store / "ro").chmod(0o755)

    assert caught.value.problems == (f"{store / 'ro'}: can't write the import there: permission denied",)
    assert left == []


def test_sigterm_during_an_import_leaves_nothing_beside_its_destination(store, monkeypatch):
    before = _left(store)
    monkeypatch.setattr(importer_module, "check_bundle", lambda resolved: os.kill(os.getpid(), signal.SIGTERM))

    result = CliRunner().invoke(cli, ["bundle", "init", str(store / "dest"), "--from-task", "triage"])

    assert result.exit_code == 1 and "Aborted!" in result.output
    assert _left(store) == before


class _Undeclared(TaskStep):
    """A plugin step type that declares no references."""

    type: ClassVar[str] = "undeclared_import_test"

    def __init__(self, target=None, **base):
        super().__init__(**base)
        self.target = target

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), target=data.get("target"))

    def to_dict(self):
        return {**super().to_dict(), "target": self.target}

    async def execute(self, context):
        return context


def test_a_step_whose_type_declares_no_references_is_copied_and_reported(store, monkeypatch):
    steps = {**Config().task_step_registry(), _Undeclared.type: _Undeclared}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    _task("plugin", [{"id": "p", "type": _Undeclared.type, "target": "crm-env"}])

    imported = import_task("plugin", store / "plugin")

    assert _steps(imported.root, "plugin")[0]["target"] == "crm-env"
    assert imported.undeclared == (f"tasks/plugin.json: step 'p' ({_Undeclared.type})",)


def test_the_cli_imports_and_says_what_it_pinned_left_out_and_floats(store):
    result = CliRunner().invoke(cli, ["bundle", "init", str(store / "triage"), "--from-task", "triage"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == f"Imported task triage v1 into {store / 'triage'}:"
    assert lines[1:3] == ["  README.md", "  tasks/triage.json"]
    assert "Pinned:" in lines and "Floating:" in lines and "Left out:" in lines
    assert "  env crm-env v2: tasks/triage.json: step 'env': env_id" in lines
    assert lines[-1] == f"Run it with: agent-env run {store / 'triage'}"


@pytest.mark.parametrize("args", [[], ["--from-task", "triage", "--from-eval", "e"]])
def test_the_cli_takes_exactly_one_source(store, args):
    result = CliRunner().invoke(cli, ["bundle", "init", str(store / "x"), *args])

    assert result.exit_code == 2
    assert "give one of --from-task or --from-eval" in result.output


def test_the_cli_says_in_one_line_when_the_store_has_no_such_task(store):
    result = CliRunner().invoke(cli, ["bundle", "init", str(store / "x"), "--from-task", "nope"])

    assert result.exit_code == 1
    assert result.output.startswith("Error: ") and "Traceback" not in result.output
    assert not (store / "x").exists()


def test_reading_the_store_writes_nothing_to_it(store):
    documents = get_config().get_document_store()
    counts = {name: documents.count(name, Filter()) for name in ("tasks", "envs", "a2a_agents", "artifacts")}

    import_task("triage", store / "triage")

    assert {name: documents.count(name, Filter()) for name in counts} == counts
