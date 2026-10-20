"""Importing a stored task, or an eval and its tasks, as a bundle: each step as the store holds it, every entity it
references written into a folder of its own and named by it, an id a step of the task writes left as it is, secrets left
out, and a folder that plans and materializes back to the stored steps. A failed import writes nothing."""

import json
import os
import signal
import tomllib
from typing import ClassVar, Literal

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact import FileArtifact
from agent_env.bundle import BundleError, parse_bundle
from agent_env.bundle import importer as importer_module
from agent_env.bundle.importer import ImportedEval, import_eval, import_task
from agent_env.bundle.installed import InstalledBundle
from agent_env.bundle.materialize import materialize
from agent_env.bundle.plan import plan_bundle
from agent_env.bundle.resolve import resolve_bundle
from agent_env.cli import cli
from agent_env.config import get_config
from agent_env.config.runtime import _DEFAULT_A2A_AGENT_ID as DEFAULT_AGENT, Config
from agent_env.env.envs.mcp_server import MCPServerEnv
from agent_env.eval import Eval
from agent_env.eval.eval import EvalTask
from agent_env.entity.scopes import FieldScope
from agent_env.plugins import TASK_STEPS, Contribution, Distribution, Inventory
from agent_env.store import Filter
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task_step.task_step import TaskStep
from tst.unit.bundle._support import context_image

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


@pytest.fixture
def store(local_stores, tmp_path, monkeypatch):
    """A registry of bare ids in the configured local store, holding the entities STEPS names: crm-env at v2."""
    monkeypatch.setenv("HOME", str(tmp_path))
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image"), environment_name="crm")
    MCPServerEnv.put(id="crm-env", docker_image_artifact=context_image(tmp_path, "crm-image"), environment_name="crm")
    A2AAgent.put(id="solver", docker_image_artifact=context_image(tmp_path, "solver-image"))
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


def test_a_task_imports_as_its_stored_steps_naming_each_entity_it_uses_by_the_folder_it_is_written_to(store):
    imported = import_task("triage", store / "triage", include_scopes=[FieldScope.USER])
    steps = _steps(store / "triage", "triage")
    stored = Task.get("triage").to_dict()["steps"]

    assert imported.files == ("README.md", "provenance.json", "tasks/triage.json")
    assert [(step["id"], step["type"]) for step in steps] == [(step["id"], step["type"]) for step in stored]
    assert (steps[0]["env_id"], steps[0]["env_version"]) == ("crm-env", None)
    assert [(item["id"], item.get("version")) for item in steps[1]["artifacts"]] == [("crm-data", None)]
    assert (steps[2]["a2a_agent_id"], steps[2]["a2a_agent_version"]) == ("solver", None)
    assert {k: v for k, v in steps[4].items() if k != "criteria"} == {
        k: v for k, v in stored[4].items() if k != "criteria"}
    assert sorted((entity.folder, entity.kind.value, entity.id, entity.version) for entity in imported.written) == [
        ("agents/solver", "agent", "solver", 1), ("artifacts/crm-data", "artifact", "crm-data", 1),
        ("envs/crm-env", "env", "crm-env", 2)]
    assert imported.pinned == ()


def test_an_id_a_step_of_the_task_writes_stays_unpinned_and_unchanged(store):
    steps = _steps(import_task("triage", store / "triage").root, "triage")

    assert steps[5]["cli_artifact_id"] == "crm-cli"
    assert steps[6]["cli_artifact_ids"] == ["crm-cli"]


def test_a_reference_that_takes_no_version_and_a_step_that_names_no_agent_are_reported_as_floating(store):
    imported = import_task("triage", store / "triage")

    assert sorted((ref.where, ref.kind.value, ref.id) for ref in imported.floating) == [
        ("tasks/triage.json: step 'agent': a2a_agent_id", "agent", DEFAULT_AGENT),
        ("tasks/triage.json: step 'helper': a2a_agent_id", "agent", DEFAULT_AGENT),
        ("tasks/triage.json: step 'judge': judge_a2a_agent_id", "agent", DEFAULT_AGENT),
    ]
    kept = import_task("triage", store / "kept", include_scopes=[FieldScope.USER])
    assert sorted((ref.where, ref.kind.value, ref.id) for ref in kept.floating) == [
        ("tasks/triage.json: step 'helper': a2a_agent_id", "agent", DEFAULT_AGENT),
    ]


def test_env_vars_that_look_like_secrets_are_left_out_and_a_urls_credentials_masked_elsewhere(store):
    imported = import_task("triage", store / "triage", include_scopes=[FieldScope.USER])
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

    imported = import_task("creds", store / "creds", include_scopes=[FieldScope.USER])
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

    imported = import_task("kept", store / "kept", include_scopes=[FieldScope.USER])
    steps = _steps(imported.root, "kept")

    assert list(steps[2]["env_vars"].values()) == kept
    assert [criterion["description"] for criterion in steps[3]["criteria"]] == kept
    assert imported.dropped == ()


def test_the_readme_says_where_the_folder_came_from_how_to_run_it_and_what_it_holds(store):
    readme = (import_task("triage", store / "triage").root / "README.md").read_text()

    assert readme.splitlines()[0] == "Imported from the store's task triage v1."
    assert "## Run it" in readme and "## Share it" in readme and "LITELLM_BASE_URL" in readme
    assert "- envs/crm-env: mcp_server crm-env v2" in readme
    assert "## Floating" in readme and "## Left out" in readme and "## Still needs the store" not in readme
    assert parse_bundle(store / "triage").description == "Imported from the store's task triage v1."


def test_the_folder_materializes_back_to_the_stored_steps(store):
    imported = import_task("triage", store / "triage", include_scopes=[FieldScope.USER])
    bundle = parse_bundle(imported.root)
    with namespace_routing():
        materialization = materialize(plan_bundle(resolve_bundle(bundle)))
        written = Task.get(f"{bundle.id_root}/triage", materialization.version_of("task", f"{bundle.id_root}/triage"))
    expected = _steps(imported.root, "triage")
    local = {name: f"{bundle.id_root}/{name}" for name in ("crm-env", "crm-data", "solver")}
    for step in expected:
        for key in ("env_id", "a2a_agent_id", "judge_a2a_agent_id"):
            step[key] = local.get(step.get(key), step.get(key)) if key in step else None
        step.get("env_ids") and step.update(env_ids=[local[env] for env in step["env_ids"]])
    expected[1]["artifacts"][0]["id"] = local["crm-data"]
    expected = [{key: value for key, value in step.items() if value is not None or key in written_step}
                for step, written_step in zip(expected, written.to_dict()["steps"])]
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
    assert [(item["id"], item.get("version")) for item in steps[1]["artifacts"]] == [("crm-data", None)]


def test_an_env_a_reference_names_with_no_version_is_written_at_its_latest(store):
    _task("live", [{"id": "agent", "type": "deploy_agent", "env_ids": ["crm-env"], "a2a_agent_id": "solver"}])

    imported = import_task("live", store / "live")

    assert [(entity.folder, entity.version) for entity in imported.written] == [("envs/crm-env", 2)]
    assert _steps(imported.root, "live")[0]["env_ids"] == ["crm-env"]
    assert [(ref.kind.value, ref.id, ref.where) for ref in imported.floating] == [
        ("agent", DEFAULT_AGENT, "tasks/live.json: step 'agent': a2a_agent_id")]


def test_a_pin_the_task_has_writes_that_version_and_is_dropped_and_one_the_store_doesnt_hold_is_refused(store):
    _task("pinned", [{"id": "env", "type": "deploy_env", "env_id": "crm-env", "env_version": 1}])
    _task("gone", [{"id": "env", "type": "deploy_env", "env_id": "crm-env", "env_version": 9}])

    kept = import_task("pinned", store / "pinned")
    with pytest.raises(BundleError) as e:
        import_task("gone", store / "gone")

    assert [(entity.id, entity.version) for entity in kept.written] == [("crm-env", 1)]
    assert _steps(kept.root, "pinned")[0]["env_version"] is None
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

    assert imported.files == ("README.md", "evals/regression.toml", "provenance.json", "tasks/RUN-3.json",
                              "tasks/run-2.json", "tasks/run.json", "tasks/triage.json")
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

    assert import_task("triage", store / "dest").files == ("README.md", "provenance.json", "tasks/triage.json")


def test_a_reference_the_store_doesnt_hold_is_refused_with_every_problem_and_nothing_is_written(store):
    _task("broken", [{"id": "env", "type": "deploy_env", "env_id": "gone"},
                     {"id": "load", "type": "load_artifact", "env_id": "gone", "artifacts": [{"id": "lost"}]}])
    before = _left(store)

    with pytest.raises(BundleError) as e:
        import_task("broken", store / "broken")

    assert e.value.problems == (
        "tasks/broken.json: step 'env': env_id: there is no env 'gone' in the store",
        "tasks/broken.json: step 'load': artifacts[0].id: there is no artifact 'lost' in the store",
        "tasks/broken.json: step 'load': env_id: there is no env 'gone' in the store",
    )
    assert _left(store) == before


def test_a_folder_that_fails_its_check_is_not_moved_into_place_and_its_staging_is_removed(store, monkeypatch):
    (store / "dest").mkdir()
    before = _left(store)

    def refuse(resolved, **_):
        raise BundleError(["refused"])

    monkeypatch.setattr(importer_module, "check_bundle", refuse)
    with pytest.raises(BundleError):
        import_task("triage", store / "dest")

    assert _left(store) == before and not any((store / "dest").iterdir())


def test_an_empty_destination_is_put_back_when_the_move_fails(store, monkeypatch):
    (store / "dest").mkdir()
    before = _left(store)

    replace = os.replace

    def fail(src, dst):
        if dst == store / "dest":
            raise OSError("disk gone")
        replace(src, dst)

    monkeypatch.setattr(importer_module.os, "replace", fail)
    with pytest.raises(BundleError) as caught:
        import_task("triage", store / "dest")

    assert caught.value.problems == (f"{store / 'dest'}: can't move the import into place: disk gone",)
    assert _left(store) == before and (store / "dest").is_dir()


def test_an_empty_destination_is_put_back_when_an_interrupt_stops_the_move(store, monkeypatch):
    (store / "dest").mkdir()
    before = _left(store)

    replace = os.replace

    def interrupt(src, dst):
        if dst == store / "dest":
            raise KeyboardInterrupt
        replace(src, dst)

    monkeypatch.setattr(importer_module.os, "replace", interrupt)
    with pytest.raises(KeyboardInterrupt):
        import_task("triage", store / "dest")

    assert _left(store) == before and (store / "dest").is_dir()


@pytest.mark.parametrize("empty", [False, True])
def test_a_destination_something_else_writes_during_the_import_is_left_as_it_wrote_it(store, monkeypatch, empty):
    dest = store / "dest"
    if empty:
        dest.mkdir()

    def land(resolved, **_):
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
    monkeypatch.setattr(importer_module, "check_bundle", lambda resolved, **_: os.kill(os.getpid(), signal.SIGTERM))

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


def test_the_cli_imports_and_says_what_it_wrote_left_out_and_floats(store):
    result = CliRunner().invoke(cli, ["bundle", "init", str(store / "triage"), "--from-task", "triage"])

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("Downloading 1 file(s) and 1 build context(s), ")
    imported = lines.index(f"Imported task triage v1 into {store / 'triage'}:")
    listed = lines[imported + 1:lines.index("Written:")]
    assert listed == sorted(listed)
    assert {"  README.md", "  provenance.json", "  tasks/triage.json", "  envs/crm-env/", "  artifacts/crm-data/"} <= {
        *listed}
    assert "Written:" in lines and "Images:" in lines and "Floating:" in lines and "Left out:" in lines
    assert "  envs/crm-env: mcp_server crm-env v2" in lines
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


SCOPED = [
    {"id": "env", "type": "deploy_env", "env_id": "crm-env", "sandbox_type": "modal_vm,modal",
     "metadata": {"attribution": {"project_id": "p-123"}, "note": "kept"}},
    {"id": "agent", "type": "deploy_agent", "env_ids": ["crm-env"], "a2a_agent_id": "solver", "sandbox_type": "local",
     "litellm_base_url": "https://litellm.internal.example.com", "metadata": {"attribution": {"project_id": "p-123"}},
     "agent_changelog_object_url": "s3://their-bucket/changelog/", "agent_changelog_toolcall_position_exclusive": 0},
    {"id": "prompt", "type": "prompt_agent", "prompt": "close the duplicates", "agent_name": "default-agent",
     "trajectory_output_prefix": "s3://their-bucket/trajectories/", "user_a2a_url": "https://hub.example.com/a2a",
     "max_thinking_tokens": 0},
    {"id": "judge", "type": "rubrics_verifier", "verifier_id": "judge", "prompt_id": "prompt",
     "judge_sandbox_type": "modal", "criteria": [{"id": "c1", "description": "closed every duplicate"}]},
    {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"},
    {"id": "install", "type": "install_agent", "sandbox_name": "box", "a2a_agent_id": "solver", "agent_name": "installed"},
    {"id": "fetch", "type": "load_artifact", "sandbox_name": "box", "urls": ["s3://their-bucket/rows.csv"],
     "destination_path": "/data"},
    {"id": "build", "type": "run_docker_container", "sandbox_name": "box", "docker_context_url": "s3://their-bucket/ctx.zip"},
    {"id": "skills", "type": "add_skills", "agent_name": "installed",
     "skills": [{"name": "notes", "description": "how notes are kept", "object_url": "s3://their-bucket/skills/notes/"}]},
]
SCOPED_FIELDS = {"env": ["sandbox_type", "metadata.attribution"],
                 "agent": ["a2a_agent_id", "sandbox_type", "litellm_base_url", "metadata.attribution"],
                 "prompt": ["trajectory_output_prefix", "max_thinking_tokens"],  # a set zero is set
                 # A stored judge always holds a model: its author's, or the one its store's config named for judges.
                 "judge": ["default_model", "judge_sandbox_type"],
                 "install": ["a2a_agent_id"],
                 "fetch": ["urls"],
                 "build": ["docker_context_url"],
                 # A skill is stored under both spellings of its object_url.
                 "skills": ["skills[0].object_url", "skills[0].s3_url"]}


def _value(step, path):
    for key in path.split("."):
        key, _, index = key.removesuffix("]").partition("[")
        step = step.get(key) if isinstance(step, dict) else None
        if index and isinstance(step, list):
            step = step[int(index)]
    return step


def test_a_steps_user_scoped_fields_are_left_out_and_reported_and_the_rest_of_it_is_kept(store):
    _task("scoped", SCOPED)
    stored = {step["id"]: step for step in Task.get("scoped").to_dict()["steps"]}

    imported = import_task("scoped", store / "scoped")

    steps = {step["id"]: step for step in _steps(imported.root, "scoped")}
    assert {id: [path for path in paths if _value(steps[id], path) is not None]
            for id, paths in SCOPED_FIELDS.items()} == {id: [] for id in SCOPED_FIELDS}
    assert sorted(drop.where for drop in imported.dropped) == sorted(
        f"tasks/scoped.json: step {id!r}: {path}" for id, paths in SCOPED_FIELDS.items() for path in paths)
    assert {drop.why for drop in imported.dropped} == {
        "dropped: user scope, so a run takes its own; --include-scope user keeps it"}
    assert steps["env"]["metadata"] == {"note": "kept"}
    assert steps["prompt"]["prompt"] == stored["prompt"]["prompt"]
    assert steps["prompt"]["user_a2a_url"] == stored["prompt"]["user_a2a_url"]  # it also says the user is a human
    assert steps["judge"]["criteria"] == stored["judge"]["criteria"]
    assert steps["skills"]["skills"] == [{"name": "notes", "description": "how notes are kept"}]
    assert steps["agent"]["agent_changelog_object_url"] == "s3://their-bucket/changelog/"  # the state it starts from
    assert steps["agent"]["agent_changelog_toolcall_position_exclusive"] == 0
    assert {(floating.where, floating.id) for floating in imported.floating} >= {
        ("tasks/scoped.json: step 'agent': a2a_agent_id", DEFAULT_AGENT),
        ("tasks/scoped.json: step 'install': a2a_agent_id", DEFAULT_AGENT)}


def test_including_the_user_scope_keeps_each_scoped_field_as_stored(store):
    _task("scoped", SCOPED)
    stored = {step["id"]: step for step in Task.get("scoped").to_dict()["steps"]}

    imported = import_task("scoped", store / "scoped", include_scopes=[FieldScope.USER])

    steps = {step["id"]: step for step in _steps(imported.root, "scoped")}
    assert {(id, path): _value(steps[id], path) for id, paths in SCOPED_FIELDS.items() for path in paths} == {
        (id, path): _value(stored[id], path) for id, paths in SCOPED_FIELDS.items() for path in paths}
    assert imported.dropped == ()


class _AcmeFile(FileArtifact):
    """A plugin's artifact type, which a config file registers here."""

    type: Literal["acme_file_import_test"] = "acme_file_import_test"


def test_what_only_a_plugin_or_a_config_file_registers_is_named_as_a_need_by_what_registers_it(store, monkeypatch):
    steps = {**Config().task_step_registry(), _Undeclared.type: _Undeclared}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    artifacts = {**Config().artifact_registry(), "acme_file_import_test": _AcmeFile}
    monkeypatch.setattr(Config, "artifact_registry", lambda self: artifacts)
    acme = Distribution("agentenv-acme", "1.0", (
        Contribution(TASK_STEPS, _Undeclared.type, f"{_Undeclared.__module__}:{_Undeclared.__qualname__}", "active"),))
    monkeypatch.setattr(importer_module, "inventory", lambda load: Inventory(
        distributions=(acme,), group_errors={}, discovery_errors={}, config_path=None, config_error=None, loaded=load))
    (store / "acme.json").write_text("{}")
    _AcmeFile.put("acme-data", description="acme rows", file_path=str(store / "acme.json"))
    _task("acme", [{"id": "env", "type": "deploy_env", "env_id": "crm-env"},
                   {"id": "load", "type": "load_artifact", "env_id": "crm-env", "artifacts": [{"id": "acme-data"}]},
                   {"id": "p", "type": _Undeclared.type, "depends_on": [{"task_step_id": "load"}]}])

    imported = import_task("acme", store / "acme")

    assert sorted((need.by, need.name, need.what, need.where) for need in imported.needs) == [
        ("agentenv-acme", _Undeclared.type, "step type", "tasks/acme.json: step 'p'"),
        ("configuration", "acme_file_import_test", "artifact type", "tasks/acme.json: step 'load': artifacts[0].id"),
    ]
    assert imported.report()[0] == ("Needs beyond core", [
        "agentenv-acme: undeclared_import_test (step type), at tasks/acme.json: step 'p'",
        "configuration: acme_file_import_test (artifact type), at tasks/acme.json: step 'load': artifacts[0].id",
    ])
    assert "## Needs beyond core" in (imported.root / "README.md").read_text()


def test_a_plugins_type_a_config_file_replaces_is_credited_to_configuration(store, monkeypatch):
    steps = {**Config().task_step_registry(), _Undeclared.type: _Undeclared}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    replaced = Distribution("agentenv-acme", "1.0", (
        Contribution(TASK_STEPS, _Undeclared.type, f"{_AcmeFile.__module__}:{_AcmeFile.__qualname__}", "replaced"),))
    monkeypatch.setattr(importer_module, "inventory", lambda load: Inventory(
        distributions=(replaced,), group_errors={}, discovery_errors={}, config_path=None, config_error=None,
        loaded=load))
    _task("acme", [{"id": "p", "type": _Undeclared.type}])

    assert [need.by for need in import_task("acme", store / "acme").needs] == ["configuration"]


def test_a_skipped_plugins_type_a_config_file_supplies_is_credited_to_configuration_without_loading_it(
        store, monkeypatch):
    steps = {**Config().task_step_registry(), _Undeclared.type: _Undeclared}
    monkeypatch.setattr(Config, "task_step_registry", lambda self: steps)
    site = store / "site"
    site.mkdir()
    (site / "acme_broken_import_test.py").write_text("raise RuntimeError('acme is broken')\n")
    monkeypatch.syspath_prepend(str(site))
    broken = Distribution("agentenv-acme", "1.0", (
        Contribution(TASK_STEPS, _Undeclared.type, "acme_broken_import_test:Step", "unloaded"),))
    monkeypatch.setattr(importer_module, "inventory", lambda load: Inventory(
        distributions=(broken,), group_errors={}, discovery_errors={}, config_path=None, config_error=None,
        loaded=load))
    monkeypatch.setattr(importer_module, "load_failures", lambda: {TASK_STEPS: {_Undeclared.type: "failed to load"}})
    _task("acme", [{"id": "p", "type": _Undeclared.type}])

    assert [need.by for need in import_task("acme", store / "acme").needs] == ["configuration"]


def test_a_task_that_uses_only_what_core_ships_needs_nothing(store):
    assert import_task("triage", store / "triage").needs == ()


def test_the_cli_leaves_out_scoped_fields_unless_told_to_include_their_scope(store):
    _task("scoped", SCOPED)

    left = CliRunner().invoke(cli, ["bundle", "init", str(store / "left"), "--from-task", "scoped"])
    kept = CliRunner().invoke(cli, ["bundle", "init", str(store / "kept"), "--from-task", "scoped",
                                    "--include-scope", "user"])

    assert left.exit_code == 0 and kept.exit_code == 0, left.output + kept.output
    assert ("  tasks/scoped.json: step 'agent': litellm_base_url: dropped: user scope, so a run takes its own; "
            "--include-scope user keeps it") in left.output.splitlines()
    assert "user scope" not in kept.output
    assert _steps(store / "kept", "scoped")[1]["litellm_base_url"] == "https://litellm.internal.example.com"


@pytest.mark.parametrize("pin", [None, 3])
def test_an_installed_bundles_entity_floats_unpinned_so_the_import_runs_it_from_there(store, monkeypatch, pin):
    kit = store / "site" / "kit"
    (kit / "agents/solver").mkdir(parents=True)
    (kit / "agents/solver/Dockerfile").write_text("FROM scratch\n")
    monkeypatch.setattr(importer_module, "installed_bundles",
                        lambda: (InstalledBundle("kit", "demo-dist", "1.0", "demo", kit, None, None),))
    agent = "@local/demo-dist/kit/solver"
    with namespace_routing():
        _task("@local/~/runs/t", [{"id": "agent", "type": "deploy_agent", "env_ids": [], "a2a_agent_id": agent,
                                   "a2a_agent_version": pin}])
        imported = import_task("@local/~/runs/t", store / "imported", include_scopes=[FieldScope.USER])

    assert [(ref.where, ref.id) for ref in imported.floating] == [("tasks/t.json: step 'agent': a2a_agent_id", agent)]
    assert ("dropped" in imported.floating[0].why) == (pin is not None) and imported.pinned == ()
    assert _steps(imported.root, "t")[0]["a2a_agent_version"] is None

