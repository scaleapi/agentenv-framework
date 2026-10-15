"""``agent-env bundle init`` on tasks and an eval held in a store, with no infra: the local stores and a local VM-mode
sandbox. The imported folder runs, reruns writing nothing, and keeps running what the store held at import after the
store moves on; an older version imports as asked, a task a bundle ran imports into another folder and runs there, and
secrets left out never reach the run. Needs `bash` and the usual shell tools on PATH."""

import json
import logging
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

import agent_env
from agent_env.artifact import FileArtifactUniverse
from agent_env.artifact.store import reset_artifact_store
from agent_env.bundle import parse_bundle
from agent_env.bundle.resolve import build_step
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.eval import Eval
from agent_env.eval.eval import EvalTask
from agent_env.store.routing import namespace_routing
from agent_env.task import Task

pytestmark = pytest.mark.integration

HELLO = Path(agent_env.__file__).parent / "examples" / "hello"


@pytest.fixture
def registry(monkeypatch, tmp_path):
    """hello's task and greeting, put under bare ids in the configured local store. Whatever ran, no sandbox work
    folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    configure()
    reset_artifact_store()
    greeting = HELLO / "artifacts" / "greeting"
    FileArtifactUniverse.put_bundled("greeting", files={path.name: path for path in greeting.iterdir()})
    Task.put(id="greet", steps=[build_step(step) for step in _hello_steps()])
    Eval.put(id="smoke", tasks=[EvalTask("greet")])
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield tmp_path
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        reset_artifact_store()
        reset_config()


def _hello_steps():
    return json.loads((HELLO / "tasks" / "hello.json").read_text())


def _cli(*args):
    result = CliRunner().invoke(cli, [*map(str, args)])
    assert result.exit_code == 0, result.output
    return result.output


def test_an_imported_task_runs_reruns_writing_nothing_and_keeps_its_pins_when_the_store_moves_on(registry):
    imported = _cli("bundle", "init", registry / "greet", "--from-task", "greet")

    assert "  artifact greeting v1: tasks/greet.json: step 'load': artifacts[0].id" in imported.splitlines()
    first = _cli("run", registry / "greet")
    assert "tasks/greet.json v1: passed (hello: 1)" in first
    second = _cli("run", registry / "greet")
    assert "tasks/greet.json: v1, unchanged" in second
    assert "tasks/greet.json v1: passed (hello: 1)" in second

    (registry / "bye.txt").write_text("bye\n")
    FileArtifactUniverse.put_bundled("greeting", files={"hello.txt": registry / "bye.txt"})
    assert "tasks/greet.json v1: passed (hello: 1)" in _cli("run", registry / "greet")


def test_an_older_version_imports_as_it_was_stored(registry):
    steps = _hello_steps()
    steps[2]["criteria"][0]["expected"] = "goodbye"
    Task.put(id="greet", steps=[build_step(step) for step in steps])

    _cli("bundle", "init", registry / "old", "--from-task", "greet", "--version", 1)
    _cli("bundle", "init", registry / "new", "--from-task", "greet")

    assert (registry / "old" / "README.md").read_text().splitlines()[0] == "Imported from the store's task greet v1."
    assert "tasks/greet.json v1: passed (hello: 1)" in _cli("run", registry / "old")
    assert "tasks/greet.json v1: scored below 1" in _cli("run", registry / "new")


def test_a_task_a_bundle_ran_imports_into_another_folder_and_runs_there(registry):
    shutil.copytree(HELLO, registry / "first")
    assert "tasks/hello.json v1: passed (hello: 1)" in _cli("run", registry / "first")
    first = parse_bundle(registry / "first").id_root

    imported = _cli("bundle", "init", registry / "second", "--from-task", f"{first}/hello")

    assert f"  artifact {first}/greeting v1: tasks/hello.json: step 'load': artifacts[0].id" in imported.splitlines()
    assert "tasks/hello.json v1: passed (hello: 1)" in _cli("run", registry / "second")
    assert "tasks/hello.json: v1, unchanged" in _cli("run", registry / "second")


def test_secrets_left_out_of_an_import_stay_out_of_the_task_its_run_writes(registry):
    key = "AK" + "IA" + "Q4ZX7" * 3 + "W"  # an AWS access key's shape, joined so no scanner takes it for one
    url = "https://u:" + "pw-1234@example.com/x"
    steps = _hello_steps()
    steps[0]["env_vars"] = {"API_TOKEN": "a-stored-token", "NOTE": key, "MODE": "kept"}
    steps[2]["criteria"].append({"type": "bash_cmd_succeeds", "criterion": "fetches", "bash_cmd": f"echo {url}"})
    Task.put(id="guarded", steps=[build_step(step) for step in steps])

    imported = _cli("bundle", "init", registry / "guarded", "--from-task", "guarded").splitlines()
    ran = _cli("run", registry / "guarded")

    at = imported.index("Left out:")
    assert imported[at + 1:at + 4] == [
        "  tasks/guarded.json: step 'box': env_vars.API_TOKEN: dropped: its name looks like a secret",
        "  tasks/guarded.json: step 'box': env_vars.NOTE: dropped: its value holds an AWS access key",
        "  tasks/guarded.json: step 'hello': criteria[2].bash_cmd: masked: a URL's credentials"]
    assert "tasks/guarded.json v1: passed (hello: 1)" in ran
    with namespace_routing():
        run_steps = Task.get(f"{parse_bundle(registry / 'guarded').id_root}/guarded").to_dict()["steps"]
    assert run_steps[0]["env_vars"] == {"MODE": "kept"}
    assert run_steps[2]["criteria"][2]["bash_cmd"] == "echo https://***@example.com/x"
    written = "".join(path.read_text() for path in (registry / "guarded").rglob("*") if path.is_file())
    assert not any(secret in written for secret in ("a-stored-token", key, "pw-1234"))


def test_an_imported_eval_runs_its_tasks(registry):
    imported = _cli("bundle", "init", registry / "nightly", "--from-eval", "smoke")

    assert imported.splitlines()[0] == f"Imported eval smoke v1 and its tasks into {registry / 'nightly'}:"
    output = _cli("run", registry / "nightly")
    assert "tasks/greet.json v1: passed (hello: 1)" in output
    assert "evals/smoke.toml v1: 1 of 1 passed" in output


def test_an_import_that_fails_leaves_nothing_beside_its_destination(registry):
    Task.put(id="broken", steps=[build_step({"id": "load", "type": "load_artifact", "artifact_id": "gone",
                                             "sandbox_name": "box", "destination_path": "/app"})])
    before = sorted(path.name for path in registry.iterdir())

    result = CliRunner().invoke(cli, ["bundle", "init", str(registry / "broken"), "--from-task", "broken"])

    assert result.exit_code == 1
    assert "there is no artifact 'gone' in the store" in result.output
    assert sorted(path.name for path in registry.iterdir()) == before
