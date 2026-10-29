"""What `agent-env task run` checks of a stored task before its first step: the plugin packages it requires, the types
of the envs and artifacts its steps read, and, in local stores, the infra envs its deploys run on."""

import logging

import click
import pytest
from click.testing import CliRunner

from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store
from agent_env.cli import cli, submitted_runs
from agent_env.cli.task import _checks
from agent_env.config import get_config
from agent_env.env import MCPServerEnv
from agent_env.entity.refs import EntityKind
from agent_env.env.bootstrap import GATEWAY, SERVICE_DB
from agent_env.env.store import ENVS_COLLECTION
from agent_env.plugins import _requirements
from agent_env.store.document_store import UnknownDocumentTypeError
from agent_env.store.routing import namespace_routing
from agent_env.task import Task
from agent_env.task.store import TASKS_COLLECTION
from tst.unit._terminal import at_a_terminal

MISSING = "agentenv-e2e-missing>=1"
REFUSED = f"task t v1: requires '{MISSING}', which isn't installed; install it with: agent-env plugin add '{MISSING}'"


@pytest.fixture
def stores(local_stores, monkeypatch, tmp_path):
    """Local stores holding no infra envs, only agentenv-e2e-acme 1.2 installed beside agent-env, and a run that
    fails the test."""
    monkeypatch.setattr(_requirements, "installed", {"agentenv-framework": "0.9.1380", "agentenv-e2e-acme": "1.2"}.get)

    def ran(*args, **kwargs):
        raise AssertionError("the task ran")

    monkeypatch.setattr(submitted_runs, "follow", ran)
    logging.disable(logging.CRITICAL)
    yield tmp_path
    logging.disable(logging.NOTSET)


def _stored(steps, **fields):
    """Store task ``t`` v1 as its document, steps and all, so one naming a type nothing registers can be stored."""
    get_config().get_document_store().insert(TASKS_COLLECTION, {"id": "t", "type": "task", "version": 1,
                                                                "steps": steps, **fields})


def _run(*args, **kwargs):
    return CliRunner().invoke(cli, ["task", "run", "--id", "t", *args], **kwargs)


def test_a_task_requiring_what_isnt_installed_is_refused_before_it_runs(stores):
    Task.put(id="t", steps=[], requires=[MISSING])

    result = _run()

    assert result.exit_code == 1
    assert f"Error: {REFUSED}" in result.output


def test_a_step_type_nothing_registers_names_the_package_the_task_requires(stores):
    _stored([{"id": "s", "type": "acme_task_run_test"}], requires=[MISSING])
    (stores / "seeds.csv").write_text("n\n1\n")

    ran = _run()
    batch = CliRunner().invoke(cli, ["task", "run-batch", "--id", "t", "--seeds", str(stores / "seeds.csv")])

    for result in (ran, batch):
        assert result.exit_code == 1
        assert f"Error: Unknown task step type: acme_task_run_test; {REFUSED}" in result.output


def test_without_a_record_a_step_type_nothing_registers_says_only_that(stores):
    _stored([{"id": "s", "type": "acme_task_run_test"}])

    result = _run()

    assert result.exit_code == 1
    assert "Unknown task step type: acme_task_run_test" in result.output
    assert "plugin add" not in result.output


def test_an_env_a_step_reads_whose_type_nothing_registers_is_refused_before_it_runs(stores):
    get_config().get_document_store().insert(ENVS_COLLECTION, {"id": "crm", "type": "acme_env_task_run_test",
                                                               "version": 1})
    _stored([{"id": "env", "type": "deploy_env", "env_id": "crm"}], requires=[MISSING])

    result = _run()

    assert result.exit_code == 1
    assert f"Error: {REFUSED}" in result.output
    assert "task t v1: step 'env': env_id 'crm': Unknown env type: acme_env_task_run_test" in result.output


def test_one_the_store_doesnt_hold_or_the_run_fills_in_is_left_to_its_step(stores, monkeypatch):
    _stored([{"id": "env", "type": "deploy_env", "env_id": "absent"},
             {"id": "seeded", "type": "deploy_env", "env_id": "{{seed.env}}"}])
    monkeypatch.setattr(_checks, "stores_are_local", lambda: False)

    assert _checks.runnable_task("t", None, {}, say=print).id == "t"


def test_a_task_whose_requirements_are_met_is_run(stores, monkeypatch):
    Task.put(id="t", steps=[], requires=["agentenv-e2e-acme>=1"])

    assert _checks.runnable_task("t", None, {}, say=print).requires == ["agentenv-e2e-acme>=1"]


@pytest.fixture
def terminal(stores, monkeypatch):
    return at_a_terminal(monkeypatch)


@pytest.mark.parametrize("store", [
    lambda: Task.put(id="t", steps=[], requires=[MISSING]),
    lambda: _stored([{"id": "s", "type": "acme_task_run_test"}], requires=[MISSING]),
], ids=["requirement", "step-type"])
def test_at_a_terminal_it_offers_to_install_what_the_task_requires_and_runs_it_again(terminal, store):
    store()

    result = _run(input="y\n")

    assert result.exit_code == 0, result.output
    assert f"Install {MISSING} with agent-env plugin add? [y/N]: y" in result.output
    assert terminal["added"] == [([MISSING], True)] and len(terminal["exec"]) == 1


def test_a_task_needing_a_newer_agent_env_says_to_upgrade_it_and_never_asks(terminal):
    Task.put(id="t", steps=[], requires=["agentenv-framework>=99"])

    result = _run(input="y\n")

    assert result.exit_code == 1
    assert ("task t v1: requires 'agentenv-framework>=99', and agentenv-framework 0.9.1380 is installed; upgrade "
            "agent-env with the installer that installed it") in result.output
    assert "Install" not in result.output and terminal["added"] == []


@pytest.mark.parametrize("local, overrides, kinds", [
    (True, {}, {GATEWAY, SERVICE_DB}),
    (True, {"gateway_env_id": "mine"}, {SERVICE_DB}),
    (True, {"gateway_env_id": "mine", "service_db_env_id": "mine-db"}, None),
    (True, {"env_state_type": "external_db"}, {GATEWAY}),
    (False, {}, None),
], ids=["local", "own-gateway", "own-both", "external-state", "shared"])
def test_the_infra_envs_its_deploys_run_on_are_set_up_in_local_stores_but_for_those_it_is_given(
        stores, monkeypatch, local, overrides, kinds):
    _mcp_env("crm")
    _stored([{"id": "env", "type": "deploy_env", "env_id": "crm"}])
    set_up = []
    monkeypatch.setattr(_checks, "stores_are_local", lambda: local)
    monkeypatch.setattr(_checks, "ensure_default_envs",
                        lambda kinds, say, missing_only: set_up.append((set(kinds), missing_only)))

    _checks.runnable_task("t", None, {"user_overrides": overrides}, say=print)

    assert set_up == ([(kinds, True)] if kinds else [])


def _mcp_env(id):
    with namespace_routing():
        image = get_artifact_store().put_document(DockerImageArtifact(
            id=f"{id}-image", description=id, image_name=f"{id}:latest", tar_gz_s3_url=None, platform=None))
        MCPServerEnv.put(id=id, docker_image_artifact=image, environment_name=id)


_UNKNOWN_ENV = [{"id": "env", "type": "deploy_env", "env_id": "acme-env"},
                {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"},
                {"id": "load", "type": "load_artifact", "sandbox_name": "box", "artifact_id": "acme-data",
                 "depends_on": [{"task_step_id": "box"}]}]


@pytest.mark.parametrize("metadata", [
    {"start_step": 3},
    {"start_step": 1, "user_overrides": {"step_params": {"load": {"artifact_id": "absent"}}}},
], ids=["resumed-past-them", "replaced"])
def test_what_only_the_steps_a_run_skips_or_its_step_params_replace_read_is_left_alone(stores, monkeypatch, metadata):
    store = get_config().get_document_store()
    store.insert(ENVS_COLLECTION, {"id": "acme-env", "type": "acme_env_task_run_test", "version": 1})
    store.insert("artifacts", {"id": "acme-data", "type": "acme_artifact_task_run_test", "version": 1})
    _stored(_UNKNOWN_ENV)
    monkeypatch.setattr(_checks, "stores_are_local", lambda: False)

    with pytest.raises(click.ClickException) as caught:
        _checks.runnable_task("t", None, {}, say=print)
    assert _checks.runnable_task("t", None, metadata, say=print).id == "t"
    env, load = caught.value.message.splitlines()
    assert env == "task t v1: step 'env': env_id 'acme-env': Unknown env type: acme_env_task_run_test"
    assert load.startswith("task t v1: step 'load': artifacts[0].id 'acme-data': Unknown artifact type "
                           "'acme_artifact_task_run_test'")


@pytest.mark.parametrize("params", [{"load": {"destination_path": "/tmp/data"}}, {"env": {"memory_mb": 4096}}],
                         ids=["load-elsewhere", "other-params"])
def test_step_params_that_leave_what_a_step_references_as_stored_check_it(stores, monkeypatch, params):
    store = get_config().get_document_store()
    store.insert(ENVS_COLLECTION, {"id": "acme-env", "type": "acme_env_task_run_test", "version": 1})
    store.insert("artifacts", {"id": "acme-data", "type": "acme_artifact_task_run_test", "version": 1})
    _stored(_UNKNOWN_ENV)
    monkeypatch.setattr(_checks, "stores_are_local", lambda: False)

    with pytest.raises(click.ClickException) as caught:
        _checks.runnable_task("t", None, {"user_overrides": {"step_params": params}}, say=print)

    assert [line.split(":")[1].strip() for line in caught.value.message.splitlines()] == ["step 'env'", "step 'load'"]


def test_a_reference_many_steps_make_is_read_once_and_named_at_each(stores, monkeypatch):
    reads = []

    def get(id, version):
        reads.append((id, version))
        raise UnknownDocumentTypeError("Unknown artifact type 'acme_artifact_task_run_test'")

    monkeypatch.setitem(_checks._GETTERS, EntityKind.ARTIFACT, get)
    _stored([{"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm"},
             *({"id": f"load{i}", "type": "load_artifact", "sandbox_name": "box", "artifact_id": "acme-data",
                "depends_on": [{"task_step_id": "box"}]} for i in range(3))])

    with pytest.raises(click.ClickException) as caught:
        _checks.runnable_task("t", None, {}, say=print)

    assert reads == [("acme-data", None)]
    assert [line.split(":")[1].strip() for line in caught.value.message.splitlines()] == [
        "step 'load0'", "step 'load1'", "step 'load2'"]


def test_the_infra_envs_are_set_up_only_where_the_store_doesnt_hold_them(stores, monkeypatch):
    _mcp_env("crm")
    _stored([{"id": "env", "type": "deploy_env", "env_id": "crm"}])
    set_up = []
    monkeypatch.setattr(_checks, "stores_are_local", lambda: True)
    monkeypatch.setattr(_checks, "ensure_default_envs",
                        lambda kinds, say, missing_only: set_up.append((set(kinds), missing_only)))

    _checks.runnable_task("t", None, {}, say=print)

    assert set_up == [({GATEWAY, SERVICE_DB}, True)]


def test_a_record_that_isnt_a_requirement_is_refused_and_never_offered(terminal):
    Task.put(id="t", steps=[], requires=["not a requirement!", MISSING])

    result = _run(input="y\n")

    assert result.exit_code == 0, result.output
    assert "task t v1: requires 'not a requirement!', which isn't a pip requirement that can be shown as written" in \
        result.output
    assert terminal["added"] == [([MISSING], True)]
