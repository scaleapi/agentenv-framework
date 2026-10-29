"""What `agent-env task run` checks of a stored task before its first step: the plugin packages it requires, the types
of the envs and artifacts its steps read, and, in local stores, the infra envs its deploys run on."""

import logging

import pytest
from click.testing import CliRunner

from agent_env.cli import _install_prompt, cli, submitted_runs
from agent_env.cli.task import _checks
from agent_env.config import get_config
from agent_env.env.bootstrap import GATEWAY, SERVICE_DB
from agent_env.env.store import ENVS_COLLECTION
from agent_env.plugins import _requirements
from agent_env.task import Task
from agent_env.task.store import TASKS_COLLECTION

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

    assert _checks.runnable_task("t", None, say=print).id == "t"


def test_a_task_whose_requirements_are_met_is_run(stores, monkeypatch):
    Task.put(id="t", steps=[], requires=["agentenv-e2e-acme>=1"])

    assert _checks.runnable_task("t", None, say=print).requires == ["agentenv-e2e-acme>=1"]


@pytest.fixture
def terminal(stores, monkeypatch):
    """A person at a terminal, an install that records what it's asked for, and a re-exec that stops the test."""
    calls = {"added": [], "exec": []}

    def execv(path, argv):
        calls["exec"].append(argv)
        raise SystemExit(0)

    monkeypatch.setattr(_install_prompt, "_interactive", lambda: True)
    monkeypatch.setattr(_install_prompt._plugin_changes, "add", lambda specs, **kwargs: calls["added"].append(specs))
    monkeypatch.setattr(_install_prompt.os, "execv", execv)
    monkeypatch.setenv(_install_prompt._ASKED, "")
    return calls


@pytest.mark.parametrize("store", [
    lambda: Task.put(id="t", steps=[], requires=[MISSING]),
    lambda: _stored([{"id": "s", "type": "acme_task_run_test"}], requires=[MISSING]),
], ids=["requirement", "step-type"])
def test_at_a_terminal_it_offers_to_install_what_the_task_requires_and_runs_it_again(terminal, store):
    store()

    result = _run(input="y\n")

    assert result.exit_code == 0, result.output
    assert f"Install {MISSING} with agent-env plugin add? [y/N]: y" in result.output
    assert terminal["added"] == [[MISSING]] and len(terminal["exec"]) == 1


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
    (False, {}, None),
], ids=["local", "own-gateway", "own-both", "shared"])
def test_the_infra_envs_its_deploys_run_on_are_set_up_in_local_stores_but_for_those_it_is_given(
        stores, monkeypatch, local, overrides, kinds):
    Task.put(id="t", steps=[])
    set_up = []
    monkeypatch.setattr(_checks, "stores_are_local", lambda: local)
    monkeypatch.setattr(_checks, "stored_infra", lambda steps: frozenset({GATEWAY, SERVICE_DB}))
    monkeypatch.setattr(_checks, "ensure_default_envs", lambda kinds, say: set_up.append(set(kinds)))

    _checks.runnable_task("t", None, say=print, **overrides)

    assert set_up == ([kinds] if kinds else [])
