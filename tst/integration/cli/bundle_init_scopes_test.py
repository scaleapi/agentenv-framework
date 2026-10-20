"""``agent-env bundle init`` for another machine, with no infra: two machines as two local stores, the source's config
registering a sandbox provider and a step type of its own and the receiver's registering nothing. A folder imported by
default leaves the source's setup out and runs on the receiver, which supplies its own; one imported with
``--include-scope user`` runs only where it came from. A step type only the source registers, or an input left out that
has no default, is refused on the receiver before anything runs, naming what's missing. Needs `bash` on PATH."""

import logging
from types import SimpleNamespace
from typing import ClassVar

import pytest
from click.testing import CliRunner

from agent_env.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.artifact.store import get_artifact_store, reset_artifact_store
from agent_env.bundle.resolve import build_step
from agent_env.cli import cli
from agent_env.config import configure, reset_config
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox_provider import (
    reset_agent_sandbox_provider,
    reset_env_sandbox_provider,
    reset_sandbox_provider,
)
from agent_env.task import Task
from agent_env.task_step.task_step import TaskStep

pytestmark = pytest.mark.integration

_HERE = "tst.integration.cli.bundle_init_scopes_test"


class _AcmeVm(LocalSandboxProvider):
    """The source's own sandbox provider: local sandboxes, under its own name."""

    async def create_vm(self, *args, **kwargs):
        sandbox = await super().create_vm(*args, **kwargs)
        sandbox.type = "acme_vm"
        return sandbox

    async def get_sandbox(self, sandbox_id):
        sandbox = await super().get_sandbox(sandbox_id)
        sandbox.type = "acme_vm"
        return sandbox


class _AcmeNote(TaskStep):
    """A step type only the source registers."""

    type: ClassVar[str] = "acme_note_integ_test"
    entity_refs = ()

    def __init__(self, note=None, **base):
        super().__init__(**base)
        self.note = note

    @classmethod
    def from_dict(cls, data):
        return cls(**cls._base_from_dict(data), note=data.get("note"))

    def to_dict(self):
        return {**super().to_dict(), "note": self.note}

    async def execute(self, context):
        return context


SANDBOX = {"id": "box", "type": "deploy_sandbox", "sandbox_name": "box", "sandbox_mode": "vm", "sandbox_type": "acme_vm",
           "metadata": {"attribution": {"project_id": "p-acme"}}}
CHECK = {"id": "check", "type": "verify_sandbox", "sandbox_name": "box", "verifier_id": "check", "base_dir": "/tmp",
         "criteria": [{"type": "bash_cmd_succeeds", "criterion": "the sandbox runs a command", "bash_cmd": "true"}],
         "depends_on": [{"task_step_id": "box"}]}


@pytest.fixture
def machines(monkeypatch, tmp_path):
    """The source machine, current until ``on("receiver")``; each has its own local store. Whatever ran, no sandbox
    work folder may be left behind."""
    sandboxes = tmp_path / "sandboxes"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("AGENT_ENV_DOCUMENT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_OBJECT_STORE", "local")
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(sandboxes))
    monkeypatch.setenv("LITELLM_BASE_URL", "http://127.0.0.1:4000")  # each machine's model endpoint, which no run calls
    monkeypatch.setenv("LITELLM_API_KEY", "sk-test")
    config = tmp_path / "source.toml"
    config.write_text(f'[sandbox.providers]\nacme_vm = "{_HERE}:_AcmeVm"\n\n[task_steps]\nimpls = ["{_HERE}:_AcmeNote"]\n')

    def on(machine):
        monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / machine))
        if machine == "source":
            monkeypatch.setenv("AGENT_ENV_CONFIG", str(config))
        else:
            monkeypatch.delenv("AGENT_ENV_CONFIG", raising=False)
        _reset()
        configure()

    on("source")
    logging.disable(logging.CRITICAL)  # pytest's live logging would take CliRunner's stdout
    try:
        yield SimpleNamespace(root=tmp_path, on=on)
        assert not sandboxes.exists() or not any(sandboxes.iterdir()), "a run left a sandbox work folder"
    finally:
        logging.disable(logging.NOTSET)
        _reset()


def _reset():
    for reset in (reset_config, reset_artifact_store, reset_sandbox_provider, reset_env_sandbox_provider,
                  reset_agent_sandbox_provider):
        reset()


def _put(task_id, steps):
    Task.put(id=task_id, steps=[build_step(dict(step)) for step in steps])


def _agent(agent_id, image_name):
    image = get_artifact_store().put_document(DockerImageArtifact(
        id=f"{agent_id}-image", description=agent_id, image_name=image_name))
    A2AAgent.put(id=agent_id, docker_image_artifact=image)


def _import(machines, task_id):
    """``task_id`` imported by default into a folder of its name, and with the user scope into ``<name>-user``."""
    _ok("bundle", "init", machines.root / task_id, "--from-task", task_id)
    _ok("bundle", "init", machines.root / f"{task_id}-user", "--from-task", task_id, "--include-scope", "user")


def _ok(*args):
    result = CliRunner().invoke(cli, [*map(str, args)])
    assert result.exit_code == 0, result.output
    return result.output


def _refused(*args):
    result = CliRunner().invoke(cli, [*map(str, args)])
    assert result.exit_code == 1, result.output
    return result.output


def test_a_task_imported_without_its_users_setup_runs_on_a_machine_without_that_setup(machines):
    _put("portable", [SANDBOX, CHECK])
    _import(machines, "portable")
    assert "tasks/portable.json v1: passed (check: 1)" in _ok("run", machines.root / "portable-user")

    machines.on("receiver")

    assert "tasks/portable.json v1: passed (check: 1)" in _ok("run", machines.root / "portable")
    assert "Unknown sandbox backend: 'acme_vm'" in _refused("run", machines.root / "portable-user")
    written = "".join(path.read_text() for path in (machines.root / "portable").rglob("*") if path.is_file())
    assert "acme" not in written


def test_an_agent_task_imported_without_its_users_agents_takes_the_receivers_default_agent(machines):
    _agent("acme-solver", "registry.acme.example/solver:v1")
    _put("agents", [
        {"id": "agent", "type": "deploy_agent", "agent_name": "solver", "env_ids": [], "a2a_agent_id": "acme-solver",
         "sandbox_type": "acme_vm", "litellm_base_url": "https://litellm.acme.example", "env_vars": {"ACME_MODE": "1"}},
        {"id": "prompt", "type": "prompt_agent", "agent_name": "solver", "prompt": "Say hello.", "model": "acme/model",
         "trajectory_output_prefix": "s3://acme-trajectories/runs/", "depends_on": [{"task_step_id": "agent"}]},
        {"id": "judge", "type": "rubrics_verifier", "verifier_id": "judge", "prompt_id": "prompt",
         "judge_a2a_agent_id": "acme-solver", "judge_sandbox_type": "acme_vm",
         "criteria": [{"id": "c1", "description": "it says hello"}], "depends_on": [{"task_step_id": "prompt"}]},
    ])
    _import(machines, "agents")
    assert "Would run:" in _ok("run", machines.root / "agents-user", "--dry-run")

    machines.on("receiver")

    shipped = _ok("run", machines.root / "agents", "--dry-run")
    assert "agentenv-framework/agents:agents/claude-code: v1 (new)" in shipped and "Would run:" in shipped
    assert "registry.acme.example/solver:v1" in (machines.root / "agents-user" / "agents" / "acme-solver" /
                                                 "agent.toml").read_text()
    assert "Unknown sandbox backend: 'acme_vm'" in _refused("run", machines.root / "agents-user", "--dry-run")

    configure(default_a2a_agent_id="receiver-default")
    bare = _refused("run", machines.root / "agents", "--dry-run")
    assert "names no agent, so it deploys the default, 'receiver-default', and there is no agent 'receiver-default'" in bare
    _agent("receiver-default", "docker.io/library/python:3.12-slim")
    assert "Would run:" in _ok("run", machines.root / "agents", "--dry-run")
    assert "Unknown sandbox backend: 'acme_vm'" in _refused("run", machines.root / "agents-user", "--dry-run")


def test_a_task_needing_a_step_type_only_its_source_registers_is_refused_on_the_receiver_naming_it(machines):
    _put("noted", [SANDBOX, {"id": "note", "type": _AcmeNote.type, "note": "kept",
                             "depends_on": [{"task_step_id": "box"}]}, CHECK])
    imported = _ok("bundle", "init", machines.root / "noted", "--from-task", "noted")
    assert f"  configuration: {_AcmeNote.type} (step type), at tasks/noted.json: step 'note'" in imported.splitlines()
    assert "tasks/noted.json v1: passed (check: 1)" in _ok("run", machines.root / "noted")

    machines.on("receiver")

    assert f"'{_AcmeNote.type}' is not a known step type" in _refused("run", machines.root / "noted")


def test_an_input_left_out_that_has_no_default_is_refused_on_the_receiver_naming_what_to_supply(machines):
    _put("fetches", [SANDBOX, {"id": "fetch", "type": "load_artifact", "sandbox_name": "box",
                               "urls": ["s3://acme-bucket/rows.csv"], "destination_path": "/tmp/rows",
                               "depends_on": [{"task_step_id": "box"}]}])
    _import(machines, "fetches")
    assert "Would run:" in _ok("run", machines.root / "fetches-user", "--dry-run")
    assert "step 'fetch': urls: dropped: user scope" in (machines.root / "fetches" / "README.md").read_text()

    machines.on("receiver")

    assert "load_artifact 'fetch': names nothing to load" in _refused("run", machines.root / "fetches", "--dry-run")
