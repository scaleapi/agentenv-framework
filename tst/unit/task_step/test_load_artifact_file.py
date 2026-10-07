"""A single-file artifact loads like a one-file universe: at ``<destination>/<filename>``, on every
target ``load_artifact`` supports, through each target's real loader."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import store as agent_store
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.env import env as env_module
from agent_env.env.env import DeployedGatewayEnv, Env
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep

CHECK = FileArtifact(
    id="check", version=2, description="checker", filename="check.py",
    content_type="text/x-python", s3_url="s3://bucket/check.py",
)


class _Sandbox:
    """Records every command and S3 pull a loader sends to a sandbox."""

    def scoped_name(self, name):
        return name

    def __init__(self):
        self.commands: list[str] = []
        self.pulls: list[tuple[str, str]] = []

    async def exec_script(self, script, **kw):
        self.commands.append(script)
        return ""

    async def exec(self, *command):
        self.commands.append(" ".join(command))

    async def docker_cp(self, source, destination, *, remove_source=False):
        self.commands.append(f"docker cp {source} {destination}")

    async def load_object_file(self, object_url, destination_path):
        self.pulls.append((object_url, destination_path))

    async def write_file_from_object(self, object_url, destination_path):
        self.pulls.append((object_url, destination_path))


def _provider(sandbox: _Sandbox):
    async def _get_sandbox(sandbox_id):
        return sandbox

    return lambda: SimpleNamespace(get_sandbox=_get_sandbox)


@pytest.fixture(autouse=True)
def stored(monkeypatch):
    monkeypatch.setattr(Artifact, "get", classmethod(lambda cls, id, version=None: CHECK))


@pytest.fixture
def box(monkeypatch):
    """A VM sandbox named ``box`` in context."""
    sandbox = _Sandbox()
    monkeypatch.setattr(sandbox_provider, "get_sandbox_provider", _provider(sandbox))
    context = TaskStepContext()
    context.deployed_sandboxes = [DeployedSandbox(sandbox_name="box", sandbox_id="sb-1", sandbox_mode="vm")]
    return context, sandbox


def _entry(context: TaskStepContext) -> dict:
    (entry,) = context.metadata["loaded_file_artifact_universes"]
    return entry


def test_a_file_is_a_one_file_tree():
    assert CHECK.get_file_artifacts() == {"check.py": CHECK}


@pytest.mark.asyncio
async def test_onto_a_sandbox_vm_host(box):
    context, sandbox = box
    step = LoadArtifactTaskStep(
        id="stage", version=None, sandbox_name="box", artifact_id="check", destination_path="/app/greeting",
    )
    context = await step.execute(context)

    assert sandbox.pulls == [("s3://bucket/check.py", "/app/greeting/check.py")]
    entry = _entry(context)
    assert (entry["id"], entry["version"], entry["artifact_type"]) == ("check", 2, "file")
    assert (entry["sandbox_name"], entry["destination_path"], entry["files"]) == ("box", "/app/greeting", ["check.py"])


@pytest.mark.asyncio
async def test_into_a_container_on_a_sandbox(box):
    context, sandbox = box
    context.metadata["deployed_docker_containers"] = [{"container_name": "app", "sandbox_name": "box"}]
    step = LoadArtifactTaskStep(
        id="stage", version=None, sandbox_name="box", container_name="app",
        artifact_id="check", destination_path="/app/greeting",
    )
    context = await step.execute(context)

    copies = [command for command in sandbox.commands if "docker cp" in command]
    assert len(copies) == 1 and "app:/app/greeting/check.py" in copies[0]
    assert (_entry(context)["container_name"], _entry(context)["files"]) == ("app", ["check.py"])


@pytest.mark.asyncio
async def test_into_a_deployed_env(monkeypatch):
    sandbox = _Sandbox()

    class _Env:
        load_file_artifact_universe = Env.load_file_artifact_universe

        @classmethod
        async def from_deployed_env(cls, deployed):
            env = cls()
            env._sandbox = sandbox
            return env

    monkeypatch.setattr(env_module.Env, "get", classmethod(lambda cls, id, version=None: _Env()))
    context = TaskStepContext()
    context.deployed_envs = [DeployedGatewayEnv(
        env_id="tickets", env_version=1, gateway_url="https://gw", mcp_url="https://gw/mcp",
        db_web_url=None, sandbox_id="sb-env",
    )]
    step = LoadArtifactTaskStep(id="stage", version=None, env_id="tickets", artifact_id="check", destination_path="/data")
    context = await step.execute(context)

    assert sandbox.pulls == [("s3://bucket/check.py", "/data/check.py")]
    entry = _entry(context)
    assert (entry["env_id"], entry["artifact_type"], entry["files"]) == ("tickets", "file", {"check.py": "/data/check.py"})


@pytest.mark.asyncio
async def test_into_an_agent(monkeypatch):
    sandbox = _Sandbox()
    monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", _provider(sandbox))
    deployed = SimpleNamespace(sandbox_type=None, sandbox_id="sb-agent")
    monkeypatch.setattr(
        agent_store, "get_a2a_agent_instance_store",
        lambda: SimpleNamespace(get=lambda instance_id: deployed),
    )
    context = TaskStepContext()
    context.deployed_agents = [DeployedAgent(agent_name="solver", api_url="https://agent", instance_id="inst-1")]
    step = LoadArtifactTaskStep(id="stage", version=None, agent_name="solver", artifact_id="check")
    context = await step.execute(context)

    assert sandbox.pulls == [("s3://bucket/check.py", "/tmp/file_artifacts/check.py")]
    entry = _entry(context)
    assert (entry["agent_name"], entry["artifact_type"], entry["files"]) == (
        "solver", "file", {"check.py": "/tmp/file_artifacts/check.py"},
    )
