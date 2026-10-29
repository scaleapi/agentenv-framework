"""A single-file artifact loads like a one-file universe: at ``<destination>/<filename>``, on every
target ``load_artifact`` supports, through each target's real loader."""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from agent_env.a2a_agent import store as agent_store
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifact import Artifact
from agent_env.artifact.artifacts.file import FileArtifact
from agent_env.env import env as env_module
from agent_env.env.env import DeployedGatewayEnv, Env
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.sandbox import CURL_RETRY_FLAGS, VmSandbox
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from tst.util.exec_scripts import script_run

CHECK = FileArtifact(
    id="check", version=2, description="checker", filename="check.py",
    content_type="text/x-python", s3_url="s3://bucket/check.py",
)
TOOL = SimpleNamespace(id="tool", version=1, command_name="tool", entrypoint="bin/tool",
                       get_cli_files=lambda: SimpleNamespace(get_file_artifacts=lambda: {"bin/tool": CHECK}))


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


class _Vm(VmSandbox):
    """A VM that records the scripts it runs on its host and the objects it pulls there."""

    def __init__(self):
        self.sandbox_id = "sb-agent"
        self.scripts: list[str] = []

    async def terminate(self) -> None:  # pragma: no cover
        pass

    async def exec(self, *command):  # pragma: no cover
        return None

    async def exec_with_output(self, *args):
        self.scripts.append(script_run(args) if args[:3] == ("sudo", "bash", "-c") else " ".join(args))
        return 0, "", ""

    async def load_object_file(self, object_url, destination_path):
        self.scripts.append(f"pull {object_url} {destination_path}")


async def _load_into_agent(monkeypatch, installed_in: str | None, on_host: bool = False, **load) -> list[str]:
    """The scripts a load aimed at an agent runs on its VM, its staging paths as one name."""
    sandbox = _Vm()
    monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", _provider(sandbox))
    monkeypatch.setattr(agent_store, "get_a2a_agent_instance_store",
                        lambda: SimpleNamespace(get=lambda instance_id: SimpleNamespace(sandbox_type=None,
                                                                                         sandbox_id="sb-agent")))
    context = TaskStepContext()
    context.deployed_agents = [DeployedAgent(agent_name="solver", api_url="https://agent", sandbox_id="sb-agent",
                                             instance_id="inst-1", installed_container=installed_in, on_host=on_host)]
    await LoadArtifactTaskStep(id="stage", version=None, agent_name="solver", **load).execute(context)
    return _unstaged(sandbox.scripts)


def _unstaged(scripts: list[str]) -> list[str]:
    """``scripts`` with each write's staging path, unique per call, as one name."""
    return [re.sub(r"/tmp/_(\w+?)_[0-9a-f]{12}_", r"/tmp/_\1_", script) for script in scripts]


@pytest.mark.asyncio
@pytest.mark.parametrize("load", [{"artifact_id": "check"}, {"urls": ["https://example.com/check.py"]}],
                         ids=["artifact", "url"])
async def test_into_an_agent_installed_into_a_container(monkeypatch, load):
    deployed = await _load_into_agent(monkeypatch, None, **load)
    in_its_own = await _load_into_agent(monkeypatch, "agent-api", **load)
    in_another = await _load_into_agent(monkeypatch, "task", **load)

    assert in_its_own == deployed
    assert [s.split(" ")[3] for s in in_another if s.startswith("docker cp ")] == [
        "task:/tmp/file_artifacts/check.py"]
    assert not any("agent-api" in s for s in in_another)


@pytest.mark.asyncio
@pytest.mark.parametrize("load, onto_the_host", [
    ({"artifact_id": "check"}, "pull s3://bucket/check.py /tmp/file_artifacts/check.py"),
    ({"urls": ["https://example.com/check.py"]},
     f"curl -fsSL {CURL_RETRY_FLAGS} https://example.com/check.py -o /tmp/file_artifacts/check.py"),
], ids=["artifact", "url"])
async def test_into_an_agent_installed_on_the_vm_host(monkeypatch, load, onto_the_host):
    assert await _load_into_agent(monkeypatch, None, on_host=True, **load) == [
        "mkdir -p /tmp/file_artifacts", onto_the_host]


@pytest.mark.asyncio
async def test_a_cli_reaches_an_agent_installed_into_a_container(monkeypatch):
    async def _install(installed_in: str | None) -> list[str]:
        sandbox = _Vm()
        monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", _provider(sandbox))
        await A2AAgent.install_cli(SimpleNamespace(sandbox_type=None, sandbox_id="sb-agent"), TOOL, "https://gateway",
                                   installed_in)
        return _unstaged(sandbox.scripts)

    deployed, in_its_own, in_another = await _install(None), await _install("agent-api"), await _install("task")

    assert in_its_own == deployed
    assert "docker exec -u 0 task chmod +x /opt/cli/tool/bin/tool" in in_another
    assert not any("agent-api" in s for s in in_another)


@pytest.mark.asyncio
async def test_a_cli_installs_on_the_vm_host_of_an_agent_installed_there(monkeypatch):
    sandbox = _Vm()
    monkeypatch.setattr(sandbox_provider, "get_agent_sandbox_provider", _provider(sandbox))

    path = await A2AAgent.install_cli(SimpleNamespace(sandbox_type=None, sandbox_id="sb-agent"), TOOL, "https://gateway",
                                      on_host=True)

    assert path == "/opt/cli/tool/bin/tool"
    assert [s for s in sandbox.scripts if not s.startswith("base64 -d")] == [
        "mkdir -p /opt/cli/tool/bin", "mkdir -p /opt/cli/tool", "mkdir -p /opt/cli/tool/bin",
        "pull s3://bucket/check.py /opt/cli/tool/bin/tool", "chmod +x /opt/cli/tool/bin/tool"]
    assert [s.split("\n")[0] for s in sandbox.scripts if s.startswith("base64 -d")] == [
        "base64 -d <<'ENDB64' > /opt/cli/tool/bin/.env"]
