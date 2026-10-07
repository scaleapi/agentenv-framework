"""Every local sandbox shares this machine's Docker and /tmp. What a step names on one (a container, its network,
image and build context, an agent's Docker sidecar) is made that sandbox's own, published ports are the sandbox's
mapped ones, a sandbox rebuilt from its id keeps that map, and a build uses this machine's platform."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import AsyncMock

import pytest

from agent_env.a2a_agent import a2a_agent as a2a_agent_module
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.env.env import DeployedEnv
from agent_env.providers.sandbox_providers import local_sandbox as local_sandbox_module
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.providers.sandbox_providers.sandbox import VmSandbox
from agent_env.task_step.context import DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.deploy_agent import _choose_mcp_url
from agent_env.task_step.task_steps.install_agent import InstallAgentTaskStep
from agent_env.task_step.task_steps.run_docker_container import RunDockerContainerTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)

SANDBOX_ID = "local-ab12cd34"
HOST_PORT = 50123


class _RecordingSandbox(LocalSandbox):
    host_ips = ("127.0.0.1",)

    def __init__(self, work_dir, *, build_error: str | None = None):
        super().__init__(sandbox_id=SANDBOX_ID, work_dir=work_dir, port_map={8000: HOST_PORT})
        self.scripts: list[str] = []
        self._build_error = build_error

    async def exec_script(self, script, *, max_retries=0):
        self.scripts.append(script)
        if self._build_error and "docker build " in script and "--platform" not in script:
            raise RuntimeError(self._build_error)
        return ""

    async def exec_with_output(self, *args):
        self.scripts.append(args[-1])
        return 0, "", ""

    async def docker_cp(self, source, destination, *, remove_source=False):
        self.scripts.append(f"docker cp {source} {destination}")


def _on(monkeypatch, sandbox) -> TaskStepContext:
    """A context whose sandbox 'h' is ``sandbox``."""
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider",
                        lambda _type: types.SimpleNamespace(get_sandbox=AsyncMock(return_value=sandbox)))
    return TaskStepContext(deployed_sandboxes=[DeployedSandbox(
        sandbox_name="h", sandbox_id=SANDBOX_ID, sandbox_mode="vm", sandbox_type="local",
        tunnel_urls={"8000": f"http://127.0.0.1:{HOST_PORT}"})])


def test_a_local_sandbox_rebuilt_from_its_id_keeps_its_ports(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_ENV_LOCAL_SANDBOX_DIR", str(tmp_path))
    monkeypatch.setattr(local_sandbox_module, "_free_host_port", lambda: HOST_PORT)
    created = asyncio.run(LocalSandboxProvider().create_vm(exposed_ports=[8000]))

    rebuilt = asyncio.run(LocalSandboxProvider().get_sandbox(created.sandbox_id))

    assert rebuilt.host_port(8000) == HOST_PORT and rebuilt.tunnel_urls == created.tunnel_urls


def test_only_a_shared_host_scopes_names(tmp_path):
    local = LocalSandbox(sandbox_id=SANDBOX_ID, work_dir=tmp_path)

    assert local.scoped_name("task-container") == f"task-container-{SANDBOX_ID}"
    assert VmSandbox.scoped_name(local, "task-container") == "task-container"


@pytest.mark.asyncio
async def test_a_step_container_on_a_local_sandbox_is_the_sandboxs_own(tmp_path, monkeypatch):
    sandbox = _RecordingSandbox(tmp_path)
    context = _on(monkeypatch, sandbox)
    monkeypatch.setattr(RunDockerContainerTaskStep, "_stage_from_universe", AsyncMock())

    context = await RunDockerContainerTaskStep(
        id="box", version=None, sandbox_name="h", docker_context_artifact_id="u", docker_context_artifact_version=1,
        container_name="task-container", network="task-net", ports=[8000],
    ).execute(context)

    own = f"task-container-{SANDBOX_ID}"
    [build] = [s for s in sandbox.scripts if "docker build " in s]
    [run] = [s for s in sandbox.scripts if "docker run -d" in s]
    assert f"cd /tmp/docker-context-{own} && docker build --label" in build and f"-t {own}:latest" in build
    assert f"rm -rf /tmp/docker-context-{own}" in sandbox.scripts[sandbox.scripts.index(build) + 1]
    assert f"--name {own} " in run and f"--network task-net-{SANDBOX_ID} " in run
    assert f"-p 127.0.0.1:{HOST_PORT}:8000" in run
    assert context.metadata["deployed_docker_containers"][0]["container_name"] == "task-container"


@pytest.mark.asyncio
@pytest.mark.parametrize("error, falls_back", [
    ("ERROR: failed to solve: no match for platform in manifest: not found", True),
    ("no matching manifest for linux/arm64/v8 in the manifest list entries", True),
    ("ERROR: failed to solve: process \"/bin/sh -c make\" did not complete successfully", False),
])
async def test_a_build_falls_back_to_amd64_only_for_a_base_image_without_this_platform(
    tmp_path, monkeypatch, error, falls_back,
):
    sandbox = _RecordingSandbox(tmp_path, build_error=error)
    context = _on(monkeypatch, sandbox)
    monkeypatch.setattr(RunDockerContainerTaskStep, "_stage_from_universe", AsyncMock())
    step = RunDockerContainerTaskStep(id="box", version=None, sandbox_name="h", docker_context_artifact_id="u",
                                      docker_context_artifact_version=1)

    if falls_back:
        await step.execute(context)
    else:
        with pytest.raises(RuntimeError, match="did not complete"):
            await step.execute(context)

    builds = [s for s in sandbox.scripts if "docker build " in s]
    assert ["--platform linux/amd64" in b for b in builds] == ([False, True] if falls_back else [False])


@pytest.mark.asyncio
async def test_a_verifier_runs_in_the_sandboxs_own_container(tmp_path, monkeypatch, local_stores):
    sandbox = _RecordingSandbox(tmp_path)
    context = _on(monkeypatch, sandbox)
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    monkeypatch.setattr(RunContainerUnitTestsVerifierTaskStep, "_upload_text_artifact", staticmethod(
        lambda text, artifact_id, description, object_url: types.SimpleNamespace(
            id=artifact_id, version=1, object_url=object_url)))

    await RunContainerUnitTestsVerifierTaskStep(id="t", version=None, sandbox_name="h", container_name="c",
                                                command="true", reward_path="/tmp/reward").execute(context)

    assert any(f"c-{SANDBOX_ID} timeout --kill-after" in s for s in sandbox.scripts)
    assert any(s.startswith(f"docker cp c-{SANDBOX_ID}:/tmp/reward ") for s in sandbox.scripts)


@pytest.mark.asyncio
@pytest.mark.parametrize("container_name, command, expected", [
    ("c", "docker exec {container} serve --port {a2a_port}", f"docker exec c-{SANDBOX_ID} serve --port 8000"),
    (None, "serve --port {a2a_port}", f"serve --port {HOST_PORT}"),
], ids=["into-a-container", "on-the-host"])
async def test_an_installed_agent_uses_the_sandboxs_container_and_published_port(
    tmp_path, monkeypatch, container_name, command, expected,
):
    sandbox = _RecordingSandbox(tmp_path)
    sandbox.load_docker_images = AsyncMock()
    sandbox.load_object_file = AsyncMock()
    context = _on(monkeypatch, sandbox)
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    agent = types.SimpleNamespace(id="agent", version=1, metadata={"default_model": "m"},
                                  docker_image_artifact=types.SimpleNamespace(image_name="img",
                                                                              build_context_object_url="file:///ctx"))
    monkeypatch.setattr(A2AAgent, "get", staticmethod(lambda *a, **k: agent))
    key = "install_commands" if container_name else "install_commands_host"
    params = {key: [command], "a2a_port": 8000, "required_params" if container_name else "required_params_host":
              ["container", "a2a_port"] if container_name else ["a2a_port"]}
    monkeypatch.setattr(InstallAgentTaskStep, "_extract_install_extension", AsyncMock(return_value={"params": params}))
    monkeypatch.setattr(InstallAgentTaskStep, "_wait_for_agent_card", AsyncMock(return_value={}))
    monkeypatch.setattr(InstallAgentTaskStep, "_register_instance", lambda *a: types.SimpleNamespace(instance_id="i"))

    await InstallAgentTaskStep(id="install", version=None, sandbox_name="h", container_name=container_name,
                               a2a_agent_id="agent").execute(context)

    assert expected in sandbox.scripts
    removed = f"rm -rf /tmp/install-agent-default-agent-{SANDBOX_ID}" in sandbox.scripts
    assert removed is (container_name is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_docker", [False, True])
async def test_an_agent_placed_on_a_local_sandbox_is_published_and_labeled_as_its_own(
    tmp_path, monkeypatch, enable_docker,
):
    monkeypatch.setattr(a2a_agent_module, "local_grant_trust", lambda: None)
    monkeypatch.setattr(a2a_agent_module.asyncio, "sleep", AsyncMock())
    sandbox = _RecordingSandbox(tmp_path)
    agent = A2AAgent.__new__(A2AAgent)
    agent._sandbox = sandbox

    await agent._run_container("img", 8000, {}, enable_docker=enable_docker)

    script = sandbox.scripts[0]
    assert f"-p 127.0.0.1:{HOST_PORT}:8000" in script and f"--label agentenv.sandbox={SANDBOX_ID}" in script
    if enable_docker:
        assert f"DIND_CONTAINER=agent-dind-{SANDBOX_ID}" in script and f"--network agent-docker-net-{SANDBOX_ID}" in script
        assert f"DOCKER_HOST='tcp://agent-dind-{SANDBOX_ID}:2375'" in script
        assert f"DIND_LABEL=agentenv.sandbox={SANDBOX_ID}" in script


@pytest.mark.parametrize("sandbox_type, expected", [
    ("local", "http://host.docker.internal:8000/mcp"), ("modal", "http://localhost:8000/mcp"),
])
def test_an_env_outside_our_sandboxes_is_at_a_url_on_this_machine(sandbox_type, expected):
    env = DeployedEnv(env_id="mine", env_version=1, mcp_url="http://localhost:8000/mcp")

    assert _choose_mcp_url(types.SimpleNamespace(sandbox_type=sandbox_type), env) == expected
