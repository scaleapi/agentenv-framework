"""A URL on this machine that agent-env hands into a container on it reaches the host: an agent's, a unit-test
verifier's and an installed agent's model endpoint, a peer's and a trigger executor's A2A URL, and a CLI's gateway
URL. Containers elsewhere get them as they are."""

from __future__ import annotations

import asyncio
import json
import types

import httpx
import pytest

from agent_env.a2a_agent import store as agent_store
from agent_env.a2a_agent.a2a_agent import A2AAgent
from agent_env.artifact.artifacts.cli import CliArtifact
from agent_env.artifact.artifacts.docker_image import DockerImageArtifact
from agent_env.env.env import DeployedGatewayEnv
from agent_env.providers.sandbox_providers import sandbox_provider
from agent_env.providers.sandbox_providers.local_sandbox import LocalSandbox, LocalSandboxProvider
from agent_env.task_step.context import DeployedAgent, DeployedSandbox, TaskStepContext
from agent_env.task_step.task_steps.install_agent import _build_param_resolvers
from agent_env.task_step.task_steps.load_artifact import LoadArtifactTaskStep
from agent_env.task_step.task_steps.peer_agents import PeerAgentsTaskStep
from agent_env.task_step.task_steps.verifiers.run_container_unit_tests_verifier import (
    RunContainerUnitTestsVerifierTaskStep,
)

MODEL = "http://localhost:4000/v1"
MODEL_IN_CONTAINER = "http://host.docker.internal:4000/v1"


@pytest.fixture
def model_endpoint(monkeypatch):
    monkeypatch.setenv("LITELLM_BASE_URL", MODEL)
    monkeypatch.setenv("LITELLM_API_KEY", "k")


class _Started(Exception):
    """Raised once a fake has seen the container's env, which is all a test reads."""


class _LocalProvider(LocalSandboxProvider):
    async def create_sandbox(self, *, env, **_):
        self.env = env
        raise _Started


class _RemoteProvider:
    async def create_sandbox(self, *, env, **_):
        self.env = env
        raise _Started


def _agent(**fields) -> A2AAgent:
    return A2AAgent(id="solver", version=1, docker_image_artifact=DockerImageArtifact(id="img", description="d", image_name="img"), **fields)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_cls, expected", [(_LocalProvider, MODEL_IN_CONTAINER), (_RemoteProvider, MODEL)])
@pytest.mark.parametrize("passed", [False, True], ids=["configured", "passed"])
async def test_an_agent_gets_the_model_endpoint_its_container_reaches(
    monkeypatch, local_stores, model_endpoint, provider_cls, expected, passed,
):
    """Whether configured or passed in, as a deploy_agent step and an agent judge pass it."""
    provider = provider_cls()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)

    with pytest.raises(_Started):
        await _agent().deploy(sandbox_type="any", env_vars={"LITELLM_BASE_URL": MODEL} if passed else None)

    assert provider.env["LITELLM_BASE_URL"] == expected


@pytest.mark.asyncio
async def test_an_agent_placed_on_a_local_sandbox_reaches_the_model_endpoint(monkeypatch, local_stores, model_endpoint, tmp_path):
    started = {}

    async def run_container(self, image_name, a2a_port, merged_env, enable_docker=False):
        started.update(merged_env)
        raise _Started

    async def no_images(artifacts):
        return None

    sandbox = LocalSandbox(work_dir=tmp_path)
    sandbox.load_docker_images = no_images
    monkeypatch.setattr(A2AAgent, "_run_container", run_container)

    with pytest.raises(_Started):
        await _agent().deploy(sandbox=sandbox)

    assert started["LITELLM_BASE_URL"] == MODEL_IN_CONTAINER


@pytest.mark.asyncio
async def test_an_agents_own_model_endpoint_is_left_as_it_declares(monkeypatch, local_stores, model_endpoint):
    provider = _LocalProvider()
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)

    with pytest.raises(_Started):
        await _agent(default_env_vars={"LITELLM_BASE_URL": "http://localhost:9000"}).deploy(sandbox_type="local")

    assert provider.env["LITELLM_BASE_URL"] == "http://localhost:9000"


class _VerifierSandbox:
    def scoped_name(self, name):
        return name

    def __init__(self, sandbox_type: str):
        self.type = sandbox_type
        self.commands: list[str] = []

    async def exec_script(self, script: str) -> str:
        self.commands.append(script)
        return ""

    async def exec_with_output(self, *args) -> tuple[int, str, str]:
        self.commands.append(args[-1])
        return 0, "ok", ""


def _run_verifier(monkeypatch, sandbox_type: str, **step_fields) -> str:
    """The verifier's test command, as it ran on a sandbox of ``sandbox_type``."""
    sandbox = _VerifierSandbox(sandbox_type)
    provider = types.SimpleNamespace(get_sandbox=lambda sandbox_id: asyncio.sleep(0, result=sandbox))
    monkeypatch.setattr(sandbox_provider, "build_sandbox_provider", lambda name: provider)
    monkeypatch.setattr(RunContainerUnitTestsVerifierTaskStep, "_upload_text_artifact", staticmethod(
        lambda text, artifact_id, description, object_url: types.SimpleNamespace(
            id=artifact_id, version=1, object_url=object_url)))
    context = TaskStepContext()
    context.metadata["deployed_docker_containers"] = [{"container_name": "c", "sandbox_name": "h"}]
    context.deployed_sandboxes.append(
        DeployedSandbox(sandbox_name="h", sandbox_id="sb-1", sandbox_mode="vm", sandbox_type=sandbox_type))

    asyncio.run(RunContainerUnitTestsVerifierTaskStep(
        id="tests", version=None, sandbox_name="h", container_name="c", command="true", **step_fields).execute(context))

    [run] = [c for c in sandbox.commands if "timeout --kill-after" in c]
    return run


@pytest.mark.parametrize("sandbox_type, expected", [("local", MODEL_IN_CONTAINER), ("modal_vm", MODEL)])
def test_a_unit_test_verifier_gets_the_model_endpoint_its_container_reaches(
    monkeypatch, local_stores, model_endpoint, sandbox_type, expected,
):
    run = _run_verifier(monkeypatch, sandbox_type)

    assert f"-e LITELLM_BASE_URL={expected} " in run and f"-e ANTHROPIC_BASE_URL={expected} " in run


def test_a_model_endpoint_the_verifier_sets_itself_is_reached_through_the_host(monkeypatch, local_stores, model_endpoint):
    run = _run_verifier(monkeypatch, "local", env_vars={"LITELLM_BASE_URL": "http://127.0.0.1:5001"})

    assert "-e LITELLM_BASE_URL=http://host.docker.internal:5001 " in run


@pytest.mark.parametrize("sandbox_type, expected", [
    ("local", MODEL_IN_CONTAINER), ("modal_vm", MODEL), (None, MODEL),
], ids=["local-container", "remote-container", "host-install"])
def test_an_installed_agent_gets_the_model_endpoint_it_reaches(sandbox_type, expected):
    resolvers = _build_param_resolvers(
        container_name="c", work_dir="/w", agent_ctx_tar="/w/a.tgz", a2a_port=8000, agent_name="solver",
        config=types.SimpleNamespace(get_litellm_base_url=lambda: MODEL), sandbox_type=sandbox_type,
    )

    assert resolvers["litellm_base_url"]() == expected


_PEERS_CARD = {"capabilities": {"extensions": [
    {"uri": A2AAgent.EXT_PEER_AGENTS, "params": {"endpoint": "/ext/peer-agents"}}]}}
_HUMAN = "http://localhost:18000/api/v1/a2a/human/instance/i-1"


def _record_posts(monkeypatch) -> list[httpx.Request]:
    sent, real = [], httpx.AsyncClient

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: real(transport=httpx.MockTransport(handle)))
    return sent


def _peering_context(source_type: str) -> TaskStepContext:
    return TaskStepContext(deployed_agents=[
        DeployedAgent(agent_name="source", api_url="http://source", a2a_url="http://source", sandbox_id="sb-source",
                      sandbox_type=source_type, a2a_card=_PEERS_CARD),
        DeployedAgent(agent_name="local", api_url="http://127.0.0.1:41001", a2a_url="http://127.0.0.1:41001",
                      sandbox_id="local-1", sandbox_type="local"),
        DeployedAgent(agent_name="human", api_url=_HUMAN, a2a_url=_HUMAN),
        DeployedAgent(agent_name="remote", api_url="https://peer.modal.run", a2a_url="https://peer.modal.run",
                      sandbox_id="sb-2", sandbox_type="modal"),
    ])


async def _peer_urls(monkeypatch, source_type: str) -> dict[str, str]:
    sent = _record_posts(monkeypatch)
    step = PeerAgentsTaskStep(id="peers", version=None, peerings=[
        {"source_agent_name": "source", "peer_agent_names": ["local", "human", "remote"]}])

    await step.execute(_peering_context(source_type))

    [request] = sent
    return {peer["name"]: peer["url"] for peer in json.loads(request.content)["peers"]}


@pytest.mark.asyncio
async def test_a_local_agent_reaches_its_peers_on_this_machine_through_the_host(monkeypatch):
    assert await _peer_urls(monkeypatch, "local") == {
        "local": "http://host.docker.internal:41001",
        "human": "http://host.docker.internal:18000/api/v1/a2a/human/instance/i-1",
        "remote": "https://peer.modal.run",
    }


@pytest.mark.asyncio
async def test_an_agent_installed_on_a_local_sandboxs_host_gets_peer_urls_as_they_are(monkeypatch):
    sent = _record_posts(monkeypatch)
    context = _peering_context("local")
    context.deployed_agents[0].on_host = True

    await PeerAgentsTaskStep(id="peers", version=None, peerings=[
        {"source_agent_name": "source", "peer_agent_names": ["local", "human"]}]).execute(context)

    assert [peer["url"] for peer in json.loads(sent[0].content)["peers"]] == ["http://127.0.0.1:41001", _HUMAN]


@pytest.mark.asyncio
async def test_a_remote_agent_gets_peer_urls_from_beyond_this_machine_as_they_are(monkeypatch):
    urls = await _peer_urls(monkeypatch, "modal")

    assert (urls["human"], urls["remote"]) == (_HUMAN, "https://peer.modal.run")


@pytest.mark.asyncio
@pytest.mark.parametrize("sandbox_type, gateway_url, expected", [
    ("local", "http://127.0.0.1:41500", "http://host.docker.internal:41500"),
    ("modal_vm", "https://gw.modal.host", "https://gw.modal.host"),
])
async def test_a_cli_gets_the_gateway_url_its_agent_reaches(
    monkeypatch, local_stores, tmp_path, sandbox_type, gateway_url, expected,
):
    (tmp_path / "cli").mkdir()
    (tmp_path / "cli" / "run.sh").write_text("#!/bin/sh\n")
    cli = CliArtifact.put("slack-cli", command_name="slack", entrypoint="run.sh", cli_dir=tmp_path / "cli")
    installed = []

    async def install_cli(deployed, cli_artifact, gateway_url):
        installed.append(gateway_url)
        return "/opt/cli/slack/run.sh"

    monkeypatch.setattr(A2AAgent, "install_cli", staticmethod(install_cli))
    monkeypatch.setattr(agent_store, "get_a2a_agent_instance_store",
                        lambda: types.SimpleNamespace(get=lambda instance_id: object()))
    context = TaskStepContext(
        deployed_envs=[DeployedGatewayEnv(env_id="env-x", env_version=1, gateway_url=gateway_url, mcp_url="",
                                          sandbox_id="sb-env", sandbox_type=sandbox_type)],
        deployed_agents=[DeployedAgent(agent_name="solver", api_url="http://agent", sandbox_id="sb-agent",
                                       sandbox_type=sandbox_type, instance_id="inst-1")],
    )

    await LoadArtifactTaskStep(id="cli", version=None, env_id="env-x", artifact_id=cli.id,
                               artifact_version=cli.version, agent_name="solver").execute(context)

    assert installed == [expected]
